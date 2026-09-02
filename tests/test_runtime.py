import pytest

from hexagon_kit.catalog import get_spec
from hexagon_kit.runtime import ModelPool, PoolBudgetExceeded, reset_process_pool


def _seed_slot(tmp_path, slot: str):
    spec = get_spec(slot)
    dest = tmp_path / spec.slot
    dest.mkdir(parents=True, exist_ok=True)
    for name in spec.expected_files:
        (dest / name).write_bytes(b"x")
    return dest


def test_acquire_reuses_instance(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    _seed_slot(tmp_path, "stt")
    created = []

    pool = ModelPool(max_ram_mb=1000, cache_dir=tmp_path)
    pool.register("stt", lambda path: created.append(path) or object())

    a = pool.acquire("stt", force=True)
    b = pool.acquire("stt", force=True)
    assert a is b
    assert len(created) == 1
    assert pool.status()[0]["refs"] == 2
    pool.release("stt")
    pool.release("stt")
    assert pool.status()[0]["refs"] == 0


def test_evicts_unused_to_fit_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    _seed_slot(tmp_path, "stt")
    _seed_slot(tmp_path, "tts")
    unloaded = []

    pool = ModelPool(max_ram_mb=200, cache_dir=tmp_path)
    pool.register("stt", lambda path: "whisper", unloader=lambda inst: unloaded.append(inst), ram_mb=150)
    pool.register("tts", lambda path: "kokoro", ram_mb=150)

    pool.acquire("stt", force=True)
    pool.release("stt")
    got = pool.acquire("tts", force=True)
    assert got == "kokoro"
    assert unloaded == ["whisper"]
    slots = {row["slot"] for row in pool.status()}
    assert slots == {"tts"}


def test_in_use_model_blocks_over_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    _seed_slot(tmp_path, "stt")
    _seed_slot(tmp_path, "tts")

    pool = ModelPool(max_ram_mb=200, cache_dir=tmp_path)
    pool.register("stt", lambda path: "whisper", ram_mb=150)
    pool.register("tts", lambda path: "kokoro", ram_mb=150)

    pool.acquire("stt", force=True)
    with pytest.raises(PoolBudgetExceeded):
        pool.acquire("tts", force=True)


def test_reset_process_pool():
    reset_process_pool()


def test_acquire_blocks_when_preflight_fails(tmp_path, monkeypatch):
    from hexagon_kit.preflight import PreflightBlocked, PreflightResult

    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    _seed_slot(tmp_path, "stt")
    pool = ModelPool(max_ram_mb=1000, cache_dir=tmp_path)
    pool.register("stt", lambda path: "whisper")
    monkeypatch.setattr(
        "hexagon_kit.runtime.preflight",
        lambda model_id: PreflightResult(
            ok=False,
            ram_fit="unsafe",
            disk_ok=True,
            can_force=True,
            message="too little RAM",
        ),
    )
    with pytest.raises(PreflightBlocked):
        pool.acquire("stt")
    assert pool.acquire("stt", force=True) == "whisper"

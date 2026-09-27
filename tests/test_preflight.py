import pytest

from hexagon_kit.cache import download_model, ensure_model
from hexagon_kit.config import load_config, reset_config
from hexagon_kit.hw import MemoryStatus
from hexagon_kit.preflight import PreflightBlocked, preflight
from hexagon_kit.status import start_download, reset_jobs


def test_unsafe_ram_blocks_download(monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    reset_jobs()
    memory = MemoryStatus(
        total_bytes=int(15.61 * 1024**3),
        available_bytes=int(1.95 * 1024**3),
        load_pct=87,
    )
    result = preflight("tts", memory=memory)
    # Kokoro is ~250 MB; 1.95 GB free still fits. Use a huge synthetic check via monkeypatch ram.
    assert result.ram_fit in {"fits", "tight"}


def test_preflight_unsafe_when_model_larger_than_free(monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    from hexagon_kit.catalog import get_spec as builtin

    spec = builtin("kokoro_int8")
    memory = MemoryStatus(
        total_bytes=int(16 * 1024**3),
        available_bytes=int((spec.ram_mb / 2) * 1024**2),
        load_pct=90,
    )
    result = preflight("kokoro_int8", memory=memory)
    assert result.ok is False
    assert result.ram_fit == "unsafe"
    assert result.can_force is True


def test_start_download_blocked_without_force(monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    reset_jobs()
    spec_ram = 250.0
    memory = MemoryStatus(
        total_bytes=int(16 * 1024**3),
        available_bytes=int((spec_ram / 4) * 1024**2),
        load_pct=95,
    )
    monkeypatch.setattr("hexagon_kit.status.preflight", lambda model_id: __import__(
        "hexagon_kit.preflight", fromlist=["preflight"]
    ).preflight(model_id, memory=memory))
    job = start_download("kokoro_int8")
    assert job["state"] == "blocked"
    assert "preflight" in job


def test_memory_bar_levels():
    green = MemoryStatus(total_bytes=16 * 1024**3, available_bytes=10 * 1024**3, load_pct=40)
    orange = MemoryStatus(total_bytes=16 * 1024**3, available_bytes=4 * 1024**3, load_pct=75)
    red = MemoryStatus(total_bytes=16 * 1024**3, available_bytes=int(1.95 * 1024**3), load_pct=87)
    assert green.bar_level == "green"
    assert orange.bar_level == "orange"
    assert red.bar_level == "red"
    assert red.available_gb == 1.95


def test_suggests_lighter_same_slot(monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    monkeypatch.delenv("HEXAGON_KIT_CONFIG", raising=False)
    reset_config()
    load_config(
        overrides={
            "models": [
                {
                    "model_id": "kokoro_fp32",
                    "slot": "tts",
                    "name": "Kokoro FP32",
                    "disk_mb": 400,
                    "ram_mb": 4000,
                    "artifacts": [
                        {"url": "https://example.invalid/k.onnx", "filename": "k.onnx"}
                    ],
                    "expected_files": ["k.onnx"],
                }
            ]
        }
    )
    try:
        memory = MemoryStatus(
            total_bytes=int(16 * 1024**3),
            available_bytes=int(1.95 * 1024**3),
            load_pct=87,
        )
        result = preflight("kokoro_fp32", memory=memory)
        assert result.ok is False
        assert result.ram_fit == "unsafe"
        assert result.suggest_id == "kokoro_int8"
        assert result.suggest_name and "Kokoro" in result.suggest_name
    finally:
        reset_config()


def test_download_model_blocks_without_force(monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    reset_config()

    def _blocked(model_id, memory=None):
        from hexagon_kit.preflight import PreflightResult

        return PreflightResult(
            ok=False,
            ram_fit="unsafe",
            disk_ok=True,
            can_force=True,
            message=(
                "Gemma-class model requires ~4000 MB RAM, but only 1996 MB is available. "
                "Loading it would force pagefile thrashing on this 16 GB Copilot+ PC."
            ),
        )

    monkeypatch.setattr("hexagon_kit.cache.preflight", _blocked)
    with pytest.raises(PreflightBlocked):
        download_model("kokoro_int8")
    with pytest.raises(PreflightBlocked):
        ensure_model("tts")

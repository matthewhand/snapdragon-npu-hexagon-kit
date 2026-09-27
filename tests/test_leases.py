import os

import pytest

from hexagon_kit.leases import (
    ModelInUse,
    Lease,
    drop_lease,
    holder_for,
    list_leases,
    pid_alive,
    take_lease,
)
from hexagon_kit.preflight import preflight
from hexagon_kit.runtime import ModelPool, reset_process_pool
from hexagon_kit.cache import delete_model
from hexagon_kit.catalog import get_spec


def test_pid_alive_self():
    assert pid_alive(os.getpid()) is True
    assert pid_alive(0) is False


def test_take_and_drop_lease(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    lease = take_lease("stt", ram_mb=150, model_id="whisper_tiny_int8", cache_dir=tmp_path)
    assert lease.pid == os.getpid()
    assert holder_for("stt", tmp_path).pid == os.getpid()
    drop_lease("stt", cache_dir=tmp_path)
    assert holder_for("stt", tmp_path) is None


def test_exclusive_lease_blocks_other_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    take_lease("stt", ram_mb=150, model_id="whisper_tiny_int8", cache_dir=tmp_path)
    fake = Lease(
        slot="stt",
        pid=os.getpid(),
        exe="persona.exe",
        ram_mb=150,
        model_id="whisper_tiny_int8",
        acquired_at=0.0,
    )
    # Same pid refreshes rather than blocking.
    again = take_lease("stt", ram_mb=150, model_id="whisper_tiny_int8", cache_dir=tmp_path)
    assert again.pid == fake.pid
    drop_lease("stt", cache_dir=tmp_path)


def test_preflight_acquire_reports_peer(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    monkeypatch.setenv("HEXAGON_KIT_MAX_RAM_MB", "3500")
    from hexagon_kit.config import reset_config

    reset_config()
    take_lease("stt", ram_mb=150, model_id="whisper_tiny_int8", cache_dir=tmp_path)
    # Same process is not a peer holder.
    result = preflight("stt", for_acquire=True)
    assert result.held_by is None
    drop_lease("stt", cache_dir=tmp_path)


def test_delete_refuses_live_lease(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    spec = get_spec("stt")
    dest = tmp_path / spec.slot
    dest.mkdir()
    for name in spec.expected_files:
        (dest / name).write_bytes(b"x")
    take_lease("stt", ram_mb=150, model_id=spec.model_id, cache_dir=tmp_path)
    with pytest.raises(ModelInUse):
        delete_model("stt", cache_dir=tmp_path)
    assert (dest / spec.expected_files[0]).is_file()
    drop_lease("stt", cache_dir=tmp_path)
    delete_model("stt", cache_dir=tmp_path)
    assert dest.exists() is False


def test_pool_acquire_writes_lease(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    spec = get_spec("stt")
    dest = tmp_path / spec.slot
    dest.mkdir()
    for name in spec.expected_files:
        (dest / name).write_bytes(b"x")
    reset_process_pool()
    pool = ModelPool(max_ram_mb=1000, cache_dir=tmp_path)
    pool.register("stt", lambda path: "whisper")
    pool.acquire("stt", force=True)
    held = holder_for("stt", tmp_path)
    assert held is not None
    assert held.pid == os.getpid()
    pool.unload("stt", force=True)
    assert holder_for("stt", tmp_path) is None
    reset_process_pool()


def test_preflight_acquire_blocks_foreign_holder(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    from hexagon_kit.config import reset_config
    from hexagon_kit.leases import Lease

    reset_config()
    fake = Lease(
        slot="stt",
        pid=99999,
        exe="persona.exe",
        ram_mb=150,
        model_id="whisper_tiny_int8",
        acquired_at=0.0,
    )
    monkeypatch.setattr("hexagon_kit.leases.holder_for", lambda slot, cache_dir=None: fake)
    monkeypatch.setattr(
        "hexagon_kit.leases.peer_ram_mb",
        lambda cache_dir=None, exclude_pid=None: 150.0,
    )
    blocked = preflight("stt", for_acquire=True)
    assert blocked.ok is False
    assert blocked.can_force is True
    assert blocked.held_by["exe"] == "persona.exe"
    assert "persona.exe" in blocked.message
    download = preflight("stt", for_acquire=False)
    assert download.held_by["exe"] == "persona.exe"
    assert download.ok is True or download.ram_fit in {"fits", "tight", "unsafe"}


def test_list_leases_reaps_dead_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    from hexagon_kit.leases import _leases_path, _save

    _save(
        _leases_path(tmp_path),
        {
            "version": 1,
            "slots": {
                "stt": {
                    "pid": 1,
                    "exe": "dead.exe",
                    "ram_mb": 150,
                    "model_id": "whisper_tiny_int8",
                    "acquired_at": 0,
                }
            },
        },
    )
    assert list_leases(tmp_path) == []

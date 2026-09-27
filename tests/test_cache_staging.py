import os

import pytest

from hexagon_kit.cache import download_model, is_installed, peer_download
from hexagon_kit.catalog import get_spec

PEER_PID = 424242
DEAD_PID = 434343


def _alive_only(*pids):
    live = set(pids) | {os.getpid()}
    return lambda pid: pid in live


def _no_preflight(monkeypatch):
    monkeypatch.setattr("hexagon_kit.cache._require_preflight", lambda model_id, force=False: None)


def test_interrupt_removes_staging(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    _no_preflight(monkeypatch)

    def interrupted(artifact, staging, progress):
        raise KeyboardInterrupt

    monkeypatch.setattr("hexagon_kit.cache._fetch_artifact", interrupted)
    with pytest.raises(KeyboardInterrupt):
        download_model("tts", cache_dir=tmp_path)
    assert not list(tmp_path.glob(".tts-staging-*"))
    assert is_installed("tts", tmp_path) is False


def test_dead_staging_is_reaped_live_peer_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    _no_preflight(monkeypatch)
    monkeypatch.setattr("hexagon_kit.cache.pid_alive", _alive_only(PEER_PID))
    dead = tmp_path / f".tts-staging-{DEAD_PID}"
    live = tmp_path / f".tts-staging-{PEER_PID}"
    dead.mkdir()
    live.mkdir()

    def fake_fetch(artifact, staging, progress):
        path = staging / artifact.filename
        path.write_bytes(b"x")
        return path

    monkeypatch.setattr("hexagon_kit.cache._fetch_artifact", fake_fetch)
    download_model("tts", cache_dir=tmp_path)
    assert not dead.exists()
    assert live.exists()
    assert is_installed("tts", tmp_path)


def test_peer_download_reports_live_other_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    monkeypatch.setattr("hexagon_kit.cache.pid_alive", _alive_only(PEER_PID))
    assert peer_download("stt") is None

    (tmp_path / f".stt-staging-{DEAD_PID}").mkdir()
    (tmp_path / f".stt-staging-{os.getpid()}").mkdir()
    assert peer_download("stt") is None

    staging = tmp_path / f".stt-staging-{PEER_PID}" / "_dl"
    staging.mkdir(parents=True)
    (staging / "part.bin").write_bytes(b"x" * 2048)
    peer = peer_download("stt")
    assert peer is not None
    assert peer["pid"] == PEER_PID
    assert peer["bytes"] == 2048
    assert 0 <= peer["estimatedPct"] < 100
    assert peer_download("tts") is None


def test_peer_download_pct_capped_below_100(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    monkeypatch.setattr("hexagon_kit.cache.pid_alive", _alive_only(PEER_PID))
    spec = get_spec("vision")
    staging = tmp_path / f".vision-staging-{PEER_PID}"
    staging.mkdir()
    (staging / "big.bin").write_bytes(b"x" * int(spec.disk_mb * 1024 * 1024 * 1.2))
    assert peer_download("vision")["estimatedPct"] == 99


def test_vendor_status_does_not_import_modules(monkeypatch):
    import builtins

    from hexagon_kit.hub import vendor_status

    real_import = builtins.__import__
    imported: list[str] = []

    def spy(name, *args, **kwargs):
        if name.split(".")[0] in {"qai_hub_models", "qai_hub", "qai_hub_apps"}:
            imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", spy)
    status = vendor_status()
    assert "modelsSdk" in status
    assert not [name for name in imported if name != "qai_hub_models_cli"]

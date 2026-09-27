import json
import os

import pytest

from hexagon_kit.cli import main
from hexagon_kit.config import reset_config
from hexagon_kit.status import reset_jobs


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path / "models"))
    monkeypatch.setenv("HEXAGON_KIT_CONFIG", str(tmp_path / "cfg" / "config.json"))
    monkeypatch.delenv("HEXAGON_KIT_MAX_RAM_MB", raising=False)
    reset_config()
    reset_jobs()
    yield tmp_path
    reset_config()


def test_status_text_summary(env, capsys):
    assert main(["status", "--text"]) == 0
    out = capsys.readouterr().out
    assert "SLOT" in out and "ACTIONS" in out
    for slot in ("stt", "tts", "llm", "vision"):
        assert f"\n{slot} " in out
    assert "Loaded by: nobody" in out


def test_models_status_matches_snapshot_cards(env, capsys):
    assert main(["models", "status"]) == 0
    cards = json.loads(capsys.readouterr().out)
    assert {c["slot"] for c in cards} >= {"stt", "tts", "llm", "vision"}
    assert main(["models", "status", "llm"]) == 0
    card = json.loads(capsys.readouterr().out)
    assert card["id"] == "smollm2_135m_int8"
    assert card["chatCapable"] is True
    assert card["statusLabel"] == "Downloadable"


def test_unknown_model_error_is_readable(env, capsys):
    assert main(["models", "status", "nope"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: Unknown model or slot 'nope'")


def test_config_set_unset_and_settings(env, capsys):
    assert main(["config", "set", "max_ram_mb", "12"]) == 2
    assert "Must be at least" in json.loads(capsys.readouterr().err)["errors"]["max_ram_mb"]

    assert main(["config", "set", "max_ram_mb", "4096"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    saved = json.loads((env / "cfg" / "config.json").read_text(encoding="utf-8"))
    assert saved == {"max_ram_mb": 4096}

    assert main(["config", "settings"]) == 0
    snap = json.loads(capsys.readouterr().out)
    ram = next(f for f in snap["fields"] if f["key"] == "max_ram_mb")
    assert ram["value"] == 4096 and ram["source"] == "file"

    assert main(["config", "unset", "max_ram_mb"]) == 0
    capsys.readouterr()
    assert json.loads((env / "cfg" / "config.json").read_text(encoding="utf-8")) == {}


def test_config_set_warns_when_env_locked(env, capsys):
    assert main(["config", "set", "cache_dir", str(env / "elsewhere")]) == 0
    assert "HEXAGON_KIT_CACHE" in capsys.readouterr().err


def test_keyboard_interrupt_exits_130(env, capsys, monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("hexagon_kit.cli.ui_snapshot", interrupted)
    assert main(["status"]) == 130
    assert "cancelled" in capsys.readouterr().err


def test_async_download_spawns_a_detached_worker(env, capsys, monkeypatch):
    spawned: dict = {}

    def fake_spawn(argv):
        spawned["argv"] = argv
        return 515151

    monkeypatch.setattr("hexagon_kit.cli.spawn_detached", fake_spawn)
    assert main(["models", "download", "tts", "--async", "--force"]) == 0
    job = json.loads(capsys.readouterr().out)
    assert job["state"] == "downloading"
    assert job["pid"] == 515151
    assert "--detach-worker" in spawned["argv"]
    assert "kokoro_int8" in spawned["argv"]
    stored = json.loads((env / "models" / "jobs" / "kokoro_int8.json").read_text(encoding="utf-8"))
    assert stored["pid"] == 515151


def test_async_download_reports_spawn_failure(env, capsys, monkeypatch):
    def boom(_argv):
        raise OSError("spawn failed")

    monkeypatch.setattr("hexagon_kit.cli.spawn_detached", boom)
    assert main(["models", "download", "tts", "--async", "--force"]) == 1
    failed = json.loads(capsys.readouterr().err)
    assert failed["state"] == "failed"
    assert "detached" in failed["error"]


def test_detach_worker_runs_download_to_completion(env, monkeypatch):
    def ok(artifact, staging, progress):
        path = staging / artifact.filename
        path.write_bytes(b"x")
        if progress:
            progress(artifact.filename, 1, 1)
        return path

    monkeypatch.setattr("hexagon_kit.cache._fetch_artifact", ok)
    assert main(["models", "download", "tts", "--detach-worker", "--force"]) == 0
    from hexagon_kit.cache import is_installed

    assert is_installed("kokoro_int8")


def test_json_lines_progress(env, capsys, monkeypatch):
    def fake_download(model, progress=None, force=False, **kwargs):
        if progress:
            progress("voices.bin", 5, 10)
        dest = env / "models" / "tts"
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    monkeypatch.setattr("hexagon_kit.cache.download_model", fake_download)
    assert main(["models", "download", "tts", "--json-lines", "--force"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[0]["file"] == "voices.bin"
    assert lines[0]["pct"] == 50
    assert lines[-1]["state"] == "ready"


def test_jobs_cli_lists_a_shared_job(env, capsys):
    payload = {
        "id": "kokoro_int8",
        "slot": "tts",
        "role": "owner",
        "state": "ready",
        "pct": 100,
        "pid": os.getpid(),
        "updatedAt": 1,
    }
    jobs = env / "models" / "jobs"
    jobs.mkdir(parents=True)
    (jobs / "kokoro_int8.json").write_text(json.dumps(payload), encoding="utf-8")
    assert main(["jobs", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["id"] == "kokoro_int8"
    assert main(["jobs", "status", "tts"]) == 0
    assert json.loads(capsys.readouterr().out)["pct"] == 100


def test_spawn_detached_outlives_the_caller(tmp_path):
    import os
    import subprocess
    import sys
    import time

    from hexagon_kit.cli import spawn_detached
    from hexagon_kit.leases import pid_alive

    marker = tmp_path / "alive.txt"
    code = f"import pathlib, time; pathlib.Path({str(marker)!r}).write_text('ok'); time.sleep(30)"
    pid = spawn_detached([sys.executable, "-c", code])
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not marker.is_file():
            time.sleep(0.05)
        assert marker.is_file()
        assert pid_alive(pid)
    finally:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            try:
                os.kill(pid, 15)
            except OSError:
                pass

import json

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

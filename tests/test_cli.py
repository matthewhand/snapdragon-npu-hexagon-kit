import json

from hexagon_kit.cli import main


def test_status_cli(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    assert main(["status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "models" in payload
    assert "storage" in payload


def test_hw_cli(capsys):
    assert main(["hw"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "preferred_provider" in payload
    assert "is_snapdragon" in payload


def test_config_show_cli(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path / "models"))
    monkeypatch.delenv("HEXAGON_KIT_CONFIG", raising=False)
    assert main(["config", "show"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cache_dir"].endswith("models") or "models" in payload["cache_dir"]
    assert "models" in payload


def test_models_cache_cli(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path / "models"))
    assert main(["models", "cache"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "models")


def test_models_list_cli(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    assert main(["models", "list"]) == 0
    rows = json.loads(capsys.readouterr().out)
    ids = {row["id"] for row in rows}
    assert ids == {"whisper_tiny_int8", "kokoro_int8"}
    assert all(row["installed"] is False for row in rows)


def test_preflight_cli(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    code = main(["preflight", "tts"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ramFit"] in {"fits", "tight", "unsafe"}
    assert "ok" in payload
    assert code in {0, 2}


def test_models_path_missing(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    assert main(["models", "path", "stt"]) == 1
    err = capsys.readouterr().err
    assert "not installed" in err.lower() or "hexagon models download" in err

import json

import pytest

import hexagon_kit
from hexagon_kit.cli import main


def test_status_cli(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    assert main(["status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "models" in payload
    assert "storage" in payload


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert hexagon_kit.__version__ in out
    assert "0.3.0" in out


def test_hw_cli(capsys):
    assert main(["hw"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "preferred_provider" in payload
    assert "is_snapdragon" in payload
    assert payload["ep_kind"] in {"qnn", "directml", "cpu"}
    assert "qnn_package" in payload
    assert "ort_package" in payload
    if payload["ep_kind"] != "qnn":
        assert "QNNExecutionProvider" not in payload.get("providers", [])


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
    slots = {row["slot"] for row in rows}
    assert {
        "whisper_tiny_int8",
        "kokoro_int8",
        "smollm2_135m_int8",
        "rapidocr_ppocrv4_mobile",
    } <= ids
    assert {"stt", "tts", "llm", "vision"} <= slots
    assert all(row["installed"] is False for row in rows)
    by_id = {row["id"]: row for row in rows}
    assert by_id["whisper_tiny_int8"]["origin"] == "github"
    assert by_id["kokoro_int8"]["origin"] == "github"
    assert by_id["smollm2_135m_int8"]["origin"] == "huggingface"
    assert by_id["rapidocr_ppocrv4_mobile"]["origin"] == "huggingface"
    assert by_id["smollm2_135m_int8"]["ram_mb"] < 1024
    assert by_id["rapidocr_ppocrv4_mobile"]["expected_files"]
    assert any("k2-fsa/sherpa-onnx" in url for row in rows for url in row["urls"])
    assert any("thewh1teagle/kokoro-onnx" in url for row in rows for url in row["urls"])


def test_preflight_cli(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    code = main(["preflight", "tts"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ramFit"] in {"fits", "tight", "unsafe"}
    assert "ok" in payload
    assert code in {0, 2}


def test_preflight_cli_llm_and_vision_shape(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    for slot in ("llm", "vision"):
        code = main(["preflight", slot])
        payload = json.loads(capsys.readouterr().out)
        assert payload["ramFit"] in {"fits", "tight", "unsafe"}
        assert "requiredRamMb" in payload
        assert payload["requiredRamMb"] < 1024
        assert payload["requiredDiskMb"] > 0
        assert "canForce" in payload
        assert code in {0, 2}


def test_models_path_missing(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    assert main(["models", "path", "stt"]) == 1
    err = capsys.readouterr().err
    assert "not installed" in err.lower() or "hexagon models download" in err

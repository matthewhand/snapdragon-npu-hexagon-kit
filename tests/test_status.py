from hexagon_kit.status import model_card, reset_jobs, storage_report, ui_snapshot


def test_ui_snapshot_shape(monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    reset_jobs()
    snap = ui_snapshot()
    assert "hardware" in snap
    assert "storage" in snap
    assert "models" in snap
    assert "pool" in snap
    assert "config" in snap
    ids = {card["id"] for card in snap["models"]}
    assert "whisper_tiny_int8" in ids
    assert "kokoro_int8" in ids
    whisper = next(c for c in snap["models"] if c["id"] == "whisper_tiny_int8")
    assert whisper["installed"] is False
    assert "download" in whisper["actions"]
    assert whisper["slot"] == "stt"
    assert whisper["ramFit"] in {"fits", "tight", "unsafe"}
    assert "preflight" in whisper


def test_model_card_ready_when_files_present(monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    reset_jobs()
    stt = tmp_path / "stt"
    stt.mkdir()
    (stt / "tiny.en-encoder.int8.onnx").write_bytes(b"x")
    (stt / "tiny.en-decoder.int8.onnx").write_bytes(b"x")
    (stt / "tiny.en-tokens.txt").write_text("x", encoding="utf-8")
    card = model_card("stt")
    assert card["installed"] is True
    assert card["actions"] == ["delete"]
    assert card["status"] == "ready"


def test_storage_report_uses_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    report = storage_report()
    assert report["cacheDir"] == str(tmp_path)
    assert "diskFreeGb" in report

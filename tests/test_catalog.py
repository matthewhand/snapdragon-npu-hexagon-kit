from hexagon_kit.catalog import get_spec, list_specs


def test_catalog_has_stt_and_tts_slots():
    specs = list_specs()
    slots = {s.slot for s in specs}
    ids = {s.model_id for s in specs}
    assert {"stt", "tts", "llm", "vision"} <= slots
    assert "whisper_tiny_int8" in ids
    assert "kokoro_int8" in ids
    assert get_spec("llm").model_id == "smollm2_135m_int8"
    assert get_spec("vision").model_id == "rapidocr_ppocrv4_mobile"


def test_llm_and_vision_are_copilot_plus_safe():
    llm = get_spec("llm")
    vision = get_spec("vision")
    assert llm.slot == "llm"
    assert vision.slot == "vision"
    # First-gen 16 GB Snapdragon X: SmolLM-class / small OCR. No 8 GB defaults.
    assert llm.ram_mb < 1024
    assert vision.ram_mb < 1024
    assert llm.disk_mb < 512
    assert vision.disk_mb < 64
    assert llm.expected_files
    assert vision.expected_files
    for spec in (llm, vision):
        for art in spec.artifacts:
            assert art.sha256 and len(art.sha256) == 64, art.filename


def test_get_spec_by_id_or_slot():
    assert get_spec("stt").model_id == "whisper_tiny_int8"
    assert get_spec("TTS").slot == "tts"
    assert get_spec("kokoro_int8").expected_files[0].endswith(".onnx")


def test_unknown_spec_lists_known():
    try:
        get_spec("gemma")
        assert False, "expected KeyError"
    except KeyError as exc:
        assert "whisper_tiny_int8" in str(exc)


def test_builtin_artifacts_have_sha256():
    for spec in list_specs():
        assert spec.artifacts, spec.model_id
        for art in spec.artifacts:
            assert art.sha256 and len(art.sha256) == 64, art.filename


def test_builtin_disk_mb_covers_upstream_archives():
    # Preflight uses disk_mb * 1.15. Underestimating would allow a download
    # that does not fit. Whisper tiny.en tar.bz2 is ~118 MB; Kokoro int8 +
    # voices is ~120 MB. Do not treat "folder > 1 MB" as installed.
    whisper = get_spec("whisper_tiny_int8")
    kokoro = get_spec("kokoro_int8")
    assert whisper.disk_mb >= 118
    assert kokoro.disk_mb >= 120
    assert whisper.ram_mb > 0 and kokoro.ram_mb > 0
    assert whisper.expected_files and kokoro.expected_files
    llm = get_spec("smollm2_135m_int8")
    vision = get_spec("rapidocr_ppocrv4_mobile")
    assert llm.disk_mb >= 132
    assert vision.disk_mb >= 15

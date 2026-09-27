from hexagon_kit.catalog import get_spec, list_specs


def test_catalog_has_stt_tts_llm_and_vision_slots():
    specs = list_specs()
    slots = {s.slot for s in specs}
    ids = {s.model_id for s in specs}
    assert slots == {"stt", "tts", "llm", "vision"}
    assert "whisper_tiny_int8" in ids
    assert "kokoro_int8" in ids
    assert "smollm2_135m_q4" in ids
    assert "ppocrv4_det_mobile" in ids


def test_get_spec_by_id_or_slot():
    assert get_spec("stt").model_id == "whisper_tiny_int8"
    assert get_spec("TTS").slot == "tts"
    assert get_spec("kokoro_int8").expected_files[0].endswith(".onnx")
    assert get_spec("llm").model_id == "smollm2_135m_q4"
    assert get_spec("VISION").slot == "vision"


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


def test_llm_and_vision_are_copilot_plus_safe():
    llm = get_spec("llm")
    vision = get_spec("vision")
    assert llm.model_id == "smollm2_135m_q4"
    assert vision.model_id == "ppocrv4_det_mobile"
    assert llm.slot == "llm"
    assert vision.slot == "vision"
    assert llm.disk_mb >= 176
    assert llm.disk_mb < 500
    assert vision.disk_mb >= 5
    assert vision.disk_mb < 50
    # First-gen 16 GB baseline: no 8 GB default models.
    for spec in (llm, vision):
        assert spec.ram_mb < 2048, spec.model_id
        assert spec.expected_files
        for art in spec.artifacts:
            assert art.sha256 and len(art.sha256) == 64, art.filename
            assert art.url.startswith("https://")

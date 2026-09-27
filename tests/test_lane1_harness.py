"""Lane-1 CI harness — shared mocks so apps do not invent fake EP probes."""

from hexagon_kit.cache import is_installed, resolve
from hexagon_kit.config import get_spec, reset_config
from hexagon_kit.hw import (
    CPU_PROVIDER,
    DML_PROVIDER,
    EP_KIND_CPU,
    EP_KIND_DIRECTML,
    EP_KIND_QNN,
    QNN_PROVIDER,
)
from hexagon_kit.testing import (
    FIXTURE_CATALOG,
    LISTING_CPU,
    LISTING_DIRECTML,
    LISTING_QNN,
    assert_cold_honesty,
    classify_providers,
    detect_hardware,
    ensure_fixture,
    fixture_bytes,
    hexagon_qnn,
    install_fixture_catalog,
    provider_chain_for,
    stub_ensure_resolve,
)


def test_prefer_qnn_without_listed_qnn_stays_cpu():
    choice = classify_providers(LISTING_CPU, prefer=QNN_PROVIDER)
    assert choice.ep_kind == EP_KIND_CPU
    assert hexagon_qnn(choice) is False
    assert_cold_honesty(choice)

    probe = detect_hardware(LISTING_CPU, prefer=QNN_PROVIDER)
    assert probe.ep_kind == EP_KIND_CPU
    assert QNN_PROVIDER not in probe.providers
    assert hexagon_qnn(probe) is False
    assert_cold_honesty(probe)
    chain = provider_chain_for(LISTING_CPU, prefer=QNN_PROVIDER)
    assert chain == [CPU_PROVIDER]
    assert QNN_PROVIDER not in chain


def test_listed_qnn_sets_hexagon_qnn_true():
    choice = classify_providers(LISTING_QNN, is_snapdragon=True, is_arm64=True)
    assert choice.ep_kind == EP_KIND_QNN
    assert choice.preferred == QNN_PROVIDER
    assert hexagon_qnn(choice) is True

    probe = detect_hardware(LISTING_QNN)
    assert probe.ep_kind == EP_KIND_QNN
    assert QNN_PROVIDER in probe.providers
    assert probe.qnn_package is True
    assert hexagon_qnn(probe) is True
    assert probe.has_npu is True
    chain = provider_chain_for(LISTING_QNN)
    assert chain[0] == QNN_PROVIDER
    assert chain.index(DML_PROVIDER) < chain.index(CPU_PROVIDER)


def test_directml_listing_is_not_hexagon_qnn():
    choice = classify_providers(LISTING_DIRECTML, is_snapdragon=True)
    assert choice.ep_kind == EP_KIND_DIRECTML
    assert hexagon_qnn(choice) is False
    probe = detect_hardware(LISTING_DIRECTML)
    assert probe.ep_kind == EP_KIND_DIRECTML
    assert hexagon_qnn(probe) is False
    assert_cold_honesty(probe)


def test_cpu_ort_never_paints_hexagon():
    probe = detect_hardware(LISTING_CPU, qnn_package=False)
    assert probe.ep_kind == EP_KIND_CPU
    assert hexagon_qnn(probe) is False
    assert probe.qnn_package is False
    assert "Hexagon" not in (probe.provider_label or "")
    assert "QNN" not in (probe.provider_label or "")
    assert_cold_honesty(probe)


def test_listed_qnn_without_qnn_package_is_not_hexagon_qnn():
    probe = detect_hardware(LISTING_QNN, qnn_package=False)
    assert QNN_PROVIDER in probe.providers
    assert probe.ep_kind == EP_KIND_QNN
    assert probe.qnn_package is False
    assert hexagon_qnn(probe) is False


def test_fixture_ensure_path_works_offline(tmp_path, monkeypatch):
    install_fixture_catalog(tmp_path, monkeypatch=monkeypatch)
    try:
        detect_dir = ensure_fixture("detect", tmp_path)
        classify_dir = ensure_fixture("classify", tmp_path)
        vision_dir = ensure_fixture("vision", tmp_path)
        assert detect_dir == resolve("detect", tmp_path)
        assert classify_dir == resolve("classify", tmp_path)
        assert vision_dir == resolve("vision", tmp_path)
        assert (detect_dir / "detect.onnx").read_bytes() == fixture_bytes()
        assert (classify_dir / "classify.onnx").read_bytes() == fixture_bytes()
        for name in get_spec("vision").expected_files:
            assert (vision_dir / name).read_bytes() == fixture_bytes()
        assert is_installed("ci_detect", tmp_path)
        assert is_installed("ci_classify", tmp_path)
        assert get_spec("detect").disk_mb == 1
        assert get_spec("vision").ram_mb == 1
        for spec in FIXTURE_CATALOG:
            for art in spec.artifacts:
                assert not art.url.startswith("https://")
    finally:
        reset_config()


def test_stub_ensure_resolve_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    reset_config()
    try:
        path = stub_ensure_resolve(tmp_path, "llm")
        assert resolve("llm") == path
        from hexagon_kit import ensure_model

        assert ensure_model("llm") == path
        assert (path / "model_int8.onnx").is_file()
        assert (path / "tokenizer.json").is_file()
    finally:
        reset_config()


def test_harness_is_importable_without_hexagon_hw():
    import hexagon_kit.testing as harness

    assert harness.hexagon_qnn is hexagon_qnn
    assert harness.LISTING_QNN[0] == QNN_PROVIDER
    assert {spec.slot for spec in harness.FIXTURE_CATALOG} >= {
        "classify",
        "detect",
        "stt",
        "tts",
        "llm",
        "vision",
    }

from hexagon_kit.hw import (
    CPU_PROVIDER,
    DML_PROVIDER,
    EP_KIND_CPU,
    EP_KIND_DIRECTML,
    EP_KIND_QNN,
    HardwareProbe,
    QNN_PROVIDER,
    _honesty_notes,
    _is_snapdragon,
    _ort_package_name,
    choose_execution_provider,
    probe_hardware,
)


def test_snapdragon_detection():
    assert _is_snapdragon("Qualcomm Snapdragon X Elite", "ARM64") is True
    assert _is_snapdragon("Intel(R) Core(TM) i7", "AMD64") is False
    assert _is_snapdragon("X1E78100", "arm64") is True


def test_probe_hardware_shape():
    probe = probe_hardware()
    assert isinstance(probe, HardwareProbe)
    assert probe.platform
    assert probe.preferred_provider
    assert "CPUExecutionProvider" in probe.providers or probe.providers
    assert probe.ep_kind in {EP_KIND_QNN, EP_KIND_DIRECTML, EP_KIND_CPU}
    assert probe.qnn_package is False or probe.qnn_package is True
    data = probe.to_dict()
    assert "ram_gb" in data
    assert "is_snapdragon" in data
    assert "preferred_provider" in data
    assert "providers" in data
    assert "has_npu" in data
    assert data["ep_kind"] == probe.ep_kind
    if data.get("memory"):
        assert "availableGb" in data["memory"]
        assert "loadPct" in data["memory"]
        assert data["memory"]["barLevel"] in {"green", "orange", "red"}


def test_choose_provider_qnn_is_hexagon_not_cpu():
    choice = choose_execution_provider(
        [QNN_PROVIDER, CPU_PROVIDER],
        is_snapdragon=True,
        is_arm64=True,
    )
    assert choice.ep_kind == EP_KIND_QNN
    assert choice.preferred == QNN_PROVIDER
    assert "Hexagon" in choice.label
    assert "QNN" in choice.label
    assert choice.has_npu is True


def test_choose_provider_directml_is_not_qnn():
    choice = choose_execution_provider(
        [DML_PROVIDER, CPU_PROVIDER],
        is_snapdragon=True,
        is_arm64=True,
    )
    assert choice.ep_kind == EP_KIND_DIRECTML
    assert choice.preferred == DML_PROVIDER
    assert QNN_PROVIDER not in choice.preferred
    assert "QNN" not in choice.label
    assert "DirectML" in choice.label
    assert choice.has_npu is True  # Snapdragon + DML keeps existing has_npu


def test_choose_provider_cpu_ort_is_not_hexagon():
    # Typical `[ort]` / upstream onnxruntime listing.
    choice = choose_execution_provider(
        [CPU_PROVIDER],
        is_snapdragon=True,
        is_arm64=True,
    )
    assert choice.ep_kind == EP_KIND_CPU
    assert choice.preferred == CPU_PROVIDER
    assert choice.has_npu is False
    assert "Hexagon" not in choice.label
    assert "QNN" not in choice.label


def test_choose_provider_empty_defaults_to_cpu():
    choice = choose_execution_provider([])
    assert choice.ep_kind == EP_KIND_CPU
    assert choice.preferred == CPU_PROVIDER
    assert choice.has_npu is False


def test_prefer_override_does_not_rewrite_ep_kind():
    choice = choose_execution_provider(
        [CPU_PROVIDER],
        prefer_override=QNN_PROVIDER,
    )
    assert choice.preferred == QNN_PROVIDER
    assert choice.ep_kind == EP_KIND_CPU
    assert choice.has_npu is False


def test_cpu_ort_package_is_not_qnn():
    assert (
        _ort_package_name([CPU_PROVIDER], qnn_package=False, ort_importable=True)
        == "onnxruntime"
    )
    assert _ort_package_name([CPU_PROVIDER], qnn_package=False, ort_importable=False) is None
    assert _ort_package_name([DML_PROVIDER, CPU_PROVIDER], qnn_package=False) == (
        "onnxruntime-directml"
    )
    assert _ort_package_name([QNN_PROVIDER, CPU_PROVIDER], qnn_package=True) == (
        "onnxruntime_qnn"
    )


def test_honesty_notes_reject_cpu_ort_as_hexagon():
    notes = _honesty_notes(
        providers=[CPU_PROVIDER],
        ort_package="onnxruntime",
        qnn_package=False,
        ep_kind=EP_KIND_CPU,
    )
    blob = " ".join(notes)
    assert "CPU-oriented" in blob
    assert "onnxruntime_qnn" in blob
    assert "QNNExecutionProvider" in blob

    dml = _honesty_notes(
        providers=[DML_PROVIDER, CPU_PROVIDER],
        ort_package="onnxruntime-directml",
        qnn_package=False,
        ep_kind=EP_KIND_DIRECTML,
    )
    assert any("DirectML" in item and "QNN" in item for item in dml)


def test_probe_without_qnn_package_does_not_list_qnn(monkeypatch):
    monkeypatch.setattr("hexagon_kit.hw._onnx_providers", lambda: [CPU_PROVIDER])
    monkeypatch.setattr("hexagon_kit.hw._qnn_package_available", lambda: False)
    monkeypatch.setattr("hexagon_kit.hw._try_register_qnn", lambda htp: None)
    probe = probe_hardware()
    assert QNN_PROVIDER not in probe.providers
    assert probe.ep_kind == EP_KIND_CPU
    assert probe.qnn_package is False
    assert probe.preferred_provider == CPU_PROVIDER
    assert probe.has_npu is False


def test_probe_qnn_listing_sets_hexagon(monkeypatch):
    monkeypatch.setattr(
        "hexagon_kit.hw._onnx_providers",
        lambda: [QNN_PROVIDER, CPU_PROVIDER],
    )
    monkeypatch.setattr("hexagon_kit.hw._qnn_package_available", lambda: True)
    monkeypatch.setattr("hexagon_kit.hw._try_register_qnn", lambda htp: None)
    probe = probe_hardware()
    assert probe.ep_kind == EP_KIND_QNN
    assert probe.preferred_provider == QNN_PROVIDER
    assert probe.ort_package == "onnxruntime_qnn"
    assert probe.qnn_package is True
    assert probe.has_npu is True
    data = probe.to_dict()
    assert data["preferred_provider"] == QNN_PROVIDER
    assert data["ep_kind"] == "qnn"

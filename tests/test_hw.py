from hexagon_kit.hw import (
    CPU_EP,
    DML_EP,
    HardwareProbe,
    QNN_EP,
    _is_snapdragon,
    _ort_honesty_note,
    classify_providers,
    detect_ort_package,
    probe_hardware,
)


def test_snapdragon_detection():
    assert _is_snapdragon("Qualcomm Snapdragon X Elite", "ARM64") is True
    assert _is_snapdragon("Intel(R) Core(TM) i7", "AMD64") is False
    assert _is_snapdragon("X1E78100", "arm64") is True


def test_classify_providers_distinguishes_qnn_dml_cpu():
    qnn_pref, qnn_label, qnn_kind = classify_providers(
        [QNN_EP, DML_EP, CPU_EP],
        is_arm64=True,
        is_snapdragon=True,
    )
    assert qnn_kind == "qnn"
    assert qnn_pref == QNN_EP
    assert "QNN" in qnn_label

    dml_pref, dml_label, dml_kind = classify_providers(
        [DML_EP, CPU_EP],
        is_arm64=True,
        is_snapdragon=True,
    )
    assert dml_kind == "directml"
    assert dml_pref == DML_EP
    assert "DirectML" in dml_label
    assert dml_kind != "qnn"

    cpu_pref, cpu_label, cpu_kind = classify_providers(
        [CPU_EP],
        is_arm64=False,
        is_snapdragon=False,
    )
    assert cpu_kind == "cpu"
    assert cpu_pref == CPU_EP
    assert cpu_label == "CPU"


def test_cpu_ort_extra_is_not_hexagon_qnn():
    note = _ort_honesty_note("cpu", "onnxruntime", [CPU_EP])
    assert note is not None
    assert "[ort]" in note
    assert "onnxruntime_qnn" in note
    assert "not Qualcomm Hexagon QNN" in note
    assert _ort_honesty_note("qnn", "onnxruntime_qnn", [QNN_EP, CPU_EP]) is None


def test_probe_hardware_shape():
    probe = probe_hardware()
    assert isinstance(probe, HardwareProbe)
    assert probe.platform
    assert probe.preferred_provider
    assert CPU_EP in probe.providers or probe.providers
    data = probe.to_dict()
    assert "ram_gb" in data
    assert "is_snapdragon" in data
    assert "preferred_provider" in data
    assert data["provider_kind"] in {"qnn", "directml", "cpu"}
    assert data["ort_package"] in {"onnxruntime_qnn", "onnxruntime", "none"}
    assert data["hexagon_qnn"] is bool(data["has_qnn"])
    if QNN_EP not in probe.providers:
        assert data["hexagon_qnn"] is False
        assert data["provider_kind"] != "qnn"
    if data.get("memory"):
        assert "availableGb" in data["memory"]
        assert "loadPct" in data["memory"]
        assert data["memory"]["barLevel"] in {"green", "orange", "red"}


def test_detect_ort_package_without_qnn():
    package = detect_ort_package()
    assert package in {"onnxruntime_qnn", "onnxruntime", "none"}
    # CI / this tree does not ship onnxruntime_qnn.
    if package != "onnxruntime_qnn":
        probe = probe_hardware()
        assert probe.hexagon_qnn is False
        assert probe.has_qnn is False

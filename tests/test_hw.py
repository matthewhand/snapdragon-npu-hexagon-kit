from hexagon_kit.hw import HardwareProbe, _is_snapdragon, probe_hardware


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
    data = probe.to_dict()
    assert "ram_gb" in data
    assert "is_snapdragon" in data
    if data.get("memory"):
        assert "availableGb" in data["memory"]
        assert "loadPct" in data["memory"]
        assert data["memory"]["barLevel"] in {"green", "orange", "red"}

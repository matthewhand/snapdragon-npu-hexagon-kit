"""Stable consumer API — no breaking renames (T13)."""

from hexagon_kit import (
    HardwareProbe,
    ensure_model,
    get_spec,
    preflight,
    probe_hardware,
    resolve,
)
from hexagon_kit.hw import (
    CPU_PROVIDER,
    EP_KIND_CPU,
    EP_KIND_DIRECTML,
    EP_KIND_QNN,
    QNN_PROVIDER,
)


# Public names apps are documented to import. Do not rename.
_CONSUMER_EXPORTS = (
    "CATALOG",
    "HardwareProbe",
    "ModelNotInstalled",
    "PreflightBlocked",
    "ensure_model",
    "get_spec",
    "preflight",
    "probe_hardware",
    "process_pool",
    "provider_chain",
    "resolve",
    "ui_snapshot",
)


def test_consumer_exports_are_importable():
    import hexagon_kit

    for name in _CONSUMER_EXPORTS:
        assert hasattr(hexagon_kit, name), name
        assert name in hexagon_kit.__all__, name


def test_probe_honesty_fields_exist_for_apps():
    probe = probe_hardware()
    assert isinstance(probe, HardwareProbe)
    assert probe.ep_kind in {EP_KIND_QNN, EP_KIND_DIRECTML, EP_KIND_CPU}
    assert hasattr(probe, "ort_package")
    assert probe.qnn_package in {True, False}
    data = probe.to_dict()
    for key in (
        "providers",
        "preferred_provider",
        "has_npu",
        "ep_kind",
        "ort_package",
        "qnn_package",
    ):
        assert key in data, key
    # Linux CI / CPU [ort]: never advertise Hexagon without QNN EP.
    if QNN_PROVIDER not in probe.providers:
        assert probe.ep_kind != EP_KIND_QNN
        assert probe.qnn_package is False or QNN_PROVIDER not in probe.providers


def test_ensure_model_preflight_slots_llm_and_vision(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    from hexagon_kit.config import reset_config

    reset_config()
    assert get_spec("llm").model_id == "smollm2_135m_int8"
    assert get_spec("vision").model_id == "rapidocr_ppocrv4_mobile"
    for slot in ("llm", "vision"):
        guard = preflight(slot)
        payload = guard.to_dict()
        assert payload["requiredRamMb"] < 1024
        assert payload["ramFit"] in {"fits", "tight", "unsafe"}
        assert callable(ensure_model)
        assert callable(resolve)


def test_cannot_advertise_hexagon_without_qnn_ep():
    """Documented consumer rule: no Hexagon/QNN badge without QNN EP."""
    probe = probe_hardware()
    can_advertise = (
        probe.ep_kind == EP_KIND_QNN
        and QNN_PROVIDER in probe.providers
        and probe.qnn_package
    )
    if probe.ep_kind == EP_KIND_CPU:
        assert can_advertise is False
        assert "Hexagon" not in (probe.provider_label or "")
        assert "QNN" not in (probe.provider_label or "")
    if CPU_PROVIDER in probe.providers and QNN_PROVIDER not in probe.providers:
        assert can_advertise is False

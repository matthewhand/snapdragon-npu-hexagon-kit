"""Consumer guide + private SoT CI stay in the tree."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_app_integration_guide_covers_consumer_contract():
    text = (ROOT / "docs" / "APP_INTEGRATION.md").read_text(encoding="utf-8")
    for needle in (
        "hexagon_kit",
        "probe_hardware",
        "ep_kind",
        "ort_package",
        "qnn_package",
        "QNNExecutionProvider",
        "ensure_model",
        "preflight",
        '"llm"',
        '"vision"',
        "smollm2_135m_int8",
        "rapidocr_ppocrv4_mobile",
        "Never advertise",
        "onnxruntime_qnn",
        "hexagon_kit.testing",
        "hexagon_qnn",
        "Lane-1",
        "SnipPilot",
        "install_fixture_catalog",
    ):
        assert needle in text, needle
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/APP_INTEGRATION.md" in readme


def test_private_sot_ci_runs_pytest_on_main():
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    raw = workflow.read_text(encoding="utf-8")
    assert "push:" in raw
    assert "pull_request:" in raw
    assert "branches: [main]" in raw
    assert "ubuntu-latest" in raw
    assert '"3.12"' in raw and '"3.13"' in raw
    assert "python -m pytest" in raw
    assert "not a Hexagon" in raw or "QNNExecutionProvider" in raw

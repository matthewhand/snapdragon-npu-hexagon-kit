"""Live QNN / Hexagon checks. Skip on Linux CI without the QNN EP."""

import pytest

from hexagon_kit.hw import QNN_PROVIDER, probe_hardware


@pytest.mark.npu
def test_live_qnn_listing_is_honest():
    probe = probe_hardware()
    assert QNN_PROVIDER in probe.providers
    assert probe.ep_kind == "qnn"
    assert probe.qnn_package is True
    assert probe.has_npu is True

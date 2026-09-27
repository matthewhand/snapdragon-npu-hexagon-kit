"""Pytest hooks for hardware-agnostic CI.

Linux GitHub runners have no Hexagon QNN EP. Tests marked ``npu`` skip
unless ORT actually lists ``QNNExecutionProvider``.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "npu: requires a live Hexagon / QNNExecutionProvider; skipped on Linux CI",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    from hexagon_kit.hw import QNN_PROVIDER, probe_hardware

    try:
        probe = probe_hardware()
        has_qnn = QNN_PROVIDER in probe.providers and probe.ep_kind == "qnn"
    except Exception:
        has_qnn = False
    if has_qnn:
        return
    skip_npu = pytest.mark.skip(
        reason="No QNNExecutionProvider on this runner (Linux CI is not Hexagon)"
    )
    for item in items:
        if item.get_closest_marker("npu"):
            item.add_marker(skip_npu)

"""ONNX Runtime session factory using the probed execution-provider order."""

from __future__ import annotations

from pathlib import Path

from .hw import probe_hardware


def provider_chain(prefer: str | None = None) -> list[str]:
    probe = probe_hardware()
    preferred = prefer or probe.preferred_provider
    chain = [preferred]
    for name in ("QNNExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"):
        if name not in chain:
            chain.append(name)
    available = set(probe.providers)
    ordered = [name for name in chain if name in available or name == "CPUExecutionProvider"]
    return ordered or ["CPUExecutionProvider"]


def open_onnx(model_path: str | Path, prefer: str | None = None):
    """Create an InferenceSession. Requires the optional `ort` extra."""
    import onnxruntime as ort

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return ort.InferenceSession(str(path), providers=provider_chain(prefer))

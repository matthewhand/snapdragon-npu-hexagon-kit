"""ONNX Runtime session factory using the probed execution-provider order."""

from __future__ import annotations

from pathlib import Path

from .hw import CPU_PROVIDER, DML_PROVIDER, QNN_PROVIDER, probe_hardware


def provider_chain(prefer: str | None = None) -> list[str]:
    """Prefer QNN, then DirectML, then CPU — only names ORT actually listed.

    ``[ort]`` / ``onnxruntime`` is CPU-oriented. ``QNNExecutionProvider`` is
    included only when ``onnxruntime_qnn`` registered it.
    """
    probe = probe_hardware()
    preferred = prefer or probe.preferred_provider
    chain = [preferred]
    for name in (QNN_PROVIDER, DML_PROVIDER, CPU_PROVIDER):
        if name not in chain:
            chain.append(name)
    available = set(probe.providers)
    ordered = [name for name in chain if name in available or name == CPU_PROVIDER]
    return ordered or [CPU_PROVIDER]


def open_onnx(model_path: str | Path, prefer: str | None = None):
    """Create an InferenceSession.

    The optional ``[ort]`` extra installs CPU-oriented ``onnxruntime``.
    Hexagon QNN sessions need ``onnxruntime_qnn`` so ``provider_chain()``
    can list ``QNNExecutionProvider``.
    """
    import onnxruntime as ort

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return ort.InferenceSession(str(path), providers=provider_chain(prefer))

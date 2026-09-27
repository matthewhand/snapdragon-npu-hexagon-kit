"""Lane-1 CI harness for consumer apps (SnipPilot, Persona, audience).

Linux GitHub runners have no Hexagon. Import this module instead of
inventing a fake EP probe or downloading multi-GB catalog weights.

Typical pytest usage::

    from hexagon_kit.testing import (
        CPU_PROVIDER,
        QNN_PROVIDER,
        detect_hardware,
        hexagon_qnn,
        install_fixture_catalog,
        ensure_fixture,
    )

    def test_prefer_qnn_without_listing_stays_cpu():
        probe = detect_hardware([CPU_PROVIDER], prefer=QNN_PROVIDER)
        assert probe.ep_kind == "cpu"
        assert hexagon_qnn(probe) is False

    def test_listed_qnn_is_hexagon():
        probe = detect_hardware([QNN_PROVIDER, CPU_PROVIDER])
        assert hexagon_qnn(probe) is True

    def test_fixture_ensure_offline(tmp_path):
        install_fixture_catalog(tmp_path)
        path = ensure_fixture("detect", tmp_path)
        assert (path / "detect.onnx").is_file()

T5 honesty fields on ``HardwareProbe`` (``ep_kind``, ``ort_package``,
``qnn_package``, ``preferred_provider``, ``providers``, ``has_npu``) are
not renamed or removed. ``hexagon_qnn()`` is a shared boolean on top of
those fields: CPU / ``[ort]`` never paints Hexagon.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from .cache import COMPLETE_NAME, ensure_model, is_installed, resolve, slot_dir
from .catalog import Artifact, ModelSpec
from .config import get_spec, load_config, reset_config
from .hw import (
    CPU_PROVIDER,
    DML_PROVIDER,
    EP_KIND_CPU,
    EP_KIND_DIRECTML,
    EP_KIND_QNN,
    HardwareProbe,
    ProviderChoice,
    QNN_PROVIDER,
    choose_execution_provider,
    probe_hardware,
)
from .session import provider_chain

__all__ = [
    "CPU_PROVIDER",
    "DML_PROVIDER",
    "EP_KIND_CPU",
    "EP_KIND_DIRECTML",
    "EP_KIND_QNN",
    "FIXTURE_CATALOG",
    "LISTING_CPU",
    "LISTING_DIRECTML",
    "LISTING_QNN",
    "QNN_PROVIDER",
    "assert_cold_honesty",
    "classify_providers",
    "detect_hardware",
    "ensure_fixture",
    "fake_ort_listing",
    "fixture_bytes",
    "hexagon_qnn",
    "install_fixture_catalog",
    "provider_chain_for",
    "stub_ensure_resolve",
]

# Preference / detect order the kit actually uses: QNN → DirectML → CPU.
LISTING_QNN: tuple[str, ...] = (QNN_PROVIDER, DML_PROVIDER, CPU_PROVIDER)
LISTING_DIRECTML: tuple[str, ...] = (DML_PROVIDER, CPU_PROVIDER)
LISTING_CPU: tuple[str, ...] = (CPU_PROVIDER,)

_FIXTURE_PAYLOAD = b"hexagon-kit-ci-fixture\n"
_FIXTURE_SHA256 = hashlib.sha256(_FIXTURE_PAYLOAD).hexdigest()


def _tiny_spec(
    model_id: str,
    slot: str,
    *filenames: str,
    name: str | None = None,
) -> ModelSpec:
    files = tuple(filenames)
    return ModelSpec(
        model_id=model_id,
        slot=slot,
        name=name or f"CI fixture {slot}",
        description="Tiny Lane-1 fixture. Not production weights. Offline CI only.",
        disk_mb=1,
        ram_mb=1,
        artifacts=tuple(
            Artifact(
                url=f"file:///{filename}",
                filename=filename,
                kind="file",
                sha256=_FIXTURE_SHA256,
            )
            for filename in files
        ),
        expected_files=files,
        notes="Lane-1 CI fixture. Never download real Hexagon / Hub weights.",
    )


# Tiny catalog stand-ins. classify / detect are extra slots for app paths.
# Builtin ids (stt/tts/llm/vision) can be overlaid so ensure_model("vision")
# stays offline on Linux CI.
FIXTURE_CATALOG: tuple[ModelSpec, ...] = (
    _tiny_spec("ci_classify", "classify", "classify.onnx", name="CI classify fixture"),
    _tiny_spec("ci_detect", "detect", "detect.onnx", name="CI detect fixture"),
    _tiny_spec(
        "whisper_tiny_int8",
        "stt",
        "tiny.en-encoder.int8.onnx",
        "tiny.en-decoder.int8.onnx",
        "tiny.en-tokens.txt",
        name="CI STT fixture",
    ),
    _tiny_spec(
        "kokoro_int8",
        "tts",
        "kokoro-v1.0.int8.onnx",
        "voices-v1.0.bin",
        name="CI TTS fixture",
    ),
    _tiny_spec(
        "smollm2_135m_int8",
        "llm",
        "model_int8.onnx",
        "tokenizer.json",
        name="CI LLM fixture",
    ),
    _tiny_spec(
        "rapidocr_ppocrv4_mobile",
        "vision",
        "ch_PP-OCRv4_det_infer.onnx",
        "ch_PP-OCRv4_rec_infer.onnx",
        name="CI vision fixture",
    ),
)


def hexagon_qnn(probe: HardwareProbe | ProviderChoice) -> bool:
    """True only when ORT listed QNN and the kit classified it as Hexagon.

    CPU / ``[ort]`` / DirectML / a config ``prefer=QNN`` override without a
    listing must stay false. Do not invent a second rule in the app.
    """
    ep_kind = getattr(probe, "ep_kind", None)
    if ep_kind != EP_KIND_QNN:
        return False
    providers = getattr(probe, "providers", None)
    if providers is not None and QNN_PROVIDER not in providers:
        return False
    if getattr(probe, "qnn_package", None) is False:
        return False
    return True


def assert_cold_honesty(probe: HardwareProbe | ProviderChoice) -> None:
    """CPU / ``[ort]`` must not paint Hexagon. Raises ``AssertionError``."""
    providers = list(getattr(probe, "providers", ()) or ())
    label = getattr(probe, "provider_label", None) or getattr(probe, "label", "") or ""
    if QNN_PROVIDER not in providers:
        assert hexagon_qnn(probe) is False
        assert getattr(probe, "ep_kind", None) != EP_KIND_QNN
    if getattr(probe, "ep_kind", None) == EP_KIND_CPU:
        assert hexagon_qnn(probe) is False
        # Override can rewrite preferred_provider; ep_kind / hexagon_qnn stay cold.
        if QNN_PROVIDER not in providers:
            # Live probe_hardware label follows listed EPs unless prefer override.
            # classify_providers() with prefer=QNN may still set a Hexagon label;
            # honesty is ep_kind + hexagon_qnn, not that label.
            if getattr(probe, "preferred_provider", None) != QNN_PROVIDER and getattr(
                probe, "preferred", None
            ) != QNN_PROVIDER:
                assert "Hexagon" not in label
                assert "QNN" not in label


def classify_providers(
    providers: Sequence[str],
    *,
    prefer: str | None = None,
    is_snapdragon: bool = False,
    is_arm64: bool = False,
) -> ProviderChoice:
    """Kit classify path: QNN → DirectML → CPU from a mocked listing."""
    return choose_execution_provider(
        list(providers),
        is_snapdragon=is_snapdragon,
        is_arm64=is_arm64,
        prefer_override=prefer,
    )


@contextmanager
def fake_ort_listing(
    providers: Sequence[str],
    *,
    qnn_package: bool | None = None,
    prefer: str | None = None,
) -> Iterator[None]:
    """Patch ORT provider discovery. No Hexagon hardware required.

    ``prefer`` sets ``HEXAGON_KIT_PROVIDER`` (same as a config override).
    ``ep_kind`` stays tied to the listing, not the override.
    """
    listed = list(providers)
    if qnn_package is None:
        qnn_package = QNN_PROVIDER in listed
    env_missing = object()
    old_prefer = os.environ.get("HEXAGON_KIT_PROVIDER", env_missing)
    try:
        if prefer is not None:
            os.environ["HEXAGON_KIT_PROVIDER"] = prefer
            reset_config()
        with (
            patch("hexagon_kit.hw._onnx_providers", lambda: list(listed)),
            patch("hexagon_kit.hw._qnn_package_available", lambda: bool(qnn_package)),
            patch("hexagon_kit.hw._try_register_qnn", lambda htp: None),
        ):
            yield
    finally:
        if prefer is not None:
            if old_prefer is env_missing:
                os.environ.pop("HEXAGON_KIT_PROVIDER", None)
            else:
                os.environ["HEXAGON_KIT_PROVIDER"] = str(old_prefer)
            reset_config()


def detect_hardware(
    providers: Sequence[str] | None = None,
    *,
    qnn_package: bool | None = None,
    prefer: str | None = None,
) -> HardwareProbe:
    """Kit detect path under a mocked ORT listing (QNN → DirectML → CPU)."""
    listed = list(providers) if providers is not None else list(LISTING_CPU)
    with fake_ort_listing(listed, qnn_package=qnn_package, prefer=prefer):
        return probe_hardware()


def provider_chain_for(
    providers: Sequence[str],
    *,
    prefer: str | None = None,
    qnn_package: bool | None = None,
) -> list[str]:
    """``provider_chain()`` against a mocked listing. Does not invent QNN."""
    with fake_ort_listing(providers, qnn_package=qnn_package, prefer=prefer):
        return provider_chain(prefer)


def fixture_bytes() -> bytes:
    """Payload written for every Lane-1 fixture file."""
    return _FIXTURE_PAYLOAD


def _spec_overlay(spec: ModelSpec, artifact_root: Path) -> dict:
    artifacts = []
    for filename in spec.expected_files:
        path = artifact_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_FIXTURE_PAYLOAD)
        artifacts.append(
            {
                "url": path.resolve().as_uri(),
                "filename": filename,
                "kind": "file",
                "sha256": _FIXTURE_SHA256,
            }
        )
    return {
        "model_id": spec.model_id,
        "slot": spec.slot,
        "name": spec.name,
        "description": spec.description,
        "disk_mb": spec.disk_mb,
        "ram_mb": spec.ram_mb,
        "expected_files": list(spec.expected_files),
        "artifacts": artifacts,
        "notes": spec.notes,
    }


def install_fixture_catalog(
    cache_dir: str | Path,
    *,
    catalog: tuple[ModelSpec, ...] | None = None,
    monkeypatch: object | None = None,
) -> Path:
    """Activate tiny fixture specs so ``ensure_model`` never hits the network.

    Writes a few dozen bytes per expected file under ``cache_dir / _ci_artifacts``
    and overlays the catalog via ``load_config``. Builtin slot names (``stt``,
    ``tts``, ``llm``, ``vision``) plus ``classify`` / ``detect`` resolve to
    those bytes. Call from a pytest test that uses ``tmp_path``.

    Pass pytest's ``monkeypatch`` so ``HEXAGON_KIT_CACHE`` is restored after
    the test.
    """
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifact_root = root / "_ci_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    specs = catalog if catalog is not None else FIXTURE_CATALOG
    overlays = [_spec_overlay(spec, artifact_root) for spec in specs]
    if monkeypatch is not None and hasattr(monkeypatch, "setenv"):
        monkeypatch.setenv("HEXAGON_KIT_CACHE", str(root))
    else:
        os.environ["HEXAGON_KIT_CACHE"] = str(root)
    load_config(overrides={"cache_dir": str(root), "models": overlays})
    return root


def stub_ensure_resolve(
    cache_dir: str | Path,
    model_id_or_slot: str,
) -> Path:
    """Materialize expected files + ``COMPLETE`` without downloading.

    After this, ``ensure_model`` / ``resolve`` return the slot dir. Use when
    the app only needs a path, not the real ensure/fetch path.
    """
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HEXAGON_KIT_CACHE", str(root))
    spec = get_spec(model_id_or_slot)
    dest = slot_dir(spec, root)
    dest.mkdir(parents=True, exist_ok=True)
    for name in spec.expected_files:
        target = dest / name
        if not target.is_file():
            target.write_bytes(_FIXTURE_PAYLOAD)
    (dest / COMPLETE_NAME).write_text("ok\n", encoding="utf-8")
    return dest


def ensure_fixture(model_id_or_slot: str, cache_dir: str | Path) -> Path:
    """Run the real ``ensure_model`` path against the fixture catalog.

    Requires ``install_fixture_catalog(cache_dir)`` first (or an equivalent
    overlay). Offline: artifacts are ``file://`` bytes, not Hub / GitHub.
    Hugging Face mirrors on builtin specs are ignored so a leftover
    ``HF_TOKEN`` cannot pull multi-GB weights.
    """
    root = Path(cache_dir)
    with patch("hexagon_kit.cache.hf_token", lambda: None):
        if not is_installed(model_id_or_slot, root):
            return ensure_model(model_id_or_slot, cache_dir=root, force=True)
        return resolve(model_id_or_slot, cache_dir=root)

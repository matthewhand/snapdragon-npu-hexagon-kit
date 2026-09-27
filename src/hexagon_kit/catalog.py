"""Model catalog for first-generation Copilot+ / Hexagon NPU apps."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Artifact:
    url: str
    filename: str
    kind: str = "file"  # file | tar.bz2 | zip
    sha256: str | None = None


@dataclass(frozen=True)
class HfFile:
    """Hugging Face mirror of an expected file (used when HF_TOKEN is set)."""

    repo_id: str
    remote_path: str
    local_name: str
    revision: str = "main"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    slot: str
    name: str
    description: str
    disk_mb: float
    ram_mb: float
    artifacts: tuple[Artifact, ...]
    expected_files: tuple[str, ...]
    hf_files: tuple[HfFile, ...] = ()
    target_hardware: str = "Hexagon NPU / DirectML / CPU"
    notes: str = ""


CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="whisper_tiny_int8",
        slot="stt",
        name="Whisper Tiny EN INT8 (sherpa-onnx)",
        description=(
            "On-device English speech-to-text. Sherpa-ONNX INT8 encoder/decoder "
            "sized for 16 GB first-gen Copilot+ PCs."
        ),
        disk_mb=120,  # upstream .tar.bz2 is ~118 MB; preflight must cover the download
        ram_mb=150,
        artifacts=(
            Artifact(
                url="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2",
                filename="sherpa-onnx-whisper-tiny.en.tar.bz2",
                kind="tar.bz2",
                sha256="2bd6cf965c8bb3e068ef9fa2191387ee63a9dfa2a4e37582a8109641c20005dd",
            ),
        ),
        expected_files=(
            "tiny.en-encoder.int8.onnx",
            "tiny.en-decoder.int8.onnx",
            "tiny.en-tokens.txt",
        ),
        hf_files=(
            HfFile("csukuangfj/sherpa-onnx-whisper-tiny.en", "tiny.en-encoder.int8.onnx", "tiny.en-encoder.int8.onnx"),
            HfFile("csukuangfj/sherpa-onnx-whisper-tiny.en", "tiny.en-decoder.int8.onnx", "tiny.en-decoder.int8.onnx"),
            HfFile("csukuangfj/sherpa-onnx-whisper-tiny.en", "tiny.en-tokens.txt", "tiny.en-tokens.txt"),
        ),
        notes="Unpacks the sherpa archive and keeps only the INT8 encoder, decoder, and tokens.",
    ),
    ModelSpec(
        model_id="kokoro_int8",
        slot="tts",
        name="Kokoro v1.0 INT8 + voices",
        description=(
            "82M-parameter neural TTS with 54 studio voices and word timings. "
            "INT8 ONNX plus voices-v1.0.bin."
        ),
        disk_mb=125,  # int8 onnx ~92 MB + voices-v1.0.bin ~28 MB
        ram_mb=250,
        artifacts=(
            Artifact(
                url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
                filename="kokoro-v1.0.int8.onnx",
                kind="file",
                sha256="6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb",
            ),
            Artifact(
                url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
                filename="voices-v1.0.bin",
                kind="file",
                sha256="bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
            ),
        ),
        expected_files=(
            "kokoro-v1.0.int8.onnx",
            "voices-v1.0.bin",
        ),
        hf_files=(
            HfFile(
                "MoYoYoTech/VoiceDialogue",
                "assets/models/tts/kokoro/kokoro-v1.0.int8.onnx",
                "kokoro-v1.0.int8.onnx",
            ),
            HfFile("leonelhs/kokoro-thewh1teagle", "voices-v1.0.bin", "voices-v1.0.bin"),
        ),
    ),
    ModelSpec(
        model_id="smollm2_135m_int8",
        slot="llm",
        name="SmolLM2 135M Instruct INT8",
        description=(
            "SmolLM-class on-device chat for first-gen 16 GB Snapdragon X. "
            "INT8 ONNX (~135M). Not an 8 GB default."
        ),
        disk_mb=140,  # onnx ~131 MB + tokenizer.json ~2 MB
        ram_mb=400,
        artifacts=(
            Artifact(
                url="https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/onnx/model_int8.onnx",
                filename="model_int8.onnx",
                kind="file",
                sha256="a7c33f9ef85d06734cc9d1f943f7e2ba57c77e769df727f4d3e217d9f672b0cc",
            ),
            Artifact(
                url="https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json",
                filename="tokenizer.json",
                kind="file",
                sha256="9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
            ),
        ),
        expected_files=(
            "model_int8.onnx",
            "tokenizer.json",
        ),
        hf_files=(
            HfFile(
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                "onnx/model_int8.onnx",
                "model_int8.onnx",
            ),
            HfFile(
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                "tokenizer.json",
                "tokenizer.json",
            ),
        ),
        notes="Kit-owned llm slot. Apps supply the loader; this package does not run inference.",
    ),
    ModelSpec(
        model_id="rapidocr_ppocrv4_mobile",
        slot="vision",
        name="RapidOCR PP-OCRv4 mobile",
        description=(
            "Small OCR (det + rec) for first-gen 16 GB Snapdragon X. "
            "Mobile-scale ONNX, not a server or 8 GB vision stack."
        ),
        disk_mb=20,  # det ~4.5 MB + rec ~10.4 MB
        ram_mb=200,
        artifacts=(
            Artifact(
                url="https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_det_infer.onnx",
                filename="ch_PP-OCRv4_det_infer.onnx",
                kind="file",
                sha256="d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9",
            ),
            Artifact(
                url="https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx",
                filename="ch_PP-OCRv4_rec_infer.onnx",
                kind="file",
                sha256="48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b",
            ),
        ),
        expected_files=(
            "ch_PP-OCRv4_det_infer.onnx",
            "ch_PP-OCRv4_rec_infer.onnx",
        ),
        hf_files=(
            HfFile(
                "SWHL/RapidOCR",
                "PP-OCRv4/ch_PP-OCRv4_det_infer.onnx",
                "ch_PP-OCRv4_det_infer.onnx",
            ),
            HfFile(
                "SWHL/RapidOCR",
                "PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx",
                "ch_PP-OCRv4_rec_infer.onnx",
            ),
        ),
        notes="Kit-owned vision slot. Apps supply the OCR runner.",
    ),
)


def list_specs(catalog: tuple[ModelSpec, ...] | None = None) -> tuple[ModelSpec, ...]:
    return catalog if catalog is not None else CATALOG


def get_spec(
    model_id_or_slot: str,
    catalog: tuple[ModelSpec, ...] | None = None,
) -> ModelSpec:
    specs = list_specs(catalog)
    key = model_id_or_slot.strip().lower()
    for spec in specs:
        if spec.model_id == key or spec.slot == key:
            return spec
    known = ", ".join(f"{s.model_id} ({s.slot})" for s in specs)
    raise KeyError(f"Unknown model or slot {model_id_or_slot!r}. Known: {known}")


def artifact_from_mapping(data: dict) -> Artifact:
    if not data.get("url") or not data.get("filename"):
        raise ValueError("Artifact requires url and filename")
    kind = data.get("kind", "file")
    if kind not in {"file", "tar.bz2", "zip"}:
        raise ValueError(f"Unsupported artifact kind: {kind}")
    return Artifact(
        url=str(data["url"]),
        filename=str(data["filename"]),
        kind=str(kind),
        sha256=str(data["sha256"]) if data.get("sha256") else None,
    )


def spec_from_mapping(data: dict, base: ModelSpec | None = None) -> ModelSpec:
    """Build or overlay a ModelSpec from a config mapping."""
    if not isinstance(data, dict):
        raise ValueError("Model override must be an object")
    model_id = str(data.get("model_id") or (base.model_id if base else "")).strip()
    if not model_id:
        raise ValueError("Model override requires model_id")

    artifacts = base.artifacts if base else None
    if "artifacts" in data:
        artifacts = tuple(artifact_from_mapping(item) for item in data["artifacts"])
    expected = base.expected_files if base else None
    if "expected_files" in data:
        expected = tuple(str(name) for name in data["expected_files"])

    slot = str(data.get("slot") or (base.slot if base else "")).strip()
    if not slot:
        raise ValueError(f"Model {model_id!r} requires slot")
    if not artifacts or not expected:
        raise ValueError(f"Model {model_id!r} requires artifacts and expected_files")

    kwargs = {
        "model_id": model_id,
        "slot": slot,
        "name": str(data.get("name") or (base.name if base else model_id)),
        "description": str(data.get("description") or (base.description if base else "")),
        "disk_mb": float(data.get("disk_mb", base.disk_mb if base else 0)),
        "ram_mb": float(data.get("ram_mb", base.ram_mb if base else 0)),
        "artifacts": artifacts,
        "expected_files": expected,
        "target_hardware": str(
            data.get("target_hardware") or (base.target_hardware if base else "Hexagon NPU / DirectML / CPU")
        ),
        "notes": str(data.get("notes") or (base.notes if base else "")),
    }
    return ModelSpec(**kwargs) if base is None else replace(base, **kwargs)


def merge_catalog(
    builtin: tuple[ModelSpec, ...],
    overrides: list[dict] | tuple[dict, ...],
) -> tuple[ModelSpec, ...]:
    by_id = {spec.model_id: spec for spec in builtin}
    for raw in overrides:
        model_id = str(raw.get("model_id", "")).strip()
        if not model_id:
            raise ValueError("models[] entry requires model_id")
        by_id[model_id] = spec_from_mapping(raw, base=by_id.get(model_id))
    return tuple(by_id.values())

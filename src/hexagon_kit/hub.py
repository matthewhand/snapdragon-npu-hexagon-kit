"""Qualcomm AI Hub adapter.

Wraps the official ``qai_hub_models_cli`` catalog/fetch API (and probes
``qai_hub_apps`` / ``qai_hub``) so this kit owns cache, locks, COMPLETE, and
leases. The heavy ``qai_hub_models`` torch SDK is never imported here.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import COMPLETE_NAME, _DOWNLOAD_THREAD_LOCK, _mark_complete, _publish_slot
from .config import active
from .credentials import apply_hub_credentials, credentials_status, hf_token
from .lock import FileLock

HF_MODELS_API = (
    "https://huggingface.co/api/models?filter=qai-hub-models&sort=lastModified&limit=100"
)
HF_MODEL_API = "https://huggingface.co/api/models/{repo}"
HF_ZIP = "https://huggingface.co/{repo}/archive/refs/heads/main.zip"

_PIPELINE_USE_CASE = {
    "automatic-speech-recognition": "Speech Recognition",
    "text-to-speech": "Audio Generation",
    "text-generation": "Text Generation",
    "image-classification": "Image Classification",
    "object-detection": "Object Detection",
    "image-to-text": "Image To Text",
    "audio-classification": "Audio Classification",
}

_PIPELINE_DOMAIN = {
    "automatic-speech-recognition": "Audio",
    "text-to-speech": "Audio",
    "audio-classification": "Audio",
    "audio-to-audio": "Audio",
    "text-generation": "Generative AI",
    "image-classification": "Computer Vision",
    "object-detection": "Computer Vision",
    "image-to-text": "Multimodal",
}

HUB_DIRNAME = "hub"
INSTALL_CLI = "pip install 'snapdragon-npu-hexagon-kit[hub]'  # or: pip install qai_hub_models_cli"
INSTALL_APPS = "pip install qai-hub-apps"
INSTALL_SDK = "pip install qai-hub"

# Hub ids that map onto kit slots. Artifact layout still differs from sherpa/Kokoro;
# this is a hint for Settings cards, not a loader contract.
HUB_SLOT_HINTS: dict[str, str] = {
    "whisper_tiny": "stt",
    "whisper_tiny_en": "stt",
    "whisper_base": "stt",
    "whisper_base_en": "stt",
    "whisper_small": "stt",
    "whisper_small_en": "stt",
    "whisper_medium": "stt",
    "distil_whisper": "stt",
    "melotts_en": "tts",
    "melotts_es": "tts",
    "melotts_zh": "tts",
    "pipertts_en": "tts",
    "pipertts_de": "tts",
    "pipertts_it": "tts",
    "easyocr": "vision",
    "gemma_3n_e2b_it": "llm",
    "gemma_3n_e4b_it": "llm",
}

_USE_CASE_SLOTS: dict[str, str] = {
    "speech recognition": "stt",
    "audio generation": "tts",
    "text generation": "llm",
    "image classification": "vision",
    "object detection": "vision",
    "semantic segmentation": "vision",
    "image to text": "vision",
    "super resolution": "vision",
}

DEFAULT_HUB_RUNTIME = "onnx"
DEFAULT_HUB_PRECISION = "float"
DEFAULT_QNN_CHIPSET = "qualcomm-snapdragon-x-elite"


class HubUnavailable(RuntimeError):
    """qai_hub_models_cli is not installed."""


@dataclass(frozen=True)
class HubBackend:
    get_manifest: Any
    get_model_info: Any
    fetch: Any
    get_asset_url: Any
    domain_proto_to_str: Any
    use_case_proto_to_str: Any
    tag_proto_to_str: Any
    runtime_proto_to_str: Any
    domain_str_match: Any
    use_case_str_to_proto: Any


def _import_cli() -> HubBackend | None:
    try:
        from qai_hub_models_cli.fetch import fetch, get_asset_url
        from qai_hub_models_cli.proto_helpers.manifest import get_manifest
        from qai_hub_models_cli.proto_helpers.info import get_model_info
        from qai_hub_models_cli.proto_helpers.platform_enums import (
            domain_proto_to_str,
            normalize_label,
            runtime_proto_to_str,
            tag_proto_to_str,
            use_case_proto_to_str,
            use_case_str_to_proto,
        )
    except ImportError:
        return None
    return HubBackend(
        get_manifest=get_manifest,
        get_model_info=get_model_info,
        fetch=fetch,
        get_asset_url=get_asset_url,
        domain_proto_to_str=domain_proto_to_str,
        use_case_proto_to_str=use_case_proto_to_str,
        tag_proto_to_str=tag_proto_to_str,
        runtime_proto_to_str=runtime_proto_to_str,
        domain_str_match=normalize_label,
        use_case_str_to_proto=use_case_str_to_proto,
    )


def hub_available() -> bool:
    return _import_cli() is not None


def _module_status(name: str, install: str) -> dict[str, Any]:
    try:
        mod = __import__(name)
    except ImportError:
        return {"available": False, "install": install}
    return {
        "available": True,
        "version": getattr(mod, "__version__", None),
        "module": name,
    }


def vendor_status() -> dict[str, Any]:
    """Probe official Qualcomm packages without calling their network APIs."""
    cli = hub_available()
    apps = _module_status("qai_hub_apps", INSTALL_APPS)
    sdk = _module_status("qai_hub", INSTALL_SDK)
    heavy = _module_status("qai_hub_models", "pip install qai_hub_models  # x64 Python on Windows")
    return {
        "available": cli,
        "cli": {
            "available": cli,
            "package": "qai_hub_models_cli",
            "install": INSTALL_CLI,
        },
        "apps": apps,
        "sdk": sdk,
        "modelsSdk": heavy,
        "credentials": credentials_status(),
        "note": (
            "Hub is optional. Builtin stt/tts download from public GitHub Releases "
            "(k2-fsa/sherpa-onnx and thewh1teagle/kokoro-onnx) with SHA-256 pins — no token. "
            "Official Hub list/fetch (qai_hub_models_cli) uses public S3 and needs no token. "
            "Set an HF token (`hexagon hub configure --hf-token` or HF_TOKEN) to list/fetch "
            "newer community recipes tagged qai-hub-models on Hugging Face, including gated ones. "
            "A Qualcomm Workbench token is only for compile/profile via qai-hub; this kit does not call that."
        ),
    }


def _cli() -> HubBackend:
    apply_hub_credentials()
    backend = _import_cli()
    if backend is None:
        raise HubUnavailable(
            "Qualcomm AI Hub CLI is not installed. " + INSTALL_CLI
            + " Or configure an HF token to list community models: hexagon hub configure --hf-token <token>"
        )
    return backend


def _hf_headers() -> dict[str, str]:
    apply_hub_credentials()
    token = hf_token()
    headers = {"User-Agent": "snapdragon-npu-hexagon-kit/0.2"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_json(url: str) -> Any:
    req = urllib.request.Request(url, headers=_hf_headers())
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers=_hf_headers())
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=60) as response, dest.open("wb") as out:
        while True:
            chunk = response.read(512 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _pipeline_meta(pipeline: str | None) -> tuple[str, str]:
    key = (pipeline or "").strip().lower()
    return _PIPELINE_DOMAIN.get(key, "Generative AI" if "text" in key else ""), _PIPELINE_USE_CASE.get(
        key, key.replace("-", " ").title() if key else ""
    )


def list_hf_community_models(*, limit: int = 100) -> list[dict[str, Any]]:
    """Hugging Face models tagged qai-hub-models (newer / community recipes)."""
    url = HF_MODELS_API.replace("limit=100", f"limit={int(limit)}")
    try:
        payload = _http_json(url)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise HubUnavailable(f"Hugging Face catalog request failed: {exc}") from exc
    if not isinstance(payload, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        repo = str(item.get("id") or "")
        if not repo:
            continue
        pipeline = item.get("pipeline_tag")
        domain, use_case = _pipeline_meta(str(pipeline) if pipeline else None)
        tags = [str(t) for t in list(item.get("tags") or []) if t]
        model_id = repo.replace("/", "--").lower()
        rows.append(
            {
                "id": repo,
                "cacheId": model_id,
                "name": repo.split("/")[-1],
                "source": "huggingface",
                "domain": domain,
                "useCase": use_case,
                "quantized": any("quant" in t.lower() for t in tags),
                "tags": tags,
                "runtimes": [],
                "chipsets": [],
                "slotHint": _slot_for(repo.split("/")[-1].replace("-", "_"), use_case),
                "installed": hub_is_installed(model_id),
                "downloads": item.get("downloads"),
                "url": f"https://huggingface.co/{repo}",
            }
        )
    return rows


def hub_dir(cache_dir: Path | None = None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else active().cache_dir
    return root / HUB_DIRNAME


def _cache_id(model_id: str) -> str:
    return model_id.strip().lower().replace("/", "--")


def hub_model_dir(model_id: str, cache_dir: Path | None = None) -> Path:
    return hub_dir(cache_dir) / _cache_id(model_id)


def hub_is_installed(model_id: str, cache_dir: Path | None = None) -> bool:
    dest = hub_model_dir(model_id, cache_dir)
    return (dest / COMPLETE_NAME).is_file()


def _label(backend: HubBackend, converter: Any, value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(converter(value) or "")
    except Exception:
        return str(value)


def _slot_for(model_id: str, use_case: str) -> str | None:
    hinted = HUB_SLOT_HINTS.get(model_id.lower())
    if hinted:
        return hinted
    return _USE_CASE_SLOTS.get(use_case.lower()) if use_case else None


def _entry_to_dict(backend: HubBackend, entry: Any, *, installed: bool) -> dict[str, Any]:
    model_id = str(getattr(entry, "id", "") or "")
    domain = _label(backend, backend.domain_proto_to_str, getattr(entry, "domain", 0))
    use_case = _label(backend, backend.use_case_proto_to_str, getattr(entry, "use_case", 0))
    tags = [
        _label(backend, backend.tag_proto_to_str, tag)
        for tag in list(getattr(entry, "tags", []) or [])
    ]
    runtimes = [
        _label(backend, backend.runtime_proto_to_str, runtime)
        for runtime in list(getattr(entry, "supported_runtimes", []) or [])
    ]
    tags = [t for t in tags if t]
    runtimes = [r for r in runtimes if r]
    return {
        "id": model_id,
        "name": str(getattr(entry, "display_name", "") or model_id),
        "source": "hub",
        "domain": domain,
        "useCase": use_case,
        "quantized": bool(getattr(entry, "is_quantized", False)),
        "tags": tags,
        "runtimes": runtimes,
        "chipsets": list(getattr(entry, "supported_chipsets", []) or []),
        "slotHint": _slot_for(model_id, use_case),
        "installed": installed,
        "path": str(hub_model_dir(model_id)) if installed else None,
    }


def _filter_row(
    row: dict[str, Any],
    *,
    domain: str | None,
    use_case: str | None,
    quantized: bool | None,
    llm: bool | None,
    tag: str | None,
) -> bool:
    def norm(value: str) -> str:
        return value.lower().replace("-", " ").replace("_", " ").strip()

    if domain and norm(row.get("domain") or "") != norm(domain):
        return False
    if use_case and norm(row.get("useCase") or "") != norm(use_case):
        return False
    if quantized is True and not row.get("quantized"):
        return False
    if quantized is False and row.get("quantized"):
        return False
    tags = [str(t) for t in row.get("tags") or []]
    if llm is True:
        tag_l = {t.lower() for t in tags}
        if "llm" not in tag_l and (row.get("useCase") or "").lower() != "text generation":
            return False
    if tag and norm(tag) not in {norm(t) for t in tags}:
        return False
    return True


def list_hub_models(
    *,
    domain: str | None = None,
    use_case: str | None = None,
    quantized: bool | None = None,
    llm: bool | None = None,
    tag: str | None = None,
    community: bool | None = None,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Official Hub catalog (CLI) plus optional Hugging Face community recipes."""
    apply_hub_credentials()
    rows: list[dict[str, Any]] = []
    backend = _import_cli()
    if backend is not None:
        manifest = backend.get_manifest()
        domain_key = backend.domain_str_match(domain) if domain else None
        use_case_val = backend.use_case_str_to_proto(use_case) if use_case else None
        tag_key = backend.domain_str_match(tag) if tag else None
        for entry in manifest.models:
            row = _entry_to_dict(
                backend, entry, installed=hub_is_installed(str(entry.id), cache_dir)
            )
            if domain_key and backend.domain_str_match(row["domain"]) != domain_key:
                continue
            if use_case_val is not None and getattr(entry, "use_case", None) != use_case_val:
                continue
            if quantized is True and not row["quantized"]:
                continue
            if quantized is False and row["quantized"]:
                continue
            if llm is True and "LLM" not in row["tags"] and "llm" not in {
                t.lower() for t in row["tags"]
            }:
                if (row["useCase"] or "").lower() != "text generation":
                    continue
            if tag_key and tag_key not in {backend.domain_str_match(t) for t in row["tags"]}:
                continue
            rows.append(row)
    include_community = community is True or (community is None and bool(hf_token()))
    if include_community or (backend is None and community is not False):
        if include_community or hf_token():
            try:
                for row in list_hf_community_models():
                    if _filter_row(
                        row,
                        domain=domain,
                        use_case=use_case,
                        quantized=quantized,
                        llm=llm,
                        tag=tag,
                    ):
                        rows.append(row)
            except HubUnavailable:
                if backend is None and not rows:
                    raise
    if not rows and backend is None:
        raise HubUnavailable(
            "No Hub CLI and no Hugging Face catalog. "
            + INSTALL_CLI
            + " Or: hexagon hub configure --hf-token <token>"
        )
    rows.sort(key=lambda item: (item.get("source") or "", item.get("domain") or "", item.get("id") or ""))
    return rows


def _hf_info(repo: str, cache_dir: Path | None = None) -> dict[str, Any]:
    try:
        data = _http_json(HF_MODEL_API.format(repo=repo))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise HubUnavailable(f"Hugging Face info failed for {repo}: {exc}") from exc
    if not isinstance(data, dict):
        raise HubUnavailable(f"Unexpected Hugging Face payload for {repo}")
    pipeline = data.get("pipeline_tag")
    domain, use_case = _pipeline_meta(str(pipeline) if pipeline else None)
    tags = [str(t) for t in list(data.get("tags") or []) if t]
    cache_id = _cache_id(repo)
    card = data.get("cardData") if isinstance(data.get("cardData"), dict) else {}
    return {
        "id": repo,
        "cacheId": cache_id,
        "name": str(card.get("pretty_name") or repo.split("/")[-1]),
        "source": "huggingface",
        "headline": str(card.get("pretty_name") or ""),
        "description": str((data.get("description") or card.get("description") or "")),
        "domain": domain,
        "useCase": use_case,
        "tags": tags,
        "licenseUrl": "",
        "sourceRepo": f"https://huggingface.co/{repo}",
        "llm": "text-generation" in (pipeline or "") or "llm" in {t.lower() for t in tags},
        "technicalDetails": [],
        "slotHint": _slot_for(repo.split("/")[-1].replace("-", "_"), use_case),
        "installed": hub_is_installed(cache_id, cache_dir),
        "path": str(hub_model_dir(cache_id, cache_dir)) if hub_is_installed(cache_id, cache_dir) else None,
    }


def hub_info(model_id: str, cache_dir: Path | None = None) -> dict[str, Any]:
    apply_hub_credentials()
    if "/" in model_id.strip():
        return _hf_info(model_id.strip(), cache_dir=cache_dir)
    backend = _cli()
    info = backend.get_model_info(model_id)
    domain = _label(backend, backend.domain_proto_to_str, getattr(info, "domain", 0))
    use_case = _label(backend, backend.use_case_proto_to_str, getattr(info, "use_case", 0))
    tags = [
        _label(backend, backend.tag_proto_to_str, tag)
        for tag in list(getattr(info, "tags", []) or [])
        if tag
    ]
    mid = str(getattr(info, "id", "") or model_id)
    details = []
    for item in list(getattr(info, "technical_details", []) or []):
        key = str(getattr(item, "key", "") or "")
        value = (
            getattr(item, "string_value", None)
            or getattr(item, "int_value", None)
            or getattr(item, "float_value", None)
        )
        if key:
            details.append({"key": key, "value": value})
    return {
        "id": mid,
        "name": str(getattr(info, "name", "") or mid),
        "source": "hub",
        "headline": str(getattr(info, "headline", "") or ""),
        "description": str(getattr(info, "description", "") or ""),
        "domain": domain,
        "useCase": use_case,
        "tags": [t for t in tags if t],
        "licenseUrl": str(getattr(info, "license_url", "") or ""),
        "sourceRepo": str(getattr(info, "source_repo", "") or ""),
        "llm": bool(getattr(info, "model_type_llm", False)),
        "technicalDetails": details,
        "slotHint": _slot_for(mid, use_case),
        "installed": hub_is_installed(mid, cache_dir),
        "path": str(hub_model_dir(mid, cache_dir)) if hub_is_installed(mid, cache_dir) else None,
    }


def _fetch_hf_repo(repo: str, cache_dir: Path | None = None) -> Path:
    apply_hub_credentials()
    key = _cache_id(repo)
    dest = hub_model_dir(key, cache_dir)
    lock = FileLock(hub_dir(cache_dir) / f"{key}.lock")
    with _DOWNLOAD_THREAD_LOCK:
        with lock:
            if hub_is_installed(key, cache_dir):
                return dest
            staging = hub_dir(cache_dir) / f".{key}-staging-{os.getpid()}"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            archive = staging / "repo.zip"
            try:
                try:
                    from .hf import hf_snapshot

                    hf_snapshot(repo, staging / "repo")
                except Exception:
                    _http_download(HF_ZIP.format(repo=repo), archive)
                    with zipfile.ZipFile(archive) as zf:
                        zf.extractall(staging / "repo")
                    archive.unlink(missing_ok=True)
                _mark_complete(staging)
                _publish_slot(staging, dest)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
    return dest


def fetch_hub_model(
    model_id: str,
    *,
    runtime: str | None = None,
    precision: str | None = None,
    chipset: str | None = None,
    device: str | None = None,
    extract: bool = True,
    cache_dir: Path | None = None,
) -> Path:
    """Download a Hub asset into the shared XDG cache under ``hub/<id>/``."""
    apply_hub_credentials()
    raw = model_id.strip()
    if "/" in raw:
        return _fetch_hf_repo(raw, cache_dir=cache_dir)
    backend = _cli()
    key = _cache_id(raw)
    dest = hub_model_dir(key, cache_dir)
    runtime = runtime or DEFAULT_HUB_RUNTIME
    precision = precision or DEFAULT_HUB_PRECISION
    if runtime.lower().startswith("qnn") and not chipset and not device:
        chipset = DEFAULT_QNN_CHIPSET
    lock = FileLock(hub_dir(cache_dir) / f"{key}.lock")
    with _DOWNLOAD_THREAD_LOCK:
        with lock:
            if hub_is_installed(key, cache_dir):
                return dest
            staging = hub_dir(cache_dir) / f".{key}-staging-{os.getpid()}"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            fetch_dir = staging / "_dl"
            fetch_dir.mkdir()
            try:
                kwargs: dict[str, Any] = {
                    "model": key,
                    "runtime": runtime,
                    "precision": precision,
                    "output_dir": str(fetch_dir),
                    "extract": extract,
                }
                if chipset:
                    kwargs["chipset"] = chipset
                if device:
                    kwargs["device"] = device
                result = Path(backend.fetch(**kwargs))
                if result.exists():
                    target = staging / result.name
                    if result.resolve() != target.resolve():
                        if result.is_dir():
                            shutil.copytree(result, target, dirs_exist_ok=True)
                        else:
                            shutil.copy2(result, target)
                _mark_complete(staging)
                _publish_slot(staging, dest)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
    return dest


def delete_hub_model(model_id: str, cache_dir: Path | None = None) -> None:
    dest = hub_model_dir(model_id, cache_dir)
    if dest.exists():
        shutil.rmtree(dest)


def hub_snapshot(cache_dir: Path | None = None) -> dict[str, Any]:
    """Cheap status for ui_snapshot: no catalog download."""
    status = vendor_status()
    root = hub_dir(cache_dir)
    installed: list[str] = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / COMPLETE_NAME).is_file():
                installed.append(child.name)
    status["installed"] = installed
    status["cacheDir"] = str(root)
    return status

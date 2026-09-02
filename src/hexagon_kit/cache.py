"""Shared model cache under the XDG cache directory."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from .catalog import Artifact, ModelSpec
from .config import get_spec
from .preflight import PreflightBlocked, preflight

__all__ = [
    "ModelNotInstalled",
    "default_cache_dir",
    "delete_model",
    "download_model",
    "ensure_model",
    "is_installed",
    "resolve",
    "slot_dir",
]

ProgressFn = Callable[[str, int, int], None]

USER_AGENT = "snapdragon-npu-hexagon-kit/0.1 (Windows ARM64; Copilot+ PC)"


class ModelNotInstalled(FileNotFoundError):
    pass


def default_cache_dir() -> Path:
    from .config import active

    return active().cache_dir


def slot_dir(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    root = cache_dir or default_cache_dir()
    return root / spec.slot


def is_installed(model_id_or_slot: str, cache_dir: Path | None = None) -> bool:
    spec = get_spec(model_id_or_slot)
    dest = slot_dir(spec, cache_dir)
    return all((dest / name).is_file() for name in spec.expected_files)


def resolve(model_id_or_slot: str, cache_dir: Path | None = None) -> Path:
    spec = get_spec(model_id_or_slot)
    dest = slot_dir(spec, cache_dir)
    if not is_installed(spec.model_id, cache_dir):
        raise ModelNotInstalled(
            f"{spec.name} is not installed in {dest}. "
            f"Run: hexagon models download {spec.model_id}"
        )
    return dest


def _require_preflight(model_id: str, *, force: bool) -> None:
    if force:
        return
    guard = preflight(model_id)
    if not guard.ok:
        raise PreflightBlocked(guard)


def ensure_model(
    model_id_or_slot: str,
    cache_dir: Path | None = None,
    progress: ProgressFn | None = None,
    *,
    force: bool = False,
) -> Path:
    spec = get_spec(model_id_or_slot)
    if is_installed(spec.model_id, cache_dir):
        return slot_dir(spec, cache_dir)
    return download_model(
        spec.model_id, cache_dir=cache_dir, progress=progress, force=force
    )


def delete_model(model_id_or_slot: str, cache_dir: Path | None = None) -> None:
    spec = get_spec(model_id_or_slot)
    dest = slot_dir(spec, cache_dir)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)


def download_model(
    model_id_or_slot: str,
    cache_dir: Path | None = None,
    progress: ProgressFn | None = None,
    *,
    force: bool = False,
) -> Path:
    spec = get_spec(model_id_or_slot)
    _require_preflight(spec.model_id, force=force)
    dest = slot_dir(spec, cache_dir)
    dest.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"hexagon-{spec.model_id}-"))
    try:
        for artifact in spec.artifacts:
            local = _fetch_artifact(artifact, staging, progress)
            _place_artifact(artifact, local, dest, spec)
        missing = [name for name in spec.expected_files if not (dest / name).is_file()]
        if missing:
            raise RuntimeError(
                f"Download finished but expected files missing in {dest}: {missing}"
            )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return dest


def _fetch_artifact(artifact: Artifact, staging: Path, progress: ProgressFn | None) -> Path:
    dest = staging / artifact.filename
    temp = dest.with_suffix(dest.suffix + ".downloading")
    req = urllib.request.Request(artifact.url, headers={"User-Agent": USER_AGENT})
    hasher = hashlib.sha256() if artifact.sha256 else None
    with urllib.request.urlopen(req, timeout=60) as response, open(temp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(512 * 1024)
            if not chunk:
                break
            out.write(chunk)
            if hasher is not None:
                hasher.update(chunk)
            downloaded += len(chunk)
            if progress:
                progress(artifact.filename, downloaded, total)
    if artifact.sha256:
        digest = hasher.hexdigest() if hasher is not None else _sha256_file(temp)
        if digest.lower() != artifact.sha256.lower():
            temp.unlink(missing_ok=True)
            raise ValueError(
                f"SHA-256 mismatch for {artifact.filename}: got {digest}, "
                f"expected {artifact.sha256}"
            )
    temp.replace(dest)
    return dest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sha256(path: Path, expected: str, filename: str) -> None:
    digest = _sha256_file(path)
    if digest.lower() != expected.lower():
        raise ValueError(
            f"SHA-256 mismatch for {filename}: got {digest}, expected {expected}"
        )


def _place_artifact(artifact: Artifact, local: Path, dest: Path, spec: ModelSpec) -> None:
    if artifact.kind == "file":
        shutil.copy2(local, dest / artifact.filename)
        return
    extract_root = local.parent / f"{local.name}-extracted"
    extract_root.mkdir(exist_ok=True)
    if artifact.kind == "tar.bz2":
        with tarfile.open(local, "r:bz2") as tar:
            tar.extractall(extract_root, filter="data")
    elif artifact.kind == "zip":
        with zipfile.ZipFile(local) as zf:
            zf.extractall(extract_root)
    else:
        raise ValueError(f"Unsupported artifact kind: {artifact.kind}")
    _promote_expected_files(extract_root, dest, spec.expected_files)


def _promote_expected_files(extract_root: Path, dest: Path, expected: tuple[str, ...]) -> None:
    found: dict[str, Path] = {}
    for path in extract_root.rglob("*"):
        if path.is_file() and path.name in expected and path.name not in found:
            found[path.name] = path
    missing = [name for name in expected if name not in found]
    if missing:
        raise RuntimeError(f"Archive did not contain: {missing}")
    for name, src in found.items():
        shutil.copy2(src, dest / name)

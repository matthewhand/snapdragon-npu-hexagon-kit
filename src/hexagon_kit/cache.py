"""Shared model cache under the XDG cache directory."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import threading
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from .catalog import Artifact, ModelSpec
from .config import get_spec
from .credentials import hf_token
from .hf import hf_hub_file, link_or_copy
from .leases import ModelInUse, holder_for
from .lock import FileLock
from .preflight import PreflightBlocked, preflight

__all__ = [
    "COMPLETE_NAME",
    "ModelInUse",
    "ModelNotInstalled",
    "default_cache_dir",
    "delete_model",
    "download_model",
    "ensure_model",
    "is_installed",
    "resolve",
    "slot_dir",
]

COMPLETE_NAME = "COMPLETE"
_FETCH_ATTEMPTS = 3
_FETCH_TIMEOUT_S = 60
_DOWNLOAD_THREAD_LOCK = threading.RLock()

ProgressFn = Callable[[str, int, int], None]

USER_AGENT = "snapdragon-npu-hexagon-kit/0.2 (Windows ARM64; Copilot+ PC)"


class ModelNotInstalled(FileNotFoundError):
    pass


def default_cache_dir() -> Path:
    from .config import active

    return active().cache_dir


def slot_dir(spec: ModelSpec, cache_dir: Path | None = None) -> Path:
    root = cache_dir or default_cache_dir()
    return root / spec.slot


def _slot_lock(spec: ModelSpec, cache_dir: Path | None = None) -> FileLock:
    root = cache_dir or default_cache_dir()
    return FileLock(Path(root) / f"{spec.slot}.lock")


def _files_present(dest: Path, spec: ModelSpec) -> bool:
    return all((dest / name).is_file() for name in spec.expected_files)


def _mark_complete(dest: Path) -> None:
    (dest / COMPLETE_NAME).write_text("ok\n", encoding="utf-8")


def is_installed(model_id_or_slot: str, cache_dir: Path | None = None) -> bool:
    spec = get_spec(model_id_or_slot)
    dest = slot_dir(spec, cache_dir)
    if not _files_present(dest, spec):
        return False
    complete = dest / COMPLETE_NAME
    if complete.is_file():
        return True
    # Heal 0.1.0 caches that wrote expected_files without a sentinel.
    try:
        _mark_complete(dest)
    except OSError:
        return False
    return True


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
    holder = holder_for(spec.slot, cache_dir)
    if holder is not None:
        raise ModelInUse(spec.slot, holder)
    if not dest.exists():
        return
    shutil.rmtree(dest)


def _publish_slot(staging: Path, dest: Path) -> None:
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        old = parent / f".{dest.name}-old-{os.getpid()}"
        if old.exists():
            shutil.rmtree(old)
        dest.rename(old)
        try:
            staging.rename(dest)
        except OSError:
            old.rename(dest)
            raise
        shutil.rmtree(old, ignore_errors=True)
        return
    staging.rename(dest)


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
    with _DOWNLOAD_THREAD_LOCK:
        with _slot_lock(spec, cache_dir):
            if is_installed(spec.model_id, cache_dir):
                return dest
            staging = dest.parent / f".{spec.slot}-staging-{os.getpid()}"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            fetch_dir = staging / "_dl"
            fetch_dir.mkdir()
            try:
                used_hf = _try_hf_files(spec, staging)
                if not used_hf:
                    for artifact in spec.artifacts:
                        local = _fetch_artifact(artifact, fetch_dir, progress)
                        _place_artifact(artifact, local, staging, spec)
                missing = [
                    name for name in spec.expected_files if not (staging / name).is_file()
                ]
                if missing:
                    raise RuntimeError(
                        f"Download finished but expected files missing in {staging}: {missing}"
                    )
                shutil.rmtree(fetch_dir, ignore_errors=True)
                _mark_complete(staging)
                _publish_slot(staging, dest)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
    return dest


def _try_hf_files(spec: ModelSpec, staging: Path) -> bool:
    """If HF_TOKEN is set and the spec has mirrors, fill staging from the HF cache."""
    if not hf_token() or not spec.hf_files:
        return False
    try:
        for item in spec.hf_files:
            cached = hf_hub_file(item.repo_id, item.remote_path, revision=item.revision)
            dest = staging / item.local_name
            link_or_copy(cached, dest)
            expected_hash = None
            for art in spec.artifacts:
                if art.filename == item.local_name and art.sha256:
                    expected_hash = art.sha256
            if expected_hash:
                digest = _sha256_file(dest)
                if digest.lower() != expected_hash.lower():
                    raise ValueError(
                        f"HF SHA-256 mismatch for {item.local_name}: "
                        f"got {digest}, expected {expected_hash}"
                    )
        return all((staging / name).is_file() for name in spec.expected_files)
    except Exception:
        for name in spec.expected_files:
            path = staging / name
            if path.exists() or path.is_symlink():
                path.unlink()
        return False


def _fetch_artifact(artifact: Artifact, staging: Path, progress: ProgressFn | None) -> Path:
    dest = staging / artifact.filename
    last_error: Exception | None = None
    for attempt in range(_FETCH_ATTEMPTS):
        temp = dest.with_suffix(dest.suffix + f".downloading.{attempt}")
        try:
            req = urllib.request.Request(artifact.url, headers={"User-Agent": USER_AGENT})
            hasher = hashlib.sha256() if artifact.sha256 else None
            with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_S) as response, open(
                temp, "wb"
            ) as out:
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
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            temp.unlink(missing_ok=True)
    assert last_error is not None
    raise last_error


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

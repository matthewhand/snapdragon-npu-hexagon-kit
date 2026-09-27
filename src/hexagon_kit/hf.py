"""Hugging Face hub cache helpers.

When an HF token is configured, downloads go through huggingface_hub so blobs
land in the shared HF cache (~/.cache/huggingface/hub) and are hard-linked
into the kit slot. That avoids a second network fetch if transformers or
another tool already pulled the same file.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .credentials import apply_hub_credentials, hf_token


def link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
        return
    except OSError:
        pass
    try:
        os.symlink(src, dest)
        return
    except OSError:
        pass
    shutil.copy2(src, dest)


def hf_hub_file(repo_id: str, filename: str, *, revision: str = "main") -> Path:
    """Return a local path in the Hugging Face cache, downloading if needed."""
    apply_hub_credentials()
    token = hf_token()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. pip install huggingface_hub "
            "(or snapdragon-npu-hexagon-kit[hf])"
        ) from exc
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
    )
    return Path(path)


def hf_snapshot(repo_id: str, dest: Path, *, revision: str = "main") -> Path:
    """Snapshot a repo into dest using the HF cache (no second blob store)."""
    apply_hub_credentials()
    token = hf_token()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. pip install huggingface_hub"
        ) from exc
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=str(dest),
        local_dir_use_symlinks=True,
    )
    return dest

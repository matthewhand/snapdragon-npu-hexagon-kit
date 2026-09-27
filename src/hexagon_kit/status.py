"""
UI-agnostic snapshot for model cards, storage, and hardware.

Qt (SnapDrago) and Electron (Persona) should bind this JSON, not invent
parallel catalogs. Widgets stay in each app.

Schema changes are additive. ``SNAPSHOT_SCHEMA_VERSION`` bumps only when a
field changes meaning or is removed.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from .cache import delete_model, download_model, is_installed, peer_download, slot_dir
from .catalog import ModelSpec
from .config import active, get_spec, list_specs
from .hw import probe_hardware
from .preflight import preflight
from .hub import hub_snapshot
from .leases import Lease, list_leases
from .runtime import process_pool
from .xdg import default_config_path

SNAPSHOT_SCHEMA_VERSION = 1

# Keeps STT/TTS/OCR weights from being offered as chat engines.
SLOT_INFO: dict[str, dict[str, Any]] = {
    "stt": {"slotLabel": "Speech recognition (STT)", "modality": "speech-to-text", "chatCapable": False},
    "tts": {"slotLabel": "Text to speech (TTS)", "modality": "text-to-speech", "chatCapable": False},
    "llm": {"slotLabel": "Chat / text generation (LLM)", "modality": "text-generation", "chatCapable": True},
    "vision": {"slotLabel": "Vision / OCR", "modality": "ocr", "chatCapable": False},
}

ACTIVE_JOB_STATES = frozenset({"downloading", "cancelling"})


class DownloadCancelled(RuntimeError):
    """A background download stopped because ``cancel_download`` was called."""


class DownloadInProgress(RuntimeError):
    """Delete refused while this process is still downloading the slot."""


def slot_info(slot: str) -> dict[str, Any]:
    """``slot``, ``slotLabel``, ``modality``, ``chatCapable``. Unknown slots are ``other``."""
    key = slot.strip().lower()
    info = SLOT_INFO.get(key, {"slotLabel": key, "modality": "other", "chatCapable": False})
    return {"slot": key, **info}


def _hub_slice() -> dict[str, Any]:
    try:
        return hub_snapshot()
    except Exception as exc:
        return {"available": False, "error": str(exc)}


_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def _dir_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def storage_report() -> dict[str, Any]:
    cache = active().cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(cache)
    cache_bytes = _dir_size_bytes(cache)
    return {
        "cacheDir": str(cache),
        "cacheBytes": cache_bytes,
        "cacheMb": round(cache_bytes / (1024 * 1024), 1),
        "diskTotalBytes": usage.total,
        "diskUsedBytes": usage.used,
        "diskFreeBytes": usage.free,
        "diskFreeGb": round(usage.free / (1024**3), 1),
        "diskTotalGb": round(usage.total / (1024**3), 1),
        "diskUsedPct": int(usage.used * 100 / usage.total) if usage.total else 0,
    }


def _public(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if not key.startswith("_")}


def _job_for(model_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(model_id)
        return _public(job) if job else None


def get_job(model_id_or_slot: str) -> dict[str, Any] | None:
    """Latest in-process download job for a model, or None. Jobs are per process."""
    return _job_for(get_spec(model_id_or_slot).model_id)


def download_jobs() -> list[dict[str, Any]]:
    """All in-process download jobs (active and finished) since start or ``reset_jobs``."""
    with _LOCK:
        return [_public(job) for job in _JOBS.values()]


def _lease_dict(lease: Lease) -> dict[str, Any]:
    data = lease.to_dict()
    data["isSelf"] = lease.pid == os.getpid()
    return data


def _card(spec: ModelSpec, leases: list[Lease]) -> dict[str, Any]:
    installed = is_installed(spec.model_id)
    job = _job_for(spec.model_id)
    job_state = str(job.get("state")) if job else None
    own_active = job_state in ACTIVE_JOB_STATES
    peer = None if (installed or own_active) else peer_download(spec.model_id)
    holder = next((lease for lease in leases if lease.slot == spec.slot), None)

    error = None
    error_type = None
    if own_active:
        state = "downloading"
        pct = int(job.get("pct") or 0)
        label = "Cancelling…" if job_state == "cancelling" else f"Downloading {pct}%"
    elif installed:
        state, label, pct = "ready", "Ready / Installed", 100
    elif job_state == "failed":
        state, label, pct = "failed", "Download failed", int(job.get("pct") or 0)
        error = job.get("error")
        error_type = job.get("errorType")
    elif job_state == "cancelled":
        state, label, pct = "cancelled", "Download cancelled", 0
    elif peer is not None:
        state = "downloading"
        pct = int(peer["estimatedPct"])
        label = f"Downloading in another app (pid {peer['pid']})"
    else:
        state, label, pct = "downloadable", "Downloadable", 0

    if own_active:
        actions = ["cancel"]
    elif state == "downloading":
        actions = []
    elif installed:
        actions = [] if holder is not None else ["delete"]
    else:
        actions = ["download"]

    delete_blocked = None
    if installed and holder is not None:
        delete_blocked = f"In use by {holder.exe} (pid {holder.pid})"
    elif installed and own_active:
        delete_blocked = "Download in progress"

    guard = preflight(spec.model_id)
    ram_labels = {
        "fits": "Fits into RAM",
        "tight": f"Tight memory (~{spec.ram_mb:.0f} MB needed, {guard.available_ram_mb or 0:.0f} MB free)",
        "unsafe": f"Too little RAM (~{spec.ram_mb:.0f} MB needed, {guard.available_ram_mb or 0:.0f} MB free)",
    }
    dest = slot_dir(spec)
    return {
        "id": spec.model_id,
        **slot_info(spec.slot),
        "name": spec.name,
        "description": spec.description,
        "diskMb": spec.disk_mb,
        "ramMb": spec.ram_mb,
        "targetHardware": spec.target_hardware,
        "installed": installed,
        "path": str(dest) if installed else None,
        "sizeBytes": _dir_size_bytes(dest) if installed else 0,
        "status": state,
        "statusLabel": label,
        "progressPct": pct,
        "downloadedBytes": int(job.get("downloaded") or 0) if job else 0,
        "totalBytes": int(job.get("total") or 0) if job else 0,
        "error": error,
        "errorType": error_type,
        "actions": actions,
        "canDelete": installed and delete_blocked is None,
        "deleteBlockedReason": delete_blocked,
        "heldBy": _lease_dict(holder) if holder is not None else None,
        "peerDownload": peer,
        "job": job,
        "ramFit": guard.ram_fit,
        "ramFitLabel": ram_labels[guard.ram_fit],
        "diskOk": guard.disk_ok,
        "preflight": guard.to_dict(),
    }


def model_card(model_id_or_slot: str) -> dict[str, Any]:
    return _card(get_spec(model_id_or_slot), list_leases())


def slots_summary(leases: list[Lease] | None = None) -> list[dict[str, Any]]:
    """One row per catalog slot: slot semantics, the kit pin, install and holder state."""
    held = {lease.slot: lease for lease in (list_leases() if leases is None else leases)}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in list_specs():
        if spec.slot in seen:
            continue
        seen.add(spec.slot)
        pin = get_spec(spec.slot)
        holder = held.get(pin.slot)
        rows.append(
            {
                **slot_info(pin.slot),
                "modelId": pin.model_id,
                "installed": is_installed(pin.model_id),
                "heldBy": _lease_dict(holder) if holder is not None else None,
            }
        )
    return rows


def ui_snapshot() -> dict[str, Any]:
    """Single payload for Settings / SnapDrago model manager."""
    from . import __version__

    cfg = active()
    pool = process_pool()
    hw = probe_hardware()
    leases = list_leases()
    return {
        "schemaVersion": SNAPSHOT_SCHEMA_VERSION,
        "kitVersion": __version__,
        "generatedAt": time.time(),
        "hardware": hw.to_dict(),
        "storage": storage_report(),
        "pool": {
            "budgetMb": pool.max_ram_mb,
            "residentMb": pool.resident_ram_mb(),
            "slots": pool.status(),
            "peers": [_lease_dict(item) for item in leases],
        },
        "config": {
            "path": str(cfg.config_path or default_config_path()),
            "cacheDir": str(cfg.cache_dir),
            "sources": list(cfg.sources),
            "preferredProvider": cfg.preferred_provider or hw.preferred_provider,
            "maxRamMb": cfg.max_ram_mb,
        },
        "slots": slots_summary(leases),
        "models": [_card(spec, leases) for spec in list_specs()],
        "jobs": download_jobs(),
        "hub": _hub_slice(),
    }


class _Progress:
    """Aggregate per-artifact callbacks into one monotonic job percentage."""

    def __init__(self, spec: ModelSpec, job: dict[str, Any], cancel: threading.Event):
        self._spec = spec
        self._job = job
        self._cancel = cancel
        self._files: dict[str, tuple[int, int]] = {}

    def __call__(self, name: str, downloaded: int, total: int) -> None:
        if self._cancel.is_set():
            raise DownloadCancelled(f"Download of {self._spec.model_id} was cancelled")
        self._files[name] = (downloaded, total)
        done = sum(d for d, _ in self._files.values())
        known = sum(max(t, d) for d, t in self._files.values())
        unseen = len(self._files) < len(self._spec.artifacts)
        if unseen or any(t <= 0 for _, t in self._files.values()):
            known = max(known, int(self._spec.disk_mb * 1024 * 1024))
        pct = min(99, int(done * 100 / known)) if known else 0
        with _LOCK:
            self._job["downloaded"] = done
            self._job["total"] = known
            self._job["currentFile"] = name
            self._job["pct"] = max(int(self._job.get("pct") or 0), pct)


def start_download(model_id_or_slot: str, *, force: bool = False) -> dict[str, Any]:
    spec = get_spec(model_id_or_slot)
    guard = preflight(spec.model_id)
    if not guard.ok and not force:
        return {
            "id": spec.model_id,
            "slot": spec.slot,
            "state": "blocked",
            "preflight": guard.to_dict(),
            "error": guard.message,
        }
    with _LOCK:
        current = _JOBS.get(spec.model_id)
        if current and current.get("state") in ACTIVE_JOB_STATES:
            return _public(current)
        cancel = threading.Event()
        job: dict[str, Any] = {
            "id": spec.model_id,
            "slot": spec.slot,
            "state": "downloading",
            "pct": 0,
            "downloaded": 0,
            "total": 0,
            "currentFile": None,
            "error": None,
            "errorType": None,
            "force": force,
            "startedAt": time.time(),
            "finishedAt": None,
            "_cancel": cancel,
        }
        _JOBS[spec.model_id] = job

    progress = _Progress(spec, job, cancel)

    def _run() -> None:
        try:
            download_model(spec.model_id, progress=progress, force=force)
            with _LOCK:
                job["state"] = "ready"
                job["pct"] = 100
                job["error"] = None
                job["errorType"] = None
        except DownloadCancelled:
            with _LOCK:
                job["state"] = "cancelled"
                job["pct"] = 0
        except Exception as exc:
            with _LOCK:
                job["state"] = "failed"
                job["error"] = str(exc) or type(exc).__name__
                job["errorType"] = type(exc).__name__
        finally:
            with _LOCK:
                job["finishedAt"] = time.time()

    threading.Thread(target=_run, name=f"hexagon-dl-{spec.model_id}", daemon=True).start()
    with _LOCK:
        return _public(job)


def cancel_download(model_id_or_slot: str) -> dict[str, Any]:
    """
    Ask this process's background download to stop.

    Cooperative: takes effect at the next streamed chunk. Returns the job with
    ``cancelRequested``; the job reaches ``cancelled`` once staging is removed.
    A Hugging Face hub-cache fill has no chunk callbacks and finishes first.
    """
    spec = get_spec(model_id_or_slot)
    with _LOCK:
        job = _JOBS.get(spec.model_id)
        if job is None or job.get("state") not in ACTIVE_JOB_STATES:
            base = _public(job) if job else {"id": spec.model_id, "slot": spec.slot, "state": "idle"}
            return {**base, "cancelRequested": False}
        job["_cancel"].set()
        job["state"] = "cancelling"
        return {**_public(job), "cancelRequested": True}


def delete_cached(model_id_or_slot: str) -> dict[str, Any]:
    spec = get_spec(model_id_or_slot)
    job = _job_for(spec.model_id)
    if job and job.get("state") in ACTIVE_JOB_STATES:
        raise DownloadInProgress(
            f"{spec.name} is still downloading. Cancel the download before deleting."
        )
    delete_model(spec.model_id)
    with _LOCK:
        _JOBS.pop(spec.model_id, None)
    return model_card(spec.model_id)


def reset_jobs() -> None:
    with _LOCK:
        for job in _JOBS.values():
            cancel = job.get("_cancel")
            if cancel is not None:
                cancel.set()
        _JOBS.clear()

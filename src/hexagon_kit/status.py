"""
UI-agnostic snapshot for model cards, storage, and hardware.

Qt (SnapDrago) and Electron (Persona) should bind this JSON, not invent
parallel catalogs. Widgets stay in each app.

Schema changes are additive. ``SNAPSHOT_SCHEMA_VERSION`` bumps only when a
field changes meaning or is removed.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .cache import delete_model, download_model, is_installed, peer_download, slot_dir
from .catalog import ModelSpec
from .config import active, get_spec, list_specs
from .hw import probe_hardware
from .preflight import preflight
from .hub import COMPLETE_NAME, HUB_SLOT_HINTS, delete_hub_model, fetch_hub_model, hub_dir, hub_is_installed, hub_model_dir, hub_snapshot
from .jobs import (
    ACTIVE_STATES,
    cancel_requested,
    foreign_active_owner,
    load_shared_jobs,
    note_owner_pid,
    read_owner_job,
    remove_jobs_for,
    remove_our_job_files,
    request_cancel,
    save_owner_job,
    save_waiter_job,
    seed_owner_job,
)
from .leases import Lease, list_leases, pid_alive
from .runtime import process_pool
from .settings import settings_snapshot
from .xdg import default_config_path

SNAPSHOT_SCHEMA_VERSION = 1

# Keeps STT/TTS/OCR weights from being offered as chat engines.
SLOT_INFO: dict[str, dict[str, Any]] = {
    "stt": {"slotLabel": "Speech recognition (STT)", "modality": "speech-to-text", "chatCapable": False},
    "tts": {"slotLabel": "Text to speech (TTS)", "modality": "text-to-speech", "chatCapable": False},
    "llm": {"slotLabel": "Chat / text generation (LLM)", "modality": "text-generation", "chatCapable": True},
    "vision": {"slotLabel": "Vision / OCR", "modality": "ocr", "chatCapable": False},
}

ACTIVE_JOB_STATES = ACTIVE_STATES
_PERSIST_INTERVAL_S = 0.2


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


def _blank_hub_preflight() -> dict[str, Any]:
    return {
        "ok": True,
        "ramFit": "fits",
        "diskOk": True,
        "canForce": False,
        "message": "",
        "suggestId": None,
        "suggestName": None,
        "availableRamMb": None,
        "requiredRamMb": 0,
        "availableDiskMb": None,
        "requiredDiskMb": 0,
        "heldBy": None,
        "peerRamMb": 0,
    }


def _job_for(model_id: str) -> dict[str, Any] | None:
    """In-process job if it should win, otherwise the shared owner record."""
    with _LOCK:
        local = _JOBS.get(model_id)
        local_public = _public(local) if local else None
    owner = None
    for item in load_shared_jobs():
        if item.get("id") != model_id or item.get("role") == "waiter":
            continue
        if item.get("state") in ACTIVE_JOB_STATES:
            try:
                alive = pid_alive(int(item.get("pid") or 0))
            except (TypeError, ValueError):
                alive = False
            if alive:
                owner = item
                break
        owner = owner or item
    if local_public and local_public.get("state") == "waiting_on_lock":
        if owner and owner.get("state") == "downloading":
            local_public = {
                **local_public,
                "pct": owner.get("pct") or local_public.get("pct") or 0,
                "downloaded": owner.get("downloaded") or 0,
                "total": owner.get("total") or 0,
                "currentFile": owner.get("currentFile"),
            }
            if not local_public.get("lockHolder"):
                local_public["lockHolder"] = {"pid": owner.get("pid"), "exe": owner.get("exe")}
        return local_public
    if local_public and local_public.get("state") == "failed" and owner and owner.get("state") in ACTIVE_JOB_STATES:
        return owner
    if local_public:
        return local_public
    return owner


def get_job(model_id_or_slot: str) -> dict[str, Any] | None:
    """Latest download job for a model, including one persisted by another process."""
    try:
        model_id = get_spec(model_id_or_slot).model_id
    except KeyError:
        model_id = hub_model_dir(model_id_or_slot).name
    return _job_for(model_id)


def download_jobs() -> list[dict[str, Any]]:
    """Shared download jobs (this process and any other app using the same cache)."""
    with _LOCK:
        memory = [_public(job) for job in _JOBS.values()]
    disk = load_shared_jobs()
    merged: dict[str, dict[str, Any]] = {}
    for job in disk:
        merged[_job_key(job)] = job
    for job in memory:
        key = _job_key(job)
        previous = merged.get(key)
        if previous is None or job.get("pid") == os.getpid():
            merged[key] = {**previous, **job} if previous else job
    return list(merged.values())


def poll_jobs() -> list[dict[str, Any]]:
    """One-shot read of the shared job list. Same payload as ``download_jobs``."""
    return download_jobs()


def watch_jobs(interval: float = 0.5) -> Iterator[list[dict[str, Any]]]:
    """Yield batches of jobs whose public payload changed.

    The first batch is the current jobs, when there are any. The iterator
    blocks for ``interval`` seconds between checks. Break or close it to stop.
    There is no HTTP server; apps that already poll ``ui_snapshot`` can call
    ``poll_jobs`` instead of consuming this iterator.
    """
    if interval < 0:
        raise ValueError("interval must be >= 0")
    previous: dict[str, str] = {}
    while True:
        changed: list[dict[str, Any]] = []
        current: dict[str, str] = {}
        for job in download_jobs():
            key = _job_key(job)
            blob = json.dumps(job, sort_keys=True, default=str)
            current[key] = blob
            if previous.get(key) != blob:
                changed.append(job)
        previous = current
        if changed:
            yield changed
        time.sleep(interval)


def _job_key(job: dict[str, Any]) -> str:
    role = job.get("role") or "owner"
    pid = job.get("pid")
    return f"{role}:{pid}:{job.get('id')}"


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
    lock_holder = job.get("lockHolder") if job and isinstance(job.get("lockHolder"), dict) else None
    if job_state == "waiting_on_lock":
        state = "waiting_on_lock"
        pct = int(job.get("pct") or 0)
        pid = (lock_holder or {}).get("pid")
        label = (
            f"Waiting on download lock (pid {pid})"
            if pid
            else "Waiting on another download's lock"
        )
    elif own_active:
        state = "downloading"
        pct = int(job.get("pct") or 0)
        label = "Cancelling…" if job_state == "cancelling" else f"Downloading {pct}%"
    elif installed:
        state, label, pct = "ready", "Ready / Installed", 100
    elif job_state == "failed":
        state, label, pct = "failed", "Download failed", int(job.get("pct") or 0)
        error = job.get("error")
        error_type = job.get("errorType")
        if error_type == "LockWaitExceeded":
            label = "Gave up waiting for the other download's lock"
    elif job_state == "cancelled":
        state, label, pct = "cancelled", "Download cancelled", 0
    elif peer is not None:
        state = "downloading"
        pct = int(peer["estimatedPct"])
        label = f"Downloading in another app (pid {peer['pid']})"
    else:
        state, label, pct = "downloadable", "Downloadable", 0

    if job_state in ACTIVE_JOB_STATES:
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
        "source": "catalog",
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
        "lockHolder": lock_holder if state == "waiting_on_lock" else None,
        "job": job,
        "ramFit": guard.ram_fit,
        "ramFitLabel": ram_labels[guard.ram_fit],
        "diskOk": guard.disk_ok,
        "preflight": guard.to_dict(),
    }


def model_card(model_id_or_slot: str) -> dict[str, Any]:
    try:
        spec = get_spec(model_id_or_slot)
    except KeyError:
        cached = hub_model_dir(model_id_or_slot).name
        if hub_is_installed(model_id_or_slot) or _job_for(cached):
            return _hub_card(model_id_or_slot)
        raise
    return _card(spec, list_leases())


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


def _installed_hub_ids() -> list[str]:
    root = hub_dir()
    if not root.is_dir():
        return []
    found: list[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith(".") and (child / COMPLETE_NAME).is_file():
            found.append(child.name)
    return found


def _hub_card(model_id: str) -> dict[str, Any]:
    """Catalog-shaped card for one Hub id. ``source`` is ``hub``."""
    key = hub_model_dir(model_id).name
    installed = hub_is_installed(key)
    job = _job_for(key)
    job_state = str(job.get("state")) if job else None
    active = job_state in ACTIVE_JOB_STATES
    dest = hub_model_dir(key)
    size = _dir_size_bytes(dest) if installed else 0
    error = None
    error_type = None
    lock_holder = job.get("lockHolder") if job and isinstance(job.get("lockHolder"), dict) else None
    if job_state == "waiting_on_lock":
        state = "waiting_on_lock"
        pct = int(job.get("pct") or 0)
        pid = (lock_holder or {}).get("pid")
        label = f"Waiting on download lock (pid {pid})" if pid else "Waiting on another download's lock"
    elif active:
        state = "downloading"
        pct = int(job.get("pct") or 0)
        phase = str(job.get("phase") or "")
        if phase == "fetching_hub" and pct <= 0:
            label = "Fetching from Hub…"
        else:
            label = "Cancelling…" if job_state == "cancelling" else f"Downloading {pct}%"
    elif installed:
        state, label, pct = "ready", "Ready / Installed", 100
    elif job_state == "failed":
        state, label, pct = "failed", "Download failed", int(job.get("pct") or 0)
        error = job.get("error")
        error_type = job.get("errorType")
    elif job_state == "cancelled":
        state, label, pct = "cancelled", "Download cancelled", 0
    else:
        state, label, pct = "downloadable", "Downloadable", 0

    if job_state in ACTIVE_JOB_STATES:
        actions = ["cancel"]
    elif installed:
        actions = ["delete"]
    else:
        actions = ["fetch"]

    slot = HUB_SLOT_HINTS.get(key, "hub")
    info = slot_info(slot)
    return {
        "id": key,
        "source": "hub",
        **info,
        "name": key,
        "description": "Qualcomm AI Hub model in the shared cache.",
        "diskMb": round(size / (1024 * 1024), 1) if installed else 0,
        "ramMb": 0,
        "targetHardware": "hub",
        "installed": installed,
        "path": str(dest) if installed else None,
        "sizeBytes": size,
        "status": state,
        "statusLabel": label,
        "progressPct": pct,
        "downloadedBytes": int(job.get("downloaded") or 0) if job else 0,
        "totalBytes": int(job.get("total") or 0) if job else 0,
        "error": error,
        "errorType": error_type,
        "actions": actions,
        "canDelete": installed and not active,
        "deleteBlockedReason": "Download in progress" if installed and active else None,
        "heldBy": None,
        "peerDownload": None,
        "lockHolder": lock_holder if state == "waiting_on_lock" else None,
        "job": job,
        "ramFit": "fits",
        "ramFitLabel": "RAM not estimated for Hub models",
        "diskOk": True,
        "preflight": _blank_hub_preflight(),
    }


def _snapshot_models(leases: list[Lease]) -> list[dict[str, Any]]:
    catalog = [_card(spec, leases) for spec in list_specs()]
    known = {card["id"] for card in catalog}
    hub_ids = set(_installed_hub_ids())
    for job in download_jobs():
        if job.get("source") == "hub" and job.get("id"):
            hub_ids.add(str(job["id"]))
    hub_cards = [_hub_card(model_id) for model_id in sorted(hub_ids) if model_id not in known]
    return catalog + hub_cards


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
        "settings": settings_snapshot(),
        "slots": slots_summary(leases),
        "models": _snapshot_models(leases),
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
        if self._job.get("id") and (
            self._cancel.is_set() or cancel_requested(str(self._job["id"]))
        ):
            self._cancel.set()
            raise DownloadCancelled(f"Download of {self._spec.model_id} was cancelled")
        callback = None
        with _LOCK:
            self._job["downloaded"] = done
            self._job["total"] = known
            self._job["currentFile"] = name
            self._job["pct"] = max(int(self._job.get("pct") or 0), pct)
            if self._job.get("phase") not in {"linking", "verifying", "resolving_hf"}:
                self._job["phase"] = "downloading"
            callback = self._job.get("_on_progress")
        if self._job.get("id"):
            _commit(self._job)
        if callback is not None:
            try:
                callback(_public(self._job))
            except Exception:
                pass


def _emit(callback: Callable[[dict[str, Any]], None] | None, job: dict[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback(_public(job))
    except Exception:
        return


def _commit(job: dict[str, Any], *, force: bool = False) -> None:
    """Persist the public job. State and phase changes always hit disk."""
    if not job.get("id"):
        return
    now = time.monotonic()
    state = job.get("state")
    phase = job.get("phase")
    changed = job.get("_persisted_state") != state or job.get("_persisted_phase") != phase
    if not force and not changed and now - float(job.get("_last_persist") or 0) < _PERSIST_INTERVAL_S:
        return
    job["_last_persist"] = now
    job["_persisted_state"] = state
    job["_persisted_phase"] = phase
    public = _public(job)
    if job.get("role") == "waiter" or state == "waiting_on_lock":
        save_waiter_job(public)
        return
    stored = save_owner_job(public)
    try:
        stored_pid = int(stored.get("pid") or 0)
    except (TypeError, ValueError):
        stored_pid = 0
    if stored_pid and stored_pid != os.getpid() and stored.get("state") in ACTIVE_JOB_STATES:
        job["role"] = "waiter"
        if job.get("state") == "downloading":
            job["state"] = "waiting_on_lock"
            job["phase"] = "waiting_on_lock"
            job["lockHolder"] = {"pid": stored_pid, "exe": stored.get("exe")}
        save_waiter_job(_public(job))
        return
    if stored.get("cancelRequested"):
        cancel = job.get("_cancel")
        if cancel is not None:
            cancel.set()
        job["cancelRequested"] = True


def _finish(job: dict[str, Any], state: str, exc: BaseException | None = None) -> None:
    with _LOCK:
        job["state"] = state
        job["finishedAt"] = time.time()
        job["role"] = "owner"
        if state == "ready":
            job["pct"] = 100
            job["phase"] = "verifying"
            job["error"] = None
            job["errorType"] = None
        elif state == "cancelled":
            job["pct"] = 0
            job["phase"] = None
            job["error"] = None
            job["errorType"] = None
        elif exc is not None:
            job["error"] = str(exc) or type(exc).__name__
            job["errorType"] = type(exc).__name__
    _commit(job, force=True)
    _emit(job.get("_on_done"), job)


def _raise_if_cancelled(job: dict[str, Any]) -> None:
    cancel = job.get("_cancel")
    if cancel is not None and cancel.is_set():
        raise DownloadCancelled(f"Download of {job.get('id')} was cancelled")
    if job.get("id") and cancel_requested(str(job["id"])):
        if cancel is not None:
            cancel.set()
        raise DownloadCancelled(f"Download of {job.get('id')} was cancelled")


def _bind_callbacks(job: dict[str, Any]) -> tuple[Callable[[str], None], Callable[[dict[str, Any]], None]]:
    def on_phase(phase: str) -> None:
        _raise_if_cancelled(job)
        with _LOCK:
            job["phase"] = phase
            if phase == "waiting_on_lock":
                job["state"] = "waiting_on_lock"
                job["role"] = "waiter"
            elif job.get("state") in {"waiting_on_lock", "downloading", "cancelling"}:
                if job.get("state") != "cancelling":
                    job["state"] = "downloading"
                job["role"] = "owner"
        _commit(job, force=True)
        _emit(job.get("_on_progress"), job)

    def on_lock_wait(holder: dict[str, Any]) -> None:
        _raise_if_cancelled(job)
        owner = foreign_active_owner(str(job["id"]))
        with _LOCK:
            if job.get("state") != "cancelling":
                job["state"] = "waiting_on_lock"
                job["phase"] = "waiting_on_lock"
                job["role"] = "waiter"
            job["lockHolder"] = {"pid": holder.get("pid"), "exe": holder.get("exe")}
            if owner and owner.get("state") == "downloading":
                job["pct"] = owner.get("pct") or job.get("pct") or 0
                job["downloaded"] = owner.get("downloaded") or 0
                job["total"] = owner.get("total") or 0
                job["currentFile"] = owner.get("currentFile")
        _commit(job)
        _emit(job.get("_on_progress"), job)

    return on_phase, on_lock_wait


def _execute_catalog(job: dict[str, Any], *, force: bool) -> None:
    on_phase, on_lock_wait = _bind_callbacks(job)
    progress = _Progress(get_spec(str(job["id"])), job, job["_cancel"])
    try:
        download_model(
            str(job["id"]),
            progress=progress,
            force=force,
            on_phase=on_phase,
            on_lock_wait=on_lock_wait,
        )
    except DownloadCancelled:
        _finish(job, "cancelled")
        return
    except Exception as exc:
        _finish(job, "failed", exc)
        return
    _finish(job, "ready")


def _remember(job: dict[str, Any]) -> dict[str, Any] | None:
    """Store ``job`` unless this process already has an active one."""
    with _LOCK:
        current = _JOBS.get(str(job["id"]))
        if current and current.get("state") in ACTIVE_JOB_STATES:
            return _public(current)
        _JOBS[str(job["id"])] = job
        return None


def _prepare_job(
    *,
    model_id: str,
    slot: str,
    source: str,
    force: bool,
    on_progress: Callable[[dict[str, Any]], None] | None,
    on_done: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    import sys

    foreign = foreign_active_owner(model_id)
    waiting = foreign is not None
    try:
        exe = Path(sys.argv[0]).name or Path(sys.executable).name
    except Exception:
        exe = "python"
    return {
        "id": model_id,
        "slot": slot,
        "source": source,
        "role": "waiter" if waiting else "owner",
        "state": "waiting_on_lock" if waiting else "downloading",
        "phase": "waiting_on_lock" if waiting else "starting",
        "pct": int(foreign.get("pct") or 0) if foreign else 0,
        "downloaded": int(foreign.get("downloaded") or 0) if foreign else 0,
        "total": int(foreign.get("total") or 0) if foreign else 0,
        "currentFile": foreign.get("currentFile") if foreign else None,
        "error": None,
        "errorType": None,
        "force": force,
        "startedAt": time.time(),
        "finishedAt": None,
        "pid": os.getpid(),
        "exe": exe,
        "lockHolder": (
            {"pid": foreign.get("pid"), "exe": foreign.get("exe")} if foreign else None
        ),
        "cancelRequested": False,
        "_cancel": threading.Event(),
        "_on_progress": on_progress,
        "_on_done": on_done,
    }


def start_download(
    model_id_or_slot: str,
    *,
    force: bool = False,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    on_done: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Start a background download and return its public job.

    ``on_progress`` and ``on_done`` receive the public job dict in this
    process. The same job is written under the cache ``jobs/`` directory so
    other apps' ``ui_snapshot`` calls see the same progress. Returns the
    existing job when one is already active in this process.
    """
    spec = get_spec(model_id_or_slot)
    guard = preflight(spec.model_id)
    if not guard.ok and not force:
        return {
            "id": spec.model_id,
            "slot": spec.slot,
            "source": "catalog",
            "state": "blocked",
            "preflight": guard.to_dict(),
            "error": guard.message,
        }
    job = _prepare_job(
        model_id=spec.model_id,
        slot=spec.slot,
        source="catalog",
        force=force,
        on_progress=on_progress,
        on_done=on_done,
    )
    existing = _remember(job)
    if existing is not None:
        return existing
    _commit(job, force=True)

    def _run() -> None:
        _execute_catalog(job, force=force)

    threading.Thread(target=_run, name=f"hexagon-dl-{spec.model_id}", daemon=True).start()
    return _public(job)


def run_download_foreground(model_id_or_slot: str, *, force: bool = False) -> dict[str, Any]:
    """Download to completion in this process. Used by the detached CLI worker."""
    spec = get_spec(model_id_or_slot)
    guard = preflight(spec.model_id)
    if not guard.ok and not force:
        return {
            "id": spec.model_id,
            "slot": spec.slot,
            "source": "catalog",
            "state": "blocked",
            "preflight": guard.to_dict(),
            "error": guard.message,
        }
    job = _prepare_job(
        model_id=spec.model_id,
        slot=spec.slot,
        source="catalog",
        force=force,
        on_progress=None,
        on_done=None,
    )
    existing = _remember(job)
    if existing is not None:
        job = _JOBS[spec.model_id]
    else:
        _commit(job, force=True)
    _execute_catalog(job, force=force)
    return _public(job)


def seed_detached_download(model_id_or_slot: str, *, force: bool) -> dict[str, Any]:
    """Write a placeholder owner job before a detached worker process starts."""
    spec = get_spec(model_id_or_slot)
    payload = {
        "id": spec.model_id,
        "slot": spec.slot,
        "source": "catalog",
        "role": "owner",
        "state": "downloading",
        "phase": "starting",
        "pct": 0,
        "downloaded": 0,
        "total": 0,
        "currentFile": None,
        "error": None,
        "errorType": None,
        "force": force,
        "pid": None,
        "startedAt": time.time(),
        "finishedAt": None,
        "cancelRequested": False,
    }
    stored = seed_owner_job(payload)
    if stored.get("pid") not in (None, os.getpid()) and stored.get("state") in ACTIVE_JOB_STATES:
        return stored
    return stored


def adopt_detached_pid(model_id: str, pid: int) -> dict[str, Any] | None:
    return note_owner_pid(model_id, pid)


def fail_detached_download(model_id: str, message: str) -> dict[str, Any]:
    current = read_owner_job(model_id) or {"id": model_id, "source": "catalog"}
    current.update(
        {
            "state": "failed",
            "phase": None,
            "error": message,
            "errorType": "OSError",
            "finishedAt": time.time(),
            "pid": os.getpid(),
        }
    )
    return seed_owner_job(current)


def start_hub_download(
    model_id: str,
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    on_done: Callable[[dict[str, Any]], None] | None = None,
    runtime: str | None = None,
    precision: str | None = None,
) -> dict[str, Any]:
    """Background Hub fetch using the same job records as catalog downloads."""
    key = hub_model_dir(model_id).name
    slot = HUB_SLOT_HINTS.get(key, "hub")
    if hub_is_installed(key):
        ready = _prepare_job(
            model_id=key,
            slot=slot,
            source="hub",
            force=True,
            on_progress=on_progress,
            on_done=on_done,
        )
        with _LOCK:
            ready["state"] = "ready"
            ready["pct"] = 100
            ready["phase"] = "verifying"
            ready["finishedAt"] = time.time()
            ready["role"] = "owner"
        existing = _remember(ready)
        if existing is not None and existing.get("state") in ACTIVE_JOB_STATES:
            return existing
        _commit(ready, force=True)
        _emit(on_done, ready)
        return _public(ready)
    job = _prepare_job(
        model_id=key,
        slot=slot,
        source="hub",
        force=True,
        on_progress=on_progress,
        on_done=on_done,
    )
    existing = _remember(job)
    if existing is not None:
        return existing
    _commit(job, force=True)

    def _run() -> None:
        on_phase, on_lock_wait = _bind_callbacks(job)
        stop = threading.Event()

        def _poll() -> None:
            while not stop.wait(0.4):
                staging = hub_dir() / f".{key}-staging-{os.getpid()}"
                if not staging.is_dir():
                    continue
                size = 0
                for item in staging.rglob("*"):
                    try:
                        if item.is_file():
                            size += item.stat().st_size
                    except OSError:
                        continue
                with _LOCK:
                    job["downloaded"] = size
                    job["phase"] = job.get("phase") or "fetching_hub"
                _commit(job)
                _emit(job.get("_on_progress"), job)

        threading.Thread(target=_poll, name=f"hexagon-hub-bytes-{key}", daemon=True).start()
        try:
            fetch_hub_model(
                model_id,
                runtime=runtime,
                precision=precision,
                on_phase=on_phase,
                on_lock_wait=on_lock_wait,
            )
        except DownloadCancelled:
            _finish(job, "cancelled")
        except Exception as exc:
            _finish(job, "failed", exc)
        else:
            _finish(job, "ready")
        finally:
            stop.set()

    threading.Thread(target=_run, name=f"hexagon-hub-{key}", daemon=True).start()
    return _public(job)


def cancel_download(model_id_or_slot: str) -> dict[str, Any]:
    """
    Ask a download to stop.

    Cooperative: takes effect at the next streamed chunk or lock-wait poll.
    A shared job owned by another process gets ``cancelRequested`` on disk;
    that process stops on its next progress callback. Returns the job with
    ``cancelRequested``. The job reaches ``cancelled`` once staging is removed.
    A Hugging Face hub-cache fill has no chunk callbacks and finishes the
    current file first.
    """
    try:
        spec = get_spec(model_id_or_slot)
        model_id = spec.model_id
        slot = spec.slot
    except KeyError:
        model_id = hub_model_dir(model_id_or_slot).name
        slot = HUB_SLOT_HINTS.get(model_id, "hub")
        if _job_for(model_id) is None and not hub_is_installed(model_id):
            raise
    local: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None
    with _LOCK:
        found = _JOBS.get(model_id)
        if found is not None and found.get("state") in ACTIVE_JOB_STATES:
            found["_cancel"].set()
            found["state"] = "cancelling"
            found["cancelRequested"] = True
            local = found
        elif found is not None:
            terminal = _public(found)
    if local is not None:
        _commit(local, force=True)
        return {**_public(local), "cancelRequested": True}
    if terminal is not None:
        return {**terminal, "cancelRequested": False}
    remote = request_cancel(model_id)
    if remote is not None:
        return {**remote, "cancelRequested": True}
    current = _job_for(model_id)
    base = current if current else {"id": model_id, "slot": slot, "state": "idle"}
    return {**base, "cancelRequested": False}


def delete_cached(model_id_or_slot: str) -> dict[str, Any]:
    try:
        spec = get_spec(model_id_or_slot)
    except KeyError:
        key = hub_model_dir(model_id_or_slot).name
        if not hub_is_installed(key) and _job_for(key) is None:
            raise
        job = _job_for(key)
        if job and job.get("state") in ACTIVE_JOB_STATES:
            raise DownloadInProgress(
                f"{key} is still downloading. Cancel the download before deleting."
            )
        delete_hub_model(key)
        with _LOCK:
            _JOBS.pop(key, None)
        remove_jobs_for(key)
        return _hub_card(key)
    job = _job_for(spec.model_id)
    if job and job.get("state") in ACTIVE_JOB_STATES:
        raise DownloadInProgress(
            f"{spec.name} is still downloading. Cancel the download before deleting."
        )
    delete_model(spec.model_id)
    with _LOCK:
        _JOBS.pop(spec.model_id, None)
    remove_jobs_for(spec.model_id)
    return model_card(spec.model_id)


def reset_jobs() -> None:
    with _LOCK:
        for job in _JOBS.values():
            cancel = job.get("_cancel")
            if cancel is not None:
                cancel.set()
        _JOBS.clear()
    remove_our_job_files()

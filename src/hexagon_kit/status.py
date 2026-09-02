"""
UI-agnostic snapshot for model cards, storage, and hardware.

Qt (SnapDrago) and Electron (Persona) should bind this JSON, not invent
parallel catalogs. Widgets stay in each app.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any

from .cache import delete_model, download_model, is_installed
from .config import active, get_spec, list_specs
from .hw import probe_hardware
from .preflight import preflight
from .runtime import process_pool
from .xdg import default_config_path

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


def _job_for(model_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _JOBS.get(model_id)
        return dict(job) if job else None


def model_card(model_id_or_slot: str) -> dict[str, Any]:
    spec = get_spec(model_id_or_slot)
    installed = is_installed(spec.model_id)
    job = _job_for(spec.model_id)
    state = "ready" if installed else "downloadable"
    label = "Ready / Installed" if installed else "Downloadable"
    pct = 100 if installed else 0
    error = None
    if job:
        state = str(job.get("state") or state)
        pct = int(job.get("pct") or pct)
        error = job.get("error")
        if state == "downloading":
            label = f"Downloading {pct}%"
        elif state == "failed":
            label = "Download failed"
            state = "failed"
        elif state == "ready":
            label = "Ready / Installed"
            installed = True
    actions = []
    if state == "downloading":
        actions = []
    elif installed:
        actions = ["delete"]
    else:
        actions = ["download"]
    guard = preflight(spec.model_id)
    ram_labels = {
        "fits": "Fits into RAM",
        "tight": f"Tight memory (~{spec.ram_mb:.0f} MB needed, {guard.available_ram_mb or 0:.0f} MB free)",
        "unsafe": f"Too little RAM (~{spec.ram_mb:.0f} MB needed, {guard.available_ram_mb or 0:.0f} MB free)",
    }
    return {
        "id": spec.model_id,
        "slot": spec.slot,
        "name": spec.name,
        "description": spec.description,
        "diskMb": spec.disk_mb,
        "ramMb": spec.ram_mb,
        "targetHardware": spec.target_hardware,
        "installed": installed,
        "status": state,
        "statusLabel": label,
        "progressPct": pct,
        "error": error,
        "actions": actions,
        "ramFit": guard.ram_fit,
        "ramFitLabel": ram_labels[guard.ram_fit],
        "diskOk": guard.disk_ok,
        "preflight": guard.to_dict(),
    }


def ui_snapshot() -> dict[str, Any]:
    """Single payload for Settings / SnapDrago model manager."""
    cfg = active()
    pool = process_pool()
    hw = probe_hardware()
    return {
        "hardware": hw.to_dict(),
        "storage": storage_report(),
        "pool": {
            "budgetMb": pool.max_ram_mb,
            "residentMb": pool.resident_ram_mb(),
            "slots": pool.status(),
        },
        "config": {
            "path": str(cfg.config_path or default_config_path()),
            "cacheDir": str(cfg.cache_dir),
            "sources": list(cfg.sources),
            "preferredProvider": cfg.preferred_provider or hw.preferred_provider,
            "maxRamMb": cfg.max_ram_mb,
        },
        "models": [model_card(spec.model_id) for spec in list_specs()],
    }


def start_download(model_id_or_slot: str, *, force: bool = False) -> dict[str, Any]:
    spec = get_spec(model_id_or_slot)
    guard = preflight(spec.model_id)
    if not guard.ok and not force:
        return {
            "id": spec.model_id,
            "state": "blocked",
            "preflight": guard.to_dict(),
            "error": guard.message,
        }
    with _LOCK:
        current = _JOBS.get(spec.model_id)
        if current and current.get("state") == "downloading":
            return dict(current)
        job: dict[str, Any] = {
            "id": spec.model_id,
            "state": "downloading",
            "pct": 0,
            "downloaded": 0,
            "total": 0,
            "error": None,
        }
        _JOBS[spec.model_id] = job

    def _progress(name: str, downloaded: int, total: int) -> None:
        with _LOCK:
            job["downloaded"] = downloaded
            job["total"] = total
            job["pct"] = int(downloaded * 100 / total) if total else 0

    def _run() -> None:
        try:
            download_model(spec.model_id, progress=_progress, force=force)
            with _LOCK:
                job["state"] = "ready"
                job["pct"] = 100
                job["error"] = None
        except Exception as exc:
            with _LOCK:
                job["state"] = "failed"
                job["error"] = str(exc)

    threading.Thread(target=_run, name=f"hexagon-dl-{spec.model_id}", daemon=True).start()
    return dict(job)


def delete_cached(model_id_or_slot: str) -> dict[str, Any]:
    spec = get_spec(model_id_or_slot)
    delete_model(spec.model_id)
    with _LOCK:
        _JOBS.pop(spec.model_id, None)
    return model_card(spec.model_id)


def reset_jobs() -> None:
    with _LOCK:
        _JOBS.clear()

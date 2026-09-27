"""Durable download jobs under the shared cache.

Owner records live at ``<cache>/jobs/<id>.json``. A process blocked on
another app's slot lock writes ``<id>.wait-<pid>.json`` instead, so it
cannot clobber the download that is actually moving bytes. ``jobs/.lock``
guards read-modify-write. Schema is additive: unknown keys are kept.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .leases import pid_alive
from .lock import FileLock

JOBS_DIRNAME = "jobs"
ACTIVE_STATES = frozenset({"downloading", "cancelling", "waiting_on_lock"})
_STALE_PIDLESS_S = 15.0


def jobs_dir(cache_dir: Path | None = None) -> Path:
    from .config import active

    root = Path(cache_dir) if cache_dir is not None else active().cache_dir
    return root / JOBS_DIRNAME


def _safe(model_id: str) -> str:
    name = str(model_id).strip().replace("\\", "/").replace("/", "--")
    if not name or name in {".", ".."} or any(ch in name for ch in '<>:"|?*'):
        raise ValueError(f"Unsafe job id {model_id!r}")
    return name


def _owner_path(root: Path, model_id: str) -> Path:
    return root / f"{_safe(model_id)}.json"


def _waiter_path(root: Path, model_id: str, pid: int) -> Path:
    return root / f"{_safe(model_id)}.wait-{int(pid)}.json"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("id"):
        return None
    return data


def _write(path: Path, job: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in job.items() if not str(key).startswith("_")}
    payload["updatedAt"] = time.time()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _with_root(cache_dir: Path | None, fn):
    root = jobs_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / ".lock", timeout_s=30, record_holder=False):
        return fn(root)


def _mark_stale(job: dict[str, Any]) -> dict[str, Any] | None:
    """Return a failed copy when the owner process is gone, else None."""
    if job.get("state") not in ACTIVE_STATES:
        return None
    pid = job.get("pid")
    try:
        pid_int = int(pid) if pid is not None else 0
    except (TypeError, ValueError):
        pid_int = 0
    if pid_int and pid_alive(pid_int):
        return None
    if not pid_int:
        updated = float(job.get("updatedAt") or job.get("startedAt") or 0)
        if time.time() - updated < _STALE_PIDLESS_S:
            return None
    failed = dict(job)
    failed["state"] = "failed"
    failed["error"] = (
        f"Download process exited before the job finished (pid {pid_int or 'unknown'})"
    )
    failed["errorType"] = "ProcessExited"
    failed["finishedAt"] = time.time()
    return failed


def load_shared_jobs(cache_dir: Path | None = None) -> list[dict[str, Any]]:
    """All persisted jobs. Dead owners of active jobs are marked failed on disk."""
    root = jobs_dir(cache_dir)
    if not root.is_dir():
        return []

    def read_all(locked: Path) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for path in sorted(locked.glob("*.json")):
            if path.name.startswith("."):
                continue
            job = _read(path)
            if job is None:
                continue
            stale = _mark_stale(job)
            if stale is not None:
                _write(path, stale)
                job = stale
            found.append(job)
        return found

    try:
        return _with_root(cache_dir, read_all)
    except TimeoutError:
        return []


def read_owner_job(model_id: str, cache_dir: Path | None = None) -> dict[str, Any] | None:
    path = _owner_path(jobs_dir(cache_dir), model_id)
    return _read(path)


def foreign_active_owner(model_id: str, cache_dir: Path | None = None) -> dict[str, Any] | None:
    """Live owner job that belongs to another process, if any."""
    job = read_owner_job(model_id, cache_dir)
    if not job or job.get("role") == "waiter" or job.get("state") not in ACTIVE_STATES:
        return None
    try:
        pid = int(job.get("pid") or 0)
    except (TypeError, ValueError):
        return None
    if pid <= 0 or pid == os.getpid() or not pid_alive(pid):
        return None
    return job


def save_owner_job(job: dict[str, Any], cache_dir: Path | None = None) -> dict[str, Any]:
    """Persist an owner record.

    If another live process already owns an active job for this id, that
    record is left untouched and returned.
    """
    model_id = str(job["id"])

    def write(root: Path) -> dict[str, Any]:
        path = _owner_path(root, model_id)
        existing = _read(path)
        payload = dict(job)
        if existing is not None:
            stale = _mark_stale(existing)
            if stale is not None:
                existing = None
            else:
                try:
                    epid = int(existing.get("pid") or 0)
                except (TypeError, ValueError):
                    epid = 0
                if (
                    existing.get("state") in ACTIVE_STATES
                    and epid
                    and epid != os.getpid()
                    and pid_alive(epid)
                ):
                    return existing
                if existing.get("cancelRequested") and payload.get("state") in ACTIVE_STATES:
                    payload["cancelRequested"] = True
        payload["role"] = "owner"
        payload["pid"] = os.getpid()
        _write(path, payload)
        stored = _read(path)
        return stored if stored is not None else payload

    return _with_root(cache_dir, write)


def save_waiter_job(job: dict[str, Any], cache_dir: Path | None = None) -> dict[str, Any]:
    """Persist this process's lock-wait record without touching the owner file."""
    model_id = str(job["id"])
    pid = os.getpid()

    def write(root: Path) -> dict[str, Any]:
        path = _waiter_path(root, model_id, pid)
        payload = dict(job)
        payload["role"] = "waiter"
        payload["pid"] = pid
        _write(path, payload)
        stored = _read(path)
        return stored if stored is not None else payload

    return _with_root(cache_dir, write)


def request_cancel(model_id: str, cache_dir: Path | None = None) -> dict[str, Any] | None:
    """Set ``cancelRequested`` on a live owner job. Returns the job, or None."""

    def write(root: Path) -> dict[str, Any] | None:
        path = _owner_path(root, model_id)
        existing = _read(path)
        if existing is None or existing.get("state") not in ACTIVE_STATES:
            return None
        existing["cancelRequested"] = True
        if int(existing.get("pid") or 0) == os.getpid():
            existing["state"] = "cancelling"
        _write(path, existing)
        return existing

    return _with_root(cache_dir, write)


def cancel_requested(model_id: str, cache_dir: Path | None = None) -> bool:
    job = read_owner_job(model_id, cache_dir)
    return bool(job and job.get("cancelRequested") and job.get("state") in ACTIVE_STATES)


def note_owner_pid(model_id: str, pid: int, cache_dir: Path | None = None) -> dict[str, Any] | None:
    """Record a detached worker pid without resetting progress the worker already wrote."""

    def write(root: Path) -> dict[str, Any] | None:
        path = _owner_path(root, model_id)
        existing = _read(path)
        if existing is None:
            return None
        current = existing.get("pid")
        if current in (None, pid):
            existing["pid"] = int(pid)
            _write(path, existing)
        stored = _read(path)
        return stored

    return _with_root(cache_dir, write)


def seed_owner_job(job: dict[str, Any], cache_dir: Path | None = None) -> dict[str, Any]:
    """Create the owner record if absent. Does not replace a live foreign job."""
    return _seed(job, cache_dir)


def _seed(job: dict[str, Any], cache_dir: Path | None) -> dict[str, Any]:
    model_id = str(job["id"])

    def write(root: Path) -> dict[str, Any]:
        path = _owner_path(root, model_id)
        existing = _read(path)
        if existing is not None:
            stale = _mark_stale(existing)
            if stale is not None:
                existing = None
            else:
                try:
                    epid = int(existing.get("pid") or 0)
                except (TypeError, ValueError):
                    epid = 0
                if (
                    existing.get("state") in ACTIVE_STATES
                    and epid
                    and epid != os.getpid()
                    and pid_alive(epid)
                ):
                    return existing
        payload = dict(job)
        payload["role"] = "owner"
        _write(path, payload)
        stored = _read(path)
        return stored if stored is not None else payload

    return _with_root(cache_dir, write)


def remove_jobs_for(model_id: str, *, pid: int | None = None, cache_dir: Path | None = None) -> None:
    """Drop this process's waiter file and, when ``pid`` matches, the owner file."""
    owner_pid = os.getpid() if pid is None else pid

    def wipe(root: Path) -> None:
        waiter = _waiter_path(root, model_id, os.getpid())
        waiter.unlink(missing_ok=True)
        path = _owner_path(root, model_id)
        existing = _read(path)
        if existing is None:
            return
        try:
            epid = int(existing.get("pid") or 0)
        except (TypeError, ValueError):
            epid = 0
        if epid in (0, owner_pid):
            path.unlink(missing_ok=True)

    root = jobs_dir(cache_dir)
    if not root.is_dir():
        return
    _with_root(cache_dir, wipe)


def remove_our_job_files(cache_dir: Path | None = None) -> None:
    root = jobs_dir(cache_dir)
    if not root.is_dir():
        return
    me = os.getpid()

    def wipe(locked: Path) -> None:
        for path in locked.glob("*.json"):
            if path.name.startswith("."):
                continue
            if f".wait-{me}.json" in path.name or path.name.endswith(f".wait-{me}.json"):
                path.unlink(missing_ok=True)
                continue
            job = _read(path)
            if job is None:
                continue
            try:
                epid = int(job.get("pid") or 0)
            except (TypeError, ValueError):
                epid = 0
            if epid == me:
                path.unlink(missing_ok=True)

    try:
        _with_root(cache_dir, wipe)
    except TimeoutError:
        return

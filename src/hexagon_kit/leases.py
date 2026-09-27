"""Cross-process slot leases for the shared cache.

One live holder per catalog slot unless the caller passes force=True.
Dead PIDs are reaped. Delete must refuse while a live lease exists.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .lock import FileLock

LEASES_NAME = "leases.json"


class ModelInUse(RuntimeError):
    """Delete or exclusive acquire refused because another process holds the slot."""

    def __init__(self, slot: str, holder: "Lease"):
        self.slot = slot
        self.holder = holder
        super().__init__(
            f"{slot} is loaded in {holder.exe} (pid {holder.pid}). "
            "Stop that app or pass force=True to load a second copy."
        )


@dataclass
class Lease:
    slot: str
    pid: int
    exe: str
    ram_mb: float
    model_id: str
    acquired_at: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _leases_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / LEASES_NAME


def _lock_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "leases.lock"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return int(code.value) == STILL_ACTIVE
            return True
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _caller_exe() -> str:
    try:
        return Path(sys.argv[0]).name or Path(sys.executable).name
    except Exception:
        return Path(sys.executable).name


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "slots": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "slots": {}}
    if not isinstance(data, dict):
        return {"version": 1, "slots": {}}
    slots = data.get("slots")
    if not isinstance(slots, dict):
        data["slots"] = {}
    return data


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _reap(slots: dict[str, Any]) -> dict[str, Any]:
    live: dict[str, Any] = {}
    for slot, raw in slots.items():
        if not isinstance(raw, dict):
            continue
        try:
            pid = int(raw.get("pid") or 0)
        except (TypeError, ValueError):
            continue
        if pid_alive(pid):
            live[slot] = raw
    return live


def _lease_from(slot: str, raw: dict[str, Any]) -> Lease:
    return Lease(
        slot=slot,
        pid=int(raw["pid"]),
        exe=str(raw.get("exe") or "unknown"),
        ram_mb=float(raw.get("ram_mb") or 0),
        model_id=str(raw.get("model_id") or ""),
        acquired_at=float(raw.get("acquired_at") or 0),
    )


def list_leases(cache_dir: Path | None = None) -> list[Lease]:
    from .config import active

    root = Path(cache_dir) if cache_dir is not None else active().cache_dir
    with FileLock(_lock_path(root)):
        data = _load(_leases_path(root))
        data["slots"] = _reap(data.get("slots") or {})
        _save(_leases_path(root), data)
        return [_lease_from(slot, raw) for slot, raw in data["slots"].items()]


def holder_for(slot: str, cache_dir: Path | None = None) -> Lease | None:
    for item in list_leases(cache_dir):
        if item.slot == slot:
            return item
    return None


def peer_ram_mb(cache_dir: Path | None = None, *, exclude_pid: int | None = None) -> float:
    me = os.getpid() if exclude_pid is None else exclude_pid
    return sum(item.ram_mb for item in list_leases(cache_dir) if item.pid != me)


def take_lease(
    slot: str,
    *,
    ram_mb: float,
    model_id: str,
    force: bool = False,
    cache_dir: Path | None = None,
) -> Lease:
    from .config import active

    root = Path(cache_dir) if cache_dir is not None else active().cache_dir
    with FileLock(_lock_path(root)):
        path = _leases_path(root)
        data = _load(path)
        data["slots"] = _reap(data.get("slots") or {})
        current = data["slots"].get(slot)
        if current is not None:
            existing = _lease_from(slot, current)
            if existing.pid == os.getpid():
                current["ram_mb"] = float(ram_mb)
                current["model_id"] = model_id
                current["acquired_at"] = time.time()
                _save(path, data)
                return _lease_from(slot, current)
            if not force:
                raise ModelInUse(slot, existing)
        record = {
            "pid": os.getpid(),
            "exe": _caller_exe(),
            "ram_mb": float(ram_mb),
            "model_id": model_id,
            "acquired_at": time.time(),
        }
        data["slots"][slot] = record
        _save(path, data)
        return _lease_from(slot, record)


def drop_lease(slot: str, cache_dir: Path | None = None) -> None:
    from .config import active

    root = Path(cache_dir) if cache_dir is not None else active().cache_dir
    with FileLock(_lock_path(root)):
        path = _leases_path(root)
        data = _load(path)
        data["slots"] = _reap(data.get("slots") or {})
        current = data["slots"].get(slot)
        if current is None:
            _save(path, data)
            return
        if int(current.get("pid") or 0) == os.getpid():
            data["slots"].pop(slot, None)
        _save(path, data)

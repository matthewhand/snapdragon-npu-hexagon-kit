"""
Pre-flight RAM and disk guard for Copilot+ PCs.

Checks live available memory (GlobalMemoryStatusEx / MemAvailable) and
free disk before download or activate, and suggests a lighter catalog
variant when the requested model would thrash the pagefile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import shutil

from .catalog import ModelSpec
from .config import active, get_spec, list_specs
from .hw import MemoryStatus, read_memory_status

# Leave headroom for Windows, the app UI, and the OS file cache.
RESERVE_RAM_MB = 1024.0
DISK_SLACK = 1.15  # extract/temp files


class PreflightBlocked(RuntimeError):
    """Download or activate refused until the caller passes force=True."""

    def __init__(self, result: "PreflightResult"):
        super().__init__(result.message)
        self.result = result


@dataclass
class PreflightResult:
    ok: bool
    ram_fit: str  # fits | tight | unsafe
    disk_ok: bool
    can_force: bool
    message: str
    suggest_id: str | None = None
    suggest_name: str | None = None
    available_ram_mb: float | None = None
    required_ram_mb: float = 0
    available_disk_mb: float | None = None
    required_disk_mb: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "ramFit": self.ram_fit,
            "diskOk": self.disk_ok,
            "canForce": self.can_force,
            "message": self.message,
            "suggestId": self.suggest_id,
            "suggestName": self.suggest_name,
            "availableRamMb": self.available_ram_mb,
            "requiredRamMb": self.required_ram_mb,
            "availableDiskMb": self.available_disk_mb,
            "requiredDiskMb": self.required_disk_mb,
        }


def _ram_fit(required_mb: float, memory: MemoryStatus | None) -> str:
    if memory is None:
        return "fits"
    available = memory.available_mb
    if required_mb > available:
        return "unsafe"
    if required_mb > max(0.0, available - RESERVE_RAM_MB):
        return "tight"
    return "fits"


def _lighter_suggestion(spec: ModelSpec, memory: MemoryStatus | None) -> tuple[str | None, str | None]:
    candidates = [
        other
        for other in list_specs()
        if other.slot == spec.slot
        and other.model_id != spec.model_id
        and other.ram_mb < spec.ram_mb
        and _ram_fit(other.ram_mb, memory) == "fits"
    ]
    candidates.sort(key=lambda item: item.ram_mb)
    if not candidates:
        return None, None
    pick = candidates[0]
    return pick.model_id, pick.name


def preflight(model_id_or_slot: str, *, memory: MemoryStatus | None = None) -> PreflightResult:
    spec = get_spec(model_id_or_slot)
    mem = memory if memory is not None else read_memory_status()
    cache = active().cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    free_disk_mb = shutil.disk_usage(cache).free / (1024 * 1024)
    need_disk = spec.disk_mb * DISK_SLACK
    disk_ok = free_disk_mb >= need_disk
    fit = _ram_fit(spec.ram_mb, mem)
    suggest_id, suggest_name = _lighter_suggestion(spec, mem)

    available_mb = mem.available_mb if mem else None
    if not disk_ok:
        message = (
            f"{spec.name} needs ~{spec.disk_mb:.0f} MB on disk "
            f"but only {free_disk_mb:.0f} MB is free."
        )
    elif fit == "unsafe":
        avail = f"{available_mb:.0f} MB" if available_mb is not None else "unknown"
        message = (
            f"{spec.name} requires ~{spec.ram_mb:.0f} MB RAM, but only {avail} is available. "
            "Loading it would force pagefile thrashing on this 16 GB Copilot+ PC."
        )
        if suggest_name:
            message += f" Suggested: {suggest_name}."
    elif fit == "tight":
        avail = f"{available_mb:.0f} MB" if available_mb is not None else "unknown"
        message = (
            f"Tight memory: {spec.name} wants ~{spec.ram_mb:.0f} MB; {avail} is free "
            f"(keeping {RESERVE_RAM_MB:.0f} MB for Windows)."
        )
        if suggest_name:
            message += f" Lighter option: {suggest_name}."
    else:
        message = f"{spec.name} fits current RAM and disk."

    ok = disk_ok and fit == "fits"
    return PreflightResult(
        ok=ok,
        ram_fit=fit,
        disk_ok=disk_ok,
        can_force=fit != "fits" or not disk_ok,
        message=message,
        suggest_id=suggest_id,
        suggest_name=suggest_name,
        available_ram_mb=available_mb,
        required_ram_mb=spec.ram_mb,
        available_disk_mb=round(free_disk_mb, 1),
        required_disk_mb=round(need_disk, 1),
    )

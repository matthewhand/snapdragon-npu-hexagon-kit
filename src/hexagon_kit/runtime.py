"""
In-process model residency pool.

Disk is shared via the XDG cache. Hexagon NPU weights still live in each
process unless a daemon owns the sessions. This pool is the in-process half:
one load per slot, refcounted, budgeted for 16 GB first-gen Copilot+ PCs.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import ensure_model, resolve
from .config import get_spec
from .preflight import PreflightBlocked, preflight

Loader = Callable[[Path], Any]
Unloader = Callable[[Any], None]


class PoolBudgetExceeded(RuntimeError):
    pass


@dataclass
class _Resident:
    slot: str
    instance: Any
    ram_mb: float
    refs: int = 0
    last_used: float = field(default_factory=time.monotonic)


class ModelPool:
    """
    Centralize load/unload for STT/TTS/LLM inside one process.

    Default budget (~3.5 GB) leaves headroom on a 16 GB Copilot+ box for
    Windows, the app UI, and the OS page cache of the same XDG files.
    """

    def __init__(
        self,
        *,
        max_ram_mb: float | None = None,
        cache_dir: Path | None = None,
    ):
        from .config import active

        cfg = active()
        self.max_ram_mb = float(cfg.max_ram_mb if max_ram_mb is None else max_ram_mb)
        self.cache_dir = cache_dir if cache_dir is not None else cfg.cache_dir
        self._lock = threading.RLock()
        self._loaders: dict[str, tuple[Loader, Unloader | None, float | None]] = {}
        self._resident: dict[str, _Resident] = {}

    def register(
        self,
        slot: str,
        loader: Loader,
        *,
        unloader: Unloader | None = None,
        ram_mb: float | None = None,
    ) -> None:
        spec = get_spec(slot)
        with self._lock:
            self._loaders[spec.slot] = (loader, unloader, ram_mb)

    def acquire(self, slot: str, *, download: bool = False, force: bool = False) -> Any:
        spec = get_spec(slot)
        with self._lock:
            if spec.slot in self._resident:
                item = self._resident[spec.slot]
                item.refs += 1
                item.last_used = time.monotonic()
                return item.instance

            loader_entry = self._loaders.get(spec.slot)
            if loader_entry is None:
                raise KeyError(
                    f"No loader registered for slot {spec.slot!r}. "
                    "Call pool.register(slot, loader) first."
                )
            if not force:
                guard = preflight(spec.model_id)
                if not guard.ok:
                    raise PreflightBlocked(guard)
            loader, _unloader, ram_override = loader_entry
            ram_mb = float(ram_override if ram_override is not None else spec.ram_mb)
            self._evict_for(ram_mb)
            path = (
                ensure_model(spec.model_id, cache_dir=self.cache_dir, force=True)
                if download
                else resolve(spec.model_id, cache_dir=self.cache_dir)
            )
            instance = loader(path)
            self._resident[spec.slot] = _Resident(
                slot=spec.slot,
                instance=instance,
                ram_mb=ram_mb,
                refs=1,
            )
            return instance

    def release(self, slot: str) -> None:
        spec = get_spec(slot)
        with self._lock:
            item = self._resident.get(spec.slot)
            if item is None:
                return
            item.refs = max(0, item.refs - 1)
            item.last_used = time.monotonic()

    def unload(self, slot: str, *, force: bool = False) -> None:
        spec = get_spec(slot)
        with self._lock:
            item = self._resident.get(spec.slot)
            if item is None:
                return
            if item.refs > 0 and not force:
                raise RuntimeError(
                    f"Slot {spec.slot!r} still has {item.refs} reference(s); "
                    "release() first or pass force=True."
                )
            self._drop(spec.slot)

    def unload_unused(self) -> list[str]:
        dropped: list[str] = []
        with self._lock:
            for slot in list(self._resident):
                if self._resident[slot].refs == 0:
                    self._drop(slot)
                    dropped.append(slot)
        return dropped

    def status(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "slot": item.slot,
                    "refs": item.refs,
                    "ram_mb": item.ram_mb,
                    "last_used": item.last_used,
                }
                for item in self._resident.values()
            ]

    def resident_ram_mb(self) -> float:
        with self._lock:
            return sum(item.ram_mb for item in self._resident.values())

    def _evict_for(self, incoming_mb: float) -> None:
        if incoming_mb > self.max_ram_mb:
            raise PoolBudgetExceeded(
                f"{incoming_mb:.0f} MB model exceeds pool budget "
                f"({self.max_ram_mb:.0f} MB). First-gen Copilot+ is 16 GB RAM."
            )
        unused = sorted(
            (item for item in self._resident.values() if item.refs == 0),
            key=lambda item: item.last_used,
        )
        used = self.resident_ram_mb()
        for item in unused:
            if used + incoming_mb <= self.max_ram_mb:
                break
            used -= item.ram_mb
            self._drop(item.slot)
        if self.resident_ram_mb() + incoming_mb > self.max_ram_mb:
            held = ", ".join(
                f"{item.slot}({item.refs} refs, {item.ram_mb:.0f} MB)"
                for item in self._resident.values()
            )
            raise PoolBudgetExceeded(
                f"Need {incoming_mb:.0f} MB but pool holds {self.resident_ram_mb():.0f} MB "
                f"of in-use models ({held}). Release a slot or raise max_ram_mb."
            )

    def _drop(self, slot: str) -> None:
        item = self._resident.pop(slot, None)
        if item is None:
            return
        entry = self._loaders.get(slot)
        if entry and entry[1] is not None:
            entry[1](item.instance)


_PROCESS_POOL: ModelPool | None = None
_PROCESS_LOCK = threading.Lock()


def process_pool() -> ModelPool:
    """Single pool per process so STT + TTS in one app share residency policy."""
    global _PROCESS_POOL
    with _PROCESS_LOCK:
        if _PROCESS_POOL is None:
            _PROCESS_POOL = ModelPool()
        return _PROCESS_POOL


def reset_process_pool() -> None:
    global _PROCESS_POOL
    with _PROCESS_LOCK:
        _PROCESS_POOL = None

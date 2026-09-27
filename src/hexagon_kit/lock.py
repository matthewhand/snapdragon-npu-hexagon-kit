"""Cross-process exclusive file lock (stdlib only).

Windows: msvcrt.locking. POSIX: fcntl.flock.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from collections.abc import Callable
from types import TracebackType
from typing import Any


class LockWaitExceeded(TimeoutError):
    """The lock was still held after a stretch with no forward progress.

    Subclasses ``TimeoutError``. ``holder`` is ``{"pid", "exe"}`` when the
    holder wrote a sidecar, otherwise empty.
    """

    def __init__(self, path: Path, *, holder: dict[str, Any] | None, waited_s: float):
        self.path = Path(path)
        self.holder = dict(holder or {})
        self.waited_s = waited_s
        pid = self.holder.get("pid")
        exe = self.holder.get("exe")
        who = ""
        if pid:
            who = f" held by pid {pid}"
            if exe:
                who += f" ({exe})"
        super().__init__(
            f"Timed out after {waited_s:.0f}s waiting for lock {self.path}{who}"
        )


def _caller_exe() -> str:
    try:
        return Path(sys.argv[0]).name or Path(sys.executable).name
    except Exception:
        return Path(sys.executable).name


class FileLock:
    """Exclusive lock on a sibling `.lock` file.

    ``timeout_s`` is a stall budget. When ``activity`` is given, it returns a
    byte count (or None). Each increase resets the budget, so a download that
    is still arriving is not failed at a fixed 300s. ``on_wait`` is called
    with the holder's ``{"pid", "exe"}`` while this process is blocked.
    """

    def __init__(
        self,
        path: Path,
        *,
        timeout_s: float = 300.0,
        on_wait: Callable[[dict[str, Any]], None] | None = None,
        activity: Callable[[], int | None] | None = None,
        record_holder: bool = True,
    ):
        self.path = Path(path)
        self.timeout_s = timeout_s
        self.on_wait = on_wait
        self.activity = activity
        self.record_holder = record_holder
        self._fh = None

    def __enter__(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+b")
        if self._fh.tell() == 0:
            self._fh.write(b"\0")
            self._fh.flush()
        started = time.monotonic()
        deadline = started + self.timeout_s
        last_activity: int | None = None
        try:
            while True:
                try:
                    self._lock()
                except OSError:
                    now = time.monotonic()
                    holder = self.read_holder()
                    if self.on_wait is not None:
                        self.on_wait(holder)
                    if self.activity is not None:
                        try:
                            current = self.activity()
                        except Exception:
                            current = None
                        if isinstance(current, int) and current >= 0:
                            if last_activity is not None and current > last_activity:
                                deadline = now + self.timeout_s
                            last_activity = current
                    if now >= deadline:
                        raise LockWaitExceeded(
                            self.path,
                            holder=holder,
                            waited_s=now - started,
                        )
                    time.sleep(0.05)
                    continue
                if self.record_holder:
                    try:
                        self._write_holder()
                    except BaseException:
                        try:
                            self._unlock()
                        finally:
                            raise
                return self
        except BaseException:
            self._close_unlocked()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fh is None:
            return
        try:
            self._unlock()
        finally:
            if self.record_holder:
                self._clear_holder()
            self._close_unlocked()

    def holder_path(self) -> Path:
        return self.path.with_name(self.path.name + ".holder")

    def read_holder(self) -> dict[str, Any]:
        path = self.holder_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"pid": None, "exe": None}
        if not isinstance(data, dict):
            return {"pid": None, "exe": None}
        pid = data.get("pid")
        try:
            pid = int(pid) if pid is not None else None
        except (TypeError, ValueError):
            pid = None
        exe = data.get("exe")
        return {"pid": pid, "exe": str(exe) if exe else None}

    def _write_holder(self) -> None:
        payload = {"pid": os.getpid(), "exe": _caller_exe(), "since": time.time()}
        path = self.holder_path()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        for attempt in range(6):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.02 * (attempt + 1))

    def _clear_holder(self) -> None:
        try:
            self.holder_path().unlink(missing_ok=True)
        except OSError:
            pass

    def _close_unlocked(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.close()
        finally:
            self._fh = None

    def _lock(self) -> None:
        assert self._fh is not None
        self._fh.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(self) -> None:
        assert self._fh is not None
        self._fh.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)

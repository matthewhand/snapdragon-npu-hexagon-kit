"""Cross-process exclusive file lock (stdlib only).

Windows: msvcrt.locking. POSIX: fcntl.flock.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import TracebackType


class FileLock:
    """Exclusive lock on a sibling `.lock` file. Blocking."""

    def __init__(self, path: Path, *, timeout_s: float = 300.0):
        self.path = Path(path)
        self.timeout_s = timeout_s
        self._fh = None

    def __enter__(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+b")
        if self._fh.tell() == 0:
            self._fh.write(b"\0")
            self._fh.flush()
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                self._lock()
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    self._fh.close()
                    self._fh = None
                    raise TimeoutError(f"Timed out waiting for lock {self.path}")
                time.sleep(0.05)

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
            self._fh.close()
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

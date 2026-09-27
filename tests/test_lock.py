import threading
import time

import pytest

from hexagon_kit.lock import FileLock, LockWaitExceeded


def test_file_lock_is_exclusive(tmp_path):
    lock_path = tmp_path / "slot.lock"
    order: list[str] = []

    def worker(name: str) -> None:
        with FileLock(lock_path):
            order.append(f"{name}-in")
            time.sleep(0.08)
            order.append(f"{name}-out")

    threads = [
        threading.Thread(target=worker, args=("a",)),
        threading.Thread(target=worker, args=("b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert order in (
        ["a-in", "a-out", "b-in", "b-out"],
        ["b-in", "b-out", "a-in", "a-out"],
    )


def _hold(path, ready: threading.Event, release_after: float) -> None:
    with FileLock(path):
        ready.set()
        time.sleep(release_after)


def test_lock_wait_extends_while_activity_grows(tmp_path):
    path = tmp_path / "slot.lock"
    ready = threading.Event()
    holder = threading.Thread(target=_hold, args=(path, ready, 0.55))
    holder.start()
    assert ready.wait(2)
    ticks = {"n": 0}

    def activity() -> int:
        ticks["n"] += 1
        return ticks["n"]

    started = time.monotonic()
    with FileLock(path, timeout_s=0.2, activity=activity):
        pass
    elapsed = time.monotonic() - started
    holder.join(timeout=2)
    assert elapsed >= 0.4


def test_lock_wait_fails_when_holder_makes_no_progress(tmp_path):
    path = tmp_path / "slot.lock"
    ready = threading.Event()
    holder = threading.Thread(target=_hold, args=(path, ready, 1.2))
    holder.start()
    assert ready.wait(2)
    waits: list[dict] = []
    with pytest.raises(LockWaitExceeded) as exc:
        with FileLock(
            path,
            timeout_s=0.3,
            activity=lambda: 10,
            on_wait=waits.append,
        ):
            pass
    holder.join(timeout=2)
    assert exc.value.holder.get("pid")
    assert waits and waits[0]["pid"]
    assert "held by pid" in str(exc.value)

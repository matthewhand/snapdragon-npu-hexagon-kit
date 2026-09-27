import threading
import time

from hexagon_kit.lock import FileLock


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

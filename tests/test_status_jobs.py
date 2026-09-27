import os
import threading
import time

import pytest

import hexagon_kit
from hexagon_kit.catalog import get_spec
from hexagon_kit.leases import drop_lease, take_lease
from hexagon_kit.status import (
    SNAPSHOT_SCHEMA_VERSION,
    DownloadInProgress,
    _Progress,
    cancel_download,
    delete_cached,
    download_jobs,
    get_job,
    model_card,
    reset_jobs,
    slot_info,
    start_download,
    ui_snapshot,
)

PEER_PID = 525252


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    monkeypatch.setattr("hexagon_kit.cache._require_preflight", lambda model_id, force=False: None)
    reset_jobs()
    yield tmp_path
    reset_jobs()


def _install(root, slot):
    spec = get_spec(slot)
    dest = root / spec.slot
    dest.mkdir(parents=True, exist_ok=True)
    for name in spec.expected_files:
        (dest / name).write_bytes(b"x")
    return spec


def _wait_state(model, states, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = get_job(model)
        if job and job["state"] in states:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job for {model} never reached {states}: {get_job(model)}")


def test_slot_info_separates_chat_from_speech():
    assert slot_info("llm")["chatCapable"] is True
    assert slot_info("llm")["modality"] == "text-generation"
    for slot in ("stt", "tts", "vision"):
        assert slot_info(slot)["chatCapable"] is False
    assert slot_info("stt")["modality"] == "speech-to-text"
    assert slot_info("tts")["modality"] == "text-to-speech"
    custom = slot_info("classify")
    assert custom["modality"] == "other" and custom["chatCapable"] is False


def test_snapshot_metadata_and_slots(cache):
    snap = ui_snapshot()
    assert snap["schemaVersion"] == SNAPSHOT_SCHEMA_VERSION
    assert snap["kitVersion"] == hexagon_kit.__version__
    assert isinstance(snap["generatedAt"], float)
    assert snap["jobs"] == []
    slots = {row["slot"]: row for row in snap["slots"]}
    assert set(slots) >= {"stt", "tts", "llm", "vision"}
    assert slots["llm"]["modelId"] == "smollm2_135m_int8"
    assert slots["stt"]["chatCapable"] is False
    card = next(c for c in snap["models"] if c["slot"] == "tts")
    for key in (
        "statusLabel", "progressPct", "error", "errorType", "slotLabel", "modality",
        "chatCapable", "canDelete", "deleteBlockedReason", "heldBy", "peerDownload",
        "job", "path", "sizeBytes", "downloadedBytes", "totalBytes",
    ):
        assert key in card, key


def test_progress_aggregates_across_artifacts():
    spec = get_spec("kokoro_int8")
    job = {"pct": 0}
    progress = _Progress(spec, job, threading.Event())
    seen = []
    for name, total in (("kokoro-v1.0.int8.onnx", 1000), ("voices-v1.0.bin", 1000)):
        for done in (250, 500, 1000):
            progress(name, done, total)
            seen.append(job["pct"])
    assert seen == sorted(seen), seen
    assert seen[-1] == 99
    assert job["downloaded"] == 2000
    assert job["currentFile"] == "voices-v1.0.bin"


def test_progress_unknown_length_uses_catalog_estimate():
    spec = get_spec("whisper_tiny_int8")
    job = {"pct": 0}
    _Progress(spec, job, threading.Event())("x.tar.bz2", 12 * 1024 * 1024, 0)
    assert job["pct"] == 10


def test_failed_download_surfaces_error(cache, monkeypatch):
    def broken(artifact, staging, progress):
        raise ValueError("SHA-256 mismatch for voices-v1.0.bin")

    monkeypatch.setattr("hexagon_kit.cache._fetch_artifact", broken)
    start_download("tts", force=True)
    _wait_state("tts", {"failed"})
    card = model_card("tts")
    assert card["status"] == "failed"
    assert card["statusLabel"] == "Download failed"
    assert "SHA-256 mismatch" in card["error"]
    assert card["errorType"] == "ValueError"
    assert card["actions"] == ["download"]
    assert card["job"]["finishedAt"] is not None
    assert [job["id"] for job in download_jobs()] == ["kokoro_int8"]


def test_cancel_download_stops_and_cleans_staging(cache, monkeypatch):
    started = threading.Event()

    def slow(artifact, staging, progress):
        started.set()
        for i in range(500):
            progress(artifact.filename, i * 1024, 500 * 1024)
            time.sleep(0.01)
        raise AssertionError("cancel was not honoured")

    monkeypatch.setattr("hexagon_kit.cache._fetch_artifact", slow)
    job = start_download("tts", force=True)
    assert job["state"] == "downloading"
    assert "_cancel" not in job
    assert started.wait(5)
    card = model_card("tts")
    assert card["status"] == "downloading"
    assert card["actions"] == ["cancel"]
    with pytest.raises(DownloadInProgress):
        delete_cached("tts")

    result = cancel_download("tts")
    assert result["cancelRequested"] is True
    _wait_state("tts", {"cancelled"})
    card = model_card("tts")
    assert card["status"] == "cancelled"
    assert card["actions"] == ["download"]
    assert not list(cache.glob(".tts-staging-*"))
    assert cancel_download("tts")["cancelRequested"] is False


def test_cancel_without_job_is_noop(cache):
    result = cancel_download("stt")
    assert result == {"id": "whisper_tiny_int8", "slot": "stt", "state": "idle", "cancelRequested": False}


def test_stale_ready_job_does_not_override_disk(cache, monkeypatch):
    def ok(artifact, staging, progress):
        path = staging / artifact.filename
        path.write_bytes(b"x")
        return path

    monkeypatch.setattr("hexagon_kit.cache._fetch_artifact", ok)
    start_download("tts", force=True)
    _wait_state("tts", {"ready"})
    assert model_card("tts")["status"] == "ready"
    import shutil

    shutil.rmtree(cache / "tts")
    card = model_card("tts")
    assert card["installed"] is False
    assert card["status"] == "downloadable"
    assert card["actions"] == ["download"]


def test_lease_blocks_delete_action(cache):
    spec = _install(cache, "stt")
    take_lease("stt", ram_mb=150, model_id=spec.model_id, cache_dir=cache)
    try:
        card = model_card("stt")
        assert card["installed"] is True
        assert card["actions"] == []
        assert card["canDelete"] is False
        assert "pid" in card["deleteBlockedReason"]
        assert card["heldBy"]["pid"] == os.getpid()
        assert card["heldBy"]["isSelf"] is True
        snap = ui_snapshot()
        assert snap["pool"]["peers"][0]["isSelf"] is True
        assert next(r for r in snap["slots"] if r["slot"] == "stt")["heldBy"]["pid"] == os.getpid()
    finally:
        drop_lease("stt", cache_dir=cache)
    card = model_card("stt")
    assert card["actions"] == ["delete"]
    assert card["canDelete"] is True
    assert card["path"] == str(cache / "stt")
    assert card["sizeBytes"] > 0


def test_peer_download_shows_on_card(cache, monkeypatch):
    live = {PEER_PID, os.getpid()}
    monkeypatch.setattr("hexagon_kit.cache.pid_alive", lambda pid: pid in live)
    (cache / f".vision-staging-{PEER_PID}").mkdir()
    card = model_card("vision")
    assert card["status"] == "downloading"
    assert card["actions"] == []
    assert card["peerDownload"]["pid"] == PEER_PID
    assert "another app" in card["statusLabel"]


def _alive_for(peer: int):
    def alive(pid: int) -> bool:
        return pid == peer or pid == os.getpid()

    return alive


def test_persisted_job_is_visible_without_local_memory(cache, monkeypatch):
    monkeypatch.setattr("hexagon_kit.jobs.pid_alive", _alive_for(PEER_PID))
    monkeypatch.setattr("hexagon_kit.status.pid_alive", _alive_for(PEER_PID))
    payload = {
        "id": "kokoro_int8",
        "slot": "tts",
        "source": "catalog",
        "role": "owner",
        "state": "downloading",
        "phase": "downloading",
        "pct": 37,
        "downloaded": 1000,
        "total": 4000,
        "currentFile": "voices-v1.0.bin",
        "error": None,
        "errorType": None,
        "pid": PEER_PID,
        "startedAt": time.time(),
        "finishedAt": None,
        "updatedAt": time.time(),
    }
    jobs = cache / "jobs"
    jobs.mkdir()
    (jobs / "kokoro_int8.json").write_text(__import__("json").dumps(payload), encoding="utf-8")
    card = model_card("tts")
    assert card["status"] == "downloading"
    assert card["progressPct"] == 37
    assert card["downloadedBytes"] == 1000
    assert card["job"]["pid"] == PEER_PID
    assert card["actions"] == ["cancel"]
    snap = ui_snapshot()
    assert snap["jobs"][0]["pct"] == 37


def test_dead_owner_job_is_marked_failed(cache):
    payload = {
        "id": "kokoro_int8",
        "slot": "tts",
        "source": "catalog",
        "role": "owner",
        "state": "downloading",
        "pct": 10,
        "downloaded": 1,
        "total": 10,
        "pid": 999999,
        "startedAt": time.time(),
        "updatedAt": time.time(),
    }
    jobs = cache / "jobs"
    jobs.mkdir()
    (jobs / "kokoro_int8.json").write_text(__import__("json").dumps(payload), encoding="utf-8")
    card = model_card("tts")
    assert card["status"] == "failed"
    assert card["errorType"] == "ProcessExited"
    assert "exited" in card["error"]


def test_progress_callback_and_watch(cache, monkeypatch):
    from hexagon_kit.status import watch_jobs

    seen: list[str] = []
    finished: list[str] = []

    def ok(artifact, staging, progress):
        path = staging / artifact.filename
        path.write_bytes(b"x")
        if progress:
            progress(artifact.filename, 4, 4)
        return path

    monkeypatch.setattr("hexagon_kit.cache._fetch_artifact", ok)
    watcher = watch_jobs(0.05)
    start_download(
        "tts",
        force=True,
        on_progress=lambda job: seen.append(str(job.get("phase"))),
        on_done=lambda job: finished.append(job["state"]),
    )
    first = next(watcher)
    watcher.close()
    assert any(job["id"] == "kokoro_int8" for job in first)
    _wait_state("tts", {"ready"})
    assert finished == ["ready"]
    assert seen


def test_lock_wait_is_distinct_until_the_budget_is_spent(cache, monkeypatch):
    monkeypatch.setattr("hexagon_kit.cache.LOCK_WAIT_S", 0.35)

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("fetch should not start while the lock is held")

    monkeypatch.setattr("hexagon_kit.cache._fetch_artifact", fail_fetch)
    ready = threading.Event()

    def hold():
        from hexagon_kit.lock import FileLock

        with FileLock(cache / "tts.lock"):
            ready.set()
            time.sleep(1.5)

    holder = threading.Thread(target=hold)
    holder.start()
    assert ready.wait(2)
    start_download("tts", force=True)
    waiting = _wait_state("tts", {"waiting_on_lock"})
    assert waiting["lockHolder"]["pid"]
    card = model_card("tts")
    assert card["status"] == "waiting_on_lock"
    assert card["error"] is None
    assert "lock" in card["statusLabel"].lower()
    failed = _wait_state("tts", {"failed"}, timeout=3)
    assert failed["errorType"] == "LockWaitExceeded"
    assert model_card("tts")["status"] == "failed"
    holder.join(timeout=3)


def test_cancel_flag_is_written_for_another_process(cache, monkeypatch):
    monkeypatch.setattr("hexagon_kit.jobs.pid_alive", _alive_for(PEER_PID))
    monkeypatch.setattr("hexagon_kit.status.pid_alive", _alive_for(PEER_PID))
    payload = {
        "id": "kokoro_int8",
        "slot": "tts",
        "role": "owner",
        "state": "downloading",
        "pct": 4,
        "pid": PEER_PID,
        "startedAt": time.time(),
        "updatedAt": time.time(),
    }
    jobs = cache / "jobs"
    jobs.mkdir()
    path = jobs / "kokoro_int8.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    result = cancel_download("tts")
    assert result["cancelRequested"] is True
    stored = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert stored["cancelRequested"] is True
    assert stored["state"] == "downloading"

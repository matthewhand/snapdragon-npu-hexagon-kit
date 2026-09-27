from pathlib import Path

from hexagon_kit.cache import download_model, is_installed
from hexagon_kit.catalog import get_spec
from hexagon_kit.hf import link_or_copy


def test_link_or_copy_hardlink_or_copy(tmp_path):
    src = tmp_path / "blob.bin"
    src.write_bytes(b"shared-bytes")
    dest = tmp_path / "slot" / "blob.bin"
    link_or_copy(src, dest)
    assert dest.read_bytes() == b"shared-bytes"


def test_download_prefers_hf_cache_when_token_set(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    spec = get_spec("whisper_tiny_int8")
    fetched: list[tuple[str, str]] = []

    def fake_hf(repo_id: str, filename: str, *, revision: str = "main") -> Path:
        fetched.append((repo_id, filename))
        blob = tmp_path / "hf-cache" / filename.replace("/", "_")
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(f"hf:{filename}".encode())
        return blob

    monkeypatch.setattr("hexagon_kit.cache.hf_hub_file", fake_hf)

    def fail_github(*args, **kwargs):
        raise AssertionError("GitHub fetch should not run when HF cache succeeds")

    monkeypatch.setattr("hexagon_kit.cache._fetch_artifact", fail_github)
    monkeypatch.setattr("hexagon_kit.cache._require_preflight", lambda *a, **k: None)

    dest = download_model("whisper_tiny_int8", cache_dir=tmp_path, force=True)
    assert dest == tmp_path / "stt"
    assert is_installed("whisper_tiny_int8", tmp_path)
    assert {name for _, name in fetched} == set(spec.expected_files)
    assert all(repo == "csukuangfj/sherpa-onnx-whisper-tiny.en" for repo, _ in fetched)
    for name in spec.expected_files:
        assert (dest / name).read_bytes() == f"hf:{name}".encode()


def test_download_falls_back_to_github_when_hf_hash_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    spec = get_spec("kokoro_int8")

    def bad_hf(repo_id: str, filename: str, *, revision: str = "main") -> Path:
        blob = tmp_path / "hf-cache" / Path(filename).name
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"wrong-bytes")
        return blob

    monkeypatch.setattr("hexagon_kit.cache.hf_hub_file", bad_hf)

    def github_fetch(artifact, staging, progress):
        path = staging / artifact.filename
        path.write_bytes(b"github-ok")
        return path

    monkeypatch.setattr("hexagon_kit.cache._fetch_artifact", github_fetch)
    monkeypatch.setattr("hexagon_kit.cache._require_preflight", lambda *a, **k: None)

    dest = download_model("kokoro_int8", cache_dir=tmp_path, force=True)
    assert is_installed("kokoro_int8", tmp_path)
    for name in spec.expected_files:
        assert (dest / name).read_bytes() == b"github-ok"

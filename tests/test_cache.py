import io
import tarfile
import zipfile

import pytest

from hexagon_kit.cache import (
    ModelNotInstalled,
    _check_sha256,
    _sha256_file,
    delete_model,
    is_installed,
    resolve,
)
from hexagon_kit.cache import _place_artifact
from hexagon_kit.catalog import Artifact, get_spec


def test_resolve_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    with pytest.raises(ModelNotInstalled):
        resolve("stt")
    assert is_installed("whisper_tiny_int8") is False


def test_file_artifact_copy(tmp_path):
    spec = get_spec("kokoro_int8")
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    payload = src / "voices-v1.0.bin"
    payload.write_bytes(b"voice-bytes")
    artifact = Artifact(url="http://example.invalid/voices-v1.0.bin", filename="voices-v1.0.bin")
    _place_artifact(artifact, payload, dest, spec)
    assert (dest / "voices-v1.0.bin").read_bytes() == b"voice-bytes"


def test_tar_bz2_promotes_expected_files(tmp_path):
    spec = get_spec("whisper_tiny_int8")
    archive = tmp_path / "sherpa-onnx-whisper-tiny.en.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        for name in spec.expected_files:
            nested = f"sherpa-onnx-whisper-tiny.en/{name}"
            data = io.BytesIO(f"payload-{name}".encode())
            info = tarfile.TarInfo(name=nested)
            info.size = data.getbuffer().nbytes
            data.seek(0)
            tar.addfile(info, data)

    dest = tmp_path / "stt"
    dest.mkdir()
    artifact = Artifact(
        url="http://example.invalid/x.tar.bz2",
        filename=archive.name,
        kind="tar.bz2",
    )
    _place_artifact(artifact, archive, dest, spec)
    for name in spec.expected_files:
        assert (dest / name).read_text(encoding="utf-8") == f"payload-{name}"


def test_zip_promotes_expected_files(tmp_path):
    spec = get_spec("kokoro_int8")
    archive = tmp_path / "kokoro.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in spec.expected_files:
            zf.writestr(f"nested/{name}", f"zip-{name}")

    dest = tmp_path / "tts"
    dest.mkdir()
    artifact = Artifact(
        url="http://example.invalid/x.zip",
        filename=archive.name,
        kind="zip",
    )
    _place_artifact(artifact, archive, dest, spec)
    for name in spec.expected_files:
        assert (dest / name).read_text(encoding="utf-8") == f"zip-{name}"


def test_sha256_mismatch_and_match(tmp_path):
    path = tmp_path / "voices-v1.0.bin"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _check_sha256(path, "0" * 64, "voices-v1.0.bin")
    _check_sha256(
        path,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "voices-v1.0.bin",
    )


def test_sha256_streams_instead_of_slurping(tmp_path):
    path = tmp_path / "chunk.bin"
    path.write_bytes(b"x" * (2 * 1024 * 1024 + 17))
    digest = _sha256_file(path)
    assert len(digest) == 64
    _check_sha256(path, digest, "chunk.bin")


def test_delete_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    spec = get_spec("tts")
    slot = tmp_path / spec.slot
    slot.mkdir()
    for name in spec.expected_files:
        (slot / name).write_bytes(b"x")
    assert is_installed("tts") is True
    delete_model("tts")
    assert is_installed("tts") is False

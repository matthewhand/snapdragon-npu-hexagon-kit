"""Tests for the Qualcomm AI Hub adapter — shipped functions, fake CLI backend."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hexagon_kit.cli import main
from hexagon_kit.hub import (
    HubUnavailable,
    fetch_hub_model,
    hub_available,
    hub_info,
    hub_is_installed,
    hub_snapshot,
    list_hub_models,
    vendor_status,
)
from hexagon_kit.status import reset_jobs, ui_snapshot


class _Entry:
    def __init__(
        self,
        model_id: str,
        display_name: str,
        domain: int,
        use_case: int,
        *,
        quantized: bool = False,
        tags: list[int] | None = None,
        runtimes: list[str] | None = None,
        chipsets: list[str] | None = None,
    ):
        self.id = model_id
        self.display_name = display_name
        self.domain = domain
        self.use_case = use_case
        self.is_quantized = quantized
        self.tags = tags or []
        self.supported_runtimes = runtimes or ["onnx"]
        self.supported_chipsets = chipsets or ["qualcomm-snapdragon-x-elite"]


def _normalize(label: str) -> str:
    return label.lower().replace("-", " ").replace("_", " ").strip()


def _fake_backend(tmp_path: Path):
    entries = [
        _Entry("whisper_tiny", "Whisper-Tiny", 3, 13),
        _Entry("melotts_en", "MeloTTS-EN", 3, 16),
        _Entry("mobilenet_v2", "MobileNet-v2", 1, 1, quantized=True),
        _Entry("qwen3_4b", "Qwen3-4B", 4, 20, tags=[4]),
    ]

    def get_manifest():
        return SimpleNamespace(models=entries)

    def get_model_info(model: str):
        key = _normalize(model)
        for entry in entries:
            if _normalize(entry.id) == key or _normalize(entry.display_name) == key:
                return SimpleNamespace(
                    id=entry.id,
                    name=entry.display_name,
                    headline=f"{entry.display_name} headline",
                    description=f"{entry.display_name} description",
                    domain=entry.domain,
                    use_case=entry.use_case,
                    tags=entry.tags,
                    license_url="https://example.invalid/license",
                    source_repo="https://github.com/qualcomm/ai-hub-models",
                    model_type_llm=entry.id == "qwen3_4b",
                    technical_details=[
                        SimpleNamespace(key="Parameters", string_value="39M", int_value=None, float_value=None)
                    ],
                )
        raise KeyError(model)

    def fetch(*, model, runtime, precision, output_dir, extract=True, chipset=None, device=None):
        dest = Path(output_dir) / f"{model}-{runtime}-{precision}"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "model.onnx").write_bytes(b"onnx")
        (dest / "meta.txt").write_text(
            f"{model} {runtime} {precision} {chipset or ''} {device or ''}",
            encoding="utf-8",
        )
        return dest

    def get_asset_url(**kwargs):
        return f"https://example.invalid/{kwargs.get('model')}.zip"

    def domain_proto_to_str(value):
        return {1: "Computer Vision", 3: "Audio", 4: "Generative AI"}.get(value, str(value))

    def use_case_proto_to_str(value):
        return {
            1: "Image Classification",
            13: "Speech Recognition",
            16: "Audio Generation",
            20: "Text Generation",
        }.get(value, "")

    def use_case_str_to_proto(value):
        target = _normalize(value)
        mapping = {
            "image classification": 1,
            "speech recognition": 13,
            "audio generation": 16,
            "text generation": 20,
        }
        if target not in mapping:
            raise KeyError(value)
        return mapping[target]

    def tag_proto_to_str(value):
        return {4: "LLM"}.get(value, str(value))

    def runtime_proto_to_str(value):
        return value if isinstance(value, str) else str(value)

    from hexagon_kit.hub import HubBackend

    return HubBackend(
        get_manifest=get_manifest,
        get_model_info=get_model_info,
        fetch=fetch,
        get_asset_url=get_asset_url,
        domain_proto_to_str=domain_proto_to_str,
        use_case_proto_to_str=use_case_proto_to_str,
        tag_proto_to_str=tag_proto_to_str,
        runtime_proto_to_str=runtime_proto_to_str,
        domain_str_match=_normalize,
        use_case_str_to_proto=use_case_str_to_proto,
    )


@pytest.fixture
def hub(monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    backend = _fake_backend(tmp_path)
    monkeypatch.setattr("hexagon_kit.hub._import_cli", lambda: backend)
    return tmp_path


def test_vendor_status_without_cli():
    status = vendor_status()
    assert "cli" in status
    assert "apps" in status
    assert "sdk" in status
    assert "modelsSdk" in status
    assert status["cli"]["package"] == "qai_hub_models_cli"
    assert "qai_hub_models_cli" in status["cli"]["install"]
    assert status["available"] is hub_available()


def test_list_hub_models_missing_cli(monkeypatch):
    monkeypatch.setattr("hexagon_kit.hub._import_cli", lambda: None)
    monkeypatch.setattr("hexagon_kit.hub.hf_token", lambda: None)
    with pytest.raises(HubUnavailable):
        list_hub_models(community=False)


def test_list_hub_models_by_domain_and_use_case(hub):
    audio = list_hub_models(domain="Audio", cache_dir=hub)
    ids = [row["id"] for row in audio]
    assert ids == ["melotts_en", "whisper_tiny"]
    assert {row["domain"] for row in audio} == {"Audio"}
    assert all(row["source"] == "hub" for row in audio)

    stt = list_hub_models(use_case="Speech Recognition", cache_dir=hub)
    assert [row["id"] for row in stt] == ["whisper_tiny"]
    assert stt[0]["slotHint"] == "stt"
    assert stt[0]["useCase"] == "Speech Recognition"

    quantized = list_hub_models(quantized=True, cache_dir=hub)
    assert [row["id"] for row in quantized] == ["mobilenet_v2"]
    assert quantized[0]["domain"] == "Computer Vision"

    llms = list_hub_models(llm=True, cache_dir=hub)
    assert [row["id"] for row in llms] == ["qwen3_4b"]
    assert llms[0]["slotHint"] == "llm"


def test_hub_info_metadata(hub):
    info = hub_info("Whisper-Tiny", cache_dir=hub)
    assert info["id"] == "whisper_tiny"
    assert info["name"] == "Whisper-Tiny"
    assert "headline" in info and "Whisper-Tiny" in info["headline"]
    assert info["domain"] == "Audio"
    assert info["useCase"] == "Speech Recognition"
    assert info["sourceRepo"].startswith("https://github.com/qualcomm")
    assert info["slotHint"] == "stt"
    assert info["installed"] is False


def test_fetch_hub_model_writes_complete_cache(hub):
    dest = fetch_hub_model("whisper_tiny", runtime="onnx", precision="float", cache_dir=hub)
    assert dest == hub / "hub" / "whisper_tiny"
    assert (dest / "COMPLETE").is_file()
    assert (dest / "whisper_tiny-onnx-float" / "model.onnx").read_bytes() == b"onnx"
    assert hub_is_installed("whisper_tiny", hub) is True
    again = fetch_hub_model("whisper_tiny", cache_dir=hub)
    assert again == dest


def test_qnn_fetch_defaults_x_elite_chipset(hub):
    dest = fetch_hub_model("whisper_tiny", runtime="qnn", cache_dir=hub)
    meta = (dest / "whisper_tiny-qnn-float" / "meta.txt").read_text(encoding="utf-8")
    assert "qualcomm-snapdragon-x-elite" in meta


def test_hub_cli_list_and_fetch(hub, capsys):
    assert main(["hub", "list", "--domain", "Audio"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["id"] for row in rows} == {"whisper_tiny", "melotts_en"}

    assert main(["hub", "info", "whisper_tiny"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["useCase"] == "Speech Recognition"

    assert main(["hub", "fetch", "whisper_tiny", "--runtime", "onnx"]) == 0
    path = Path(capsys.readouterr().out.strip())
    assert path.name == "whisper_tiny"
    assert (path / "COMPLETE").is_file()

    assert main(["hub", "path", "whisper_tiny"]) == 0
    assert "whisper_tiny" in capsys.readouterr().out

    assert main(["models", "list", "--source", "hub", "--use-case", "Speech Recognition"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["id"] == "whisper_tiny"


def test_hub_cli_missing_package(monkeypatch, capsys):
    monkeypatch.setattr("hexagon_kit.hub._import_cli", lambda: None)
    monkeypatch.setattr("hexagon_kit.hub.hf_token", lambda: None)
    monkeypatch.setattr("hexagon_kit.credentials.hf_token", lambda: None)
    assert main(["hub", "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["available"] is False
    assert "credentials" in status
    assert "no token" in status["note"]

    code = main(["hub", "list"])
    err = capsys.readouterr().err
    assert code == 2
    payload = json.loads(err)
    assert payload["available"] is False
    assert "qai_hub_models_cli" in payload["error"] or "hf-token" in payload["error"]


def test_ui_snapshot_includes_hub_without_network(monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    reset_jobs()
    snap = ui_snapshot()
    assert "hub" in snap
    assert "cli" in snap["hub"]
    assert snap["hub"]["cli"]["package"] == "qai_hub_models_cli"
    assert "installed" in snap["hub"]
    assert snap["hub"]["cacheDir"].endswith("hub") or "hub" in snap["hub"]["cacheDir"]


def test_hf_community_list_and_fetch(monkeypatch, tmp_path):
    import zipfile

    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setattr("hexagon_kit.hub._import_cli", lambda: None)

    listing = [
        {
            "id": "qualcomm/Whisper-Tiny",
            "pipeline_tag": "automatic-speech-recognition",
            "tags": ["qai-hub-models", "whisper"],
            "downloads": 42,
        }
    ]
    card = {
        "id": "qualcomm/Whisper-Tiny",
        "pipeline_tag": "automatic-speech-recognition",
        "tags": ["qai-hub-models"],
        "cardData": {"pretty_name": "Whisper Tiny (Hub)"},
    }

    def fake_json(url: str):
        if "api/models?" in url or "filter=qai-hub-models" in url:
            return listing
        if url.endswith("/qualcomm/Whisper-Tiny") or "models/qualcomm/Whisper-Tiny" in url:
            return card
        raise AssertionError(url)

    def fake_download(url: str, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("Whisper-Tiny-main/model.onnx", b"onnx")

    monkeypatch.setattr("hexagon_kit.hub._http_json", fake_json)
    monkeypatch.setattr("hexagon_kit.hub._http_download", fake_download)

    rows = list_hub_models(community=True, cache_dir=tmp_path)
    assert rows[0]["id"] == "qualcomm/Whisper-Tiny"
    assert rows[0]["source"] == "huggingface"
    assert rows[0]["useCase"] == "Speech Recognition"
    assert rows[0]["slotHint"] == "stt"

    info = hub_info("qualcomm/Whisper-Tiny", cache_dir=tmp_path)
    assert info["name"] == "Whisper Tiny (Hub)"
    assert info["source"] == "huggingface"

    dest = fetch_hub_model("qualcomm/Whisper-Tiny", cache_dir=tmp_path)
    assert dest.name == "qualcomm--whisper-tiny"
    assert (dest / "COMPLETE").is_file()
    assert any(dest.rglob("model.onnx"))


def test_models_list_builtin_still_default(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    assert main(["models", "list"]) == 0
    rows = json.loads(capsys.readouterr().out)
    ids = {row["id"] for row in rows}
    assert ids == {"whisper_tiny_int8", "kokoro_int8"}
    assert all(row.get("source") == "builtin" for row in rows)
    assert all(row.get("origin") == "github" for row in rows)

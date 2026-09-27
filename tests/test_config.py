import json
from pathlib import Path

from hexagon_kit.catalog import get_spec as builtin_get_spec
from hexagon_kit.config import get_spec, list_specs, load_config, reset_config


def test_file_overrides_cache_and_ram(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({"cache_dir": str(tmp_path / "models"), "max_ram_mb": 1200}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HEXAGON_KIT_CONFIG", str(cfg_path))
    monkeypatch.delenv("HEXAGON_KIT_CACHE", raising=False)
    monkeypatch.delenv("HEXAGON_KIT_MAX_RAM_MB", raising=False)
    reset_config()
    cfg = load_config()
    assert cfg.cache_dir == tmp_path / "models"
    assert cfg.max_ram_mb == 1200
    assert "file:" in "".join(cfg.sources)


def test_env_wins_over_file(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({"cache_dir": str(tmp_path / "from-file"), "max_ram_mb": 1000}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HEXAGON_KIT_CONFIG", str(cfg_path))
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path / "from-env"))
    monkeypatch.setenv("HEXAGON_KIT_MAX_RAM_MB", "800")
    reset_config()
    cfg = load_config()
    assert cfg.cache_dir == tmp_path / "from-env"
    assert cfg.max_ram_mb == 800


def test_overlay_existing_model_url(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "kokoro_int8",
                        "artifacts": [
                            {
                                "url": "https://mirror.example/kokoro-v1.0.int8.onnx",
                                "filename": "kokoro-v1.0.int8.onnx",
                            },
                            {
                                "url": "https://mirror.example/voices-v1.0.bin",
                                "filename": "voices-v1.0.bin",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HEXAGON_KIT_CONFIG", str(cfg_path))
    reset_config()
    spec = get_spec("tts")
    builtin = builtin_get_spec("tts")
    assert spec.slot == builtin.slot
    assert spec.artifacts[0].url.startswith("https://mirror.example/")
    assert spec.expected_files == builtin.expected_files


def test_add_custom_model(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "phi3_mini",
                        "slot": "llm",
                        "name": "Phi-3 mini",
                        "disk_mb": 2200,
                        "ram_mb": 2400,
                        "artifacts": [
                            {
                                "url": "https://example.invalid/model.onnx",
                                "filename": "model.onnx",
                            }
                        ],
                        "expected_files": ["model.onnx"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HEXAGON_KIT_CONFIG", str(cfg_path))
    reset_config()
    ids = {s.model_id for s in list_specs()}
    assert "phi3_mini" in ids
    assert "whisper_tiny_int8" in ids
    assert get_spec("llm").model_id == "phi3_mini"


def test_programmatic_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("HEXAGON_KIT_CONFIG", raising=False)
    monkeypatch.delenv("HEXAGON_KIT_CACHE", raising=False)
    reset_config()
    cfg = load_config(overrides={"max_ram_mb": 512, "preferred_provider": "CPUExecutionProvider"})
    assert cfg.max_ram_mb == 512
    assert cfg.preferred_provider == "CPUExecutionProvider"


def test_example_config_keeps_builtin_sha256(tmp_path, monkeypatch):
    example = Path(__file__).resolve().parents[1] / "config.example.json"
    data = json.loads(example.read_text(encoding="utf-8"))
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("HEXAGON_KIT_CONFIG", str(cfg_path))
    monkeypatch.delenv("HEXAGON_KIT_CACHE", raising=False)
    reset_config()
    spec = get_spec("kokoro_int8")
    builtin = builtin_get_spec("kokoro_int8")
    assert spec.ram_mb == 280
    assert spec.expected_files == builtin.expected_files
    assert len(spec.artifacts) == len(builtin.artifacts)
    for art, base in zip(spec.artifacts, builtin.artifacts, strict=True):
        assert art.sha256 == base.sha256
        assert art.sha256 and len(art.sha256) == 64


def test_overrides_win_over_env(tmp_path, monkeypatch):
    monkeypatch.delenv("HEXAGON_KIT_CONFIG", raising=False)
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path / "from-env"))
    monkeypatch.setenv("HEXAGON_KIT_MAX_RAM_MB", "800")
    monkeypatch.setenv("HEXAGON_KIT_PROVIDER", "DmlExecutionProvider")
    reset_config()
    cfg = load_config(
        overrides={
            "cache_dir": str(tmp_path / "from-overrides"),
            "max_ram_mb": 512,
            "preferred_provider": "CPUExecutionProvider",
        }
    )
    assert cfg.cache_dir == tmp_path / "from-overrides"
    assert cfg.max_ram_mb == 512
    assert cfg.preferred_provider == "CPUExecutionProvider"
    assert "overrides" in cfg.sources
    assert any(source.startswith("env:") for source in cfg.sources)

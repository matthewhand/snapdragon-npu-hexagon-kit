import json

import pytest

from hexagon_kit.config import active, reset_config
from hexagon_kit.settings import (
    KIT_SETTINGS,
    SettingField,
    SettingsError,
    save_settings,
    settings_snapshot,
    validate_settings,
    validate_values,
)
from hexagon_kit.status import ui_snapshot


@pytest.fixture
def cfg_file(tmp_path, monkeypatch):
    path = tmp_path / "cfg" / "config.json"
    monkeypatch.setenv("HEXAGON_KIT_CONFIG", str(path))
    for name in ("HEXAGON_KIT_MAX_RAM_MB", "HEXAGON_KIT_PROVIDER", "HEXAGON_QNN_HTP_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path / "models"))
    reset_config()
    yield path
    reset_config()


def _field(snap, key):
    return next(f for f in snap["fields"] if f["key"] == key)


def test_snapshot_describes_kit_fields(cfg_file):
    snap = settings_snapshot()
    assert snap["schemaVersion"] == 1
    assert snap["path"] == str(cfg_file)
    assert snap["fileExists"] is False
    assert snap["error"] is None
    assert [f["key"] for f in snap["fields"]] == [f.key for f in KIT_SETTINGS]
    ram = _field(snap, "max_ram_mb")
    assert ram["type"] == "number" and ram["unit"] == "MB"
    assert ram["value"] == 3500 and ram["source"] == "default"
    provider = _field(snap, "preferred_provider")
    assert provider["type"] == "enum"
    assert provider["options"][0] == {"value": None, "label": "Automatic"}
    assert "QNNExecutionProvider" in {o["value"] for o in provider["options"]}
    cache = _field(snap, "cache_dir")
    assert cache["locked"] is True
    assert cache["lockedBy"] == "HEXAGON_KIT_CACHE"
    assert cache["source"] == "env:HEXAGON_KIT_CACHE"
    assert cache["placeholder"].endswith("models")


def test_save_merges_and_preserves_models(cfg_file):
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text(json.dumps({"models": [{"model_id": "kokoro_int8", "ram_mb": 280}]}), encoding="utf-8")
    snap = save_settings({"max_ram_mb": "4096", "preferred_provider": "CPUExecutionProvider"})
    ram = _field(snap, "max_ram_mb")
    assert ram["value"] == 4096 and ram["source"] == "file"
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert data["max_ram_mb"] == 4096
    assert data["models"][0]["model_id"] == "kokoro_int8"
    assert active().max_ram_mb == 4096
    assert active().preferred_provider == "CPUExecutionProvider"

    save_settings({"preferred_provider": None})
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert "preferred_provider" not in data
    assert active().preferred_provider is None


def test_save_rejects_invalid_without_writing(cfg_file):
    with pytest.raises(SettingsError) as exc:
        save_settings({"max_ram_mb": 10, "preferred_provider": "Hexagon", "bogus": 1})
    assert set(exc.value.errors) == {"max_ram_mb", "preferred_provider", "bogus"}
    assert not cfg_file.exists()
    assert validate_settings({"max_ram_mb": 2048}) == {}
    assert "max_ram_mb" in validate_settings({"max_ram_mb": True})


def test_save_refuses_to_clobber_broken_file(cfg_file):
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text("{not json", encoding="utf-8")
    snap = settings_snapshot()
    assert snap["error"]
    with pytest.raises(SettingsError) as exc:
        save_settings({"max_ram_mb": 2048})
    assert "_file" in exc.value.errors
    assert cfg_file.read_text(encoding="utf-8") == "{not json"


def test_path_field_rejects_file(cfg_file, tmp_path):
    target = tmp_path / "a-file.txt"
    target.write_text("x", encoding="utf-8")
    assert "qnn_htp_dir" in validate_settings({"qnn_htp_dir": str(target)})


def test_env_lock_reported(cfg_file, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_MAX_RAM_MB", "2000")
    reset_config()
    ram = _field(settings_snapshot(), "max_ram_mb")
    assert ram["locked"] is True and ram["value"] == 2000


def test_app_defined_fields_reuse_format():
    fields = (
        SettingField("wake_word", "Wake word", "string", nullable=False),
        SettingField("volume", "Volume", "integer", minimum=0, maximum=100, default=80),
        SettingField("push_to_talk", "Push to talk", "boolean", default=False),
        SettingField("voice", "Voice", "enum", options=(("af", "AF"), ("am", "AM"))),
    )
    clean, errors = validate_values(fields, {"volume": "55", "push_to_talk": "yes", "voice": "af"})
    assert errors == {}
    assert clean == {"volume": 55, "push_to_talk": True, "voice": "af"}
    _, errors = validate_values(fields, {"volume": 101, "wake_word": None, "voice": "zz"})
    assert set(errors) == {"volume", "wake_word", "voice"}
    assert fields[1].to_dict()["maximum"] == 100
    with pytest.raises(ValueError):
        SettingField("x", "X", "color")


def test_ui_snapshot_embeds_settings(cfg_file):
    snap = ui_snapshot()
    assert snap["settings"]["schemaVersion"] == 1
    assert {f["key"] for f in snap["settings"]["fields"]} >= {"cache_dir", "max_ram_mb"}

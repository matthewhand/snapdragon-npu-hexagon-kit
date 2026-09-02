import os

import pytest

from hexagon_kit.xdg import (
    KIT_DIRNAME,
    default_cache_dir,
    default_config_path,
    xdg_cache_home,
    xdg_config_home,
    xdg_data_home,
)


def test_hexagon_kit_cache_overrides_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path / "explicit"))
    assert default_cache_dir() == tmp_path / "explicit"


def test_xdg_cache_home_env(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.delenv("HEXAGON_KIT_CACHE", raising=False)
    assert xdg_cache_home() == tmp_path / "xdg-cache"
    assert default_cache_dir() == tmp_path / "xdg-cache" / "hexagon-kit" / "models"


def test_xdg_data_home_env(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    assert xdg_data_home() == tmp_path / "xdg-data"


def test_config_path_xdg_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.delenv("HEXAGON_KIT_CONFIG", raising=False)
    assert xdg_config_home() == tmp_path / "xdg-config"
    assert default_config_path() == tmp_path / "xdg-config" / KIT_DIRNAME / "config.json"
    monkeypatch.setenv("HEXAGON_KIT_CONFIG", str(tmp_path / "custom.json"))
    assert default_config_path() == tmp_path / "custom.json"


def test_posix_fallback_is_dot_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("HEXAGON_KIT_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr("hexagon_kit.xdg.os.name", "posix")
    monkeypatch.setattr("hexagon_kit.xdg.Path.home", lambda: tmp_path)
    assert xdg_cache_home() == tmp_path / ".cache"
    assert default_cache_dir() == tmp_path / ".cache" / "hexagon-kit" / "models"


@pytest.mark.skipif(os.name != "nt", reason="pathlib.WindowsPath cannot be constructed on POSIX CI")
def test_windows_localappdata_without_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("HEXAGON_KIT_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.setattr("hexagon_kit.xdg.os.name", "nt")
    assert default_cache_dir() == tmp_path / "Local" / "hexagon-kit" / "models"


@pytest.mark.skipif(os.name != "nt", reason="pathlib.WindowsPath cannot be constructed on POSIX CI")
def test_windows_reuses_legacy_snapdragon_npu_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("HEXAGON_KIT_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    local = tmp_path / "Local"
    legacy = local / "SnapdragonNpu" / "models"
    legacy.mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setattr("hexagon_kit.xdg.os.name", "nt")
    assert default_cache_dir() == legacy


@pytest.mark.skipif(os.name != "nt", reason="pathlib.WindowsPath cannot be constructed on POSIX CI")
def test_modern_hexagon_kit_dir_wins_over_legacy(monkeypatch, tmp_path):
    monkeypatch.delenv("HEXAGON_KIT_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    local = tmp_path / "Local"
    modern = local / "hexagon-kit" / "models"
    legacy = local / "SnapdragonNpu" / "models"
    modern.mkdir(parents=True)
    legacy.mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setattr("hexagon_kit.xdg.os.name", "nt")
    assert default_cache_dir() == modern

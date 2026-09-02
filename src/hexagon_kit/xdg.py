"""XDG Base Directory paths so every Copilot+ app shares one model cache."""

from __future__ import annotations

import os
from pathlib import Path

KIT_DIRNAME = "hexagon-kit"
LEGACY_WIN_CACHE_NAMES = ("SnapdragonNpu",)


def xdg_cache_home() -> Path:
    """
    Cache root per the XDG Base Directory spec.

    1. $XDG_CACHE_HOME if set (including on Windows)
    2. %LOCALAPPDATA% on Windows (the usual XDG_CACHE_HOME analog)
    3. ~/.cache
    """
    env = os.environ.get("XDG_CACHE_HOME", "").strip()
    if env:
        return Path(env).expanduser()
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if local:
            return Path(local)
    return Path.home() / ".cache"


def xdg_config_home() -> Path:
    env = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if env:
        return Path(env).expanduser()
    if os.name == "nt":
        roaming = os.environ.get("APPDATA")
        if roaming:
            return Path(roaming)
    return Path.home() / ".config"


def default_config_path() -> Path:
    override = os.environ.get("HEXAGON_KIT_CONFIG", "").strip()
    if override:
        return Path(override).expanduser()
    return xdg_config_home() / KIT_DIRNAME / "config.json"


def xdg_data_home() -> Path:
    env = os.environ.get("XDG_DATA_HOME", "").strip()
    if env:
        return Path(env).expanduser()
    if os.name == "nt":
        roaming = os.environ.get("APPDATA")
        if roaming:
            return Path(roaming)
    return Path.home() / ".local" / "share"


def default_cache_dir() -> Path:
    """
    Shared ONNX/weights directory.

    Priority:
      1. $HEXAGON_KIT_CACHE
      2. $XDG_CACHE_HOME/hexagon-kit/models
      3. Windows: %LOCALAPPDATA%/hexagon-kit/models
      4. ~/.cache/hexagon-kit/models

    Re-downloadable weights live in *cache*, not XDG_DATA_HOME.
    """
    override = os.environ.get("HEXAGON_KIT_CACHE", "").strip()
    if override:
        return Path(override).expanduser()
    modern = xdg_cache_home() / KIT_DIRNAME / "models"
    if os.name == "nt" and not modern.exists():
        local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if local:
            for name in LEGACY_WIN_CACHE_NAMES:
                legacy = Path(local) / name / "models"
                if legacy.exists():
                    return legacy
    return modern

"""
Layered kit configuration.

Priority (highest last):
  1. Built-in catalog and 16 GB Copilot+ defaults
  2. XDG config file ($XDG_CONFIG_HOME/hexagon-kit/config.json)
  3. Environment variables
  4. load_config(overrides=...) / constructor arguments
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .catalog import CATALOG, ModelSpec, get_spec as catalog_get_spec, merge_catalog
from .xdg import default_cache_dir, default_config_path

DEFAULT_MAX_RAM_MB = 3500.0
VALID_PROVIDERS = {
    "QNNExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
}


@dataclass(frozen=True)
class KitConfig:
    cache_dir: Path
    max_ram_mb: float = DEFAULT_MAX_RAM_MB
    preferred_provider: str | None = None
    qnn_htp_dir: Path | None = None
    catalog: tuple[ModelSpec, ...] = CATALOG
    config_path: Path | None = None
    sources: tuple[str, ...] = ("defaults",)

    def spec(self, model_id_or_slot: str) -> ModelSpec:
        return catalog_get_spec(model_id_or_slot, self.catalog)

    def to_dict(self) -> dict:
        return {
            "cache_dir": str(self.cache_dir),
            "max_ram_mb": self.max_ram_mb,
            "preferred_provider": self.preferred_provider,
            "qnn_htp_dir": str(self.qnn_htp_dir) if self.qnn_htp_dir else None,
            "config_path": str(self.config_path) if self.config_path else None,
            "sources": list(self.sources),
            "models": [
                {
                    "model_id": spec.model_id,
                    "slot": spec.slot,
                    "name": spec.name,
                    "disk_mb": spec.disk_mb,
                    "ram_mb": spec.ram_mb,
                    "expected_files": list(spec.expected_files),
                    "artifacts": [
                        {
                            "url": art.url,
                            "filename": art.filename,
                            "kind": art.kind,
                            "sha256": art.sha256,
                        }
                        for art in spec.artifacts
                    ],
                }
                for spec in self.catalog
            ],
        }


_ACTIVE: KitConfig | None = None
_FINGERPRINT: tuple | None = None


def _fingerprint() -> tuple:
    path = default_config_path()
    try:
        mtime = path.stat().st_mtime if path.is_file() else 0.0
    except OSError:
        mtime = 0.0
    return (
        os.environ.get("HEXAGON_KIT_CACHE", ""),
        os.environ.get("HEXAGON_KIT_CONFIG", ""),
        os.environ.get("HEXAGON_KIT_MAX_RAM_MB", ""),
        os.environ.get("HEXAGON_KIT_PROVIDER", ""),
        os.environ.get("HEXAGON_QNN_HTP_DIR", ""),
        str(path),
        mtime,
    )


def reset_config() -> None:
    global _ACTIVE, _FINGERPRINT
    _ACTIVE = None
    _FINGERPRINT = None


def active() -> KitConfig:
    global _ACTIVE, _FINGERPRINT
    fp = _fingerprint()
    if _ACTIVE is None or _FINGERPRINT != fp:
        _ACTIVE = load_config(reload=False)
        _FINGERPRINT = fp
    return _ACTIVE


def get_spec(model_id_or_slot: str) -> ModelSpec:
    return active().spec(model_id_or_slot)


def list_specs() -> tuple[ModelSpec, ...]:
    return active().catalog


def _read_json(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a JSON object")
    return raw


def _provider(value: str | None) -> str | None:
    if not value:
        return None
    if value not in VALID_PROVIDERS:
        raise ValueError(
            f"preferred_provider must be one of {sorted(VALID_PROVIDERS)}, not {value!r}"
        )
    return value


def load_config(
    path: str | Path | None = None,
    overrides: dict | None = None,
    *,
    reload: bool = True,
) -> KitConfig:
    """
    Load layered config. `overrides` is a mapping with the same keys as the file.
    """
    sources = ["defaults"]
    cache_dir = default_cache_dir()
    max_ram_mb = DEFAULT_MAX_RAM_MB
    preferred_provider: str | None = None
    qnn_htp_dir: Path | None = None
    catalog = CATALOG
    config_path: Path | None = None

    file_path = Path(path).expanduser() if path else default_config_path()
    file_data: dict = {}
    if file_path.is_file():
        file_data = _read_json(file_path)
        config_path = file_path
        sources.append(f"file:{file_path}")

    def apply_scalars(data: dict) -> None:
        nonlocal cache_dir, max_ram_mb, preferred_provider, qnn_htp_dir
        if data.get("cache_dir"):
            cache_dir = Path(str(data["cache_dir"])).expanduser()
        if data.get("max_ram_mb") is not None:
            max_ram_mb = float(data["max_ram_mb"])
        if data.get("preferred_provider"):
            preferred_provider = _provider(str(data["preferred_provider"]))
        if data.get("qnn_htp_dir"):
            qnn_htp_dir = Path(str(data["qnn_htp_dir"])).expanduser()

    # Highest last: defaults < file < env < load_config(overrides=...).
    apply_scalars(file_data)
    if file_data.get("models"):
        catalog = merge_catalog(catalog, file_data["models"])

    env_cache = os.environ.get("HEXAGON_KIT_CACHE", "").strip()
    if env_cache:
        cache_dir = Path(env_cache).expanduser()
        sources.append("env:HEXAGON_KIT_CACHE")
    env_ram = os.environ.get("HEXAGON_KIT_MAX_RAM_MB", "").strip()
    if env_ram:
        max_ram_mb = float(env_ram)
        sources.append("env:HEXAGON_KIT_MAX_RAM_MB")
    env_prov = os.environ.get("HEXAGON_KIT_PROVIDER", "").strip()
    if env_prov:
        preferred_provider = _provider(env_prov)
        sources.append("env:HEXAGON_KIT_PROVIDER")
    env_htp = os.environ.get("HEXAGON_QNN_HTP_DIR", "").strip()
    if env_htp:
        qnn_htp_dir = Path(env_htp).expanduser()
        sources.append("env:HEXAGON_QNN_HTP_DIR")

    if overrides:
        apply_scalars(overrides)
        if overrides.get("models"):
            catalog = merge_catalog(catalog, overrides["models"])
        sources.append("overrides")

    cfg = KitConfig(
        cache_dir=cache_dir,
        max_ram_mb=max_ram_mb,
        preferred_provider=preferred_provider,
        qnn_htp_dir=qnn_htp_dir,
        catalog=catalog,
        config_path=config_path,
        sources=tuple(sources),
    )
    if reload:
        global _ACTIVE, _FINGERPRINT
        _ACTIVE = cfg
        _FINGERPRINT = _fingerprint()
    return cfg

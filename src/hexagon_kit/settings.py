"""
Shared Settings schema for the kit config file. JSON only, no widgets.

``settings_snapshot()`` describes every kit setting (type, label, help,
options, effective value, and which layer set it) so Qt / Electron / CLI
forms render the same fields. ``save_settings()`` validates and writes the
shared XDG ``config.json`` that every app on the machine reads.

Apps may describe their own settings with ``SettingField`` and validate
form input with ``validate_values``; the descriptor JSON is identical.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_MAX_RAM_MB, VALID_PROVIDERS, active, reset_config
from .lock import FileLock
from .xdg import KIT_DIRNAME, default_config_path, xdg_cache_home

SETTINGS_SCHEMA_VERSION = 1
FIELD_TYPES = frozenset({"string", "path", "number", "integer", "boolean", "enum"})
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class SettingsError(ValueError):
    """Validation failed. ``errors`` maps field key to a user-facing message."""

    def __init__(self, errors: dict[str, str]):
        self.errors = dict(errors)
        super().__init__("; ".join(f"{key}: {msg}" for key, msg in self.errors.items()))


@dataclass(frozen=True)
class SettingField:
    key: str
    label: str
    type: str
    help: str = ""
    default: Any = None
    options: tuple[tuple[Any, str], ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    unit: str | None = None
    env: str | None = None
    nullable: bool = True
    placeholder: str | None = None

    def __post_init__(self) -> None:
        if self.type not in FIELD_TYPES:
            raise ValueError(f"Unknown setting type {self.type!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "type": self.type,
            "help": self.help,
            "default": self.default,
            "options": [{"value": value, "label": label} for value, label in self.options],
            "minimum": self.minimum,
            "maximum": self.maximum,
            "unit": self.unit,
            "env": self.env,
            "nullable": self.nullable,
            "placeholder": self.placeholder,
        }

    def coerce(self, value: Any) -> Any:
        """Normalize form / CLI input. Raises ValueError with a user-facing message."""
        if value is None or (isinstance(value, str) and not value.strip() and self.type != "string"):
            if not self.nullable:
                raise ValueError("A value is required")
            return None
        if self.type in {"string", "path"}:
            if not isinstance(value, (str, os.PathLike)):
                raise ValueError("Expected text")
            text = str(value).strip()
            if self.type == "path" and Path(text).expanduser().is_file():
                raise ValueError("Must be a folder, not a file")
            return text
        if self.type == "boolean":
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in _TRUE:
                return True
            if text in _FALSE:
                return False
            raise ValueError("Expected true or false")
        if self.type in {"number", "integer"}:
            if isinstance(value, bool):
                raise ValueError("Expected a number")
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError("Expected a number") from None
            if self.type == "integer":
                if not number.is_integer():
                    raise ValueError("Expected a whole number")
                number = int(number)
            if self.minimum is not None and number < self.minimum:
                raise ValueError(f"Must be at least {self.minimum:g}")
            if self.maximum is not None and number > self.maximum:
                raise ValueError(f"Must be at most {self.maximum:g}")
            return number
        allowed = [option for option, _label in self.options]
        if value not in allowed:
            raise ValueError(f"Must be one of {[o for o in allowed if o is not None]}")
        return value


def validate_values(
    fields: tuple[SettingField, ...] | list[SettingField],
    updates: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return ``(clean, errors)``. Unknown keys are errors, not silently dropped."""
    by_key = {field.key: field for field in fields}
    clean: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for key, value in updates.items():
        field = by_key.get(key)
        if field is None:
            errors[key] = "Unknown setting"
            continue
        try:
            clean[key] = field.coerce(value)
        except ValueError as exc:
            errors[key] = str(exc)
    return clean, errors


_PROVIDER_LABELS = {
    "QNNExecutionProvider": "Hexagon NPU (QNN) — needs onnxruntime_qnn",
    "DmlExecutionProvider": "DirectML (GPU)",
    "CPUExecutionProvider": "CPU",
}

KIT_SETTINGS: tuple[SettingField, ...] = (
    SettingField(
        key="cache_dir",
        label="Model cache folder",
        type="path",
        help="Shared by every app using this kit. Changing it does not move existing downloads.",
        env="HEXAGON_KIT_CACHE",
    ),
    SettingField(
        key="max_ram_mb",
        label="Machine model RAM budget",
        type="number",
        help="Total RAM all apps may hold in loaded models (peer leases count).",
        default=DEFAULT_MAX_RAM_MB,
        minimum=256,
        maximum=65536,
        unit="MB",
        env="HEXAGON_KIT_MAX_RAM_MB",
    ),
    SettingField(
        key="preferred_provider",
        label="Preferred execution provider",
        type="enum",
        help=(
            "Overrides the automatic QNN → DirectML → CPU choice. Does not make "
            "Hexagon available; UI must still check hardware.ep_kind."
        ),
        options=((None, "Automatic"),)
        + tuple((name, _PROVIDER_LABELS[name]) for name in sorted(VALID_PROVIDERS)),
        env="HEXAGON_KIT_PROVIDER",
    ),
    SettingField(
        key="qnn_htp_dir",
        label="QNN HTP libraries folder",
        type="path",
        help="Optional folder with QnnHtp.dll when onnxruntime_qnn cannot find it.",
        env="HEXAGON_QNN_HTP_DIR",
    ),
)


def validate_settings(updates: dict[str, Any]) -> dict[str, str]:
    """Field-keyed errors for kit settings; empty when ``updates`` is valid."""
    return validate_values(KIT_SETTINGS, updates)[1]


def _read_file(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"Cannot read {path}: {exc}"
    if not isinstance(data, dict):
        return {}, f"Config {path} must be a JSON object"
    return data, None


def settings_snapshot() -> dict[str, Any]:
    """Settings form payload. ``locked`` fields are pinned by an env var and read-only."""
    path = default_config_path()
    file_data, error = _read_file(path)
    effective: dict[str, Any] = {}
    if error is None:
        try:
            cfg = active()
            effective = {
                "cache_dir": str(cfg.cache_dir),
                "max_ram_mb": cfg.max_ram_mb,
                "preferred_provider": cfg.preferred_provider,
                "qnn_htp_dir": str(cfg.qnn_htp_dir) if cfg.qnn_htp_dir else None,
            }
        except ValueError as exc:
            error = str(exc)
    fields = []
    for field in KIT_SETTINGS:
        env_value = os.environ.get(field.env or "", "").strip() if field.env else ""
        file_value = file_data.get(field.key)
        if env_value:
            source = f"env:{field.env}"
        elif file_value not in (None, ""):
            source = "file"
        else:
            source = "default"
        described = field.to_dict()
        if field.key == "cache_dir":
            described["placeholder"] = str(xdg_cache_home() / KIT_DIRNAME / "models")
        fields.append(
            {
                **described,
                "value": effective.get(field.key, field.default),
                "fileValue": file_value,
                "source": source,
                "locked": bool(env_value),
                "lockedBy": field.env if env_value else None,
            }
        )
    return {
        "schemaVersion": SETTINGS_SCHEMA_VERSION,
        "path": str(path),
        "fileExists": path.is_file(),
        "error": error,
        "fields": fields,
    }


def save_settings(updates: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and merge ``updates`` into the shared config file.

    ``None`` removes a key (back to default). Keys this schema does not own
    (for example ``models``) are preserved. Raises ``SettingsError``.
    Returns the new ``settings_snapshot()``.
    """
    clean, errors = validate_values(KIT_SETTINGS, updates)
    if errors:
        raise SettingsError(errors)
    path = default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.parent / f"{path.name}.lock"):
        data, error = _read_file(path)
        if error is not None:
            raise SettingsError({"_file": f"{error}. Fix or remove it before saving."})
        for key, value in clean.items():
            if value is None:
                data.pop(key, None)
            else:
                data[key] = value
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    reset_config()
    return settings_snapshot()

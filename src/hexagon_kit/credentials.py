"""Optional Hugging Face and Qualcomm Hub tokens.

Never stored in hexagon-kit/config.json. Env wins over the secrets file.
Builtin GitHub STT/TTS do not use these.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .xdg import xdg_config_home

KIT_DIRNAME = "hexagon-kit"
SECRETS_NAME = "secrets.json"

HF_ENV = ("HEXAGON_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
QAI_ENV = ("HEXAGON_QAI_HUB_TOKEN", "QAI_HUB_API_TOKEN", "QAI_HUB_API_KEY")


def secrets_path() -> Path:
    override = os.environ.get("HEXAGON_KIT_SECRETS", "").strip()
    if override:
        return Path(override).expanduser()
    return xdg_config_home() / KIT_DIRNAME / SECRETS_NAME


def _read_secrets_file(path: Path | None = None) -> dict[str, str]:
    dest = path or secrets_path()
    if not dest.is_file():
        return {}
    try:
        data = json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for key in ("hf_token", "qai_hub_token"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
    return out


def _token_from_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _hf_token_files() -> tuple[Path, ...]:
    home = Path.home()
    return (
        home / ".cache" / "huggingface" / "token",
        home / ".huggingface" / "token",
    )


def _read_first_file(paths: tuple[Path, ...]) -> str | None:
    for path in paths:
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    return None


def hf_token() -> str | None:
    return _token_from_env(HF_ENV) or _read_secrets_file().get("hf_token") or _read_first_file(
        _hf_token_files()
    )


def qai_hub_token() -> str | None:
    return _token_from_env(QAI_ENV) or _read_secrets_file().get("qai_hub_token") or _read_qai_ini()


def _qai_ini_path() -> Path:
    return Path.home() / ".qai_hub" / "client.ini"


def _read_qai_ini() -> str | None:
    path = _qai_ini_path()
    if not path.is_file():
        return None
    token = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("api_token"):
            _, _, value = stripped.partition("=")
            token = value.strip().strip('"').strip("'")
    return token or None


def credentials_status() -> dict[str, Any]:
    """Booleans only — never echo secrets."""
    return {
        "hf": bool(hf_token()),
        "qaiHub": bool(qai_hub_token()),
        "secretsPath": str(secrets_path()),
        "note": (
            "HF token unlocks gated Hugging Face repos tagged qai-hub-models "
            "(community / newer Hub recipes). "
            "Qualcomm Workbench token is only for compile/profile via qai-hub. "
            "Neither is required for builtin GitHub STT/TTS."
        ),
    }


def apply_hub_credentials() -> None:
    """Export tokens into the env names vendor libraries already honor."""
    hf = hf_token()
    if hf:
        os.environ.setdefault("HF_TOKEN", hf)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf)
    qai = qai_hub_token()
    if qai:
        os.environ.setdefault("QAI_HUB_API_TOKEN", qai)


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def save_tokens(*, hf_token_value: str | None = None, qai_token_value: str | None = None) -> Path:
    dest = secrets_path()
    data = _read_secrets_file(dest)
    if hf_token_value is not None:
        token = hf_token_value.strip()
        if token:
            data["hf_token"] = token
            for path in _hf_token_files():
                _write_private(path, token + "\n")
        else:
            data.pop("hf_token", None)
    if qai_token_value is not None:
        token = qai_token_value.strip()
        if token:
            data["qai_hub_token"] = token
            _write_private(
                _qai_ini_path(),
                "[api]\napi_token = " + token + "\n",
            )
        else:
            data.pop("qai_hub_token", None)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _write_private(dest, json.dumps(data, indent=2) + "\n")
    apply_hub_credentials()
    return dest

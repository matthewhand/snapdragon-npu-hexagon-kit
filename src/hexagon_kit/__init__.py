"""Hexagon NPU kit: hardware probe, model catalog, shared XDG cache, and residency pool."""

from .cache import ModelNotInstalled, ensure_model, is_installed, resolve
from .catalog import CATALOG, ModelSpec
from .config import KitConfig, active, get_spec, list_specs, load_config, reset_config
from .hw import HardwareProbe, MemoryStatus, probe_hardware, read_memory_status
from .preflight import PreflightBlocked, PreflightResult, preflight
from .runtime import ModelPool, PoolBudgetExceeded, process_pool, reset_process_pool
from .session import open_onnx, provider_chain
from .status import delete_cached, model_card, start_download, storage_report, ui_snapshot
from .xdg import default_cache_dir, default_config_path, xdg_cache_home, xdg_config_home, xdg_data_home

__all__ = [
    "CATALOG",
    "HardwareProbe",
    "KitConfig",
    "MemoryStatus",
    "PreflightBlocked",
    "PreflightResult",
    "ModelNotInstalled",
    "ModelPool",
    "ModelSpec",
    "PoolBudgetExceeded",
    "active",
    "delete_cached",
    "default_cache_dir",
    "default_config_path",
    "ensure_model",
    "get_spec",
    "is_installed",
    "list_specs",
    "load_config",
    "open_onnx",
    "preflight",
    "probe_hardware",
    "provider_chain",
    "read_memory_status",
    "process_pool",
    "reset_config",
    "reset_process_pool",
    "resolve",
    "start_download",
    "storage_report",
    "ui_snapshot",
    "model_card",
    "xdg_cache_home",
    "xdg_config_home",
    "xdg_data_home",
]

__version__ = "0.1.0"

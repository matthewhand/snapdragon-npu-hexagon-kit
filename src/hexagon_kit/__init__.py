"""Hexagon NPU kit: hardware probe, model catalog, shared XDG cache, and residency pool."""

from .cache import ModelNotInstalled, ensure_model, is_installed, peer_download, resolve
from .leases import Lease, ModelInUse, holder_for, list_leases, take_lease, drop_lease
from .catalog import CATALOG, ModelSpec
from .config import KitConfig, active, get_spec, list_specs, load_config, reset_config
from .hw import (
    HardwareProbe,
    MemoryStatus,
    ProviderChoice,
    choose_execution_provider,
    probe_hardware,
    read_memory_status,
)
from .preflight import PreflightBlocked, PreflightResult, preflight
from .runtime import ModelPool, PoolBudgetExceeded, process_pool, reset_process_pool
from .session import open_onnx, provider_chain
from .credentials import credentials_status, hf_token, qai_hub_token, save_tokens
from .hub import (
    HubUnavailable,
    fetch_hub_model,
    hub_available,
    hub_info,
    hub_snapshot,
    list_hf_community_models,
    list_hub_models,
    vendor_status,
)
from .status import (
    SNAPSHOT_SCHEMA_VERSION,
    DownloadCancelled,
    DownloadInProgress,
    cancel_download,
    delete_cached,
    download_jobs,
    get_job,
    model_card,
    slot_info,
    slots_summary,
    start_download,
    storage_report,
    ui_snapshot,
)
from .xdg import default_cache_dir, default_config_path, xdg_cache_home, xdg_config_home, xdg_data_home

__all__ = [
    "CATALOG",
    "DownloadCancelled",
    "DownloadInProgress",
    "SNAPSHOT_SCHEMA_VERSION",
    "cancel_download",
    "download_jobs",
    "get_job",
    "peer_download",
    "slot_info",
    "slots_summary",
    "HardwareProbe",
    "HubUnavailable",
    "ProviderChoice",
    "KitConfig",
    "Lease",
    "MemoryStatus",
    "ModelInUse",
    "PreflightBlocked",
    "PreflightResult",
    "ModelNotInstalled",
    "ModelPool",
    "ModelSpec",
    "PoolBudgetExceeded",
    "active",
    "choose_execution_provider",
    "credentials_status",
    "delete_cached",
    "default_cache_dir",
    "default_config_path",
    "drop_lease",
    "ensure_model",
    "fetch_hub_model",
    "holder_for",
    "hub_available",
    "hub_info",
    "hub_snapshot",
    "hf_token",
    "get_spec",
    "is_installed",
    "list_hf_community_models",
    "list_hub_models",
    "list_leases",
    "list_specs",
    "load_config",
    "open_onnx",
    "preflight",
    "probe_hardware",
    "provider_chain",
    "qai_hub_token",
    "save_tokens",
    "read_memory_status",
    "process_pool",
    "reset_config",
    "reset_process_pool",
    "resolve",
    "start_download",
    "storage_report",
    "take_lease",
    "ui_snapshot",
    "vendor_status",
    "model_card",
    "xdg_cache_home",
    "xdg_config_home",
    "xdg_data_home",
    "__version__",
]

__version__ = "0.3.0"

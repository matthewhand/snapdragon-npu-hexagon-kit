"""Hardware probe for Snapdragon X / Hexagon NPU / DirectML / CPU.

Execution-provider honesty:
  * ``QNNExecutionProvider`` means Qualcomm Hexagon HTP and requires the
    ``onnxruntime_qnn`` package (not the CPU-oriented ``[ort]`` extra).
  * ``DmlExecutionProvider`` is DirectML (typically Adreno GPU).
  * Upstream ``onnxruntime`` from ``pip install '.[ort]'`` is CPU-oriented
    and must not be reported as Hexagon QNN.
"""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

QNN_PROVIDER = "QNNExecutionProvider"
DML_PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"

# Human labels for the three EPs this kit distinguishes.
EP_KIND_QNN = "qnn"
EP_KIND_DIRECTML = "directml"
EP_KIND_CPU = "cpu"


@dataclass
class MemoryStatus:
    """Live RAM from GlobalMemoryStatusEx (Windows) or /proc/meminfo."""

    total_bytes: int
    available_bytes: int
    load_pct: int

    @property
    def total_gb(self) -> float:
        return round(self.total_bytes / (1024**3), 2)

    @property
    def available_gb(self) -> float:
        return round(self.available_bytes / (1024**3), 2)

    @property
    def available_mb(self) -> float:
        return round(self.available_bytes / (1024**2), 1)

    @property
    def bar_level(self) -> str:
        if self.load_pct >= 85:
            return "red"
        if self.load_pct >= 70:
            return "orange"
        return "green"

    def to_dict(self) -> dict:
        return {
            "totalBytes": self.total_bytes,
            "availableBytes": self.available_bytes,
            "totalGb": self.total_gb,
            "availableGb": self.available_gb,
            "availableMb": self.available_mb,
            "loadPct": self.load_pct,
            "barLevel": self.bar_level,
        }


@dataclass
class ProviderChoice:
    """QNN vs DirectML vs CPU decision from a provider list."""

    preferred: str
    label: str
    has_npu: bool
    ep_kind: str


@dataclass
class HardwareProbe:
    platform: str
    arch: str
    processor: str
    ram_gb: float | None
    ram_available_gb: float | None
    ram_load_pct: int | None
    ram_bar_level: str | None
    is_snapdragon: bool
    is_windows_arm64: bool
    providers: list[str] = field(default_factory=list)
    preferred_provider: str = CPU_PROVIDER
    provider_label: str = "CPU"
    has_npu: bool = False
    npu_tops: int | None = None
    qnn_htp_dir: str | None = None
    notes: str = ""
    memory: MemoryStatus | None = None
    # Honesty fields (additive; existing hexagon hw / probe_hardware callers
    # keep reading preferred_provider / providers / has_npu).
    ep_kind: str = EP_KIND_CPU
    ort_package: str | None = None
    qnn_package: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.memory is not None:
            data["memory"] = self.memory.to_dict()
        else:
            data.pop("memory", None)
        return data


def read_memory_status() -> MemoryStatus | None:
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return MemoryStatus(
                    total_bytes=int(stat.ullTotalPhys),
                    available_bytes=int(stat.ullAvailPhys),
                    load_pct=int(stat.dwMemoryLoad),
                )
        except Exception:
            return None
        return None
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            values: dict[str, int] = {}
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                parts = line.replace(":", " ").split()
                if len(parts) >= 2 and parts[1].isdigit():
                    values[parts[0]] = int(parts[1]) * 1024
            total = values.get("MemTotal", 0)
            available = values.get("MemAvailable") or values.get("MemFree", 0)
            if total:
                used = max(0, total - available)
                return MemoryStatus(
                    total_bytes=total,
                    available_bytes=available,
                    load_pct=int(used * 100 / total),
                )
        pages = os.sysconf("SC_PHYS_PAGES")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = pages * page_size
        available = avail_pages * page_size
        return MemoryStatus(
            total_bytes=int(total),
            available_bytes=int(available),
            load_pct=int((total - available) * 100 / total) if total else 0,
        )
    except (AttributeError, OSError, ValueError):
        return None


def _ram_gb() -> float | None:
    memory = read_memory_status()
    return memory.total_gb if memory else None


def _cpu_brand() -> str:
    processor = platform.processor() or ""
    if os.name == "nt":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            brand, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            if brand:
                return str(brand).strip()
        except OSError:
            pass
    return processor.strip() or platform.machine()


def _is_snapdragon(brand: str, machine: str) -> bool:
    text = f"{brand} {machine}".lower()
    return any(
        token in text
        for token in ("snapdragon", "qualcomm", "qcom", "x1e", "x1p")
    )


def find_qnn_htp_dir() -> Path | None:
    """Locate a Qualcomm HTP driver directory without pinning INF hashes."""
    roots = [
        Path(r"C:\Windows\System32\DriverStore\FileRepository"),
        Path(r"C:\Windows\System32"),
    ]
    for root in roots:
        if not root.exists():
            continue
        try:
            matches = sorted(
                p
                for p in root.glob("qcnspmcdm*/HTP")
                if p.is_dir()
            )
        except OSError:
            matches = []
        if matches:
            return matches[-1]
    env = os.environ.get("HEXAGON_QNN_HTP_DIR")
    if env:
        path = Path(env)
        if path.is_dir():
            return path
    return None


def _onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def _qnn_package_available() -> bool:
    try:
        import onnxruntime_qnn  # noqa: F401
    except Exception:
        return False
    return True


def _ort_package_name(
    providers: list[str],
    qnn_package: bool,
    *,
    ort_importable: bool | None = None,
) -> str | None:
    """Name the installed ORT wheel. ``[ort]`` is upstream ``onnxruntime``."""
    if qnn_package:
        return "onnxruntime_qnn"
    if DML_PROVIDER in providers:
        return "onnxruntime-directml"
    if ort_importable is None:
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            return None
        ort_importable = True
    return "onnxruntime" if ort_importable else None


def choose_execution_provider(
    providers: list[str],
    *,
    is_snapdragon: bool = False,
    is_arm64: bool = False,
    prefer_override: str | None = None,
) -> ProviderChoice:
    """Pick QNN, DirectML, or CPU from *available* providers.

    ``onnxruntime`` from the ``[ort]`` extra typically lists only
    ``CPUExecutionProvider``. That is not Hexagon QNN. QNN is listed only
    after ``onnxruntime_qnn`` registers ``QNNExecutionProvider``.
    """
    available = list(providers)
    if QNN_PROVIDER in available:
        preferred = QNN_PROVIDER
        label = "Hexagon NPU (QNN HTP)"
        has_npu = True
        ep_kind = EP_KIND_QNN
    elif DML_PROVIDER in available:
        preferred = DML_PROVIDER
        label = "DirectML (Adreno / Hexagon)" if is_snapdragon else "DirectML"
        has_npu = is_snapdragon
        ep_kind = EP_KIND_DIRECTML
    else:
        preferred = CPU_PROVIDER
        label = "ARM NEON CPU" if is_arm64 else "CPU"
        has_npu = False
        ep_kind = EP_KIND_CPU

    if prefer_override:
        preferred = prefer_override
        labels = {
            QNN_PROVIDER: "Hexagon NPU (QNN HTP)",
            DML_PROVIDER: "DirectML (Adreno / Hexagon)",
            CPU_PROVIDER: "ARM NEON CPU" if is_arm64 else "CPU",
        }
        label = labels.get(prefer_override, prefer_override)
        # ep_kind stays tied to what ORT actually listed, not the override.

    return ProviderChoice(
        preferred=preferred,
        label=label,
        has_npu=has_npu,
        ep_kind=ep_kind,
    )


def _try_register_qnn(htp_dir: Path | None) -> None:
    """Load QNN only when ``onnxruntime_qnn`` is installed.

    Importing CPU ``onnxruntime`` is not enough and must not be treated as
    a successful Hexagon EP registration.
    """
    try:
        import onnxruntime as ort
        import onnxruntime_qnn as qnn_ep
    except Exception:
        return
    try:
        if htp_dir and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(htp_dir))
            os.environ["PATH"] = str(htp_dir) + os.pathsep + os.environ.get("PATH", "")
        lib_dir = getattr(qnn_ep, "LIB_DIR_FULL_PATH", None)
        if lib_dir and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(lib_dir))
        ort.register_execution_provider_library(
            QNN_PROVIDER,
            qnn_ep.get_library_path(),
        )
    except Exception:
        return


def _honesty_notes(
    *,
    providers: list[str],
    ort_package: str | None,
    qnn_package: bool,
    ep_kind: str,
) -> list[str]:
    notes: list[str] = []
    cpu_ort = ort_package == "onnxruntime" and QNN_PROVIDER not in providers
    if cpu_ort:
        notes.append(
            "Installed onnxruntime is the CPU-oriented [ort] extra; it is not "
            "Qualcomm Hexagon QNN. True Hexagon listing needs onnxruntime_qnn "
            "(a different package) so QNNExecutionProvider appears."
        )
    if ep_kind == EP_KIND_DIRECTML:
        notes.append(
            "DirectML is listed; that is not QNNExecutionProvider / Hexagon HTP."
        )
    if QNN_PROVIDER in providers and not qnn_package:
        notes.append(
            "QNNExecutionProvider is listed without an importable onnxruntime_qnn "
            "package; treat that listing as the EP name only."
        )
    return notes


def probe_hardware() -> HardwareProbe:
    brand = _cpu_brand()
    machine = platform.machine() or ""
    is_snapdragon = _is_snapdragon(brand, machine)
    is_arm64 = machine.lower() in {"arm64", "aarch64"}
    htp_dir = find_qnn_htp_dir() if os.name == "nt" else None
    prefer_override = None
    try:
        from .config import active

        cfg = active()
        prefer_override = cfg.preferred_provider
        if cfg.qnn_htp_dir is not None:
            htp_dir = cfg.qnn_htp_dir
    except Exception:
        pass
    _try_register_qnn(htp_dir)
    providers = _onnx_providers()
    qnn_package = _qnn_package_available()
    listed = providers or [CPU_PROVIDER]
    choice = choose_execution_provider(
        listed,
        is_snapdragon=is_snapdragon,
        is_arm64=is_arm64,
        prefer_override=prefer_override,
    )
    ort_package = _ort_package_name(listed, qnn_package)

    memory = read_memory_status()
    ram = memory.total_gb if memory else None
    notes = []
    if is_snapdragon and ram is not None and ram < 16:
        notes.append(f"Copilot+ minimum is 16 GB RAM; this machine reports {ram} GB.")
    if is_snapdragon and ram is not None and 15 <= ram <= 17:
        notes.append("Matches first-gen Copilot+ 16 GB RAM baseline (2024).")
    if memory and memory.load_pct >= 85:
        notes.append(
            f"Memory load {memory.load_pct}% ({memory.available_gb} GB free of {memory.total_gb} GB)."
        )
    notes.extend(
        _honesty_notes(
            providers=listed,
            ort_package=ort_package,
            qnn_package=qnn_package,
            ep_kind=choice.ep_kind,
        )
    )

    return HardwareProbe(
        platform=platform.system(),
        arch=machine,
        processor=brand,
        ram_gb=ram,
        ram_available_gb=memory.available_gb if memory else None,
        ram_load_pct=memory.load_pct if memory else None,
        ram_bar_level=memory.bar_level if memory else None,
        is_snapdragon=is_snapdragon,
        is_windows_arm64=os.name == "nt" and is_arm64,
        providers=listed,
        preferred_provider=choice.preferred,
        provider_label=choice.label,
        has_npu=bool(choice.has_npu and providers),
        npu_tops=45 if is_snapdragon else None,
        qnn_htp_dir=str(htp_dir) if htp_dir else None,
        notes=" ".join(notes),
        memory=memory,
        ep_kind=choice.ep_kind,
        ort_package=ort_package,
        qnn_package=qnn_package,
    )

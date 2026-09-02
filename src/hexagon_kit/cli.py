"""CLI: hexagon hw | models list | download | path | delete."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .cache import is_installed, resolve
from .config import active, get_spec, list_specs
from .hw import probe_hardware
from .preflight import PreflightBlocked, preflight
from .status import delete_cached, start_download, ui_snapshot
from .xdg import default_config_path


def _progress(name: str, downloaded: int, total: int) -> None:
    if total > 0:
        pct = int(downloaded * 100 / total)
        mb = downloaded / (1024 * 1024)
        tot = total / (1024 * 1024)
        print(f"\r{name}: {pct:3d}% ({mb:.1f}/{tot:.1f} MB)", end="", file=sys.stderr)
    else:
        mb = downloaded / (1024 * 1024)
        print(f"\r{name}: {mb:.1f} MB", end="", file=sys.stderr)


def cmd_hw(_args: argparse.Namespace) -> int:
    probe = probe_hardware()
    print(json.dumps(probe.to_dict(), indent=2))
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    print(json.dumps(ui_snapshot(), indent=2))
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    result = preflight(args.model)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 2


def cmd_models_cache(_args: argparse.Namespace) -> int:
    print(active().cache_dir)
    return 0


def cmd_config_path(_args: argparse.Namespace) -> int:
    print(default_config_path())
    return 0


def cmd_config_show(_args: argparse.Namespace) -> int:
    print(json.dumps(active().to_dict(), indent=2))
    return 0


def cmd_models_list(_args: argparse.Namespace) -> int:
    rows = []
    for spec in list_specs():
        rows.append(
            {
                "id": spec.model_id,
                "slot": spec.slot,
                "name": spec.name,
                "disk_mb": spec.disk_mb,
                "installed": is_installed(spec.model_id),
            }
        )
    print(json.dumps(rows, indent=2))
    return 0


def cmd_models_download(args: argparse.Namespace) -> int:
    spec = get_spec(args.model)
    if args.async_job:
        print(json.dumps(start_download(spec.model_id, force=args.force), indent=2))
        return 0
    print(f"Downloading {spec.name} ({spec.model_id})...", file=sys.stderr)
    from .cache import download_model

    try:
        dest = download_model(spec.model_id, progress=_progress, force=args.force)
    except PreflightBlocked as exc:
        print(json.dumps(exc.result.to_dict(), indent=2), file=sys.stderr)
        return 2
    print(file=sys.stderr)
    print(str(dest))
    return 0


def cmd_models_path(args: argparse.Namespace) -> int:
    print(resolve(args.model))
    return 0


def cmd_models_delete(args: argparse.Namespace) -> int:
    spec = get_spec(args.model)
    delete_cached(spec.model_id)
    print(f"Deleted {spec.model_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hexagon",
        description="Snapdragon NPU Hexagon kit — hardware probe and model cache.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    hw = sub.add_parser("hw", help="Probe Snapdragon / Hexagon / DirectML / CPU")
    hw.set_defaults(func=cmd_hw)

    status = sub.add_parser("status", help="JSON snapshot for Settings / SnapDrago model cards")
    status.set_defaults(func=cmd_status)

    pf = sub.add_parser("preflight", help="RAM/disk guard before download or activate")
    pf.add_argument("model", help="Model id or slot")
    pf.set_defaults(func=cmd_preflight)

    cfg = sub.add_parser("config", help="Show effective configuration and overlay path")
    cfg_sub = cfg.add_subparsers(dest="config_cmd", required=True)
    cfg_show = cfg_sub.add_parser("show", help="Print the merged config (file + env + defaults)")
    cfg_show.set_defaults(func=cmd_config_show)
    cfg_path = cfg_sub.add_parser("path", help="Print the XDG config file path")
    cfg_path.set_defaults(func=cmd_config_path)

    models = sub.add_parser("models", help="List, download, resolve, or delete cached models")
    models_sub = models.add_subparsers(dest="models_cmd", required=True)

    cache = models_sub.add_parser("cache", help="Print the shared XDG model cache directory")
    cache.set_defaults(func=cmd_models_cache)

    lst = models_sub.add_parser("list", help="Show catalog and install status")
    lst.set_defaults(func=cmd_models_list)

    dl = models_sub.add_parser("download", help="Download a catalog model into the shared cache")
    dl.add_argument("model", help="Model id or slot (stt, tts, whisper_tiny_int8, kokoro_int8)")
    dl.add_argument("--async", dest="async_job", action="store_true", help="Start a background download and print job JSON")
    dl.add_argument("--force", action="store_true", help="Bypass RAM/disk preflight (may thrash this 16 GB PC)")
    dl.set_defaults(func=cmd_models_download)

    path = models_sub.add_parser("path", help="Print the installed slot directory")
    path.add_argument("model", help="Model id or slot")
    path.set_defaults(func=cmd_models_path)

    delete = models_sub.add_parser("delete", help="Remove a cached model")
    delete.add_argument("model", help="Model id or slot")
    delete.set_defaults(func=cmd_models_delete)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

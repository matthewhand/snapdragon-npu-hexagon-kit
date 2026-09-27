"""CLI: hexagon hw | models list | download | path | delete."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .cache import is_installed, resolve
from .config import active, get_spec, list_specs
from .credentials import save_tokens
from .hub import (
    HubUnavailable,
    delete_hub_model,
    fetch_hub_model,
    hub_info,
    hub_is_installed,
    hub_model_dir,
    list_hub_models,
    vendor_status,
)
from .hw import probe_hardware
from .leases import ModelInUse
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


def _builtin_rows() -> list[dict]:
    rows = []
    for spec in list_specs():
        rows.append(
            {
                "id": spec.model_id,
                "slot": spec.slot,
                "name": spec.name,
                "source": "builtin",
                "origin": "github",
                "urls": [art.url for art in spec.artifacts],
                "disk_mb": spec.disk_mb,
                "installed": is_installed(spec.model_id),
            }
        )
    return rows


def cmd_models_list(args: argparse.Namespace) -> int:
    source = getattr(args, "source", "builtin") or "builtin"
    domain = getattr(args, "domain", None)
    use_case = getattr(args, "use_case", None)
    quantized = getattr(args, "quantized", False) or None
    llm = getattr(args, "llm", False) or None
    rows: list[dict] = []
    if source in {"builtin", "all"}:
        rows.extend(_builtin_rows())
    if source in {"hub", "all"}:
        try:
            rows.extend(
                list_hub_models(domain=domain, use_case=use_case, quantized=quantized, llm=llm)
            )
        except HubUnavailable as exc:
            if source == "hub":
                print(json.dumps({"error": str(exc), "available": False}), file=sys.stderr)
                return 2
    print(json.dumps(rows, indent=2))
    return 0


def cmd_hub_status(_args: argparse.Namespace) -> int:
    payload = vendor_status()
    print(json.dumps(payload, indent=2))
    return 0


def cmd_hub_list(args: argparse.Namespace) -> int:
    try:
        rows = list_hub_models(
            domain=args.domain,
            use_case=args.use_case,
            quantized=True if args.quantized else None,
            llm=True if args.llm else None,
            tag=args.tag,
            community=True if args.community else None,
        )
    except HubUnavailable as exc:
        print(json.dumps({"error": str(exc), "available": False}), file=sys.stderr)
        return 2
    print(json.dumps(rows, indent=2))
    return 0


def cmd_hub_info(args: argparse.Namespace) -> int:
    try:
        print(json.dumps(hub_info(args.model), indent=2))
    except HubUnavailable as exc:
        print(json.dumps({"error": str(exc), "available": False}), file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_hub_fetch(args: argparse.Namespace) -> int:
    try:
        dest = fetch_hub_model(
            args.model,
            runtime=args.runtime,
            precision=args.precision,
            chipset=args.chipset,
            device=args.device,
        )
    except HubUnavailable as exc:
        print(json.dumps({"error": str(exc), "available": False}), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(str(dest))
    return 0


def cmd_hub_path(args: argparse.Namespace) -> int:
    if not hub_is_installed(args.model):
        print(f"error: hub model {args.model!r} is not installed", file=sys.stderr)
        return 1
    print(str(hub_model_dir(args.model)))
    return 0


def cmd_hub_delete(args: argparse.Namespace) -> int:
    delete_hub_model(args.model)
    print(f"Deleted hub/{args.model.strip().lower()}")
    return 0


def cmd_hub_configure(args: argparse.Namespace) -> int:
    hf = args.hf_token
    qai = args.qai_token
    if hf is None and qai is None:
        print(json.dumps(vendor_status()["credentials"], indent=2))
        return 0
    path = save_tokens(hf_token_value=hf, qai_token_value=qai)
    print(json.dumps({"ok": True, "secretsPath": str(path), "credentials": vendor_status()["credentials"]}, indent=2))
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
    try:
        delete_cached(spec.model_id)
    except ModelInUse as exc:
        print(json.dumps({"error": str(exc), "slot": exc.slot, "holder": exc.holder.to_dict()}, indent=2))
        return 2
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
    lst.add_argument(
        "--source",
        choices=("builtin", "hub", "all"),
        default="builtin",
        help="builtin kit catalog, Qualcomm AI Hub, or both",
    )
    lst.add_argument("--domain", help="Hub filter, e.g. Audio or Computer Vision")
    lst.add_argument("--use-case", dest="use_case", help="Hub filter, e.g. Speech Recognition")
    lst.add_argument("--quantized", action="store_true", help="Hub: quantized models only")
    lst.add_argument("--llm", action="store_true", help="Hub: LLM / text-generation models")
    lst.set_defaults(func=cmd_models_list)

    hub = sub.add_parser("hub", help="Qualcomm AI Hub catalog (qai_hub_models_cli)")
    hub_sub = hub.add_subparsers(dest="hub_cmd", required=True)

    hub_status = hub_sub.add_parser("status", help="Which Hub packages are importable")
    hub_status.set_defaults(func=cmd_hub_status)

    hub_list = hub_sub.add_parser("list", help="List Hub models by domain / use-case")
    hub_list.add_argument("--domain", help="e.g. Audio, Computer Vision, Generative AI")
    hub_list.add_argument("--use-case", dest="use_case", help="e.g. Speech Recognition")
    hub_list.add_argument("--quantized", action="store_true")
    hub_list.add_argument("--llm", action="store_true")
    hub_list.add_argument("--tag", help="e.g. LLM, VLM, Real-Time")
    hub_list.add_argument(
        "--community",
        action="store_true",
        help="Include Hugging Face models tagged qai-hub-models (uses HF_TOKEN if set)",
    )
    hub_list.set_defaults(func=cmd_hub_list)

    hub_cfg = hub_sub.add_parser(
        "configure",
        help="Save optional HF / Qualcomm tokens (not written to config.json)",
    )
    hub_cfg.add_argument("--hf-token", dest="hf_token", help="Hugging Face token (gated + community recipes)")
    hub_cfg.add_argument("--qai-token", dest="qai_token", help="Qualcomm AI Hub Workbench API token")
    hub_cfg.set_defaults(func=cmd_hub_configure)

    hub_info_p = hub_sub.add_parser("info", help="Hub model metadata")
    hub_info_p.add_argument("model", help="Hub model id or display name")
    hub_info_p.set_defaults(func=cmd_hub_info)

    hub_fetch = hub_sub.add_parser("fetch", help="Download a Hub asset into the shared cache")
    hub_fetch.add_argument("model", help="Hub model id")
    hub_fetch.add_argument("--runtime", default="onnx", help="onnx, qnn, tflite, …")
    hub_fetch.add_argument("--precision", default="float", help="float, w8a8, …")
    hub_fetch.add_argument("--chipset", help="e.g. qualcomm-snapdragon-x-elite (QNN AOT)")
    hub_fetch.add_argument("--device", help="Hub device name (mutually exclusive with --chipset)")
    hub_fetch.set_defaults(func=cmd_hub_fetch)

    hub_path = hub_sub.add_parser("path", help="Print installed Hub model directory")
    hub_path.add_argument("model")
    hub_path.set_defaults(func=cmd_hub_path)

    hub_del = hub_sub.add_parser("delete", help="Remove a cached Hub model")
    hub_del.add_argument("model")
    hub_del.set_defaults(func=cmd_hub_delete)

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

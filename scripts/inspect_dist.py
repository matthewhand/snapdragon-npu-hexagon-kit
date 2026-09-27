#!/usr/bin/env python3
"""Inspect sdist/wheel for a private SoT publish candidate.

T13 already runs pytest. This script is the leftover publish gap: prove
the built artifacts exist, ship consumer docs, and keep [ort] CPU-only
(no onnxruntime_qnn unless extra == 'qnn').
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PACKAGE = "snapdragon_npu_hexagon_kit"
EXPECTED_VERSION = "0.3.0"

SDIST_MUST_CONTAIN = (
    "CHANGELOG.md",
    "config.example.json",
    "docs/APP_INTEGRATION.md",
    "docs/INSTALL.md",
    "docs/PUBLISH.md",
    "pyproject.toml",
)


def _one(pattern: str) -> Path:
    matches = sorted(DIST.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected one {pattern} in {DIST}, found {matches}")
    return matches[0]


def _sdist_names(sdist: Path) -> list[str]:
    with tarfile.open(sdist, "r:gz") as tf:
        return [m.name for m in tf.getmembers() if m.isfile()]


def _wheel_metadata(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as zf:
        metas = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metas) != 1:
            raise SystemExit(f"expected one METADATA in {wheel}, found {metas}")
        return zf.read(metas[0]).decode("utf-8")


def main() -> int:
    if not DIST.is_dir():
        raise SystemExit(f"missing {DIST}; run `python -m build` first")

    sdist = _one("*.tar.gz")
    wheel = _one("*.whl")
    if EXPECTED_VERSION not in sdist.name or EXPECTED_VERSION not in wheel.name:
        raise SystemExit(f"artifact names must include {EXPECTED_VERSION}: {sdist.name} {wheel.name}")

    names = _sdist_names(sdist)
    prefix = f"{PACKAGE.replace('_', '-')}-{EXPECTED_VERSION}/"
    for rel in SDIST_MUST_CONTAIN:
        if not any(n.endswith(rel) or n == prefix + rel for n in names):
            raise SystemExit(f"sdist missing {rel}")

    meta = _wheel_metadata(wheel)
    if f"Version: {EXPECTED_VERSION}" not in meta:
        raise SystemExit("wheel METADATA version mismatch")
    if "Name: snapdragon-npu-hexagon-kit" not in meta:
        raise SystemExit("wheel METADATA name mismatch")

    ort_lines = [
        line
        for line in meta.splitlines()
        if line.startswith("Requires-Dist:") and 'extra == "ort"' in line
    ]
    qnn_lines = [
        line
        for line in meta.splitlines()
        if line.startswith("Requires-Dist:") and 'extra == "qnn"' in line
    ]
    if not ort_lines or not any("onnxruntime" in line for line in ort_lines):
        raise SystemExit(f"[ort] extra missing onnxruntime in METADATA: {ort_lines}")
    if any("onnxruntime_qnn" in line for line in ort_lines):
        raise SystemExit("[ort] must not require onnxruntime_qnn")
    if not qnn_lines or not any("onnxruntime_qnn" in line for line in qnn_lines):
        raise SystemExit(f"[qnn] extra missing onnxruntime_qnn in METADATA: {qnn_lines}")

    bare_qnn = [
        line
        for line in meta.splitlines()
        if line.startswith("Requires-Dist:")
        and "onnxruntime_qnn" in line
        and "extra ==" not in line
    ]
    if bare_qnn:
        raise SystemExit(f"onnxruntime_qnn must not be an unconditional dep: {bare_qnn}")

    print(f"ok: {sdist.name}")
    print(f"ok: {wheel.name}")
    print("ok: [ort] is CPU onnxruntime; [qnn] is onnxruntime_qnn")
    return 0


if __name__ == "__main__":
    sys.exit(main())

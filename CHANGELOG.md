# Changelog

## Unreleased

- **T5 — EP honesty.** `[ort]` stays CPU-only (`onnxruntime>=1.20`). Hexagon
  QNN is a different extra/package: `[qnn]` → `onnxruntime_qnn`. Docs and
  `hexagon hw` / `probe_hardware()` distinguish QNN vs DirectML vs CPU
  (`ep_kind`, `ort_package`, `qnn_package`) without changing existing
  caller fields.
- **T6 — catalog slots `llm` + `vision`.** Kit-owned Copilot+ pins:
  `smollm2_135m_int8` (SmolLM-class INT8, 400 MB RAM) and
  `rapidocr_ppocrv4_mobile` (small OCR, 200 MB RAM). SHA / `ram_mb` /
  `disk_mb` / `expected_files` set. Preflight gates downloads;
  `hexagon models list` shows both. No 8 GB defaults.
- **T13 — consumer integration docs + CI.** `docs/APP_INTEGRATION.md`
  shows how apps depend on `hexagon_kit`, read EP honesty fields,
  `ensure_model` / `preflight` slots `llm` and `vision`, and never
  advertise Hexagon/QNN without `QNNExecutionProvider`. Private SoT
  GitHub Actions runs pytest on PR/push to `main` (Linux; `npu` tests
  skip without QNN). Public API names stay stable.

## 0.2.0

Cross-process sharing for four Copilot+ apps on one 16 GB machine.

- Per-slot file lock, staging dir, `COMPLETE` sentinel, atomic publish. Torn downloads are not `is_installed`.
- Cross-process leases (`leases.json`). Default: one live holder per slot. `acquire` blocks with `PreflightBlocked` / `heldBy` until `force=True`.
- `delete_cached` / `hexagon models delete` refuse while a live lease exists (`ModelInUse`, exit 2). Dead PIDs are reaped.
- Machine NPU budget (`max_ram_mb`, default 3500) counts **peer** leases on acquire, not only this process.
- `ui_snapshot().pool.peers` lists who holds which slot.
- Download retries (3) on network errors. `rmtree(..., ignore_errors=True)` removed from delete.
- 0.1.0 caches without `COMPLETE` are healed on `is_installed`.
- Optional Qualcomm AI Hub wrap (`[hub]` extra = `qai_hub_models_cli` + `huggingface_hub`). **No Qualcomm API token** for list/fetch. Builtin sherpa/Kokoro still default to GitHub Releases. If `HF_TOKEN` / `hexagon hub configure --hf-token` is set, downloads prefer the Hugging Face hub cache (hard-link into the kit slot) so we do not double-fetch blobs already on disk. SHA mismatch falls back to GitHub. `hexagon hub list --community` and `fetch owner/name` use the same HF cache. Workbench token is optional. Secrets stay out of `config.json`.

Not in this release: PyPI upload, Hexagon QNN wheel, inference daemon, `hexagon ui` extra.

## 0.1.0

First tagged library slice for first-generation Copilot+ PCs (16 GB RAM, Hexagon 45 TOPS).

- Hardware probe: QNN → DirectML → CPU from ONNX Runtime when installed; live RAM via GlobalMemoryStatusEx / MemAvailable.
- XDG cache for weights (`hexagon-kit/models`), with legacy `%LOCALAPPDATA%\SnapdragonNpu\models` reuse.
- Builtin catalog: `stt` = `whisper_tiny_int8`, `tts` = `kokoro_int8`.
- Config overlays: defaults < file < env < `load_config(overrides=...)`.
- In-process `ModelPool` (default 3500 MB). Disk is shared; NPU sessions are not.
- `ui_snapshot()` / `hexagon status` JSON for app Settings pages. No widgets.
- Preflight blocks download/activate unless `force=True` (~1 GB RAM headroom, ~15% disk slack).
- SHA-256 pinned on builtin Whisper Tiny EN and Kokoro INT8 artifacts.
- Downloads hash SHA-256 in a streaming pass (no extra 100+ MB RAM spike).
- Example config overlays RAM only so copying it does not drop builtin SHA pins.
- CI builds an sdist/wheel after pytest so a missing `tests/` tree fails the job.
- sdist includes CHANGELOG and `config.example.json`.

Not in this release: PyPI upload, Hexagon QNN wheel, a cross-app NPU daemon.

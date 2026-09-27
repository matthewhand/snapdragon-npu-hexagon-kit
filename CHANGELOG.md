# Changelog

## Unreleased

App-facing UX surface. All schema changes are additive; existing keys keep
their meaning.

- **Reliable model cards.** `progressPct` aggregates across artifacts
  (no more 0→100→0 on Kokoro), disk state wins over a stale in-process
  job, failed jobs carry `error` + `errorType`. New card fields:
  `slotLabel`, `modality`, `chatCapable` (only `llm` is a chat engine),
  `canDelete`, `deleteBlockedReason`, `heldBy`, `peerDownload`, `job`,
  `path`, `sizeBytes`, `downloadedBytes`, `totalBytes`.
- **Lease-aware delete.** Cards omit `delete` while any process holds a
  lease on the slot. `delete_cached` raises `DownloadInProgress` while this
  process is still downloading.
- **Cancel.** `cancel_download(slot)`; jobs go `cancelling` → `cancelled`
  and staging is removed. `get_job()` / `download_jobs()` expose jobs.
  Card `actions` shows `cancel` during an own download.
- **Snapshot metadata.** `ui_snapshot()` adds `schemaVersion` (1),
  `kitVersion`, `generatedAt`, `slots`, `jobs`, `settings`; `pool.peers`
  rows add `isSelf`.
- **Shared Settings schema.** `settings_snapshot()`, `validate_settings()`,
  `save_settings()` (writes the shared XDG `config.json`, keeps
  `models[]`), `SettingsError`, and a reusable `SettingField` /
  `validate_values` descriptor for app-owned settings. JSON only.
- **CLI.** `hexagon status --text`, `hexagon models status [model] [--text]`,
  `hexagon config settings | set | unset`. Ctrl+C exits 130; `KeyError`
  messages are no longer repr-quoted.
- **Shared cache hygiene.** Interrupted downloads (Ctrl+C, cancel) remove
  their staging dir; dead-PID staging dirs are reaped under the slot lock.
  `peer_download()` reports another app's in-flight download.
- **Shared download jobs.** Job records live in `<cache>/jobs/` under a
  file lock. `ui_snapshot()` / `model_card()` / `download_jobs()` in any
  process show the same progress. A job whose process has exited is marked
  `failed` with `errorType: "ProcessExited"`.
- **Progress callbacks and watch.** `start_download(..., on_progress, on_done)`
  and `watch_jobs(interval)` / `poll_jobs()`. Job `phase` is one of
  `starting`, `waiting_on_lock`, `resolving_hf`, `linking`, `downloading`,
  `fetching_hub`, `verifying`.
- **Lock wait.** A download blocked on another process's slot lock reports
  `waiting_on_lock` and `lockHolder` (`pid`, `exe`) instead of a generic
  error. The wait budget resets while that process's staging directory
  grows. It fails with `LockWaitExceeded` only after a quiet budget
  (~300s).
- **Hub cards.** Installed Hub models, and Hub downloads in flight, are
  cards on `ui_snapshot()["models"]` with `source: "hub"` and the same
  fields. `start_hub_download(id)` uses the shared job records. Actions
  are `fetch`, `cancel`, and `delete`.
- **CLI downloads that outlive the command.** `hexagon models download
  --async` starts a detached process and prints the job JSON. Blocking
  downloads accept `--json-lines`. `hexagon jobs list|status|cancel`
  reads the shared jobs.
- **Cheaper snapshot.** `vendor_status()` / `hub_snapshot()` probe Hub
  packages with `find_spec` + metadata instead of importing them, so
  `ui_snapshot()` no longer loads the torch-heavy `qai_hub_models`.

## 0.3.0 — 2026-09-27

Private SoT **publish candidate**. Not a PyPI upload. Public promote stays
human-gated. Do not force-push the public mirror from this tag.

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
- **T19 — Lane-1 CI harness.** Importable `hexagon_kit.testing` mocks
  ORT listings and kit classify/detect (QNN → DirectML → CPU) so
  SnipPilot / Persona / audience do not invent fake EP probes. Cold
  honesty: CPU / `[ort]` never paints Hexagon (`hexagon_qnn` is false
  unless QNN is listed). Tiny fixture catalog + `ensure_fixture` /
  `stub_ensure_resolve` keep Linux CI offline (no multi-GB weights).
  T5 honesty fields unchanged.
- **Publish polish.** Version `0.3.0` is pinned in `pyproject.toml`,
  `hexagon_kit.__version__`, and this heading. Consumer
  [docs/INSTALL.md](docs/INSTALL.md) covers git/tree install and the
  `[ort]` CPU vs `[qnn]` Hexagon extras. [docs/PUBLISH.md](docs/PUBLISH.md)
  is the private-only checklist. CI inspects sdist/wheel metadata and
  smoke-imports the wheel. `[ort]` must not require `onnxruntime_qnn`.

Not in this candidate: PyPI upload, live Hexagon runner, public-mirror
writes.

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

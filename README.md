# snapdragon-npu-hexagon-kit

Shared Hexagon NPU **runtime kit** aimed at first-generation Copilot+ PC apps
(Persona Snapdragon, SnapDrago, npu_pipeline, and future apps). Persona’s
bridge can import `ui_snapshot` when this package is on `PYTHONPATH`. SnapDrago
still ships its own model manager.

| | |
|---|---|
| Project name | `snapdragon-npu-hexagon-kit` |
| Python import | `hexagon_kit` |
| CLI | `hexagon` |

This is a model cache + hardware probe + preflight library, not an application
framework. It does not draw widgets, load Whisper/Kokoro engines, or bind a
network port.

This tree is the **private source of truth**
(`snapdragon-npu-hexagon-kit-private`). Public name:
https://github.com/matthewhand/snapdragon-npu-hexagon-kit (human-gated
promote only — do not force-push that mirror from here).

The package is **not on PyPI** (`import hexagon_kit`, CLI `hexagon`).
Version **0.3.0** is a private publish candidate. Install notes:
[docs/INSTALL.md](docs/INSTALL.md) (`[ort]` = CPU `onnxruntime`;
`[qnn]` = Hexagon `onnxruntime_qnn`). Publish checklist:
[docs/PUBLISH.md](docs/PUBLISH.md).

**Consumer apps:** follow [docs/APP_INTEGRATION.md](docs/APP_INTEGRATION.md) —
depend on `hexagon_kit`, call `probe_hardware()` / EP honesty fields,
`ensure_model` / `preflight` for slots `llm` and `vision`, and **never
advertise Hexagon/QNN** unless `QNNExecutionProvider` is listed. Lane-1
CI: import `hexagon_kit.testing` (do not invent a fake EP probe; fixture
catalog stays offline).

GitHub Actions CI runs CPython 3.12/3.13 on `ubuntu-latest`; that is not
a Hexagon box. NPU-marked tests skip when QNN is absent. Do not upload
to PyPI until a human has a token and promotes.

---

## Hardware baseline

**Developed and tested on first-generation Copilot+ PCs (announced 20 May 2024,
shipping 18 June 2024).**

| | |
|---|---|
| SoC | Qualcomm Snapdragon X Elite / X Plus |
| NPU | Qualcomm Hexagon, 45 TOPS (Copilot+ requires ≥ 40 TOPS) |
| RAM | **16 GB** system memory (machine RAM, not a chip name) |
| Storage | ≥ 256 GB SSD |
| OS | Windows 11 ARM64 |

16 GB is the SKU this kit sizes catalog models for. Later Copilot+ silicon may
run; it is not the claimed baseline. `pytest` is hardware-agnostic (it does not
need a Hexagon).

`hexagon hw` does **not** measure TOPS. If the CPU brand looks like Snapdragon,
`npu_tops` is reported as `45` because that is the first-gen Copilot+ claim.

---

## Install

Full consumer recipe (git pin, extras, wheel): [docs/INSTALL.md](docs/INSTALL.md).

```powershell
pip install -e .
pip install -e ".[dev]"    # pytest
```

The `hexagon` script is on PATH after the editable install. Equivalent:
`python -m hexagon_kit`. `hexagon --version` prints `0.3.0` on this
candidate.

Optional ONNX Runtime extras — **these are different packages**:

| Extra | Installs | What `hexagon hw` can list |
|---|---|---|
| `[ort]` | upstream `onnxruntime>=1.20` | **CPU-only** on most wheels (`CPUExecutionProvider`). Not Hexagon QNN. |
| `[qnn]` | `onnxruntime_qnn` | `QNNExecutionProvider` on Windows ARM64 Copilot+ with HTP drivers. |

```powershell
pip install -e ".[ort]"    # CPU-oriented. Does not equal Qualcomm Hexagon QNN.
pip install -e ".[qnn]"    # Hexagon QNN EP. Separate package from [ort].
```

`[ort]` never installs `onnxruntime_qnn`. If ORT reports only
`CPUExecutionProvider`, that is the CPU extra, not a silent Hexagon path.
DirectML (`DmlExecutionProvider`, typically `onnxruntime-directml`) is a
third listing — Adreno GPU, not QNN HTP.

**Hub is not required.** Default STT/TTS downloads are public GitHub Releases
(no Qualcomm account, no Hugging Face login):

| Slot | Package | URL |
|---|---|---|
| `stt` | sherpa-onnx Whisper Tiny EN INT8 | `github.com/k2-fsa/sherpa-onnx/releases` |
| `tts` | Kokoro v1.0 INT8 + voices | `github.com/thewh1teagle/kokoro-onnx/releases` |

Optional Qualcomm AI Hub catalog (still no Workbench token for list/fetch):

```powershell
pip install -e ".[hub]"    # qai_hub_models_cli — public S3 assets
```

`hexagon hub list` / `info` / `fetch` wrap that CLI. Prebuilt Hub zips come from
`qaihub-public-assets.s3.us-west-2.amazonaws.com`.

Optional tokens (never stored in `config.json`):

```powershell
hexagon hub configure --hf-token hf_...     # gated HF + community qai-hub-models
hexagon hub configure --qai-token ...       # Workbench compile only
```

Env also works: `HF_TOKEN`, `QAI_HUB_API_TOKEN`. With an HF token, builtin
`hexagon models download stt|tts` prefers the Hugging Face hub cache
(`csukuangfj/sherpa-onnx-whisper-tiny.en`, Kokoro mirrors) so a file already
fetched by `transformers` / `huggingface_hub` is hard-linked into the kit slot
instead of downloaded again. `hexagon hub list --community` lists newer HF
recipes tagged `qai-hub-models`, and `hexagon hub fetch owner/name` uses
`snapshot_download` into that same HF cache. GitHub Releases remain the
no-token fallback. SHA-256 still applies; a mismatched HF blob falls back to
GitHub. A Qualcomm
Workbench token is **not** required for builtin GitHub STT/TTS or public Hub
S3 fetch. Kokoro is not a Hub model, so builtin `tts` stays GitHub.
Without any ORT package, `probe_hardware().providers` is
`["CPUExecutionProvider"]` and `ep_kind` is `cpu`.

`probe_hardware()` / `hexagon hw` distinguish the three EPs from what ORT
actually lists (not from “an ORT wheel is installed”):

| `ep_kind` | Provider | Means |
|---|---|---|
| `qnn` | `QNNExecutionProvider` | Hexagon HTP. Needs `onnxruntime_qnn`. |
| `directml` | `DmlExecutionProvider` | DirectML. Not QNN. |
| `cpu` | `CPUExecutionProvider` | CPU (`[ort]` or no ORT). Not Hexagon. |

`ort_package` is `onnxruntime` / `onnxruntime-directml` / `onnxruntime_qnn`.
`qnn_package` is true only when `import onnxruntime_qnn` works. Existing
fields (`providers`, `preferred_provider`, `has_npu`) stay stable for
callers.

Provider preference, when ORT reports them: **QNN → DirectML → CPU**. HTP
discovery is a glob of `qcnspmcdm*/HTP` plus `HEXAGON_QNN_HTP_DIR` / config —
not a machine-specific DriverStore INF hash.

---

## Cache

Weights are cache (re-downloadable), not `$XDG_DATA_HOME`.

| Priority | Path |
|---|---|
| 1 | `$HEXAGON_KIT_CACHE` |
| 2 | `$XDG_CACHE_HOME/hexagon-kit/models` |
| 3 | Windows, no XDG: `%LOCALAPPDATA%\hexagon-kit\models` |
| 4 | `~/.cache/hexagon-kit/models` |

If the modern path does not exist, Windows may reuse a leftover
`%LOCALAPPDATA%\SnapdragonNpu\models` directory. Same path → one copy on disk.
The OS page cache may share those file pages across processes; loaded ORT /
sherpa / Hexagon sessions are still **per process**.

Install is “the `expected_files` exist,” not “the folder is bigger than 1 MB.”
SHA-256 is hashed while the bytes stream to disk when a catalog artifact sets
`sha256`. Builtin Whisper Tiny EN (`.tar.bz2`) and Kokoro INT8 +
`voices-v1.0.bin` are pinned. Copy `config.example.json` as-is: it overlays
`ram_mb` only. A `models[]` object that includes `artifacts` replaces the
whole list and will drop those pins unless you copy the hashes too.

```powershell
hexagon models cache
```

---

## Configuration overlays

Highest last: **defaults < XDG `config.json` < environment <
`load_config(overrides=...)`**.

Config file: `$XDG_CONFIG_HOME/hexagon-kit/config.json`, or
`%APPDATA%\hexagon-kit\config.json` on Windows, or `$HEXAGON_KIT_CONFIG`.
See `config.example.json`. Never put secrets in this file.

| Key | Meaning |
|---|---|
| `cache_dir` | Shared model directory |
| `max_ram_mb` | `ModelPool` budget (default **3500**) |
| `preferred_provider` | `QNNExecutionProvider` / `DmlExecutionProvider` / `CPUExecutionProvider` |
| `qnn_htp_dir` | Qualcomm HTP driver directory |
| `models` | Overlay a builtin entry by `model_id`, or add a new slot |

A `models[]` object that includes `artifacts` **replaces** the whole artifact
list for that id. To change RAM only, omit `artifacts`:

```json
{
  "max_ram_mb": 2800,
  "preferred_provider": "CPUExecutionProvider",
  "models": [
    { "model_id": "kokoro_int8", "ram_mb": 280 }
  ]
}
```

`$HEXAGON_KIT_CONFIG` selects the file path. These env vars overlay keys
(they win over the file, lose to `load_config(overrides=...)`):
`HEXAGON_KIT_CACHE`, `HEXAGON_KIT_MAX_RAM_MB`, `HEXAGON_KIT_PROVIDER`,
`HEXAGON_QNN_HTP_DIR`.

```powershell
hexagon config path
hexagon config show
```

---

## CLI

```powershell
hexagon hw
hexagon status
hexagon preflight tts
hexagon config show
hexagon models cache
hexagon models list
hexagon models download whisper_tiny_int8
hexagon models download kokoro_int8
hexagon models download llm
hexagon models download vision
hexagon models path stt
hexagon models path tts
hexagon models path llm
hexagon models path vision
hexagon models delete whisper_tiny_int8
hexagon hub status
hexagon hub list --domain Audio
hexagon hub list --use-case "Speech Recognition"
hexagon hub info Whisper-Tiny
hexagon hub fetch whisper_tiny --runtime onnx --precision float
```

- `hexagon preflight <model>` prints JSON and exits **0** if it fits, **2** if
  RAM or disk is insufficient.
- `hexagon models download` calls the same guard. On failure it prints that JSON
  to stderr and exits 2. `--force` bypasses it; use that only as an explicit
  user choice.
- `hexagon models path` prints the slot directory or exits 1 if the files are
  not installed (`hexagon models download …` first).
- `hexagon status` is `ui_snapshot()` as JSON: hardware, live RAM, disk, catalog
  cards (`actions`, `ramFit` / `ramFitLabel`), pool, config sources. This kit
  does not draw the Settings UI.

Live RAM is `GlobalMemoryStatusEx` on Windows (`total`, `available`, `load %`,
`barLevel` green / orange / red) or `MemAvailable` on Linux.

---

## Python

```python
from hexagon_kit import (
    probe_hardware,
    ensure_model,
    resolve,
    process_pool,
    PreflightBlocked,
    list_hub_models,
    fetch_hub_model,
    HubUnavailable,
)

try:
    audio = list_hub_models(domain="Audio", use_case="Speech Recognition")
except HubUnavailable:
    audio = []  # pip install -e ".[hub]"

print(probe_hardware())

try:
    stt_dir = ensure_model("whisper_tiny_int8")  # downloads if missing
except PreflightBlocked as blocked:
    print(blocked.result.to_dict())  # suggestId, message, canForce
    raise SystemExit(blocked)

assert resolve("stt") == stt_dir  # resolve() never downloads; raises if missing

# The kit does not ship STT/TTS runtimes. The app supplies the loader.
pool = process_pool()
pool.register("stt", lambda path: path)
handle = pool.acquire("stt")
pool.release("stt")
```

`ensure_model(..., force=True)` and `pool.acquire(..., force=True)` are the
same explicit bypass as `hexagon models download --force`.

`open_onnx(path)` uses `onnxruntime.InferenceSession` with `provider_chain()`.
The optional `[ort]` extra is CPU-oriented; Hexagon QNN needs
`onnxruntime_qnn` so the chain can include `QNNExecutionProvider`. It is not
a sherpa Whisper or Kokoro wrapper.

Apps should call `resolve("stt")` / `resolve("tts")` / `resolve("llm")` /
`resolve("vision")` instead of hardcoding `C:\tmp\npu_pipeline\models`.
See [docs/APP_INTEGRATION.md](docs/APP_INTEGRATION.md).

---

## Preflight

`preflight()` / `hexagon preflight` compare the spec’s `ram_mb` / `disk_mb` to
**live** available RAM and free disk.

- RAM: keep ~**1 GB** headroom (`RESERVE_RAM_MB = 1024`). `fits` / `tight` /
  `unsafe`. `ok` is true only for `fits` with enough disk.
- Disk: require `disk_mb * 1.15` (~15% slack for extract/temp files).
- If another catalog model in the **same slot** has lower `ram_mb` and still
  `fits`, `suggestId` / `suggestName` point at it. Builtin catalog has one
  model per slot, so those fields stay empty unless you overlay a heavier
  variant.
- `canForce` is true when the guard would block.

`ensure_model` / `download_model` run this before a download.
`ModelPool.acquire` runs it before the first load of a slot.
`resolve` does not.

A 4 GB LLM added via config overlay is refused on a 16 GB box with ~2 GB free
unless the caller passes `force=True`. Builtin `llm` / `vision` stay
SmolLM-class / small OCR (`ram_mb` well under 1 GB), not 8 GB defaults.

---

## Disk vs RAM

| Layer | Shared across apps? | Who owns it |
|---|---|---|
| XDG weight files | Yes (one download) | kit cache |
| OS page cache of those files | Yes, opportunistically | kernel |
| Loaded ORT / sherpa / Hexagon sessions | **No** — per process | `ModelPool` inside each app |
| Slot lease (who has STT/TTS loaded) | Yes (`leases.json`) | kit; exclusive unless `force=True` |
| Cross-app NPU SRAM / daemon | Not in this kit | future work. This library does not bind 8765 or 47831. |

`ModelPool` is in-process: one load per slot, refcounts, evicts unused slots,
default budget 3.5 GB. It does not share Hexagon SRAM between Persona and
SnapDrago. A cross-process **lease** records which PID holds each slot so a
second app is blocked (or must Force) and delete cannot yank files in use.
`ui_snapshot()["pool"]["peers"]` lists holders.

---

## Builtin catalog

The kit owns these slots. Each entry pins `sha256`, `ram_mb`, `disk_mb`, and
`expected_files`. `hexagon models list` shows them; `hexagon preflight` /
`ensure_model` gate downloads.

| Slot | Id | Artifacts | ~RAM |
|---|---|---|---|
| `stt` | `whisper_tiny_int8` | sherpa-onnx Whisper Tiny EN INT8 (encoder + decoder + tokens), unpacked from the upstream `.tar.bz2` | 150 MB |
| `tts` | `kokoro_int8` | `kokoro-v1.0.int8.onnx` + `voices-v1.0.bin` | 250 MB |
| `llm` | `smollm2_135m_int8` | SmolLM2 135M Instruct INT8 ONNX + `tokenizer.json` | 400 MB |
| `vision` | `rapidocr_ppocrv4_mobile` | RapidOCR PP-OCRv4 mobile det + rec ONNX | 200 MB |

`llm` / `vision` are first-gen 16 GB Copilot+ sized (SmolLM-class / small OCR).
There is no 8 GB default. Preflight and `ui_snapshot` use those fields. Add
further models through `models[]` in config when a second app actually loads
the same files. Overlaying a new `model_id` on slot `llm` does not replace
the builtin slot lookup; `get_spec("llm")` still returns the kit pin.

---

## What this kit does not own

Electron/Qt widgets, VRM visemes, snipping, avatars, ElevenLabs keys, a global
“AI mode,” or listen ports. Default ports in *apps* may exist; this kit does
not bind one.

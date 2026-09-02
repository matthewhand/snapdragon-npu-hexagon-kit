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

Source: https://github.com/matthewhand/snapdragon-npu-hexagon-kit

The package is **not on PyPI**. Install from this tree.

TODO: publish `snapdragon-npu-hexagon-kit` to PyPI (`import hexagon_kit`, CLI `hexagon`). Do not upload until a PyPI token is available. GitHub Actions CI runs CPython 3.12/3.13 on `ubuntu-latest`; that is not a Hexagon box.

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

```powershell
pip install -e .
pip install -e ".[dev]"    # pytest
```

The `hexagon` script is on PATH after the editable install. Equivalent:
`python -m hexagon_kit`.

Optional ONNX Runtime extra:

```powershell
pip install -e ".[ort]"
```

That extra is upstream `onnxruntime>=1.20`. On many machines that wheel is
CPU-only. It is **not** Qualcomm’s QNN build. Hexagon listing needs
`onnxruntime_qnn` (a different package); this extra does not install it.
Without any ORT package, `probe_hardware().providers` is
`["CPUExecutionProvider"]`.

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
hexagon models path stt
hexagon models path tts
hexagon models delete whisper_tiny_int8
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
)

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

`open_onnx(path)` (optional `ort` extra) is `onnxruntime.InferenceSession` with
`provider_chain()`. It is not a sherpa Whisper or Kokoro wrapper.

Apps should call `resolve("stt")` / `resolve("tts")` instead of hardcoding
`C:\tmp\npu_pipeline\models`.

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
unless the caller passes `force=True`. The builtin catalog does not include
Gemma / LLM / vision.

---

## Disk vs RAM

| Layer | Shared across apps? | Who owns it |
|---|---|---|
| XDG weight files | Yes (one download) | kit cache |
| OS page cache of those files | Yes, opportunistically | kernel |
| Loaded ORT / sherpa / Hexagon sessions | **No** — per process | `ModelPool` inside each app |
| Cross-app NPU SRAM / daemon | Not in this kit | future work; port would come from config/env. This library does not bind 8765 or 47831. |

`ModelPool` is in-process: one load per slot, refcounts, evicts unused slots,
default budget 3.5 GB. It does not share Hexagon SRAM between Persona and
SnapDrago.

---

## Builtin catalog

| Slot | Id | Artifacts |
|---|---|---|
| `stt` | `whisper_tiny_int8` | sherpa-onnx Whisper Tiny EN INT8 (encoder + decoder + tokens), unpacked from the upstream `.tar.bz2` |
| `tts` | `kokoro_int8` | `kokoro-v1.0.int8.onnx` + `voices-v1.0.bin` |

Each entry has `slot`, `ram_mb`, `disk_mb`, and `expected_files`. Preflight and
`ui_snapshot` use those fields. Add further models through `models[]` in config
(or a future catalog slice) when a second app actually loads the same files.

---

## What this kit does not own

Electron/Qt widgets, VRM visemes, snipping, avatars, ElevenLabs keys, a global
“AI mode,” or listen ports. Default ports in *apps* may exist; this kit does
not bind one.

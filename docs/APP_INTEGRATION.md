# App integration

How consumer apps (Persona Snapdragon, SnapDrago, `npu_pipeline`, and
future Copilot+ clients) should depend on this kit. The package import is
`hexagon_kit`. Do not invent a parallel catalog, cache path, or Hexagon
badge.

This library caches weights, probes hardware, and gates downloads. It does
not run Whisper, SmolLM, RapidOCR, or bind a port. The app supplies the
loader.

Public promote of this private source of truth is **human-gated**. Keep the
Python API stable (`probe_hardware`, `ensure_model`, `preflight`,
`resolve`, slot names). Do not rename those symbols.

---

## Depend on `hexagon_kit`

The package is **not on PyPI**. Install from this tree (editable) or a git
checkout of the private SoT:

```powershell
pip install -e .
pip install -e ".[dev]"     # pytest
pip install -e ".[ort]"     # CPU-oriented onnxruntime. Not Hexagon QNN.
pip install -e ".[qnn]"     # onnxruntime_qnn — Hexagon QNN EP only.
```

| Extra | Wheel | What it is |
|---|---|---|
| `[ort]` | `onnxruntime>=1.20` | CPU. Most wheels list only `CPUExecutionProvider`. |
| `[qnn]` | `onnxruntime_qnn` | Hexagon HTP. Different package. Required for `QNNExecutionProvider`. |

`[ort]` never installs `onnxruntime_qnn`. Do not treat “an ORT extra is
installed” as Hexagon.

Import the stable surface:

```python
from hexagon_kit import (
    ensure_model,
    get_spec,
    preflight,
    probe_hardware,
    process_pool,
    resolve,
    PreflightBlocked,
    ModelNotInstalled,
)
```

CLI equivalent: `hexagon` / `python -m hexagon_kit`. Apps should call the
Python API, not scrape CLI text.

---

## Probe hardware and EP honesty

Call `probe_hardware()` once at startup (and again if the user installs
ORT). Read the **honesty** fields before any UI copy that says Hexagon or
QNN.

```python
probe = probe_hardware()

# Stable caller fields (do not rename):
probe.providers            # what ORT actually listed
probe.preferred_provider   # QNN → DirectML → CPU when listed
probe.provider_label
probe.has_npu              # True only when a non-CPU EP was listed
probe.npu_tops             # 45 on Snapdragon brand; not a measurement

# Honesty fields (T5; additive):
probe.ep_kind              # "qnn" | "directml" | "cpu"
probe.ort_package          # "onnxruntime" | "onnxruntime-directml" | "onnxruntime_qnn"
probe.qnn_package          # True only if `import onnxruntime_qnn` works
```

`hexagon hw` prints the same object as JSON (`probe.to_dict()`).

| `ep_kind` | Provider listed | UI may say |
|---|---|---|
| `qnn` | `QNNExecutionProvider` | Hexagon NPU / QNN HTP |
| `directml` | `DmlExecutionProvider` | DirectML (Adreno). **Not** QNN. |
| `cpu` | `CPUExecutionProvider` | CPU. **Not** Hexagon. Typical `[ort]` extra. |

`preferred_provider` can be overridden in config. `ep_kind` stays tied to
what ORT listed, not the override. Do not use the override as a Hexagon
badge.

---

## Never advertise Hexagon / QNN without the QNN EP

Apps must not show “Hexagon NPU”, “Qualcomm QNN”, or a Hexagon badge
unless **all** of these are true:

1. `"QNNExecutionProvider" in probe.providers`
2. `probe.ep_kind == "qnn"`
3. `probe.qnn_package is True`

```python
QNN = "QNNExecutionProvider"

def can_advertise_hexagon(probe) -> bool:
    return (
        probe.ep_kind == "qnn"
        and QNN in probe.providers
        and probe.qnn_package
    )

probe = probe_hardware()
if can_advertise_hexagon(probe):
    label = probe.provider_label  # "Hexagon NPU (QNN HTP)"
else:
    # CPU [ort], DirectML, or no ORT — never say Hexagon / QNN.
    label = {
        "directml": "DirectML",
        "cpu": "CPU",
    }.get(probe.ep_kind, "CPU")
```

Counter-examples that must **not** light a Hexagon badge:

- `[ort]` / upstream `onnxruntime` listing only `CPUExecutionProvider`
- DirectML listed (`DmlExecutionProvider`) — Adreno GPU, not HTP
- Snapdragon CPU brand / `npu_tops == 45` / `is_snapdragon` without QNN EP
- `preferred_provider` forced to `QNNExecutionProvider` in config while
  `ep_kind` is still `cpu`

`has_npu` can be true on Snapdragon + DirectML. That is still **not** QNN.
Use `ep_kind == "qnn"` for Hexagon copy.

---

## Slots `llm` and `vision`

The kit owns four builtin slots. `llm` / `vision` are first-gen 16 GB
Copilot+ pins (SmolLM-class / small OCR). There is no 8 GB default.

| Slot | Id | ~RAM | Files the app loads |
|---|---|---|---|
| `stt` | `whisper_tiny_int8` | 150 MB | sherpa INT8 encoder / decoder / tokens |
| `tts` | `kokoro_int8` | 250 MB | `kokoro-v1.0.int8.onnx` + `voices-v1.0.bin` |
| `llm` | `smollm2_135m_int8` | 400 MB | `model_int8.onnx` + `tokenizer.json` |
| `vision` | `rapidocr_ppocrv4_mobile` | 200 MB | PP-OCRv4 mobile det + rec ONNX |

Pass the **slot** name (`"llm"`, `"vision"`) to `preflight` / `ensure_model`
/ `resolve`. `get_spec("llm")` returns the kit pin (`smollm2_135m_int8`).
A config overlay with a new `model_id` on slot `llm` does not replace that
builtin slot lookup.

### Preflight before download or first load

```python
from hexagon_kit import PreflightBlocked, ensure_model, preflight, resolve

for slot in ("llm", "vision"):
    guard = preflight(slot)
    if not guard.ok:
        # JSON for Settings: ramFit, canForce, suggestId, message
        print(guard.to_dict())
        if not guard.can_force:
            continue
        # Only Force when the user explicitly chose it.

    try:
        path = ensure_model(slot)          # downloads if missing; runs preflight
    except PreflightBlocked as blocked:
        print(blocked.result.to_dict())
        raise SystemExit(blocked)

    assert resolve(slot) == path           # never downloads; raises if missing
    spec = get_spec(slot)
    for name in spec.expected_files:
        assert (path / name).is_file()
    # App-owned loader goes here. The kit does not run SmolLM or OCR.
```

`ensure_model(..., force=True)` is the same explicit bypass as
`hexagon models download llm --force`. Do not pass `force=True` by default.

`resolve("llm")` / `resolve("vision")` after a successful ensure. Do not
hardcode `C:\tmp\npu_pipeline\models` or a per-app cache.

CLI:

```powershell
hexagon preflight llm
hexagon preflight vision
hexagon models download llm
hexagon models download vision
hexagon models path llm
hexagon models path vision
```

`hexagon preflight` exits **0** if it fits, **2** if RAM or disk is
insufficient.

### Optional in-process pool

```python
pool = process_pool()
pool.register("llm", lambda path: path)       # app supplies the real loader
handle = pool.acquire("llm")                  # preflight + exclusive lease
# ... inference in this process ...
pool.release("llm")
```

Disk weights are shared (XDG cache). Loaded sessions are **per process**.
A second app that already holds the slot is blocked unless `force=True`.

---

## Settings / status JSON

`ui_snapshot()` / `hexagon status` is the Settings payload: hardware
(including honesty fields), live RAM, disk, catalog cards (`actions`,
`ramFit` / `ramFitLabel`), pool peers, config sources. Bind that JSON.
Do not draw kit widgets here.

---

## CI and hardware-less runners

Private SoT CI (`.github/workflows/ci.yml`) runs `pytest` on
`ubuntu-latest` (CPython 3.12 / 3.13). That is **not** a Hexagon box.

Tests must stay hardware-agnostic: probe helpers, catalog pins, and
preflight shape run everywhere. Tests that need a live QNN EP use the
`npu` pytest marker and **skip** when `QNNExecutionProvider` is absent.
Do not fail Linux CI for missing HTP drivers.

---

## Lane-1 CI harness (SnipPilot / Persona / audience)

Do **not** invent a fake `probe_hardware()` or a home-grown
`get_available_providers()` stub in the app. Import the kit harness:

```python
from hexagon_kit.testing import (
    CPU_PROVIDER,
    LISTING_CPU,
    LISTING_DIRECTML,
    LISTING_QNN,   # QNN → DirectML → CPU
    QNN_PROVIDER,
    assert_cold_honesty,
    classify_providers,
    detect_hardware,
    ensure_fixture,
    hexagon_qnn,
    install_fixture_catalog,
    stub_ensure_resolve,
)
```

`hexagon_kit.testing` is part of the installed package. T5 honesty fields
stay stable; `hexagon_qnn(probe)` is the shared boolean for “may paint
Hexagon”. CPU / `[ort]` is never Hexagon.

### Mock ORT listings and classify / detect

```python
def test_prefer_qnn_without_listing_stays_cpu():
    probe = detect_hardware(LISTING_CPU, prefer=QNN_PROVIDER)
    assert probe.ep_kind == "cpu"
    assert hexagon_qnn(probe) is False
    assert_cold_honesty(probe)

def test_listed_qnn_is_hexagon():
    probe = detect_hardware(LISTING_QNN)
    assert hexagon_qnn(probe) is True

def test_classify_order():
    assert classify_providers(LISTING_QNN).ep_kind == "qnn"
    assert classify_providers(LISTING_DIRECTML).ep_kind == "directml"
    assert classify_providers(LISTING_CPU).ep_kind == "cpu"
```

`detect_hardware` patches the kit detect path (`probe_hardware`).
`classify_providers` is the kit classify path (`choose_execution_provider`).
Neither needs Hexagon hardware.

### Tiny fixture catalog (no multi-GB weights)

Linux CI must not pull Whisper / Kokoro / SmolLM / RapidOCR. Install the
Lane-1 fixtures, then `ensure_model` / `resolve` stay offline (`file://`
bytes):

```python
def test_snip_pilot_vision_offline(tmp_path, monkeypatch):
    install_fixture_catalog(tmp_path, monkeypatch=monkeypatch)
    vision = ensure_fixture("vision", tmp_path)   # det + rec names, ~bytes
    detect = ensure_fixture("detect", tmp_path)
    classify = ensure_fixture("classify", tmp_path)
    # Persona / audience: same helper for stt / tts / llm.
```

If the app only needs a path (no fetch path), stub:

```python
from hexagon_kit import resolve

def test_persona_resolve_stub(tmp_path, monkeypatch):
    monkeypatch.setenv("HEXAGON_KIT_CACHE", str(tmp_path))
    path = stub_ensure_resolve(tmp_path, "stt")
    assert path == resolve("stt")
```

Use `classify` / `detect` slots for app-specific heads. Use builtin slot
names (`vision`, `llm`, …) when the app already calls `ensure_model("vision")`.
Both are tiny; both are offline.

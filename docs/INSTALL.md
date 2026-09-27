# Install (private source of truth)

This checkout is the **private SoT** for `snapdragon-npu-hexagon-kit`
(`import hexagon_kit`, CLI `hexagon`). The package is **not on PyPI**.
Do not install from the public GitHub mirror unless a human has promoted
that exact commit. Do not `pip install snapdragon-npu-hexagon-kit` from
the index — that name is reserved; it is not published.

API usage after install is in [APP_INTEGRATION.md](APP_INTEGRATION.md).
This page is only how to get the package onto `PYTHONPATH`.

---

## From this tree (editable)

```powershell
pip install -e .
pip install -e ".[dev]"    # pytest + build
```

`hexagon` is then on PATH. Equivalent: `python -m hexagon_kit`.
`hexagon --version` must print `0.3.0` for this candidate.

---

## As an app dependency (git)

Consumer apps (Persona, SnapDrago, `npu_pipeline`) should pin the
**private** repo, not the public name, until a human promote.

```powershell
# CPU-oriented extra. Linux CI / machines without Hexagon.
pip install "snapdragon-npu-hexagon-kit[ort] @ git+https://github.com/matthewhand/snapdragon-npu-hexagon-kit-private.git@v0.3.0"

# Hexagon QNN extra. Windows ARM64 Copilot+ with HTP drivers only.
pip install "snapdragon-npu-hexagon-kit[qnn] @ git+https://github.com/matthewhand/snapdragon-npu-hexagon-kit-private.git@v0.3.0"
```

Until a `v0.3.0` tag exists, pin a commit SHA on this private repo
instead of `@v0.3.0`. Read access to the private SoT is required.

`pyproject.toml` in the app:

```toml
# CPU path (default for Linux / no HTP). Not Hexagon.
# snapdragon-npu-hexagon-kit = { git = "https://github.com/matthewhand/snapdragon-npu-hexagon-kit-private.git", rev = "v0.3.0", extras = ["ort"] }

# Hexagon path (Windows ARM64 Copilot+ only).
# snapdragon-npu-hexagon-kit = { git = "https://github.com/matthewhand/snapdragon-npu-hexagon-kit-private.git", rev = "v0.3.0", extras = ["qnn"] }
```

Local sibling checkout:

```powershell
pip install -e "..\snapdragon-npu-hexagon-kit-private[ort]"
```

---

## Extras honesty — `[ort]` is CPU, `[qnn]` is Hexagon

These extras install **different wheels**. Installing one is not the other.

| Extra | Wheel | What ORT can list | Hexagon? |
|---|---|---|---|
| *(none)* | — | `CPUExecutionProvider` (kit fallback) | **No** |
| `[ort]` | `onnxruntime>=1.20` | **CPU-only** on most wheels (`CPUExecutionProvider`) | **No** |
| `[qnn]` | `onnxruntime_qnn` | `QNNExecutionProvider` on Windows ARM64 + HTP | **Yes**, if the EP is actually listed |
| `[hub]` | `qai_hub_models_cli` + `huggingface_hub` | (catalog only) | **No** |

Rules:

- `[ort]` never installs `onnxruntime_qnn`.
- `[qnn]` never equals `[ort]`. Do not write `onnxruntime` when you mean QNN.
- `pip install -e ".[ort]"` on Linux CI is expected. That runner is **not**
  a Hexagon box. `hexagon hw` will report `ep_kind: cpu`.
- `pip install -e ".[qnn]"` on a machine without HTP / `onnxruntime_qnn`
  does not create a silent Hexagon path.
- DirectML (`onnxruntime-directml`, `DmlExecutionProvider`) is a third
  listing — Adreno GPU, not QNN HTP. It is not an extra on this kit.

After install, call `probe_hardware()` and **never advertise Hexagon/QNN**
unless `QNNExecutionProvider` is listed, `ep_kind == "qnn"`, and
`qnn_package` is true. That contract is in
[APP_INTEGRATION.md](APP_INTEGRATION.md).

---

## Wheel / sdist (private artifact)

```powershell
pip install -e ".[dev]"
python -m build
pip install dist/snapdragon_npu_hexagon_kit-0.3.0-py3-none-any.whl
```

The sdist ships `docs/`, `CHANGELOG.md`, and `config.example.json`.
The wheel is the importable library only. Do not upload either file to
PyPI or the public mirror from automation.

---

## What this install is not

- Not a PyPI release.
- Not a public-mirror force-push.
- Not a guarantee of Hexagon. Hexagon requires `[qnn]` **and** a live
  `QNNExecutionProvider`.
- Lane-1 CI (T19) is in this tree: `import hexagon_kit.testing`. Do not
  invent a fake EP probe. See [APP_INTEGRATION.md](APP_INTEGRATION.md).

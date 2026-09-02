# Changelog

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

Not in this release: PyPI upload, Hexagon QNN wheel, Gemma/LLM/vision catalog entries, a cross-app NPU daemon.

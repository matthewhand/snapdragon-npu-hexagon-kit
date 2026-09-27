# Private SoT publish

This repository (`snapdragon-npu-hexagon-kit-private`) is the source of
truth. Public GitHub (`snapdragon-npu-hexagon-kit`) and PyPI are
**human-gated**. Automation in this repo must not write to those
destinations and must not `--force` push a public mirror.

T13 already runs pytest on Linux. This page is the remaining publish
checklist for a versioned candidate.

## Candidate invariants

1. `pyproject.toml` `[project].version` == `hexagon_kit.__version__` ==
   a `## X.Y.Z` heading in `CHANGELOG.md` (not only `## Unreleased`).
2. Consumer install notes live in [INSTALL.md](INSTALL.md). `[ort]` is
   CPU (`onnxruntime`); `[qnn]` is Hexagon (`onnxruntime_qnn`). Those
   extras must stay distinct in package metadata.
3. [APP_INTEGRATION.md](APP_INTEGRATION.md) (T13) stays the API contract.
   Do not rename `probe_hardware` / `ensure_model` / `preflight` /
   `resolve` or the honesty fields. T19 `hexagon_kit.testing` (Lane-1
   mock-EP + fixture catalog) stays importable and must not be renamed.
4. `.github/workflows/ci.yml` runs pytest (3.12/3.13), builds an
   sdist/wheel, inspects extras metadata, and smoke-imports the wheel.
   `ubuntu-latest` is not Hexagon; `npu` tests skip without QNN.
5. The sdist includes `docs/INSTALL.md`, `docs/APP_INTEGRATION.md`,
   `CHANGELOG.md`, and `config.example.json`.

## Do not

- Upload to PyPI (no token in this workflow; do not add one here).
- Push or force-push `matthewhand/snapdragon-npu-hexagon-kit` from CI
  or from a cloud agent unless a human explicitly promotes.
- Treat `[ort]` / CPU `onnxruntime` as Hexagon QNN in release notes.
- Fail Linux CI because `QNNExecutionProvider` is absent.

## Human promote (out of band)

Tag `v0.3.0` on this private repo after the candidate PR merges and CI
is green. Copying that tag to the public mirror is a separate, manual
step. T19 (`hexagon_kit.testing`) is already on this candidate.

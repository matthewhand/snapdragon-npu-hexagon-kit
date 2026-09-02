import tomllib
from pathlib import Path

import hexagon_kit


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_identity_matches_runtime():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert project["name"] == "snapdragon-npu-hexagon-kit"
    assert project["version"] == hexagon_kit.__version__
    assert project["requires-python"] == ">=3.12"
    assert project["scripts"]["hexagon"] == "hexagon_kit.cli:main"
    assert data["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]


def test_readme_and_license_ship_with_the_tree():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "import hexagon_kit" in readme
    assert "hexagon status" in readme
    assert "MIT License" in license_text
    assert hexagon_kit.__version__ in changelog
    assert "CHANGELOG.md" in manifest
    assert "config.example.json" in manifest

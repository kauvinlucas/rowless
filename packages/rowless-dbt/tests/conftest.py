"""Throwaway dbt projects, built around the committed artifact fixtures.

A dbt project is a directory shape as much as a manifest: `dbt_project.yml` at
the root, `target/manifest.json` beside it, and a file per model where
`original_file_path` says there is one. The fixtures commit only the artifacts,
and this builds the rest in a temporary directory, so no test depends on a
`models/` tree checked into the repository and nothing has to be excluded from
`rowless check` to keep it out of the way.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from rowless_dbt import source
from rowless_dbt.manifest import Manifest

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass(frozen=True)
class DbtProject:
    """A built project: where it is, and the manifest inside it."""

    root: Path
    manifest_path: Path

    def manifest(self) -> Manifest:
        return Manifest.load(str(self.manifest_path))

    def model_file(self, relative: str) -> Path:
        return self.root / relative


#: What `build_project` hands back: the builder, not a built project.
ProjectBuilder = Callable[..., DbtProject]


@pytest.fixture(autouse=True)
def _isolated_dbt_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's own dbt environment out of the tests.

    `find_manifest` reads `DBT_TARGET_PATH` and `DBT_PROJECT_DIR` before it
    walks the tree, which is the right order for a user and the wrong one for
    a test suite that would otherwise resolve against whatever project the
    machine happens to have.
    """
    for variable in ("DBT_TARGET_PATH", "DBT_PROJECT_DIR"):
        monkeypatch.delenv(variable, raising=False)
    source.clear_cache()


@pytest.fixture
def build_project(tmp_path: Path) -> ProjectBuilder:
    """Build a dbt project around one of the committed manifests."""

    def build(
        fixture: str = "manifest_v12.json",
        *,
        catalog: str | None = "catalog_v12.json",
        name: str = "shop",
        target_path: str = "target",
    ) -> DbtProject:
        root = tmp_path / name
        (root / target_path).mkdir(parents=True)
        lines = [f"name: {name}", "version: '1.0.0'", "profile: default"]
        if target_path != "target":
            lines.append(f"target-path: {target_path}")
        (root / "dbt_project.yml").write_text("\n".join(lines) + "\n", encoding="utf-8")

        manifest_path = root / target_path / "manifest.json"
        shutil.copyfile(FIXTURES / fixture, manifest_path)
        if catalog is not None:
            shutil.copyfile(FIXTURES / catalog, root / target_path / "catalog.json")
        _write_model_files(root, manifest_path)
        return DbtProject(root=root, manifest_path=manifest_path)

    return build


@pytest.fixture
def project(build_project: ProjectBuilder) -> DbtProject:
    """The v12 Postgres project, which most tests use as it comes."""
    return build_project()


def _write_model_files(root: Path, manifest_path: Path) -> None:
    """Create a file for every node that claims one, with its own code in it."""
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    for section in ("nodes", "sources"):
        for raw in payload.get(section, {}).values():
            relative = raw.get("original_file_path")
            if not relative:
                continue
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                continue
            path.write_text(str(raw.get("raw_code") or ""), encoding="utf-8")

"""Shared fixtures: a tiny repository and an in-process engine.

A module each test file imports from rather than a `conftest.py`. Two files
called `conftest` in one workspace are one module name to mypy, and
rowless-core already has the one at the top of that namespace - so the
fixtures live here under a name nothing else uses, and each test module names
the ones it wants on its import line. rowless-dbt solves the same problem the
other way, by making its test directory a package; that spelling is taken too,
since `tests.conftest` would then be shared with it.

The dry runs here go through `dry_run_callable` rather than the worker
subprocess, for the reason core's own `test_check` gives: a fake adapter
cannot be registered in a child process. The subprocess boundary is core's to
test; what these tests exercise is the server above it, which calls exactly
the same `rowless.check.run`.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from rowless_fake_lsp import FakeAdapter, install

#: The proposal's example, with the planted rename: the source calls the
#: column `amount_cents` and the pipeline still asks for `amount`.
PIPELINE = '''"""A pipeline that reads a column the source renamed."""

import rowless
from rowless_fake_lsp import FakeFrame

orders = rowless.source("lake://stg_orders")


@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="fake")
def daily_revenue(orders: FakeFrame) -> FakeFrame:
    return orders.select("order_id", "amount")
'''

#: The same pipeline with the column name corrected.
FIXED = PIPELINE.replace('"order_id", "amount"', '"order_id", "amount_cents"')


@pytest.fixture
def in_process(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeAdapter]:
    """Run dry runs in this process, mirroring what the worker does in a child."""
    from rowless.adapters import registry
    from rowless.runtime.dryrun import DryRunRequest, DryRunResult, dry_run_target

    original = registry._entry_points

    def in_process_engines_only(group: str) -> list[Any]:
        found: list[Any] = [] if group == "rowless.engines" else list(original(group))  # type: ignore[arg-type]
        return found

    monkeypatch.setattr(registry, "_entry_points", in_process_engines_only)
    registry.clear_cache()
    adapter = install()

    def _run(request: DryRunRequest) -> DryRunResult:
        for entry in request.sys_path:
            if entry not in sys.path:
                sys.path.insert(0, entry)
        # A worker gets a fresh interpreter per dry run; in-process, the module
        # from the previous check is still cached, and the point of a re-check
        # is that the file changed.
        sys.modules.pop(request.target.partition(":")[0], None)
        importlib.invalidate_caches()
        return dry_run_target(
            request.target,
            dict(request.inputs),
            engine=request.engine,
            options=request.options,
            uris=request.uris,
        )

    monkeypatch.setattr("rowless.check.dry_run", _run)
    yield adapter
    registry.clear_cache()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A repository with one schema file and one pipeline that reads it."""
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "stg_orders.json").write_text(
        '{"order_id": "int64", "status": "string", "amount_cents": "int64"}',
        encoding="utf-8",
    )
    (tmp_path / "rowless.toml").write_text(
        'roots = ["."]\n\n[sources]\n"lake://*" = "json://schemas/{name}.json"\n',
        encoding="utf-8",
    )
    (tmp_path / "revenue.py").write_text(PIPELINE, encoding="utf-8")
    return tmp_path


@pytest.fixture
def pipeline_path(project: Path) -> Path:
    """The pipeline module inside the fixture repository."""
    return project / "revenue.py"

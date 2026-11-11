"""Discovering dbt models, and what `rowless check` does with them today."""

from __future__ import annotations

from pathlib import Path

import pytest
from rowless.diagnostics.codes import Code
from rowless.diagnostics.models import Report
from rowless.ir.confidence import Confidence
from rowless_dbt.discovery import DbtModelDiscovery

from tests.conftest import DbtProject, ProjectBuilder

FINDER = DbtModelDiscovery()


def test_available_inside_a_dbt_project(
    project: DbtProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(project.root)
    assert FINDER.available() is True


def test_not_available_outside_one(project: DbtProject, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project.root.parent)
    assert FINDER.available() is False


def test_finds_one_pipeline_per_sql_model(project: DbtProject) -> None:
    found = {pipeline.target: pipeline for pipeline in FINDER.find([str(project.root)])}

    assert set(found) == {
        "model.shop.stg_orders",
        "model.shop.daily_revenue",
        "model.shop.finance_kpis",
    }


def test_a_discovered_model_carries_its_upstream_uris(project: DbtProject) -> None:
    """A rename upstream reaches this model through ordinary source resolution."""
    found = {pipeline.target: pipeline for pipeline in FINDER.find([str(project.root)])}

    assert found["model.shop.stg_orders"].inputs == (("orders", "dbt://source.shop.raw.orders"),)
    assert found["model.shop.daily_revenue"].inputs == (
        ("stg_orders", "dbt://model.shop.stg_orders"),
    )
    assert not found["model.shop.daily_revenue"].unbound


def test_a_discovered_model_is_a_sql_pipeline_pointing_at_its_file(project: DbtProject) -> None:
    (pipeline,) = [
        found for found in FINDER.find([str(project.root)]) if found.name.endswith("stg_orders")
    ]

    assert pipeline.engine == "sql"
    assert pipeline.framework == "dbt"
    assert pipeline.file is not None
    assert Path(pipeline.file).name == "stg_orders.sql"
    assert Path(pipeline.file).is_file()


def test_tags_carry_what_select_filters_want(project: DbtProject) -> None:
    found = {pipeline.target: pipeline for pipeline in FINDER.find([str(project.root)])}

    assert "contract" in found["model.shop.stg_orders"].tags
    assert "schema_source:contract" in found["model.shop.stg_orders"].tags
    assert "package:shop" in found["model.shop.stg_orders"].tags
    assert "contract" not in found["model.shop.finance_kpis"].tags
    assert "schema_source:none" in found["model.shop.finance_kpis"].tags


def test_python_models_are_not_yielded_yet(project: DbtProject) -> None:
    """dbt calls them `model(dbt, session)` and expects the framework to supply
    both arguments. The runtime binds inputs by parameter name and has no way
    to be handed objects, so a Python model is skipped rather than yielded as a
    pipeline nothing can call. See the module docstring."""
    targets = {pipeline.target for pipeline in FINDER.find([str(project.root)])}

    assert "model.shop.revenue_report" not in targets


def test_seeds_and_sources_are_not_pipelines(project: DbtProject) -> None:
    """They are inputs, resolved through `dbt://`, not code to check."""
    targets = {pipeline.target for pipeline in FINDER.find([str(project.root)])}

    assert not [target for target in targets if target.startswith(("seed.", "source."))]


def test_nothing_is_found_without_a_manifest(
    build_project: ProjectBuilder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A project that has never been compiled yields nothing, and does not raise."""
    project = build_project()
    project.manifest_path.unlink()
    monkeypatch.chdir(tmp_path)

    assert tuple(FINDER.find([str(project.root)])) == ()


def test_an_unsupported_manifest_does_not_break_discovery(
    build_project: ProjectBuilder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One unreadable artifact must not hide every other pipeline in the repo.

    The version is reported where it matters: by the source, when a pipeline
    actually needs a schema out of it.
    """
    project = build_project("manifest_unsupported.json", catalog=None, name="legacy")
    monkeypatch.chdir(project.root)

    assert tuple(FINDER.find([str(project.root)])) == ()


def test_models_are_discovered_in_dependency_order(project: DbtProject) -> None:
    ordered = [pipeline.target for pipeline in FINDER.find([str(project.root)])]

    assert ordered == [
        "model.shop.stg_orders",
        "model.shop.daily_revenue",
        "model.shop.finance_kpis",
    ]


@pytest.mark.parametrize(
    ("adapter", "dialect"),
    [("snowflake", "snowflake"), ("bigquery", "bigquery"), ("sqlserver", "tsql")],
)
def test_dialect_for(adapter: str, dialect: str) -> None:
    assert FINDER.dialect_for(adapter) == dialect


# --- Through `rowless check` -------------------------------------------------


def _check(root: Path) -> Report:
    """Run a check over a built project, as the CLI does."""
    from rowless.check import check
    from rowless.config import Config

    return check(Config.discover(str(root)))


def test_sql_models_report_that_the_runtime_cannot_run_one(
    project: DbtProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Today's honest outcome, and a tripwire for the day it changes.

    `rowless check` finds the models and resolves what it can of their inputs,
    and then stops: a model's target is its dbt node id, which is not an
    importable reference, and the runtime has no path for a pipeline that is a
    compiled query rather than a function. So each model is reported as
    unchecked - never as a pass, and no longer as a broken pipeline either.

    The engine half of this is done: `rowless-sql` resolves a compiled query
    against its upstream schemas with no warehouse, which is what a `dbt://`
    source uses it for (see `test_dbt_source.py`). What is left is the core
    gap in `rowless_dbt.discovery`'s docstring. When that lands, this test
    fails and the next one to write is the one that asserts a resolved SQL
    schema.
    """
    monkeypatch.chdir(project.root)
    (project.root / "rowless.toml").write_text('roots = ["."]\n', encoding="utf-8")

    report = _check(project.root)
    codes = {diagnostic.code for diagnostic in report.diagnostics}

    assert codes == {Code.UNRESOLVED_SOURCE}
    assert Code.PIPELINE_RAISED not in codes
    assert report.coverage.confidence is Confidence.UNKNOWN
    assert report.pipelines_checked == 3
    detail = " ".join(diagnostic.message or "" for diagnostic in report.diagnostics)
    assert "could not import" in detail


def test_a_python_pipeline_breaks_when_a_dbt_contract_renames_a_column(
    project: DbtProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proposal, end to end: one rename in SQL, reported in Python.

    The dbt contract calls the column `amount_cents`. The Polars pipeline still
    reads `amount`. Nothing here is dbt-aware except the `dbt://` URI, which is
    the whole point: the column graph crosses the language boundary because
    both halves resolve through the same source registry.
    """
    pytest.importorskip("polars", reason="the Python half of the seam needs an engine")

    monkeypatch.chdir(project.root)
    (project.root / "rowless.toml").write_text('roots = ["pipelines"]\n', encoding="utf-8")
    pipelines = project.root / "pipelines"
    pipelines.mkdir()
    (pipelines / "revenue.py").write_text(
        "\n".join(
            [
                '"""Revenue, reading a column the dbt contract renamed."""',
                "",
                "import polars as pl",
                "import rowless",
                "",
                'orders = rowless.source("dbt://model.shop.stg_orders")',
                "",
                "",
                "def daily_revenue(orders: pl.LazyFrame) -> pl.LazyFrame:",
                '    return orders.select("order_id", "amount")',
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = _check(project.root)
    findings = [
        diagnostic for diagnostic in report.diagnostics if diagnostic.code is Code.COLUMN_NOT_FOUND
    ]

    assert len(findings) == 1, [diagnostic.message for diagnostic in report.diagnostics]
    (finding,) = findings
    assert "amount" in finding.message
    assert finding.suggestion is not None and "amount_cents" in finding.suggestion
    assert finding.origin is not None
    assert finding.origin.file is not None
    assert Path(finding.origin.file).name == "revenue.py"

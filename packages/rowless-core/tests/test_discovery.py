"""Pipeline discovery: what `rowless check` with no arguments has to find."""

from __future__ import annotations

from pathlib import Path

from rowless.discovery.base import discover, import_root_for, module_name_for
from rowless.discovery.plain import PlainFunctionDiscovery

MODULE = '''
"""A pipeline module shaped like the proposal's example."""

import polars as pl
import rowless

orders = rowless.source("lake://stg_orders")


def daily_revenue(orders: pl.LazyFrame) -> pl.LazyFrame:
    return orders.select("amount")


def helper(value: int) -> int:
    return value + 1


@rowless.pipeline(inputs={"events": "lake://stg_events"}, engine="duckdb")
def marked(events):
    return events
'''


def _find(root: Path) -> dict[str, object]:
    """Discovered pipelines by function name."""
    return {pipeline.name: pipeline for pipeline in PlainFunctionDiscovery().find([str(root)])}


def test_finds_annotated_functions_and_binds_module_sources(tmp_path: Path) -> None:
    """The example works with no configuration: the module names its inputs."""
    (tmp_path / "revenue.py").write_text(MODULE, encoding="utf-8")

    found = _find(tmp_path)

    revenue = found["daily_revenue"]
    assert revenue.engine == "polars"  # type: ignore[attr-defined]
    assert revenue.inputs == (("orders", "lake://stg_orders"),)  # type: ignore[attr-defined]
    assert revenue.target.endswith(":daily_revenue")  # type: ignore[attr-defined]
    assert revenue.line == 10  # type: ignore[attr-defined]


def test_ignores_functions_that_are_not_pipelines(tmp_path: Path) -> None:
    """A helper taking an int is not an entry point, and must not be reported."""
    (tmp_path / "revenue.py").write_text(MODULE, encoding="utf-8")

    assert "helper" not in _find(tmp_path)


def test_honours_the_explicit_marker(tmp_path: Path) -> None:
    """The escape hatch for code whose annotations name no frame type."""
    (tmp_path / "revenue.py").write_text(MODULE, encoding="utf-8")

    marked = _find(tmp_path)["marked"]

    assert marked.engine == "duckdb"  # type: ignore[attr-defined]
    assert marked.inputs == (("events", "lake://stg_events"),)  # type: ignore[attr-defined]
    assert "decorated" in marked.tags  # type: ignore[attr-defined]


def test_resolves_aliases_rather_than_guessing(tmp_path: Path) -> None:
    """A bare `DataFrame` means pandas or Polars depending on the import."""
    (tmp_path / "a.py").write_text(
        "from polars import DataFrame\n\n\ndef one(df: DataFrame) -> DataFrame:\n    return df\n",
        encoding="utf-8",
    )
    (tmp_path / "b.py").write_text(
        "from pandas import DataFrame\n\n\ndef two(df: DataFrame) -> DataFrame:\n    return df\n",
        encoding="utf-8",
    )

    found = _find(tmp_path)

    assert found["one"].engine == "polars"  # type: ignore[attr-defined]
    assert found["two"].engine == "pandas"  # type: ignore[attr-defined]


def test_unparseable_module_does_not_stop_the_walk(tmp_path: Path) -> None:
    """A repo with one broken file still yields a pipeline list."""
    (tmp_path / "broken.py").write_text("def oops(:\n", encoding="utf-8")
    (tmp_path / "revenue.py").write_text(MODULE, encoding="utf-8")

    assert "daily_revenue" in _find(tmp_path)


def test_unbound_inputs_are_visible(tmp_path: Path) -> None:
    """A parameter rowless cannot bind is reported, not silently dropped."""
    (tmp_path / "revenue.py").write_text(
        "import polars as pl\n\n\ndef totals(orders: pl.LazyFrame) -> pl.LazyFrame:\n"
        "    return orders\n",
        encoding="utf-8",
    )

    totals = _find(tmp_path)["totals"]

    assert totals.unbound == ("orders",)  # type: ignore[attr-defined]


def test_discover_merges_finders_and_dedupes(tmp_path: Path) -> None:
    """Every available finder runs; the framework-aware record wins."""
    (tmp_path / "revenue.py").write_text(MODULE, encoding="utf-8")

    found = discover([str(tmp_path)])

    names = [pipeline.name for pipeline in found]
    assert names.count("daily_revenue") == 1
    assert found[0].file is not None


def test_module_names_follow_packages(tmp_path: Path) -> None:
    """The worker has to be able to import what discovery found."""
    package = tmp_path / "pipelines"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    module = package / "revenue.py"
    module.write_text(MODULE, encoding="utf-8")

    assert module_name_for(module) == "pipelines.revenue"
    assert import_root_for(module) == tmp_path.resolve()

    loose = tmp_path / "script.py"
    loose.write_text(MODULE, encoding="utf-8")
    assert module_name_for(loose) == "script"

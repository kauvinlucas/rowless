"""Static analysis: the half a dry run cannot see, and nothing more.

These tests are as much about what is *not* reported. The AST pass exists to
lower confidence in a dry run, and a pass that fires on correct Polars code
would make every real pipeline look unverified.
"""

from __future__ import annotations

from pathlib import Path

from rowless.analysis import analyse_module, branch_coverage, dynamic_column_names

CLEAN = """
import polars as pl


def daily_revenue(orders: pl.LazyFrame) -> pl.LazyFrame:
    return (
        orders.filter(pl.col("status") == "paid")
        .group_by(pl.col("created_at").dt.date().alias("day"))
        .agg(pl.col("amount").sum())
    )
"""

SUSPECT = """
import polars as pl


def totals(df: pl.DataFrame, currency: str) -> pl.DataFrame:
    if df.is_empty():
        return df
    if len(df) == 0:
        return df
    ceiling = df["amount"].max()
    column = f"amount_{currency}"
    return df.select(pl.col(column)).with_columns(pl.col("amount").sum() / ceiling)
"""


def _write(tmp_path: Path, name: str, source: str) -> str:
    """Write a module and return its path."""
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return str(path)


def test_correct_lazy_code_is_not_flagged(tmp_path: Path) -> None:
    """`pl.col("amount").sum()` resolves without data, so it is not a finding."""
    assert analyse_module(_write(tmp_path, "clean.py", CLEAN)) == ()


def test_row_dependent_calls_are_flagged(tmp_path: Path) -> None:
    """An eager accessor on a frame takes a branch production would not."""
    findings = analyse_module(_write(tmp_path, "suspect.py", SUSPECT))

    row_dependent = [f for f in findings if f.kind == "row_dependent_call"]
    details = " ".join(f.detail for f in row_dependent)
    assert ".is_empty()" in details
    assert "row count" in details, "len(df) == 0 is the same bug in another spelling"
    assert ".max()" in details, "an eager aggregation on a frame is row-dependent"


def test_dynamic_column_names_are_flagged(tmp_path: Path) -> None:
    """A computed column name is a lineage gap, and saying so is the honest answer."""
    findings = dynamic_column_names(_write(tmp_path, "suspect.py", SUSPECT))

    assert findings, "an f-string column name must be reported"
    assert all(f.kind == "dynamic_column" for f in findings)
    assert len({(f.line, f.kind) for f in findings}) == len(findings), "no duplicates"


def test_unparseable_module_is_reported_once(tmp_path: Path) -> None:
    """A module that cannot be parsed can never be dry-run either."""
    findings = analyse_module(_write(tmp_path, "broken.py", "def oops(:\n"))

    assert [f.kind for f in findings] == ["import_cycle"]


def test_branch_coverage_names_the_paths_that_did_not_run(tmp_path: Path) -> None:
    """A pipeline whose branches all ran on zero rows is better verified."""
    path = _write(tmp_path, "suspect.py", SUSPECT)

    nothing_ran = branch_coverage(path, frozenset())
    assert [f.kind for f in nothing_ran] == ["unreached_branch"] * len(nothing_ran)
    assert len(nothing_ran) >= 2

    everything_ran = branch_coverage(path, frozenset(range(1, 40)))
    assert everything_ran == ()

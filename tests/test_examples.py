"""End-to-end: run rowless over `examples/` and assert on what it finds.

The dogfooding suite, and the thing that keeps the examples honest. Each example
has a planted finding; rowless must report exactly that finding, with the right
file and line, and nothing else.

"And nothing else" is the half that matters. A checker that reports the planted
error alongside four spurious ones is unusable in CI, and false positives are
how linters get disabled.

The Polars example runs: `rowless-polars` is implemented. The SQL and mixed
examples are skipped until the adapters they need exist, individually rather
than by skipping the module, so that the one example that can be checked is
actually checked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rowless.check import check
from rowless.config import Config
from rowless.diagnostics.codes import Code
from rowless.diagnostics.models import Level

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

#: The planted rename in `examples/polars_demo/revenue.py`: the pipeline reads
#: `amount`, and the committed source schema calls it `amount_cents`.
POLARS_DEMO = EXAMPLES / "polars_demo"


def _check(example: Path):
    """Run `rowless check` over one example directory, as the CLI does."""
    return check(Config.discover(str(example)))


def _line_reading_amount() -> int:
    """The 1-based line of the example that reads the renamed column."""
    source = (POLARS_DEMO / "revenue.py").read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(source, start=1):
        if 'pl.col("amount")' in line:
            return number
    raise AssertionError("examples/polars_demo/revenue.py no longer reads 'amount'")


@pytest.fixture(scope="module")
def polars_report():
    """The report for the Polars example, built once for every assertion on it."""
    pytest.importorskip("polars", reason="the Polars example needs the Polars adapter")
    return _check(POLARS_DEMO)


def test_polars_demo_reports_renamed_column(polars_report) -> None:
    """polars_demo yields exactly one ColumnNotFound, at the documented line."""
    findings = [
        diagnostic
        for diagnostic in polars_report.diagnostics
        if diagnostic.code is Code.COLUMN_NOT_FOUND
    ]
    assert len(findings) == 1, [diagnostic.message for diagnostic in findings]

    (finding,) = findings
    assert finding.origin is not None
    assert finding.origin.file is not None
    assert Path(finding.origin.file).name == "revenue.py"
    # The line that reads the renamed column, which is the line to edit. Read
    # out of the source rather than written here, so editing the example moves
    # the expectation with it instead of silently loosening this assertion.
    assert finding.origin.line == _line_reading_amount()
    assert "amount" in finding.message
    assert finding.suggestion is not None and "amount_cents" in finding.suggestion


def test_polars_demo_reports_no_spurious_findings(polars_report) -> None:
    """The planted rename is the only thing wrong with the example."""
    unexpected = [
        f"{diagnostic.code.value} {diagnostic.message}"
        for diagnostic in polars_report.diagnostics
        if diagnostic.code is not Code.COLUMN_NOT_FOUND
        and diagnostic.level in {Level.ERROR, Level.UNVERIFIED}
    ]
    assert unexpected == []
    assert polars_report.pipelines_checked == 1


def test_polars_demo_is_fast_enough_to_be_a_linter(polars_report) -> None:
    """Checking the example stays under a budget.

    Deliberately a test and not a benchmark: "at linter speed" is a product
    promise, and a regression should fail CI rather than show up on a chart
    nobody reads. The budget is per pipeline and generous enough to survive a
    cold worker start on a shared CI runner; it is there to catch a dry run
    that started *executing*, which costs seconds rather than milliseconds.
    """
    assert polars_report.duration_ms < 5_000


@pytest.mark.skip(reason="examples/sql_demo is not written yet; the SQL adapter is implemented")
def test_sql_demo_resolves_without_database() -> None:
    """sql_demo resolves with no engine running and no credentials.

    `rowless-sql` is implemented and passes its conformance suite, so the
    engine half of this example works today. What is missing is a finder for
    standalone `.sql` files: nothing in `rowless.discovery` claims one, so the
    models the example's README describes would not be discovered.
    """


@pytest.mark.skip(reason="needs a runtime path for a target that is not importable")
def test_mixed_demo_diff_lists_both_languages() -> None:
    """mixed_demo diff names both the Python function and the SQL mart.

    Waiting on core rather than on an adapter: a dbt model's target is its node
    id, and the runtime can only run an importable function. The model's
    schema already resolves through a `dbt://` source.
    """

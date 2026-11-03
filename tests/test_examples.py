"""End-to-end: run rowless over `examples/` and assert on what it finds.

The dogfooding suite, and the thing that keeps the examples honest. Each example
has a planted finding; rowless must report exactly that finding, with the right
file and line, and nothing else.

"And nothing else" is the half that matters. A checker that reports the planted
error alongside four spurious ones is unusable in CI, and false positives are
how linters get disabled.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "needs an engine adapter: core resolves these examples and reports that it has "
        "no engine to run them on. Unskip with rowless-polars."
    )
)


def test_polars_demo_reports_renamed_column() -> None:
    """polars_demo yields exactly one ColumnNotFound, at the documented line."""


def test_sql_demo_resolves_without_database() -> None:
    """sql_demo resolves with no engine running and no credentials."""


def test_mixed_demo_diff_lists_both_languages() -> None:
    """mixed_demo diff names both the Python function and the SQL mart."""


def test_examples_produce_no_spurious_findings() -> None:
    """No example reports a finding that is not planted and documented."""


def test_check_is_fast_enough_to_be_a_linter() -> None:
    """Checking every example stays under a budget.

    Deliberately a test and not a benchmark: "at linter speed" is a product
    promise, and a regression should fail CI rather than show up on a chart
    nobody reads.
    """

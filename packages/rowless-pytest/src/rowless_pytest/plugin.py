"""The pytest plugin: schema contract tests that run in milliseconds.

The adoption path that asks the least of a team. No CI configuration, no new
command in the build, just a test that fails when somebody upstream renames a
column:

    def test_revenue_schema(rowless_check):
        rowless_check(daily_revenue, {"day": "date", "amount": "int64"})

Because a dry run moves no data, these tests cost about as much as a unit test
and can cover every pipeline in a repo. That is the argument for putting them
in the existing test suite rather than asking for a new pipeline stage.

The one rule worth being strict about: an unverified result fails. A contract
test that quietly stops testing because rowless lost confidence is worse than
no test at all, so `rowless_check` raises on `Confidence.UNKNOWN` unless the
test opts in with `allow_unverified=True`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest

# The session hooks below are no-ops rather than NotImplementedError, because
# pytest calls them on every invocation in any environment where this package is
# installed. A raising stub here would break unrelated test runs. The fixtures
# below do raise: they only run when a test asks for them.


def pytest_configure(config: pytest.Config) -> None:
    """Register the `rowless` marker."""
    # TODO(pytest): config.addinivalue_line("markers", "rowless: ...")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add `--rowless-min-coverage` and `--rowless-engine`."""
    # TODO(pytest): add the option group.


@pytest.fixture
def rowless_check() -> Callable[..., None]:
    """Assert a pipeline resolves to an expected schema.

    Fails with a schema diff, and fails on an unverified result unless
    `allow_unverified=True` is passed.
    """
    raise NotImplementedError("TODO(pytest)")


@pytest.fixture
def rowless_frame() -> Callable[[Mapping[str, str]], Any]:
    """Build a rowless frame from a dtype mapping, for the configured engine.

    Useful on its own: a test can pass one into a function and assert on the
    schema it gets back, without going through the dry-run machinery.
    """
    raise NotImplementedError("TODO(pytest)")


@pytest.fixture
def rowless_source() -> Callable[[str], Any]:
    """Resolve a source URI to a rowless frame, with the source cache in play."""
    raise NotImplementedError("TODO(pytest)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip rowless tests when the required engine is not installed.

    Skipping, not failing: a contributor with only pandas installed should be
    able to run the suite without the Polars tests erroring.
    """
    # TODO(pytest): deselect items whose required engine is unavailable.

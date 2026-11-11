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

What this module holds is the pytest surface and nothing else - the options,
the marker, the fixtures, the summary line and the exit code. The work is in
`rowless_pytest.session` (config, cache, tally), `rowless_pytest.checks` (the
assertion) and `rowless_pytest.engines` (which adapter, and whether it is
installed).

    fixtures    rowless_check   assert a pipeline resolves to a schema
                rowless_frame   a frame with a schema and no rows
                rowless_source  a frame, or a schema, resolved from a source URI

    marker      @pytest.mark.rowless(engine="polars")

    options     --rowless-engine NAME
                --rowless-min-coverage RATIO

Everything reads `rowless.toml` where it has an opinion already: the
`[sources]` rewrites, the per-engine options, `min_coverage`. A test that
resolved a URI differently from `rowless check` would be two tools wearing one
name.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from rowless_pytest import engines
from rowless_pytest.checks import SchemaCheck
from rowless_pytest.frames import FrameBuilder, SourceResolver
from rowless_pytest.session import RowlessSession, SessionHolder

#: Where the lazily built session state hangs off `pytest.Config`.
HOLDER: pytest.StashKey[SessionHolder] = pytest.StashKey()


# --- pytest hooks ------------------------------------------------------------
#
# The session hooks are written so that a suite with no rowless tests in it
# does nothing at all: this package is installed in an environment, so pytest
# loads it for every suite in that environment. `pytest_configure` only parks
# a holder, and the holder builds nothing until a fixture asks.


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add `--rowless-engine` and `--rowless-min-coverage`."""
    group = parser.getgroup("rowless", "rowless: schema contract tests")
    group.addoption(
        "--rowless-engine",
        dest="rowless_engine",
        default=None,
        metavar="NAME",
        help=(
            "Engine adapter for tests that do not name one (polars, pandas, duckdb, ...). "
            "A test's own @pytest.mark.rowless(engine=...) still wins."
        ),
    )
    group.addoption(
        "--rowless-min-coverage",
        dest="rowless_min_coverage",
        default=None,
        type=_ratio,
        metavar="RATIO",
        help=(
            "Fail the run when typed schema coverage across the rowless contract tests "
            "falls below this ratio (0 to 1). Defaults to min_coverage in rowless.toml."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the `rowless` marker and park the session holder."""
    config.addinivalue_line("markers", engines.MARKER_HELP)
    _holder(config)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip rowless tests when the required engine is not installed.

    Skipping, not failing: a contributor with only pandas installed should be
    able to run the suite without the Polars tests erroring. A skip with the
    registry's own reason on it also says which extra to install, which a
    deselect would not.

    A marker that cannot mean anything - a misspelled keyword - is a usage
    error instead. The failure mode it otherwise has is the bad one: the test
    silently stops being gated and starts erroring on the machine that lacks
    the engine.
    """
    availability: dict[str, str | None] = {}
    for item in items:
        problem = engines.marker_problem(item)
        if problem is not None:
            raise pytest.UsageError(f"{item.nodeid}: {problem}")
        name = engines.declared_engine(item)
        if name is None:
            continue
        if name not in availability:
            availability[name] = engines.unavailable(name)
        reason = availability[name]
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=f"rowless: {reason}"))


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    """Report what the contract tests actually verified.

    A pass count on its own overstates it: forty green contract tests over a
    schema rowless typed half of is a different state of the world from forty
    over one it typed all of, and the only way anybody finds that out is if
    the number is printed next to the passes.
    """
    session = _holder(config).active
    if session is None or not session.checks:
        return
    terminalreporter.write_sep("-", "rowless")
    terminalreporter.write_line(session.summary())
    for target in session.unverified:
        terminalreporter.write_line(f"  unverified, allowed by the test: {target}")
    shortfall = session.shortfall()
    if shortfall is not None:
        terminalreporter.write_line(f"FAILED {shortfall}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail the run when typed coverage is below the floor.

    Only when nothing else already failed it: a coverage shortfall must not
    overwrite an interrupt or a usage error, which are the two exit codes
    somebody is actually debugging when they see them.
    """
    rowless_session = _holder(session.config).active
    if rowless_session is None or exitstatus != pytest.ExitCode.OK:
        return
    if rowless_session.shortfall() is not None:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_unconfigure(config: pytest.Config) -> None:
    """Drop the scratch directory any materialized frames were built against."""
    holder = config.stash.get(HOLDER, None)
    if holder is not None:
        holder.close()


# --- fixtures ----------------------------------------------------------------


@pytest.fixture
def rowless_check(request: pytest.FixtureRequest) -> SchemaCheck:
    """Assert a pipeline resolves to an expected schema.

        def test_revenue_schema(rowless_check):
            rowless_check(daily_revenue, {"day": "date", "amount": "int64"})

    Fails with a schema diff, and fails on an unverified result unless
    `allow_unverified=True` is passed. Returns the `DryRunResult`, so a test
    that wants more than the schema - lineage, the writes the pipeline
    attempted, how long it took - has it.

    See `rowless_pytest.checks.SchemaCheck` for the full signature.
    """
    return SchemaCheck(
        session=_session(request),
        engine=engines.declared_engine(request.node),
    )


@pytest.fixture
def rowless_frame(request: pytest.FixtureRequest) -> FrameBuilder:
    """Build a rowless frame from a dtype mapping, for the configured engine.

    Useful on its own: a test can pass one into a function and assert on the
    schema it gets back, without going through the dry-run machinery.

        def test_cast(rowless_frame):
            out = normalise(rowless_frame({"amount": "int64"}))
            assert rowless.schema_of(out)["amount"].dtype == float_(64)

    The frame is a real engine object - a real `pl.LazyFrame`, a real
    `pd.DataFrame` - with every column and no rows. Nothing is recorded in the
    session tally: a frame on its own is not a result, so there is nothing to
    be confident or unconfident about. Row-dependent code called with one of
    these is not reported either, which is the reason to prefer
    `rowless_check` for anything shaped like a pipeline.
    """
    return FrameBuilder(
        session=_session(request),
        engine=engines.declared_engine(request.node),
    )


@pytest.fixture
def rowless_source(request: pytest.FixtureRequest) -> SourceResolver:
    """Resolve a source URI to a rowless frame, with the source cache in play.

        def test_contract(rowless_source):
            orders = rowless_source("lake://stg_orders")
            assert rowless.schema_of(orders)["amount"].dtype == int_(64)

    The URI goes through the `[sources]` rewrites and the cache under
    `.rowless/`, so the schema is the one `rowless check` would use and a
    suite full of these reads the catalog once.

    `rowless_source.schema(uri)` returns the schema without materializing
    anything, which is the right call for a test about an upstream table
    rather than about a pipeline: it needs no engine installed at all.
    """
    return SourceResolver(
        session=_session(request),
        engine=engines.declared_engine(request.node),
    )


# --- internals ---------------------------------------------------------------


def _session(request: pytest.FixtureRequest) -> RowlessSession:
    """The session state, built on this first request if it does not exist."""
    return _holder(request.config).get()


def _holder(config: pytest.Config) -> SessionHolder:
    """The holder on this config, created if `pytest_configure` has not run.

    It has always run under pytest itself. The get-or-create is for rowless's
    own tests and for tools that drive pytest's API directly, where a plugin
    object can outlive one configuration.
    """
    holder = config.stash.get(HOLDER, None)
    if holder is None:
        holder = SessionHolder(config)
        config.stash[HOLDER] = holder
    return holder


def _ratio(text: str) -> float:
    """Parse a 0-to-1 coverage ratio, the spelling `rowless.toml` uses."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number between 0 and 1") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"coverage floor must be between 0 and 1, got {value}")
    return value

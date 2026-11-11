"""Plugin tests, using pytest's own pytester fixture.

Each case writes a small suite, runs pytest on it and asserts on the outcome
and the output, which is the only honest way to test a plugin: what matters
about `rowless_check` is not that it raises, it is that a test fails, with a
message somebody can act on, and that the run's exit code says so.

The runs are in-process, which is what makes the fake adapter work: a fake
cannot be registered through an entry point, and an in-process run shares the
registry `plugin_fixtures` installed it in. The subprocess boundary belongs to
core's worker tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from rowless_fake_pytest import ENGINE, UNAVAILABLE_ENGINE, FakeAdapter

pytest_plugins = ["pytester", "plugin_fixtures"]

#: The schema the fixture repository's source resolves to.
ORDERS_JSON = '{"order_id": "int64", "status": "string", "amount": "int64"}'

#: The same table after somebody upstream renamed the column the pipelines
#: read. The planted finding from the proposal.
RENAMED_JSON = '{"order_id": "int64", "status": "string", "amount_cents": "int64"}'

#: A pipeline whose inputs are declared once, on the function, and resolved
#: through the `[sources]` rewrite in the fixture repository's config.
FROM_SOURCE = """
import rowless


@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="fake")
def daily_revenue(orders):
    return orders.select("order_id", "amount")
"""

#: The same pipeline with its inputs passed by the test instead, and no
#: engine named anywhere.
INLINE_INPUTS = """
ORDERS = {"order_id": "int64", "status": "string", "amount": "int64"}


def daily_revenue(orders):
    return orders.select("order_id", "amount")
"""

#: A conftest that registers a second usable engine, so a test can be run in
#: an environment where rowless must be told which adapter to use.
SECOND_ENGINE = """
from rowless.adapters.registry import register
from rowless_fake_pytest import FakeAdapter


class SecondFake(FakeAdapter):
    name = "fake2"


register("rowless.engines", "fake2", SecondFake())
"""


@pytest.fixture
def project(pytester: pytest.Pytester, fake_engine: FakeAdapter) -> pytest.Pytester:
    """A repository with a schema source and config, and no pipelines yet.

    Shaped like the one in the proposal: the schema lives in a file, config
    rewrites `lake://` onto it, and the pipelines are whatever the case writes.
    """
    _ = fake_engine
    pytester.makefile(
        ".toml",
        rowless='roots = ["."]\n\n[sources]\n"lake://*" = "json://schemas/{name}.json"\n',
    )
    (pytester.path / "schemas").mkdir()
    (pytester.path / "schemas" / "stg_orders.json").write_text(ORDERS_JSON, encoding="utf-8")
    return pytester


def source_schema(project: pytest.Pytester, payload: str) -> None:
    """Rewrite what the fixture repository's source resolves to."""
    (project.path / "schemas" / "stg_orders.json").write_text(payload, encoding="utf-8")


# --- the assertion -----------------------------------------------------------


def test_check_passes_on_matching_schema(project: pytest.Pytester) -> None:
    """A pipeline matching its contract passes."""
    project.makepyfile(
        FROM_SOURCE
        + """
def test_contract(rowless_check):
    rowless_check(daily_revenue, {"order_id": "int64", "amount": "int64"})
"""
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1)


def test_check_fails_with_schema_diff(project: pytest.Pytester) -> None:
    """A mismatch fails, and the diff names the offending column."""
    project.makepyfile(
        FROM_SOURCE
        + """
def test_contract(rowless_check):
    rowless_check(daily_revenue, {"order_id": "int64", "amount": "string"})
"""
    )
    result = project.runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "*does not match the expected schema*",
            "*amount: expected string, got int64*",
            "*resolved: order_id int64, amount int64*",
        ]
    )


def test_exactness_is_the_default_and_can_be_turned_off(project: pytest.Pytester) -> None:
    """An unnamed output column fails, until the test says it does not care."""
    project.makepyfile(
        FROM_SOURCE
        + """
def test_strict(rowless_check):
    rowless_check(daily_revenue, {"order_id": "int64"})


def test_partial(rowless_check):
    rowless_check(daily_revenue, {"order_id": "int64"}, exact=False)
"""
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(
        [
            "*unexpected column(s): amount*",
            "*pass exact=False to check only the columns this test names*",
        ]
    )


def test_a_renamed_upstream_column_fails_the_contract(project: pytest.Pytester) -> None:
    """The headline case: the source renamed it, so nothing resolves.

    Not a schema mismatch - the pipeline cannot run at all - so the failure
    has to say that, and carry the finding that explains it.
    """
    source_schema(project, RENAMED_JSON)
    project.makepyfile(
        FROM_SOURCE
        + """
def test_contract(rowless_check):
    rowless_check(daily_revenue, {"order_id": "int64", "amount": "int64"})
"""
    )
    result = project.runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "*rowless could not resolve*daily_revenue, so the contract was not tested*",
            "*RWL101*amount*",
        ]
    )


def test_a_rename_inside_the_pipeline_is_reported_as_one(project: pytest.Pytester) -> None:
    """A column that came back under another name is paired, not listed twice."""
    project.makepyfile(
        """
import rowless


@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="fake")
def renamed(orders):
    return orders.select("order_id", "amount").rename({"amount": "amount_cents"})


def test_contract(rowless_check):
    rowless_check(renamed, {"order_id": "int64", "amount": "int64"})
"""
    )
    result = project.runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*amount -> amount_cents: the resolved schema has it*"])


def test_an_any_column_is_not_evidence(project: pytest.Pytester) -> None:
    """A column rowless could not type fails, even against a matching dtype.

    The dtype algebra lets ANY match anything, which is right for inference
    and wrong for a test: an assertion that passes because rowless knows
    nothing is the failure mode this plugin is built against.
    """
    project.makepyfile(
        """
import rowless
from rowless.ir.dtypes import any_


@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="fake")
def guessed(orders):
    return orders.select("order_id").derive("guess", any_())


def test_contract(rowless_check):
    rowless_check(guessed, {"order_id": "int64", "guess": "int64"})
"""
    )
    result = project.runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*guess: rowless resolved any, which is not evidence of int64*"])


def test_inline_inputs_need_no_source(project: pytest.Pytester) -> None:
    """Inputs passed by the test are used as they are, with no source involved."""
    project.makepyfile(
        INLINE_INPUTS
        + """
def test_contract(rowless_check):
    rowless_check(
        daily_revenue,
        {"order_id": "int64", "amount": "int64"},
        inputs={"orders": ORDERS},
    )
"""
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1)
    assert not (project.path / ".rowless").exists()


def test_undeclared_inputs_say_what_to_do(project: pytest.Pytester) -> None:
    """A pipeline with no declared inputs fails with the two ways to fix it."""
    project.makepyfile(
        INLINE_INPUTS
        + """
def test_contract(rowless_check):
    rowless_check(daily_revenue, {"order_id": "int64", "amount": "int64"})
"""
    )
    result = project.runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*has no declared inputs*@rowless.pipeline(inputs=*"])


def test_the_result_comes_back(project: pytest.Pytester) -> None:
    """The `DryRunResult` is returned, so a test can assert on more than dtypes."""
    project.makepyfile(
        FROM_SOURCE
        + """
from rowless import Confidence


def test_contract(rowless_check):
    result = rowless_check(daily_revenue, {"order_id": "int64", "amount": "int64"})
    assert result.coverage.confidence is Confidence.EXACT
    assert result.output.schema["amount"].dtype.is_known
    assert result.side_effects == ()
"""
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1)


# --- unverified results ------------------------------------------------------


def test_unverified_result_fails_by_default(project: pytest.Pytester) -> None:
    """Row-dependent control flow fails the test instead of passing silently."""
    project.makepyfile(
        """
import rowless


@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="fake")
def row_dependent(orders):
    return orders.select("order_id") if not orders.is_empty() else orders


def test_contract(rowless_check):
    rowless_check(row_dependent, {"order_id": "int64"})
"""
    )
    result = project.runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "*rowless did not verify*row_dependent, so this test proves nothing*",
            "*reason:*is_empty*",
            "*pass allow_unverified=True*",
        ]
    )


def test_unverified_result_passes_when_the_test_opts_in(project: pytest.Pytester) -> None:
    """`allow_unverified=True` accepts it, and the summary counts it."""
    project.makepyfile(
        """
import rowless


@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="fake")
def row_dependent(orders):
    return orders.select("order_id") if not orders.is_empty() else orders


def test_contract(rowless_check):
    rowless_check(row_dependent, {"order_id": "int64"}, allow_unverified=True)
"""
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(
        [
            "*1 unverified result(s) allowed*",
            "*unverified, allowed by the test:*row_dependent*",
        ]
    )


def test_config_cannot_turn_the_rule_off(project: pytest.Pytester) -> None:
    """`fail_on_unverified = false` is a gating policy, not a licence here.

    It decides which findings fail a `rowless check` run. A contract test is a
    narrower question, and a line in a config file that silently stopped a
    suite of them from asserting is the failure mode this plugin exists to
    avoid - so the only way through is `allow_unverified=True`, in the test.
    """
    project.makefile(
        ".toml",
        rowless=(
            'roots = ["."]\nfail_on_unverified = false\n\n'
            '[sources]\n"lake://*" = "json://schemas/{name}.json"\n'
        ),
    )
    project.makepyfile(
        """
import rowless


@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="fake")
def row_dependent(orders):
    return orders.select("order_id") if not orders.is_empty() else orders


def test_contract(rowless_check):
    rowless_check(row_dependent, {"order_id": "int64"})
"""
    )
    result = project.runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*this test proves nothing*"])


# --- the fixtures that build frames ------------------------------------------


def test_frame_fixture_builds_a_frame(project: pytest.Pytester) -> None:
    """`rowless_frame` hands back a real engine object with no rows."""
    project.makepyfile(
        f'''
import rowless


def test_frame(rowless_frame):
    frame = rowless_frame({{"order_id": "int64"}}, name="orders", engine="{ENGINE}")
    schema = rowless.schema_of(frame)
    assert list(schema) == ["order_id"]
    assert schema.name == "orders"
'''
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1)


def test_source_fixture_resolves_through_config(project: pytest.Pytester) -> None:
    """`rowless_source` applies the `[sources]` rewrite and caches the schema."""
    project.makepyfile(
        """
import rowless


def test_source(rowless_source):
    orders = rowless_source("lake://stg_orders")
    assert list(rowless.schema_of(orders)) == ["order_id", "status", "amount"]
"""
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1)
    assert list((project.path / ".rowless" / "sources").glob("*.json"))


def test_a_source_schema_needs_no_engine(
    pytester: pytest.Pytester,
    no_engines: None,
) -> None:
    """`rowless_source.schema` answers without materializing anything.

    A test about an upstream table - is the column this repository reads still
    there - is not a question about an engine, and should not need one
    installed to ask.
    """
    _ = no_engines
    pytester.makefile(
        ".toml",
        rowless='roots = ["."]\n\n[sources]\n"lake://*" = "json://schemas/{name}.json"\n',
    )
    (pytester.path / "schemas").mkdir()
    (pytester.path / "schemas" / "stg_orders.json").write_text(ORDERS_JSON, encoding="utf-8")
    pytester.makepyfile(
        """
def test_contract(rowless_source):
    assert "amount" in rowless_source.schema("lake://stg_orders")


def test_a_frame_still_needs_one(rowless_source):
    rowless_source("lake://stg_orders")
"""
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*no engine adapter is installed*rowless-polars*"])


# --- choosing an engine ------------------------------------------------------


def test_two_engines_and_no_name_is_an_error_that_lists_them(
    project: pytest.Pytester,
) -> None:
    """With nothing to infer from, rowless names the candidates instead of picking."""
    project.makeconftest(SECOND_ENGINE)
    project.makepyfile(
        """
def test_frame(rowless_frame):
    rowless_frame({"order_id": "int64"})
"""
    )
    result = project.runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*more than one engine is installed (fake, fake2)*"])


def test_the_flag_fills_in_where_nothing_names_an_engine(project: pytest.Pytester) -> None:
    """`--rowless-engine` answers the question the test did not."""
    project.makeconftest(SECOND_ENGINE)
    project.makepyfile(
        """
def test_frame(rowless_frame):
    assert type(rowless_frame({"order_id": "int64"})).__name__ == "FakeFrame"
"""
    )
    result = project.runpytest(f"--rowless-engine={ENGINE}")
    result.assert_outcomes(passed=1)


def test_the_marker_names_the_engine_for_a_whole_module(project: pytest.Pytester) -> None:
    """A module-level marker covers every test under it."""
    project.makeconftest(SECOND_ENGINE)
    project.makepyfile(
        f'''
import pytest

pytestmark = pytest.mark.rowless(engine="{ENGINE}")


def test_frame(rowless_frame):
    assert type(rowless_frame({{"order_id": "int64"}})).__name__ == "FakeFrame"
'''
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1)


def test_the_pipeline_declaration_decides_for_a_check(project: pytest.Pytester) -> None:
    """`@rowless.pipeline(engine=...)` is enough, even with two engines installed."""
    project.makeconftest(SECOND_ENGINE)
    project.makepyfile(
        FROM_SOURCE
        + """
def test_contract(rowless_check):
    rowless_check(daily_revenue, {"order_id": "int64", "amount": "int64"})
"""
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1)


def test_skips_when_engine_missing(project: pytest.Pytester) -> None:
    """A test needing an uninstalled engine is skipped, not errored."""
    project.makepyfile(
        f'''
import pytest


@pytest.mark.rowless(engine="{UNAVAILABLE_ENGINE}")
def test_contract(rowless_frame):
    rowless_frame({{"order_id": "int64"}})
'''
    )
    result = project.runpytest("-rs")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines([f"*rowless: adapter '{UNAVAILABLE_ENGINE}' is installed*"])


def test_skips_when_the_engine_is_not_registered_at_all(project: pytest.Pytester) -> None:
    """A marker naming an engine nobody ships skips too, listing what is there."""
    project.makepyfile(
        """
import pytest


@pytest.mark.rowless(engine="nosuchengine")
def test_contract(rowless_frame):
    rowless_frame({"order_id": "int64"})
"""
    )
    result = project.runpytest("-rs")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*no adapter named 'nosuchengine'*"])


def test_a_misspelled_marker_argument_is_a_usage_error(project: pytest.Pytester) -> None:
    """A marker that cannot gate anything stops the run rather than passing."""
    project.makepyfile(
        """
import pytest


@pytest.mark.rowless(engin="fake")
def test_contract(rowless_frame):
    rowless_frame({"order_id": "int64"})
"""
    )
    result = project.runpytest()
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*unknown argument(s) engin*"])


# --- the summary and the coverage gate ---------------------------------------


def test_summary_reports_what_was_verified(project: pytest.Pytester) -> None:
    """Passes alone overstate a contract suite, so coverage prints next to them."""
    project.makepyfile(
        FROM_SOURCE
        + """
def test_contract(rowless_check):
    rowless_check(daily_revenue, {"order_id": "int64", "amount": "int64"})
"""
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*rowless: 1 contract test(s), 2 column(s), 100% typed*"])


def test_a_suite_with_no_rowless_tests_reports_nothing(project: pytest.Pytester) -> None:
    """Installed is not the same as involved: an unrelated suite pays nothing."""
    project.makepyfile(
        """
def test_arithmetic():
    assert 2 + 2 == 4
"""
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1)
    assert "rowless:" not in result.stdout.str()
    assert not (project.path / ".rowless").exists()


def test_min_coverage_gate_fails_a_run_whose_tests_passed(project: pytest.Pytester) -> None:
    """A half-typed schema passes its assertion and still fails the gate.

    Which is the point of the gate: `exact=False` on a column rowless typed
    is a true statement about that column and says nothing about the other
    one, so the floor is what notices.
    """
    project.makepyfile(
        """
import rowless
from rowless.ir.dtypes import any_


@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="fake")
def guessed(orders):
    return orders.select("order_id").derive("guess", any_())


def test_contract(rowless_check):
    rowless_check(guessed, {"order_id": "int64"}, exact=False)
"""
    )
    passing = project.runpytest()
    passing.assert_outcomes(passed=1)
    assert passing.ret == pytest.ExitCode.OK

    gated = project.runpytest("--rowless-min-coverage=0.9")
    gated.assert_outcomes(passed=1)
    assert gated.ret == pytest.ExitCode.TESTS_FAILED
    gated.stdout.fnmatch_lines(["*RWL204 typed coverage 50% is below the floor of 90%*"])


def test_min_coverage_comes_from_config_too(project: pytest.Pytester) -> None:
    """`min_coverage` in `rowless.toml` gates without a flag on the command line."""
    project.makefile(
        ".toml",
        rowless=(
            'roots = ["."]\nmin_coverage = 0.9\n\n'
            '[sources]\n"lake://*" = "json://schemas/{name}.json"\n'
        ),
    )
    project.makepyfile(
        """
import rowless
from rowless.ir.dtypes import any_


@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="fake")
def guessed(orders):
    return orders.select("order_id").derive("guess", any_())


def test_contract(rowless_check):
    rowless_check(guessed, {"order_id": "int64"}, exact=False)
"""
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED


def test_a_floor_with_no_rowless_tests_does_not_fail_the_run(project: pytest.Pytester) -> None:
    """Nothing asserted means nothing to score, not a failure."""
    project.makepyfile(
        """
def test_arithmetic():
    assert 2 + 2 == 4
"""
    )
    result = project.runpytest("--rowless-min-coverage=1.0")
    result.assert_outcomes(passed=1)
    assert result.ret == pytest.ExitCode.OK


def test_a_floor_outside_zero_to_one_is_rejected(project: pytest.Pytester) -> None:
    """The flag takes a ratio, the way `rowless.toml` does."""
    result = project.runpytest("--rowless-min-coverage=90")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*between 0 and 1*"])


def test_a_broken_config_errors_the_rowless_tests(project: pytest.Pytester) -> None:
    """A bad `rowless.toml` errors the tests that need it, and nothing else.

    The config is read when a fixture first asks for it rather than at
    start-up, so a file this plugin had no business reading cannot fail a
    suite that never mentioned rowless.
    """
    project.makefile(".toml", rowless="min_coverage = 5\n")
    project.makepyfile(
        INLINE_INPUTS
        + """
def test_contract(rowless_check):
    rowless_check(daily_revenue, {"order_id": "int64"}, inputs={"orders": ORDERS})


def test_unrelated():
    assert 2 + 2 == 4
"""
    )
    result = project.runpytest()
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*min_coverage must be between 0 and 1*"])

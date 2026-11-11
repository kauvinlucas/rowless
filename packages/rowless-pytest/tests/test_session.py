"""The parts under the fixtures, tested without a pytest-inside-pytest run.

`test_plugin.py` covers what a user sees, which is the right level for a
plugin and a slow one: every case there starts an inner pytest session. The
pieces those sessions are made of - which engine gets chosen, what the tally
adds up to, what a failure message says - are ordinary functions, and they are
cheaper and more precise to pin down directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest
from rowless.api import as_schema
from rowless.config import Config
from rowless.diagnostics.codes import Code
from rowless.diagnostics.models import Diagnostic, Level
from rowless.errors import AdapterNotFound
from rowless.ir.confidence import Confidence, Coverage
from rowless.ir.dtypes import any_, int_
from rowless.ir.schema import Field, Origin, Relation, Schema
from rowless.runtime.dryrun import DryRunResult
from rowless_pytest import checks, engines, frames
from rowless_pytest.session import RowlessSession

sys.path.insert(0, str(Path(__file__).parent))

from rowless_fake_pytest import ENGINE, UNAVAILABLE_ENGINE, FakeAdapter, FakeFrame

pytest_plugins = ["plugin_fixtures"]

ORDERS = Schema(
    fields=(
        Field(name="order_id", dtype=int_(64)),
        Field(name="amount", dtype=int_(64)),
    ),
    name="orders",
)

#: One column typed and one rowless could not type: 50% typed coverage, which
#: is what the gate is for.
HALF_TYPED = Schema(
    fields=(
        Field(name="order_id", dtype=int_(64)),
        Field(name="guess", dtype=any_()),
    ),
    name="guessed",
)


def session_for(tmp_path: Path, **overrides: object) -> RowlessSession:
    """A session over an empty repository at `tmp_path`."""
    config = Config(root_dir=str(tmp_path), **overrides)  # type: ignore[arg-type]
    return RowlessSession(config)


def result_for(
    schema: Schema,
    *,
    reasons: tuple[str, ...] = (),
    diagnostics: tuple[Diagnostic, ...] = (),
    output: bool = True,
) -> DryRunResult:
    """A `DryRunResult` shaped the way the runtime returns one."""
    return DryRunResult(
        target="pipelines.revenue:daily_revenue",
        output=Relation(schema=schema, engine=ENGINE) if output else None,
        coverage=Coverage.score(
            schema,
            confidence=Confidence.UNKNOWN if reasons else None,
            reasons=reasons,
        ),
        diagnostics=diagnostics,
    )


class Node:
    """The slice of `pytest.Item` the marker helpers actually use."""

    def __init__(self, mark: pytest.Mark | None) -> None:
        self._mark = mark

    def get_closest_marker(self, name: str) -> pytest.Mark | None:
        """The marker, if this node has one by that name."""
        if self._mark is not None and self._mark.name == name:
            return self._mark
        return None


def node_for(mark: pytest.MarkDecorator | None) -> pytest.Item:
    """A stand-in item carrying one marker."""
    return cast("pytest.Item", Node(None if mark is None else mark.mark))


# --- choosing an engine ------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"requested": "a", "pipeline": "b", "declared": "c", "flag": "d"}, "a"),
        ({"pipeline": "b", "declared": "c", "flag": "d"}, "b"),
        ({"declared": "c", "flag": "d"}, "c"),
        ({"flag": "d"}, "d"),
        ({"configured": ["e"]}, "e"),
    ],
)
def test_choose_follows_one_order(kwargs: dict[str, object], expected: str) -> None:
    """The call site, then the pipeline, then the test, then the flag, then config."""
    assert engines.choose(**kwargs) == expected  # type: ignore[arg-type]


def test_choose_will_not_pick_between_configured_engines() -> None:
    """Two configured engines name nothing, so the runtime infers instead."""
    assert engines.choose(configured=["polars", "duckdb"]) is None


def test_choose_is_none_when_nothing_names_an_engine() -> None:
    """None is an answer: `rowless_check` lets the runtime read the annotations."""
    assert engines.choose() is None


def test_require_takes_the_only_installed_engine(fake_engine: FakeAdapter) -> None:
    """One engine installed is not a guess."""
    _ = fake_engine
    assert engines.require(None) == ENGINE


def test_require_refuses_to_choose_between_two(fake_engine: FakeAdapter) -> None:
    """Two engines and no name is an error that lists them both."""
    from rowless.adapters.registry import register

    class SecondFake(FakeAdapter):
        name = "fake2"

    register("rowless.engines", "fake2", SecondFake())
    with pytest.raises(AdapterNotFound, match="more than one engine is installed"):
        engines.require(None)


def test_require_says_to_install_one_when_there_is_none(no_engines: None) -> None:
    """The first-install message names a package rather than a concept."""
    _ = no_engines
    with pytest.raises(AdapterNotFound, match="rowless-polars"):
        engines.require(None)


def test_unavailable_reports_the_registry_reason(fake_engine: FakeAdapter) -> None:
    """A usable engine is None; the others carry the message that explains them."""
    _ = fake_engine
    assert engines.unavailable(ENGINE) is None
    absent = engines.unavailable(UNAVAILABLE_ENGINE)
    assert absent is not None and "fake-engine>=1.0" in absent
    missing = engines.unavailable("nosuchengine")
    assert missing is not None and "no adapter named" in missing


def test_declared_engine_reads_a_keyword_or_a_positional() -> None:
    """Both spellings of the marker name the engine."""
    assert engines.declared_engine(node_for(pytest.mark.rowless(engine="polars"))) == "polars"
    assert engines.declared_engine(node_for(pytest.mark.rowless("duckdb"))) == "duckdb"
    assert engines.declared_engine(node_for(pytest.mark.rowless)) is None
    assert engines.declared_engine(node_for(None)) is None


def test_marker_problem_catches_what_would_silently_not_gate() -> None:
    """A misspelled keyword is reported; a plain marker is fine."""
    problem = engines.marker_problem(node_for(pytest.mark.rowless(engin="polars")))
    assert problem is not None and "engin" in problem
    assert engines.marker_problem(node_for(pytest.mark.rowless("polars", "duckdb"))) is not None
    assert engines.marker_problem(node_for(pytest.mark.rowless(engine="polars"))) is None
    assert engines.marker_problem(node_for(None)) is None


# --- the session ---------------------------------------------------------------


def test_the_flag_beats_the_configured_floor(tmp_path: Path) -> None:
    """`--rowless-min-coverage` overrides `min_coverage` in `rowless.toml`."""
    assert session_for(tmp_path, min_coverage=0.5).floor == 0.5
    config = Config(root_dir=str(tmp_path), min_coverage=0.5)
    assert RowlessSession(config, min_coverage=0.9).floor == 0.9


def test_record_adds_up_coverage_and_time(tmp_path: Path) -> None:
    """Two assertions make one tally, and the weaker confidence wins."""
    session = session_for(tmp_path)
    session.record(result_for(ORDERS))
    session.record(result_for(ORDERS, reasons=("is_empty",)), allowed_unverified=True)
    assert session.checks == 2
    assert session.coverage.columns_total == 4
    assert session.coverage.confidence is Confidence.UNKNOWN
    assert session.unverified == ["pipelines.revenue:daily_revenue"]


def test_no_assertions_means_no_gate(tmp_path: Path) -> None:
    """A suite that ran no rowless tests is not failed by a floor."""
    assert session_for(tmp_path, min_coverage=1.0).shortfall() is None


def test_shortfall_names_the_code_and_both_percentages(tmp_path: Path) -> None:
    """The message says what was measured, against what, over how many tests."""
    session = session_for(tmp_path, min_coverage=0.9)
    session.record(result_for(HALF_TYPED))
    shortfall = session.shortfall()
    assert shortfall is not None
    assert Code.COVERAGE_BELOW_THRESHOLD.value in shortfall
    assert "50%" in shortfall
    assert "90%" in shortfall


def test_a_met_floor_reports_nothing(tmp_path: Path) -> None:
    """Coverage at or above the floor is not a finding."""
    session = session_for(tmp_path, min_coverage=1.0)
    session.record(result_for(ORDERS))
    assert session.shortfall() is None


def test_summary_counts_the_unverified_results_a_suite_allowed(tmp_path: Path) -> None:
    """Opting out is visible in the one line everybody reads."""
    session = session_for(tmp_path)
    session.record(result_for(ORDERS, reasons=("is_empty",)), allowed_unverified=True)
    summary = session.summary()
    assert "1 contract test(s)" in summary
    assert "1 unverified result(s) allowed" in summary


def test_resolve_applies_the_source_map_and_caches(tmp_path: Path) -> None:
    """A URI resolves through config and lands in the cache under `.rowless/`."""
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "stg_orders.json").write_text(
        '{"order_id": "int64", "amount": "int64"}', encoding="utf-8"
    )
    session = session_for(tmp_path, source_map={"lake://*": "json://schemas/{name}.json"})
    schema = session.resolve("lake://stg_orders")
    assert list(schema) == ["order_id", "amount"]
    assert list((tmp_path / ".rowless" / "sources").glob("*.json"))


def test_frame_is_a_real_engine_object(tmp_path: Path, fake_engine: FakeAdapter) -> None:
    """The fixtures hand out the engine's own type, not a proxy."""
    _ = fake_engine
    session = session_for(tmp_path)
    frame = session.frame(ORDERS, engine=ENGINE)
    assert isinstance(frame, FakeFrame)
    assert list(frame.schema) == ["order_id", "amount"]
    session.close()


def test_frame_builder_labels_the_relation(tmp_path: Path, fake_engine: FakeAdapter) -> None:
    """`name=` relabels the schema, which is what a finding is reported against."""
    _ = fake_engine
    session = session_for(tmp_path)
    frame = frames.FrameBuilder(session, engine=ENGINE)({"order_id": "int64"}, name="orders")
    assert isinstance(frame, FakeFrame)
    assert frame.schema.name == "orders"
    session.close()


def test_source_resolver_answers_with_a_schema_or_a_frame(
    tmp_path: Path,
    fake_engine: FakeAdapter,
) -> None:
    """The schema path materializes nothing; the call path hands back a frame."""
    _ = fake_engine
    (tmp_path / "orders.json").write_text('{"order_id": "int64"}', encoding="utf-8")
    session = session_for(tmp_path)
    resolver = frames.SourceResolver(session, engine=ENGINE)
    assert list(resolver.schema("json://orders.json")) == ["order_id"]
    assert isinstance(resolver("json://orders.json"), FakeFrame)
    session.close()


def test_engine_options_come_from_config(tmp_path: Path) -> None:
    """`[engine.<name>]` reaches the adapter, the way the CLI passes it."""
    session = session_for(tmp_path, engine_options={ENGINE: {"arrow_dtypes": True}})
    assert session.options_for(ENGINE) == {"arrow_dtypes": True}
    assert session.options_for("other") == {}
    assert session.options_for(None) == {}


# --- failure messages --------------------------------------------------------


def test_unresolved_message_carries_the_finding_that_explains_it() -> None:
    """The reason a contract was not tested is usually the diagnostic."""
    diagnostic = Diagnostic(
        code=Code.COLUMN_NOT_FOUND,
        level=Level.ERROR,
        message="ColumnNotFound 'amount'",
        origin=Origin(kind="python", ref="pipelines.revenue", file="revenue.py", line=9),
        suggestion="did you mean 'amount_cents'?",
        related=(Origin(kind="sql", ref="stg_orders", file="stg_orders.sql", line=14),),
    )
    message = checks.unresolved_message(
        result_for(
            Schema.empty(),
            reasons=("ColumnNotFound amount",),
            diagnostics=(diagnostic,),
            output=False,
        )
    )
    assert "could not resolve pipelines.revenue:daily_revenue" in message
    assert "RWL101 error: ColumnNotFound 'amount' (revenue.py:9)" in message
    assert "did you mean 'amount_cents'?" in message
    assert "upstream stg_orders.sql:14" in message


def test_unverified_message_offers_the_opt_in_and_the_reason() -> None:
    """Both halves: why rowless is not sure, and what a test can do about it."""
    message = checks.unverified_message(result_for(ORDERS, reasons=("row-dependent is_empty",)))
    assert "proves nothing" in message
    assert "row-dependent is_empty" in message
    assert "allow_unverified=True" in message


def test_mismatch_message_shows_both_schemas() -> None:
    """Expected next to resolved, because the diff alone is hard to read."""
    message = checks.mismatch_message(
        result_for(ORDERS),
        as_schema({"order_id": "int64", "amount": "string"}),
        ["amount: expected string, got int64"],
        exact=True,
    )
    assert "amount: expected string, got int64" in message
    assert "expected: order_id int64, amount string" in message
    assert "resolved: order_id int64, amount int64" in message
    assert "exact=False" not in message


def test_mismatch_message_suggests_exact_false_only_when_it_would_help() -> None:
    """The hint appears for an unexpected column, not for a wrong dtype."""
    message = checks.mismatch_message(
        result_for(ORDERS),
        as_schema({"order_id": "int64"}),
        ["unexpected column(s): amount"],
        exact=True,
    )
    assert "pass exact=False" in message

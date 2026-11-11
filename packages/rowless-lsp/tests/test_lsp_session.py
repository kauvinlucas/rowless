"""The warm state: caching, re-anchoring, and coalescing work.

These are the tests about latency and staleness rather than about findings.
Both failure modes they guard against are quiet ones: a cached config that
outlives its file reports findings the user already configured away, and a
carried-over finding on a line that moved accuses code that is fine.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from lsp_fixtures import (  # noqa: F401
    FIXED,
    PIPELINE,
    in_process,
    project,
)
from rowless.diagnostics.codes import Code
from rowless.diagnostics.models import Diagnostic, Level
from rowless.ir.schema import Origin
from rowless_lsp.anchors import anchor, reanchor
from rowless_lsp.scheduler import Debouncer
from rowless_lsp.session import Session


def _finding(line: int, message: str = "ColumnNotFound 'amount'") -> Diagnostic:
    """A located finding, as a dry run produces."""
    return Diagnostic(
        code=Code.COLUMN_NOT_FOUND,
        level=Level.ERROR,
        message=message,
        origin=Origin(kind="python", ref="revenue:f", file="revenue.py", line=line),
    )


# --- Checking one file -------------------------------------------------------


def test_a_check_finds_the_planted_rename(project: Path, in_process: object) -> None:
    """The same finding `rowless check` reports, through the same call."""
    session = Session(str(project))
    path = str(project / "revenue.py")
    session.open(path, PIPELINE)
    grouped = session.check(path)

    findings = [finding for found in grouped.values() for finding in found]
    assert [finding.code for finding in findings] == [Code.COLUMN_NOT_FOUND]
    assert findings[0].origin is not None
    assert findings[0].origin.line == 11


def test_a_check_warms_the_schemas_every_other_feature_reads(
    project: Path,
    in_process: object,
) -> None:
    """Completion and hover have nothing until this has happened once."""
    session = Session(str(project))
    path = str(project / "revenue.py")
    document = session.open(path, PIPELINE)
    session.check(path)

    resolved = document.resolved("daily_revenue")
    assert resolved is not None
    assert list(resolved.inputs) == ["orders"]
    assert "amount_cents" in resolved.inputs["orders"]
    assert resolved.uris == {"orders": "lake://stg_orders"}
    assert resolved.target == "revenue:daily_revenue"


def test_the_dry_run_and_the_live_check_do_not_double_report(
    project: Path,
    in_process: object,
) -> None:
    """Both reach the same missing column. The editor must show one squiggle."""
    session = Session(str(project))
    path = str(project / "revenue.py")
    session.open(path, PIPELINE)
    session.check(path)

    found = session.diagnostics(path)
    assert [finding.code for finding in found] == [Code.COLUMN_NOT_FOUND]


def test_a_fixed_pipeline_checks_clean(project: Path, in_process: object) -> None:
    """The other half: a correct pipeline produces nothing."""
    path = project / "revenue.py"
    path.write_text(FIXED, encoding="utf-8")
    session = Session(str(project))
    session.open(str(path), FIXED)
    session.check(str(path))

    assert session.diagnostics(str(path)) == []


def test_a_workspace_check_fills_every_open_document(
    project: Path,
    in_process: object,
) -> None:
    """So the first problem list is the one CI would produce."""
    session = Session(str(project))
    path = str(project / "revenue.py")
    session.open(path, PIPELINE)
    session.check_workspace()

    document = session.document(path)
    assert document is not None
    assert document.checked is not None
    assert "daily_revenue" in document.checked.resolved


# --- Staleness ---------------------------------------------------------------


def test_a_finding_follows_its_line_when_the_buffer_moves(
    project: Path,
    in_process: object,
) -> None:
    """Insert two lines above a finding and it must not stay where it was."""
    session = Session(str(project))
    path = str(project / "revenue.py")
    session.open(path, PIPELINE)
    session.check(path)

    session.update(path, "# a comment\n# another\n" + PIPELINE, version=2)
    found = session.diagnostics(path)

    assert [finding.origin.line for finding in found] == [13]  # type: ignore[union-attr]


def test_a_finding_is_dropped_when_its_line_is_gone(
    project: Path,
    in_process: object,
) -> None:
    """Better silence until the next save than a squiggle on innocent code."""
    session = Session(str(project))
    path = str(project / "revenue.py")
    session.open(path, PIPELINE)
    session.check(path)

    without = PIPELINE.replace(
        '    return orders.select("order_id", "amount")', "    return orders"
    )
    session.update(path, without, version=2)

    assert session.diagnostics(path) == []


def test_a_buffer_that_does_not_parse_reports_nothing(
    project: Path,
    in_process: object,
) -> None:
    """Half-typed code gets no squiggles placed by stale line numbers.

    Completion still works here - it falls back to the last index that parsed,
    because a stale shape beats a flicker. A diagnostic cannot make that
    trade: placed by stale lines it lands on code that is fine.
    """
    session = Session(str(project))
    path = str(project / "revenue.py")
    session.open(path, PIPELINE)
    session.check(path)

    session.update(path, PIPELINE.replace('"amount")', '"am'), version=2)

    assert session.diagnostics(path) == []


def test_an_ignore_comment_silences_a_dry_run_finding_before_the_save(
    project: Path,
    in_process: object,
) -> None:
    """The comment was typed after the check ran, and has to work anyway."""
    session = Session(str(project))
    path = str(project / "revenue.py")
    session.open(path, PIPELINE)
    session.check(path)

    edited = PIPELINE.replace(
        '"order_id", "amount")', '"order_id", "amount")  # rowless: ignore[RWL101]'
    )
    session.update(path, edited, version=2)

    assert session.diagnostics(path) == []


def test_closing_a_document_forgets_it(project: Path, in_process: object) -> None:
    """Cached schemas for a file nobody has open are just memory."""
    session = Session(str(project))
    path = str(project / "revenue.py")
    session.open(path, PIPELINE)
    session.close(path)

    assert session.document(path) is None
    assert session.diagnostics(path) == []


# --- Configuration -----------------------------------------------------------


def test_the_config_is_discovered_once(project: Path) -> None:
    """Walking the tree for `rowless.toml` is not a per-keystroke activity."""
    session = Session(str(project))

    assert session.config is session.config


def test_reload_rediscovers_the_config(project: Path) -> None:
    """A cached config outliving its file is how settings stop taking effect."""
    session = Session(str(project))
    first = session.config
    session.reload()

    assert session.config is not first
    assert session.config.root_dir == first.root_dir


def test_configured_excludes_are_honoured(project: Path, in_process: object) -> None:
    """One policy: what CI skips, the editor skips."""
    (project / "rowless.toml").write_text(
        'roots = ["."]\nexclude = ["revenue.py"]\n\n'
        '[sources]\n"lake://*" = "json://schemas/{name}.json"\n',
        encoding="utf-8",
    )
    session = Session(str(project))
    path = str(project / "revenue.py")
    session.open(path, PIPELINE)
    grouped = session.check(path)

    codes = [finding.code for found in grouped.values() for finding in found]
    assert Code.COLUMN_NOT_FOUND not in codes


# --- Anchoring ---------------------------------------------------------------


def test_an_unmoved_line_keeps_its_finding() -> None:
    """The common case has to cost nothing."""
    lines = ["import rowless", "", "x = 1"]
    kept = reanchor(anchor([_finding(3)], lines), lines)

    assert [finding.origin.line for finding in kept] == [3]  # type: ignore[union-attr]


def test_a_moved_line_takes_its_finding_with_it() -> None:
    """Recognised by what the line says, not by where it was."""
    before = ["import rowless", "", "x = 1"]
    after = ["# new", "# new", "import rowless", "", "x = 1"]
    kept = reanchor(anchor([_finding(3)], before), after)

    assert [finding.origin.line for finding in kept] == [5]  # type: ignore[union-attr]


def test_a_deleted_line_drops_its_finding() -> None:
    """A finding pointing at the wrong line is worse than no finding."""
    before = ["import rowless", "", "x = 1"]
    after = ["import rowless", ""]

    assert reanchor(anchor([_finding(3)], before), after) == ()


def test_the_nearest_identical_line_wins() -> None:
    """Two pipelines can end in the same call, and the finding belongs to one."""
    before = ["return orders", "pass", "return orders"]
    after = ["return orders", "pass", "pass", "return orders"]
    kept = reanchor(anchor([_finding(3)], before), after)

    assert [finding.origin.line for finding in kept] == [4]  # type: ignore[union-attr]


def test_a_finding_with_no_line_passes_through() -> None:
    """A coverage score has no line to be wrong about."""
    report_level = Diagnostic(
        code=Code.COVERAGE_BELOW_THRESHOLD, level=Level.WARNING, message="below floor"
    )
    kept = reanchor(anchor([report_level], ["x = 1"]), ["y = 2"])

    assert kept == (report_level,)


def test_a_move_beyond_the_window_is_a_drop() -> None:
    """Far enough away and an identical line is a different line."""
    before = ["x = 1"]
    after = ["pass"] * 60 + ["x = 1"]

    assert reanchor(anchor([_finding(1)], before), after, window=5) == ()


# --- Debouncing --------------------------------------------------------------


def test_rapid_edits_produce_one_run() -> None:
    """A queue would make the editor fall further behind the faster you type."""
    runs: list[int] = []

    async def exercise() -> None:
        debouncer = Debouncer(delay=0.01)

        async def job() -> None:
            runs.append(1)

        for _ in range(5):
            debouncer.schedule("revenue.py", job)
        await debouncer.drain()

    asyncio.run(exercise())

    assert runs == [1]


def test_separate_documents_do_not_cancel_each_other() -> None:
    """One pending job per key, not one in total."""
    runs: list[str] = []

    async def exercise() -> None:
        debouncer = Debouncer(delay=0.01)

        def job(name: str) -> Callable[[], Awaitable[None]]:
            async def run() -> None:
                runs.append(name)

            return run

        debouncer.schedule("a.py", job("a.py"))
        debouncer.schedule("b.py", job("b.py"))
        await debouncer.drain()

    asyncio.run(exercise())

    assert sorted(runs) == ["a.py", "b.py"]


def test_a_cancelled_job_never_runs() -> None:
    """Closing a document must not leave a check queued against it."""
    runs: list[int] = []

    async def exercise() -> None:
        debouncer = Debouncer(delay=0.05)

        async def job() -> None:
            runs.append(1)

        debouncer.schedule("revenue.py", job)
        assert debouncer.pending == frozenset({"revenue.py"})
        debouncer.cancel("revenue.py")
        await debouncer.drain()

    asyncio.run(exercise())

    assert runs == []

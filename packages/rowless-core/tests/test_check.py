"""End to end: discovery, source resolution, dry run, report, graph.

The dry runs here go through `dry_run_callable` rather than the worker
subprocess, because a fake adapter cannot be registered in a child process.
The subprocess boundary itself is covered in `test_worker.py`; what is checked
here is the orchestration above it, which is the same code path the pytest
plugin and the LSP use.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from rowless.check import run
from rowless.config import Config
from rowless.diagnostics.codes import Code
from rowless.diagnostics.models import Level
from rowless.ir.confidence import Confidence
from rowless.lineage.graph import ColumnRef, Evidence, NodeKind
from rowless.runtime.dryrun import DryRunRequest, DryRunResult, dry_run_target
from rowless_fake import FakeAdapter, install


@pytest.fixture
def in_process(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeAdapter]:
    """Run dry runs in this process, mirroring what the worker does in a child."""
    from rowless.adapters.registry import clear_cache

    clear_cache()
    adapter = install()

    def _run(request: DryRunRequest) -> DryRunResult:
        for entry in request.sys_path:
            if entry not in sys.path:
                sys.path.insert(0, entry)
        # A worker gets a fresh interpreter per dry run; in-process, the module
        # from the previous test's temporary directory is still cached.
        module_name = request.target.partition(":")[0]
        sys.modules.pop(module_name, None)
        importlib.invalidate_caches()
        return dry_run_target(
            request.target,
            dict(request.inputs),
            engine=request.engine,
            options=request.options,
            uris=request.uris,
        )

    monkeypatch.setattr("rowless.check.dry_run", _run)
    yield adapter
    clear_cache()


def test_finds_the_planted_rename_and_nothing_else(project: Path, in_process: FakeAdapter) -> None:
    """The proposal's example: one finding, naming the column and the near miss."""
    outcome = run(Config.discover(str(project)))
    report = outcome.report

    assert report.exit_code == 1
    assert [d.code for d in report.diagnostics] == [Code.COLUMN_NOT_FOUND]
    finding = report.diagnostics[0]
    assert "amount" in finding.message
    assert finding.suggestion == "did you mean 'amount_cents'?"
    assert finding.level is Level.ERROR
    assert report.pipelines_checked == 1
    assert report.pipelines_skipped == 0


def test_a_working_pipeline_is_clean_and_traced(project: Path, in_process: FakeAdapter) -> None:
    """A correct pipeline exits 0, and its columns come back traced."""
    (project / "revenue.py").write_text(
        "\n".join(
            [
                "import rowless",
                "from rowless_fake import FakeFrame",
                "",
                "",
                '@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="fake")',
                "def daily_revenue(orders: FakeFrame) -> FakeFrame:",
                '    return orders.rename({"amount_cents": "revenue"})',
                "",
            ]
        ),
        encoding="utf-8",
    )

    outcome = run(Config.discover(str(project)))

    assert outcome.report.exit_code == 0
    assert outcome.report.coverage.confidence is Confidence.EXACT
    assert outcome.report.coverage.typed_ratio == 1.0

    node = outcome.graph.nodes["revenue:daily_revenue"]
    assert node.kind is NodeKind.PYTHON
    assert not node.opaque
    edges = outcome.graph.edges_into(ColumnRef("revenue:daily_revenue", "revenue"))
    assert [edge.evidence for edge in edges] == [Evidence.PLAN]
    assert edges[0].source == ColumnRef("lake://stg_orders", "amount_cents"), (
        "the graph names the table the way the pipeline does, not the rewritten path"
    )


def test_untraced_columns_fall_back_to_a_name_match(project: Path, in_process: FakeAdapter) -> None:
    """Weak evidence is recorded as weak, not as absent and not as strong."""
    outcome = run(Config.discover(str(project)))
    _ = outcome

    (project / "revenue.py").write_text(
        "\n".join(
            [
                "import rowless",
                "from rowless_fake import FakeFrame",
                "",
                "",
                '@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="fake")',
                "def passthrough(orders: FakeFrame) -> FakeFrame:",
                '    return orders.select("order_id", "status")',
                "",
            ]
        ),
        encoding="utf-8",
    )

    rebuilt = run(Config.discover(str(project)))

    edges = rebuilt.graph.edges_into(ColumnRef("revenue:passthrough", "order_id"))
    assert [edge.evidence for edge in edges] == [Evidence.NAME_MATCH]


def test_an_unresolvable_source_is_reported_not_skipped_silently(
    project: Path,
    in_process: FakeAdapter,
) -> None:
    """A skipped pipeline looks exactly like a passing one, so it must be loud."""
    (project / "schemas" / "stg_orders.json").unlink()

    report = run(Config.discover(str(project))).report

    assert report.pipelines_checked == 0
    assert report.pipelines_skipped == 1
    assert [d.code for d in report.diagnostics] == [Code.SCHEMA_NOT_FOUND]
    assert report.diagnostics[0].level is Level.UNVERIFIED
    assert report.exit_code == 1
    assert report.coverage.confidence is Confidence.UNKNOWN, (
        "a score computed only over the pipelines that worked would improve "
        "every time rowless failed to check something"
    )
    assert "not checked" in " ".join(report.coverage.reasons)


def test_an_unbound_input_says_what_to_do(project: Path, in_process: FakeAdapter) -> None:
    """Rowless cannot guess which table a bare parameter reads, and says so."""
    (project / "revenue.py").write_text(
        "\n".join(
            [
                "import rowless",
                "from rowless_fake import FakeFrame",
                "",
                "",
                "@rowless.pipeline(engine='fake')",
                "def totals(orders: FakeFrame) -> FakeFrame:",
                '    return orders.select("order_id")',
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = run(Config.discover(str(project))).report

    assert [d.code for d in report.diagnostics] == [Code.UNRESOLVED_SOURCE]
    assert "@rowless.pipeline" in (report.diagnostics[0].detail or "")


def test_row_dependence_makes_the_node_opaque(project: Path, in_process: FakeAdapter) -> None:
    """A node rowless could not see through has to be visible in the graph."""
    (project / "revenue.py").write_text(
        "\n".join(
            [
                "import rowless",
                "from rowless_fake import FakeFrame",
                "",
                "",
                '@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="fake")',
                "def totals(orders: FakeFrame) -> FakeFrame:",
                "    if orders.is_empty():",
                "        return orders",
                '    return orders.select("order_id")',
                "",
            ]
        ),
        encoding="utf-8",
    )

    outcome = run(Config.discover(str(project)))

    assert outcome.graph.nodes["revenue:totals"].opaque
    assert outcome.report.coverage.confidence is Confidence.UNKNOWN
    assert outcome.report.exit_code == 1


def test_inline_ignore_suppresses_one_line(project: Path, in_process: FakeAdapter) -> None:
    """Line-level and code-specific, like mypy's, and visible in review."""
    report = run(Config.discover(str(project))).report
    assert report.exit_code == 1

    flagged = report.diagnostics[0].origin
    assert flagged is not None and flagged.line is not None

    source = (project / "revenue.py").read_text(encoding="utf-8").splitlines()
    source[flagged.line - 1] += "  # rowless: ignore[RWL101]"
    suppressed = "\n".join(source) + "\n"
    (project / "revenue.py").write_text(suppressed, encoding="utf-8")

    assert run(Config.discover(str(project))).report.exit_code == 0


def test_coverage_floor_gates_only_when_configured(
    project: Path,
    in_process: FakeAdapter,
) -> None:
    """`min_coverage` starts at 0 so a first run reports rather than blocks."""
    (project / "revenue.py").write_text(
        "\n".join(
            [
                "import rowless",
                "from rowless.ir.dtypes import any_",
                "from rowless_fake import FakeFrame",
                "",
                "",
                '@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="fake")',
                "def totals(orders: FakeFrame) -> FakeFrame:",
                '    return orders.select("order_id").derive("guess", any_())',
                "",
            ]
        ),
        encoding="utf-8",
    )
    config = Config.discover(str(project))

    relaxed = run(config).report
    assert relaxed.coverage.typed_ratio == 0.5
    assert Code.COVERAGE_BELOW_THRESHOLD not in {d.code for d in relaxed.diagnostics}

    strict = run(config.with_overrides(min_coverage=0.9)).report
    assert Code.COVERAGE_BELOW_THRESHOLD in {d.code for d in strict.diagnostics}
    assert strict.exit_code == 1


def test_no_pipelines_is_reported_rather_than_a_silent_pass(tmp_path: Path) -> None:
    """No findings and nothing checked mean opposite things."""
    report = run(Config.discover(str(tmp_path))).report

    assert [d.code for d in report.diagnostics] == [Code.NO_PIPELINES_FOUND]
    assert report.pipelines_checked == 0


def test_select_narrows_to_one_pipeline(project: Path, in_process: FakeAdapter) -> None:
    """`--select` is how a user works on one pipeline in a large repo."""
    outcome = run(Config.discover(str(project)), select="*nothing_matches*")

    assert [d.code for d in outcome.report.diagnostics] == [Code.NO_PIPELINES_FOUND]

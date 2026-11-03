"""Renderers: one report, several surfaces, and one exit-code contract."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from rowless.diagnostics.codes import Code
from rowless.diagnostics.models import Diagnostic, Level, Report
from rowless.diagnostics.render import (
    RENDERERS,
    SCHEMA_VERSION,
    GitHubRenderer,
    JsonRenderer,
    JUnitRenderer,
    MarkdownRenderer,
    SarifRenderer,
    TerminalRenderer,
)
from rowless.ir.confidence import Confidence, Coverage
from rowless.ir.schema import Origin

FINDING = Diagnostic(
    code=Code.COLUMN_NOT_FOUND,
    level=Level.ERROR,
    message="ColumnNotFound 'amount'",
    origin=Origin(kind="python", ref="revenue:daily_revenue", file="pipelines/revenue.py", line=9),
    related=(Origin(kind="sql", ref="stg_orders", file="models/stg_orders.sql", line=14),),
    detail="upstream renamed it to amount_cents (this branch)",
    suggestion="did you mean 'amount_cents'?",
)

UNVERIFIED = Diagnostic(
    code=Code.ROW_DEPENDENT_CONTROL_FLOW,
    level=Level.UNVERIFIED,
    message="row-dependent call is_empty in revenue:daily_revenue",
)

REPORT = Report(
    diagnostics=(FINDING, UNVERIFIED),
    pipelines_checked=2,
    pipelines_skipped=1,
    coverage=Coverage(
        columns_total=10,
        columns_typed=8,
        columns_traced=5,
        confidence=Confidence.PARTIAL,
        reasons=("if df.is_empty() at pipelines/revenue.py:4",),
    ),
    duration_ms=120.0,
)


def test_terminal_output_matches_the_proposals_shape() -> None:
    """Located headline, indented cause, upstream line, then a coverage line."""
    rendered = TerminalRenderer(color=False).render(REPORT)
    lines = rendered.splitlines()

    assert lines[0].startswith("pipelines/revenue.py:9  ColumnNotFound 'amount'")
    assert lines[0].endswith("RWL101")
    assert "  upstream renamed it to amount_cents (this branch)" in lines
    assert "  upstream models/stg_orders.sql:14" in lines
    assert "2 pipeline(s) checked, 1 error(s), 0 warning(s), 1 unverified, 1 skipped" in rendered
    assert "coverage: 80% typed, 50% traced, confidence partial" in rendered
    assert "if df.is_empty()" in rendered


def test_terminal_colour_is_optional() -> None:
    """Markup only when the output is a terminal; CI logs stay readable."""
    assert "[bold red]" in TerminalRenderer(color=True).render(REPORT)
    assert "[bold red]" not in TerminalRenderer(color=False).render(REPORT)


def test_json_is_line_per_diagnostic_and_versioned() -> None:
    """The LSP and the Action parse this, so every line carries its version."""
    lines = [json.loads(line) for line in JsonRenderer().render(REPORT).splitlines()]

    assert all(line["schema_version"] == SCHEMA_VERSION for line in lines)
    assert [line["kind"] for line in lines] == ["diagnostic", "diagnostic", "summary"]
    assert lines[0]["code"] == "RWL101"
    assert lines[0]["origin"]["line"] == 9
    assert lines[-1]["exit_code"] == 1
    assert lines[-1]["coverage"]["confidence"] == "partial"
    assert lines[-1]["pipelines_skipped"] == 1


def test_github_annotations_carry_file_and_line() -> None:
    """Annotations have to land on the right line of a PR diff, with no token."""
    rendered = GitHubRenderer().render(REPORT)

    assert "::error file=pipelines/revenue.py,line=9,title=rowless RWL101::" in rendered
    assert "amount_cents" in rendered, "the cause travels on the same line"
    assert rendered.splitlines()[1].startswith("::error "), "unverified annotates as an error"


def test_markdown_comment_has_coverage_and_collapses() -> None:
    """The PR comment: findings plus the number that qualifies them."""
    rendered = MarkdownRenderer().render(REPORT)

    assert "## rowless" in rendered
    assert "80% typed" in rendered
    assert "<details>" in rendered
    assert "<code>RWL101</code>" in rendered


def test_sarif_is_valid_enough_for_code_scanning() -> None:
    """One result per finding, located, with the rule declared."""
    log = json.loads(SarifRenderer().render(REPORT))

    assert log["version"] == "2.1.0"
    run = log["runs"][0]
    assert {rule["id"] for rule in run["tool"]["driver"]["rules"]} == {"RWL101", "RWL201"}
    located = run["results"][0]["locations"][0]["physicalLocation"]
    assert located["artifactLocation"]["uri"] == "pipelines/revenue.py"
    assert located["region"]["startLine"] == 9
    assert run["results"][1]["level"] == "error", "unverified is not a note"


def test_junit_reports_failures_not_just_cases() -> None:
    """Any CI system shows findings as test failures."""
    suite = ET.fromstring(JUnitRenderer().render(REPORT))

    assert suite.attrib["failures"] == "2"
    assert suite.attrib["skipped"] == "1"
    cases = suite.findall("testcase")
    assert cases[0].attrib["classname"] == "pipelines/revenue.py:9"
    assert cases[0].find("failure") is not None


def test_every_renderer_handles_a_clean_report() -> None:
    """A clean run is the common case, and must not crash a formatter."""
    clean = Report(pipelines_checked=3, coverage=Coverage(confidence=Confidence.EXACT))

    for name, factory in RENDERERS.items():
        renderer = factory(color=False) if name == "terminal" else factory()
        rendered = renderer.render(clean)
        assert isinstance(rendered, str)
        if name == "github":
            assert rendered == "", "no findings means no annotations to place"
        else:
            assert rendered


def test_exit_codes_separate_findings_from_rowless_failing() -> None:
    """CI needs to tell a broken pipeline from a broken tool."""
    assert Report().exit_code == 0
    assert Report(diagnostics=(FINDING,)).exit_code == 1
    assert Report(diagnostics=(UNVERIFIED,)).exit_code == 1, "unverified is not a pass"

    internal = Diagnostic(code=Code.WORKER_CRASHED, level=Level.ERROR, message="worker died")
    assert Report(diagnostics=(internal,)).exit_code == 2

    warning = Diagnostic(code=Code.ANY_COLUMN_IN_OUTPUT, level=Level.WARNING, message="any")
    assert Report(diagnostics=(warning,)).exit_code == 0

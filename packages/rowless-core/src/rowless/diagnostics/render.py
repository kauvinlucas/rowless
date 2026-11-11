"""Rendering a report for each surface.

Every renderer takes the same `Report`, so a new output format is one function
here and one CLI flag, never a change to how checking works.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any, Protocol

from rowless.diagnostics.models import Diagnostic, Level, Report
from rowless.ir.confidence import Coverage
from rowless.ir.serde import coverage_to_dict, origin_to_dict

#: Bumped when the JSON or SARIF shape changes incompatibly. The LSP and the
#: GitHub Action parse this output, so the version is part of the contract.
SCHEMA_VERSION = 1

#: GitHub rejects a comment over 65536 characters, and a truncated comment that
#: still says how much was cut is more useful than a failed post.
MARKDOWN_LIMIT = 60000

_LEVEL_STYLE: dict[Level, str] = {
    Level.ERROR: "bold red",
    Level.WARNING: "yellow",
    Level.UNVERIFIED: "magenta",
    Level.INFO: "dim",
}

_GITHUB_COMMAND: dict[Level, str] = {
    Level.ERROR: "error",
    Level.WARNING: "warning",
    # GitHub has no "unverified", and a notice would read as a pass. An
    # unverified finding gates the build, so it is annotated as an error.
    Level.UNVERIFIED: "error",
    Level.INFO: "notice",
}

_SARIF_LEVEL: dict[Level, str] = {
    Level.ERROR: "error",
    Level.WARNING: "warning",
    Level.UNVERIFIED: "error",
    Level.INFO: "note",
}


class Renderer(Protocol):
    """Report in, text out."""

    def render(self, report: Report) -> str:
        """Serialise the whole report."""
        ...


class TerminalRenderer:
    """Human output, matching the proposal's mock: code, message, indented cause."""

    def __init__(self, *, color: bool = True, show_coverage: bool = True) -> None:
        self.color = color
        self.show_coverage = show_coverage

    def render(self, report: Report) -> str:
        """Render with rich markup, degrading to plain text when not a tty."""
        lines: list[str] = []
        for diagnostic in report.diagnostics:
            lines.extend(self._diagnostic(diagnostic))
        if lines and (self.show_coverage or report.diagnostics):
            lines.append("")
        lines.append(self._summary(report))
        if self.show_coverage:
            lines.append(self._coverage(report.coverage))
            lines.extend(f"  {reason}" for reason in report.coverage.reasons)
        return "\n".join(lines)

    def _diagnostic(self, diagnostic: Diagnostic) -> list[str]:
        """One finding: a located headline, then everything indented under it."""
        where = f"{diagnostic.origin}  " if diagnostic.origin is not None else ""
        lines = [f"{where}{self._styled(diagnostic.level, diagnostic.message)}"]
        for note in (diagnostic.detail or "").splitlines():
            lines.append(f"  {note}")
        for related in diagnostic.related:
            lines.append(f"  upstream {related}")
        if diagnostic.suggestion:
            lines.append(f"  {diagnostic.suggestion}")
        lines[0] = f"{lines[0]}  {self._dim(diagnostic.code.value)}"
        return lines

    def _summary(self, report: Report) -> str:
        """The count line.

        Says "skipped" out loud, because a skipped pipeline is not a passing
        one and the two look identical on a terminal.
        """
        errors = sum(1 for d in report.diagnostics if d.level is Level.ERROR)
        unverified = sum(1 for d in report.diagnostics if d.level is Level.UNVERIFIED)
        warnings = sum(1 for d in report.diagnostics if d.level is Level.WARNING)
        parts = [
            f"{report.pipelines_checked} pipeline(s) checked",
            f"{errors} error(s)",
            f"{warnings} warning(s)",
            f"{unverified} unverified",
        ]
        if report.pipelines_skipped:
            parts.append(f"{report.pipelines_skipped} skipped")
        return ", ".join(parts)

    def _coverage(self, coverage: Coverage) -> str:
        """The coverage line, reported next to every finding count.

        With no columns resolved there is nothing to take a percentage of, and
        printing "100% typed" there would be the most misleading number the
        tool could produce.
        """
        confidence = coverage.confidence.name.lower()
        if coverage.columns_total == 0:
            return f"coverage: no columns resolved, confidence {confidence}"
        return (
            f"coverage: {coverage.typed_ratio:.0%} typed, "
            f"{coverage.traced_ratio:.0%} traced, "
            f"confidence {confidence}"
        )

    def _styled(self, level: Level, text: str) -> str:
        """Apply rich markup when colour is on."""
        if not self.color:
            return text
        return f"[{_LEVEL_STYLE[level]}]{text}[/]"

    def _dim(self, text: str) -> str:
        """Secondary text: the code at the end of a headline."""
        return f"[dim]{text}[/]" if self.color else text


class JsonRenderer:
    """Machine output. The stable contract for editors and other tools.

    Versioned with a `schema_version` field, because the LSP and the GitHub
    Action parse it and must keep working across rowless releases.
    """

    def render(self, report: Report) -> str:
        """JSON Lines, one diagnostic per line, then one summary line.

        Every line carries `schema_version` and `kind`, so a consumer reading
        the stream a line at a time never has to have seen a header.
        """
        lines = [json.dumps(_diagnostic_to_dict(d), sort_keys=True) for d in report.diagnostics]
        lines.append(json.dumps(_summary_to_dict(report), sort_keys=True))
        return "\n".join(lines)


class GitHubRenderer:
    """`::error file=...,line=...::message` workflow commands.

    Gets annotations onto the right lines of a PR diff with no API token.
    """

    def render(self, report: Report) -> str:
        """Emit workflow commands.

        A clean report renders as nothing: there is no annotation to place,
        and a workflow command saying "all good" would be noise in a log.
        """
        lines: list[str] = []
        for diagnostic in report.diagnostics:
            command = _GITHUB_COMMAND[diagnostic.level]
            properties = [f"title=rowless {diagnostic.code.value}"]
            origin = diagnostic.origin
            if origin is not None and origin.file:
                properties.insert(0, f"file={origin.file}")
                if origin.line is not None:
                    properties.insert(1, f"line={origin.line}")
                if origin.column is not None:
                    properties.insert(2, f"col={origin.column}")
            lines.append(f"::{command} {','.join(properties)}::{_one_line(diagnostic)}")
        return "\n".join(lines)


class MarkdownRenderer:
    """The PR comment body: the blast-radius tree plus a coverage line."""

    def render(self, report: Report) -> str:
        """Render collapsible Markdown, kept under GitHub's comment size limit."""
        coverage = report.coverage
        confidence = coverage.confidence.name.lower()
        if coverage.columns_total == 0:
            summary = f"Coverage: **no columns resolved**, confidence `{confidence}`."
        else:
            summary = (
                f"Coverage: **{coverage.typed_ratio:.0%} typed**, "
                f"{coverage.traced_ratio:.0%} traced, confidence `{confidence}`."
            )
        head = [
            "## rowless",
            "",
            f"{len(report.failures)} finding(s) across {report.pipelines_checked} pipeline(s).",
            "",
            summary,
            "",
        ]
        body: list[str] = []
        for diagnostic in report.diagnostics:
            summary = (
                f"<code>{diagnostic.code.value}</code> "
                f"{diagnostic.level.value}: {diagnostic.message}"
            )
            body.append(f"<details><summary>{summary}</summary>")
            body.append("")
            if diagnostic.origin is not None:
                body.append(f"`{diagnostic.origin}`")
                body.append("")
            if diagnostic.detail:
                body.append("```")
                body.append(diagnostic.detail)
                body.append("```")
            for related in diagnostic.related:
                body.append(f"- upstream `{related}`")
            if diagnostic.suggestion:
                body.append(f"- {diagnostic.suggestion}")
            body.append("</details>")
            body.append("")
        for reason in coverage.reasons:
            body.append(f"> {reason}")
        rendered = "\n".join(head + body)
        if len(rendered) <= MARKDOWN_LIMIT:
            return rendered
        kept = rendered[:MARKDOWN_LIMIT]
        return f"{kept}\n\n_Truncated: run `rowless check` locally for the full report._"


class SarifRenderer:
    """SARIF 2.1.0, for GitHub code scanning and other security dashboards."""

    def render(self, report: Report) -> str:
        """Emit a SARIF log."""
        rules = {
            diagnostic.code.value: {
                "id": diagnostic.code.value,
                "name": diagnostic.code.name,
                "shortDescription": {"text": diagnostic.code.name.replace("_", " ").lower()},
            }
            for diagnostic in report.diagnostics
        }
        log = {
            "version": "2.1.0",
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "rowless",
                            "informationUri": "https://github.com/kauvinlucas/rowless",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": [_sarif_result(d) for d in report.diagnostics],
                }
            ],
        }
        return json.dumps(log, indent=2, sort_keys=True)


class JUnitRenderer:
    """JUnit XML, so any CI system shows findings as test failures."""

    def render(self, report: Report) -> str:
        """Emit a JUnit testsuite."""
        failures = report.failures
        suite = ET.Element(
            "testsuite",
            name="rowless",
            tests=str(max(len(report.diagnostics), report.pipelines_checked)),
            failures=str(len(failures)),
            skipped=str(report.pipelines_skipped),
            time=f"{report.duration_ms / 1000:.3f}",
        )
        for diagnostic in report.diagnostics:
            case = ET.SubElement(
                suite,
                "testcase",
                classname=str(diagnostic.origin) if diagnostic.origin else "rowless",
                name=f"{diagnostic.code.value} {diagnostic.message}",
            )
            if diagnostic.fails_run:
                failure = ET.SubElement(
                    case,
                    "failure",
                    type=diagnostic.code.value,
                    message=diagnostic.message,
                )
                failure.text = _one_line(diagnostic)
        return ET.tostring(suite, encoding="unicode")


def _diagnostic_to_dict(diagnostic: Diagnostic) -> dict[str, Any]:
    """The JSON shape editors and the GitHub Action parse."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "diagnostic",
        "code": diagnostic.code.value,
        "level": diagnostic.level.value,
        "message": diagnostic.message,
        "origin": origin_to_dict(diagnostic.origin) if diagnostic.origin else None,
        "related": [origin_to_dict(related) for related in diagnostic.related],
        "detail": diagnostic.detail,
        "suggestion": diagnostic.suggestion,
    }


def _summary_to_dict(report: Report) -> dict[str, Any]:
    """The trailing summary record: counts, coverage and the exit code."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "summary",
        "pipelines_checked": report.pipelines_checked,
        "pipelines_skipped": report.pipelines_skipped,
        "findings": len(report.diagnostics),
        "failures": len(report.failures),
        "coverage": coverage_to_dict(report.coverage),
        "duration_ms": report.duration_ms,
        "exit_code": report.exit_code,
    }


def _sarif_result(diagnostic: Diagnostic) -> dict[str, Any]:
    """One SARIF result, located when rowless knows where to point."""
    result: dict[str, Any] = {
        "ruleId": diagnostic.code.value,
        "level": _SARIF_LEVEL[diagnostic.level],
        "message": {"text": _one_line(diagnostic)},
    }
    origin = diagnostic.origin
    if origin is not None and origin.file:
        region: dict[str, Any] = {}
        if origin.line is not None:
            region["startLine"] = origin.line
        if origin.column is not None:
            region["startColumn"] = origin.column
        result["locations"] = [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": origin.file},
                    **({"region": region} if region else {}),
                }
            }
        ]
    return result


def _one_line(diagnostic: Diagnostic) -> str:
    """Message, detail and suggestion flattened onto one line.

    Annotation formats take a single string, and dropping the detail there
    would hide the upstream cause, which is the half users act on.
    """
    parts = [diagnostic.message]
    if diagnostic.detail:
        parts.append(diagnostic.detail.replace("\n", " "))
    if diagnostic.suggestion:
        parts.append(diagnostic.suggestion)
    return " - ".join(parts)


RENDERERS: dict[str, type] = {
    "terminal": TerminalRenderer,
    "json": JsonRenderer,
    "github": GitHubRenderer,
    "markdown": MarkdownRenderer,
    "sarif": SarifRenderer,
    "junit": JUnitRenderer,
}

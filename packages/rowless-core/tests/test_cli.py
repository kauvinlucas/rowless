"""The command line: exit codes, output formats and the honest empty cases.

The exit-code contract is the part CI depends on, so it is asserted here
rather than inferred: 0 clean, 1 findings, 2 rowless itself failed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rowless.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_help_lists_every_command() -> None:
    """A user's first contact with the tool."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("check", "diff", "explain", "graph", "list", "adapters", "doctor"):
        assert command in result.output


def test_check_on_an_empty_tree_is_loud_rather_than_green(tmp_path: Path) -> None:
    """Nothing checked must not look like nothing wrong."""
    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 0, "a repo with no pipelines is not a failure"
    assert "no pipelines found" in result.output


def test_check_reports_the_finding_and_exits_one(project: Path, in_process_cli: None) -> None:
    """The proposal's first terminal block, with the exit code CI reads."""
    result = runner.invoke(app, ["check", str(project)])

    assert result.exit_code == 1
    assert "ColumnNotFound" in result.output
    assert "amount_cents" in result.output
    assert "coverage:" in result.output


def test_check_json_output_is_parseable(project: Path, in_process_cli: None) -> None:
    """The format the LSP and the GitHub Action consume."""
    result = runner.invoke(app, ["check", str(project), "--output", "json"])

    lines = [json.loads(line) for line in result.output.strip().splitlines()]
    assert lines[0]["code"] == "RWL101"
    assert lines[-1]["kind"] == "summary"
    assert lines[-1]["exit_code"] == 1


def test_check_rejects_an_unknown_output_format(project: Path) -> None:
    """A typo in a flag is rowless failing to run, which is exit 2."""
    result = runner.invoke(app, ["check", str(project), "--output", "yaml"])

    assert result.exit_code == 2
    assert "--output must be one of" in result.output


def test_list_shows_what_was_found_and_what_is_unbound(project: Path) -> None:
    """The first thing to run in a new repo."""
    result = runner.invoke(app, ["list", "--output", "json"])

    assert result.exit_code == 0
    listed = json.loads(result.output)
    assert any(entry["target"].endswith(":daily_revenue") for entry in listed)


def test_adapters_explains_why_one_is_inactive() -> None:
    """A coverage score means something different with no adapter installed."""
    result = runner.invoke(app, ["adapters"])

    assert result.exit_code == 0
    assert "polars" in result.output
    assert "inactive" in result.output


def test_doctor_reports_every_registry() -> None:
    """Setup problems are the first thing a new user hits."""
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    for group in ("engines", "sources", "discovery", "emitters"):
        assert group in result.output
    assert "Exit codes" in result.output


def test_graph_exports_the_lineage_it_resolved(project: Path, in_process_cli: None) -> None:
    """The graph is how rowless hands its findings to a catalog."""
    result = runner.invoke(app, ["graph", "--output", "json"])

    assert result.exit_code == 0
    exported = json.loads(result.output)
    assert exported["version"] == 1


def test_graph_rejects_an_unknown_format(project: Path) -> None:
    """Same contract as check: a bad flag is exit 2."""
    result = runner.invoke(app, ["graph", "--output", "svg"])

    assert result.exit_code == 2


def test_emit_refuses_an_unusable_emitter() -> None:
    """Pointing at what is missing beats an empty traceback.

    In this environment the OpenLineage emitter is registered but cannot run,
    which is the same shape as a user who installed the package without
    configuring a transport.
    """
    result = runner.invoke(app, ["emit"])

    assert result.exit_code == 2
    assert "openlineage" in result.output

    unknown = runner.invoke(app, ["emit", "--output", "nowhere"])
    assert unknown.exit_code == 2
    assert "no emitter named" in unknown.output


@pytest.fixture
def in_process_cli(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the CLI against `project`, with dry runs in this process.

    The CLI resolves config by walking up from the path it is given, so the
    working directory moves too; and a fake adapter cannot be registered in a
    worker subprocess, so the dry run is redirected the same way
    `test_check.py` does it.
    """
    import importlib
    import sys

    from rowless.adapters.registry import clear_cache
    from rowless.runtime.dryrun import DryRunRequest, DryRunResult, dry_run_target
    from rowless_fake import install

    clear_cache()
    install()
    monkeypatch.chdir(project)

    def _run(request: DryRunRequest) -> DryRunResult:
        for entry in request.sys_path:
            if entry not in sys.path:
                sys.path.insert(0, entry)
        sys.modules.pop(request.target.partition(":")[0], None)
        importlib.invalidate_caches()
        return dry_run_target(
            request.target,
            dict(request.inputs),
            engine=request.engine,
            options=request.options,
            uris=request.uris,
        )

    monkeypatch.setattr("rowless.check.dry_run", _run)

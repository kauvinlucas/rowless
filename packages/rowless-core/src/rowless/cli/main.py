"""The `rowless` command.

Exit codes are part of the contract: 0 clean, 1 findings, 2 rowless failed.
CI needs to tell a broken pipeline from a broken tool.

Every command here is thin. Discovery, resolution and scoring live in
`rowless.check`; this module parses flags, picks a renderer and sets the exit
code, so a second front end (the LSP, the GitHub Action) gets the same
behaviour without going through argument parsing.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from rowless.adapters.registry import probe_plugins
from rowless.check import run
from rowless.config import Config
from rowless.diagnostics.codes import Code
from rowless.diagnostics.models import Diagnostic, Level, Report
from rowless.diagnostics.render import RENDERERS, TerminalRenderer
from rowless.discovery.base import discover
from rowless.errors import RowlessError
from rowless.ir.diff import SEVERITY_ORDER, Severity
from rowless.lineage.blast_radius import build_base_graph, compare
from rowless.lineage.graph import ColumnRef

app = typer.Typer(
    name="rowless",
    help="Dry-run type checker for data pipelines. Schemas without data.",
    no_args_is_help=True,
    add_completion=True,
)

#: Exit code for rowless failing, as opposed to finding something. Kept here
#: because every command path has to agree on it.
EXIT_INTERNAL = 2

#: How long the watcher waits between scans. Long enough not to spin a laptop
#: fan, short enough that a save feels answered.
WATCH_INTERVAL_S = 1.0

_SEVERITY_GATES: dict[str, frozenset[Severity]] = {
    "breaking": frozenset({Severity.BREAKING, Severity.UNVERIFIED}),
    "risky": frozenset({Severity.BREAKING, Severity.UNVERIFIED, Severity.RISKY}),
    "any": frozenset(SEVERITY_ORDER),
    "never": frozenset(),
}


@app.command()
def check(
    paths: Annotated[list[str] | None, typer.Argument(help="Files or directories.")] = None,
    select: Annotated[str | None, typer.Option(help="Only pipelines matching this glob.")] = None,
    engine: Annotated[list[str] | None, typer.Option(help="Restrict to these adapters.")] = None,
    output: Annotated[str, typer.Option(help="terminal, json, github, sarif, junit.")] = "terminal",
    min_coverage: Annotated[float | None, typer.Option(help="Fail below this ratio.")] = None,
    watch: Annotated[bool, typer.Option(help="Re-check on file change.")] = False,
    jobs: Annotated[int, typer.Option(help="Parallel dry runs. 0 means CPU count.")] = 0,
) -> None:
    """Check the working tree. The command in the proposal's first terminal block."""
    config = _config(paths, engine=engine, min_coverage=min_coverage)
    if watch:
        _watch(config, paths=paths, select=select, output=output, jobs=jobs)
        return
    report = _checked(config, paths=paths, select=select, jobs=jobs)
    _emit_report(report, output)
    raise typer.Exit(report.exit_code)


@app.command()
def diff(
    ref: Annotated[str, typer.Argument(help="Git ref to compare against, e.g. main.")],
    output: Annotated[str, typer.Option(help="terminal, json, markdown, sarif.")] = "terminal",
    fail_on: Annotated[str, typer.Option(help="breaking, risky, any, never.")] = "breaking",
) -> None:
    """Report the cross-language blast radius of schema changes versus `ref`."""
    if fail_on not in _SEVERITY_GATES:
        _fail(f"--fail-on must be one of {', '.join(_SEVERITY_GATES)}")
    config = _config(None)
    try:
        outcome = run(config)
        base = build_base_graph(ref, cache_dir=str(config.cache_path))
    except RowlessError as exc:
        _fail(str(exc))
    radius = compare(base, outcome.graph, base_ref=ref, head_ref="working tree")

    gate = _SEVERITY_GATES[fail_on]
    diagnostics = tuple(
        Diagnostic.from_change(impact.change, relation=impact.relation) for impact in radius.impacts
    )
    diagnostics += tuple(
        Diagnostic(
            code=Code.UNVERIFIED_CHANGE,
            level=Level.UNVERIFIED,
            message=f"impact of {impact.relation}.{impact.change.column} may be larger",
            detail="passes through: " + ", ".join(impact.unverified_paths),
        )
        for impact in radius.impacts
        if impact.unverified_paths
    )
    # A pipeline the head run could not check is a hole in the comparison, so
    # its findings travel with the diff. Without them, a diff over a repository
    # half of which never resolved would print "no breaking changes".
    blocked = tuple(diagnostic for diagnostic in outcome.report.diagnostics if diagnostic.fails_run)
    report = Report(
        diagnostics=blocked + diagnostics,
        pipelines_checked=outcome.report.pipelines_checked,
        pipelines_skipped=outcome.report.pipelines_skipped,
        coverage=outcome.report.coverage,
    )
    _emit_report(report, output)
    gated = any(impact.change.severity in gate for impact in radius.impacts)
    raise typer.Exit(max(1 if gated else 0, report.exit_code))


@app.command()
def explain(
    target: Annotated[str, typer.Argument(help="Pipeline, or relation.column.")],
) -> None:
    """Show how one schema or column was derived.

    The debugging command, and the honesty command: it prints each lineage
    edge with the evidence behind it, every ANY column, and every reason
    confidence was lost. If rowless is wrong about something, this is where a
    user finds out why.
    """
    console = Console()
    outcome = run(_config(None))
    node_id, _, column = target.rpartition(".")
    candidates = [name for name in outcome.graph.nodes if target in name or node_id in name]
    if not candidates:
        _fail(f"nothing named {target!r} was checked. Try `rowless list`.")

    for name in candidates:
        node = outcome.graph.nodes[name]
        engine_note = f", {node.engine}" if node.engine else ""
        console.print(f"[bold]{name}[/] ({node.kind.value}{engine_note})")
        if node.opaque:
            console.print("  [magenta]opaque[/]: rowless could not see inside this node")
        for field in node.schema.fields:
            resolved = field.dtype.is_known and not field.dtype.is_any
            marker = "" if resolved else "  [yellow]<-- unverified[/]"
            console.print(f"  {field.name}: {field.dtype}{marker}")
            for edge in outcome.graph.edges_into(ColumnRef(name, field.name)):
                transform = f" via {edge.transform}" if edge.transform else ""
                console.print(f"      from {edge.source} ({edge.evidence.value}{transform})")
            if column and field.name == column and field.dtype.reason:
                console.print(f"      reason: {field.dtype.reason}")

    for result in outcome.results:
        if not result.coverage.reasons:
            continue
        console.print(f"\n[bold]{result.target}[/] lost confidence because:")
        for reason in result.coverage.reasons:
            console.print(f"  - {reason}")
        if result.output is not None:
            for note in result.output.notes:
                console.print(f"  note: {note}")


@app.command()
def graph(
    output: Annotated[str, typer.Option(help="dot, json, mermaid.")] = "dot",
    focus: Annotated[str | None, typer.Option(help="Limit to this node's neighbours.")] = None,
) -> None:
    """Export the column lineage graph."""
    built = run(_config(None)).graph
    if focus is not None:
        built = _focused(built, focus)
    writers = {"dot": built.to_dot, "json": built.to_json, "mermaid": built.to_mermaid}
    if output not in writers:
        _fail(f"--output must be one of {', '.join(writers)}")
    typer.echo(writers[output]())


@app.command(name="list")
def list_pipelines(
    output: Annotated[str, typer.Option(help="terminal, json.")] = "terminal",
) -> None:
    """List discovered pipelines and their resolved inputs.

    The first thing to run in a new repo: it shows what rowless found, what it
    could not bind, and why.
    """
    config = _config(None)
    found = discover(
        [str(root) for root in config.resolved_roots()],
        finders=config.finders or None,
        exclude=config.exclude,
    )
    if output == "json":
        import json

        typer.echo(
            json.dumps(
                [
                    {
                        "target": pipeline.target,
                        "framework": pipeline.framework,
                        "engine": pipeline.engine,
                        "file": pipeline.file,
                        "line": pipeline.line,
                        "inputs": dict(pipeline.inputs),
                        "unbound": list(pipeline.unbound),
                    }
                    for pipeline in found
                ],
                indent=2,
            )
        )
        return

    table = Table("pipeline", "engine", "framework", "inputs", "location")
    for pipeline in found:
        inputs = ", ".join(
            f"{name}={uri}" if uri else f"[yellow]{name}=?[/]" for name, uri in pipeline.inputs
        )
        table.add_row(
            pipeline.name,
            pipeline.engine or "[yellow]unknown[/]",
            pipeline.framework,
            inputs or "-",
            f"{pipeline.file}:{pipeline.line}",
        )
    console = Console()
    console.print(table)
    if not found:
        console.print("[yellow]no pipelines found[/]; see docs/writing-an-adapter.md")


@app.command()
def adapters() -> None:
    """List installed adapters, their capabilities, and why any are inactive."""
    console = Console()
    table = Table("adapter", "status", "distribution", "capabilities")
    for info in probe_plugins("rowless.engines"):
        status = "[green]active[/]" if info.loaded else f"[yellow]inactive[/] ({info.error})"
        table.add_row(
            info.name,
            status,
            info.distribution or "-",
            _capabilities(info.name) if info.loaded else "-",
        )
    console.print(table)
    if not table.row_count:
        console.print("[yellow]no adapters installed[/]: try `pip install rowless-polars`")


@app.command()
def snapshot(
    dsn: Annotated[str, typer.Argument(help="Warehouse connection string.")],
    out: Annotated[str, typer.Option(help="Output path.")] = "rowless-catalog.json",
    include: Annotated[list[str] | None, typer.Option(help="Table glob to include.")] = None,
) -> None:
    """Write a committable catalog snapshot, so CI runs without credentials."""
    from rowless.sources.information_schema import write_snapshot

    try:
        write_snapshot(dsn, out, include=tuple(include or ()))
    except ImportError:
        _fail("snapshots need SQLAlchemy: pip install 'rowless[warehouse]'")
    except Exception as exc:
        _fail(f"snapshot failed: {exc}")
    typer.echo(f"wrote {out}. Commit it so CI needs no credentials.")


@app.command()
def doctor() -> None:
    """Diagnose setup: adapters, sources, config, cache, engine versions."""
    console = Console()
    config = Config.discover()
    console.print(f"[bold]rowless[/] {_version()} on Python {_python_version()}")
    console.print(f"config root: {config.root_dir}")
    cache_state = "present" if config.cache_path.exists() else "empty"
    console.print(f"cache: {config.cache_path} ({cache_state})")
    console.print(f"roots: {', '.join(config.roots)}")

    for group, title in (
        ("rowless.engines", "engines"),
        ("rowless.sources", "sources"),
        ("rowless.discovery", "discovery"),
        ("rowless.emitters", "emitters"),
    ):
        table = Table(title, "status", "detail")
        for info in probe_plugins(group):  # type: ignore[arg-type]
            table.add_row(
                info.name,
                "[green]ok[/]" if info.loaded else "[yellow]inactive[/]",
                info.error or "-",
            )
        console.print(table)
    console.print(
        "Exit codes: 0 clean, 1 findings, 2 rowless itself failed.",
        style="dim",
    )


@app.command()
def emit(
    output: Annotated[str, typer.Option(help="Emitter to use. Default: openlineage.")] = (
        "openlineage"
    ),
) -> None:
    """Emit column-level lineage events to an OpenLineage backend.

    The coverage score travels with the events: a catalog that cannot tell
    verified lineage from partially guessed lineage will present both as
    equally solid, which is the one way this feature could mislead people.
    """
    from rowless.adapters.registry import load, registered_names

    names = sorted(registered_names("rowless.emitters"))
    if not names:
        _fail("no emitters installed: pip install rowless-openlineage")
    if output not in names:
        _fail(f"no emitter named {output!r}; installed: {', '.join(names)}")

    try:
        emitter = load("rowless.emitters", output)
    except RowlessError as exc:
        _fail(str(exc))
    if not _emitter_available(emitter):
        requires = getattr(emitter, "requires", "")
        _fail(f"emitter {output!r} is not usable here; it needs {requires or 'configuration'}")

    outcome = run(_config(None))
    emitter.emit(outcome.graph, outcome.report.coverage)
    typer.echo(f"emitted lineage for {len(outcome.graph.nodes)} node(s) via {output}")


def _emitter_available(emitter: Any) -> bool:
    """Whether an emitter can run, treating a raise as a no."""
    checker = getattr(emitter, "available", None)
    if checker is None:
        return True
    try:
        return bool(checker())
    except Exception:
        return False


# --- Shared plumbing ---------------------------------------------------------


def _config(
    paths: Sequence[str] | None,
    *,
    engine: Sequence[str] | None = None,
    min_coverage: float | None = None,
) -> Config:
    """Load configuration, with CLI flags on top."""
    start = paths[0] if paths else "."
    try:
        config = Config.discover(start)
    except RowlessError as exc:
        _fail(str(exc))
    overrides: dict[str, object] = {}
    if engine:
        overrides["engines"] = tuple(engine)
    if min_coverage is not None:
        overrides["min_coverage"] = min_coverage
    return config.with_overrides(**overrides)


def _checked(
    config: Config,
    *,
    paths: Sequence[str] | None,
    select: str | None,
    jobs: int,
) -> Report:
    """Run a check, turning a rowless failure into an exit-2 report."""
    try:
        return run(config, paths=paths, select=select, jobs=jobs).report
    except RowlessError as exc:
        return Report(
            diagnostics=(Diagnostic(code=Code.ADAPTER_ERROR, level=Level.ERROR, message=str(exc)),),
        )


def _emit_report(report: Report, output: str) -> None:
    """Render a report in the requested format."""
    if output not in RENDERERS:
        _fail(f"--output must be one of {', '.join(RENDERERS)}")
    console = Console()
    renderer: Any
    if output == "terminal":
        renderer = TerminalRenderer(color=console.is_terminal)
        console.print(renderer.render(report), markup=console.is_terminal, highlight=False)
        return
    renderer = RENDERERS[output]()
    typer.echo(renderer.render(report))


def _watch(
    config: Config,
    *,
    paths: Sequence[str] | None,
    select: str | None,
    output: str,
    jobs: int,
) -> None:
    """Re-check whenever a watched file changes.

    Polling rather than an OS watcher: one less dependency, and a check that
    takes milliseconds does not need sub-second change detection.
    """
    console = Console()
    previous: dict[str, float] = {}
    while True:
        current = _mtimes(config)
        if current != previous:
            console.clear()
            report = _checked(config, paths=paths, select=select, jobs=jobs)
            _emit_report(report, output)
            previous = current
        try:
            time.sleep(WATCH_INTERVAL_S)
        except KeyboardInterrupt:  # pragma: no cover - interactive only
            return


def _mtimes(config: Config) -> dict[str, float]:
    """Modification times of every Python file under the configured roots."""
    found: dict[str, float] = {}
    for root in config.resolved_roots():
        for path in Path(root).rglob("*.py"):
            if config.is_excluded(path):
                continue
            try:
                found[str(path)] = path.stat().st_mtime
            except OSError:
                continue
    return found


def _focused(built: Any, focus: str) -> Any:
    """A subgraph of `built` around `focus`, for a readable export."""
    from rowless.lineage.graph import LineageGraph

    matches = [name for name in built.nodes if focus in name]
    if not matches:
        _fail(f"no node matching {focus!r}")
    keep = set(matches)
    for edge in built.edges:
        if edge.source.node in matches:
            keep.add(edge.target.node)
        if edge.target.node in matches:
            keep.add(edge.source.node)
    narrowed = LineageGraph()
    for name in keep:
        narrowed.add_node(built.nodes[name])
    for edge in built.edges:
        if edge.source.node in keep and edge.target.node in keep:
            narrowed.add_edge(edge)
    return narrowed


def _capabilities(name: str) -> str:
    """An adapter's declared capabilities, as a readable list."""
    from rowless.adapters.registry import load

    try:
        adapter = load("rowless.engines", name)
    except RowlessError:
        return "-"
    declared = getattr(adapter, "capabilities", None)
    if declared is None:
        return "-"
    flags = [
        flag
        for flag in ("column_lineage", "sql", "udf_probe", "lazy", "nested_types", "remote")
        if getattr(declared, flag, False)
    ]
    return ", ".join(flags) or "none declared"


def _version() -> str:
    """The installed rowless version."""
    from rowless import __version__

    return __version__


def _python_version() -> str:
    """The interpreter running the check, which is the one adapters import."""
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _fail(message: str) -> Any:
    """Report a setup problem and exit 2: rowless could not do its job."""
    Console(stderr=True).print(f"[red]rowless: {message}[/]")
    raise typer.Exit(EXIT_INTERNAL)


if __name__ == "__main__":
    app()

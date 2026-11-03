"""One run of rowless: discover, resolve, dry-run, score.

The layer between the CLI and the runtime. It exists so that `rowless check`,
the pytest plugin, the LSP and `rowless diff`'s base-revision build all take
the same path: discovery, then source resolution in the parent process, then
one sandboxed dry run per pipeline, then one report.

Nothing here knows about an engine, and nothing here prints. The CLI picks a
renderer; this module produces the `Report` and the lineage graph.
"""

from __future__ import annotations

import fnmatch
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from rowless.config import Config, inline_ignores
from rowless.diagnostics.codes import Code
from rowless.diagnostics.models import Diagnostic, Level, Report
from rowless.discovery.base import Pipeline, discover, import_root_for
from rowless.errors import RowlessError
from rowless.ir.confidence import Confidence, Coverage
from rowless.ir.schema import Origin, Schema
from rowless.lineage.graph import ColumnRef, Edge, Evidence, LineageGraph, Node, NodeKind
from rowless.runtime.dryrun import DryRunRequest, DryRunResult, dry_run
from rowless.sources.base import SourceCache
from rowless.sources.base import resolve as resolve_source


@dataclass(frozen=True, slots=True)
class Outcome:
    """Everything one run produced: findings, coverage and the column graph."""

    report: Report
    graph: LineageGraph = field(default_factory=LineageGraph)
    results: tuple[DryRunResult, ...] = ()
    pipelines: tuple[Pipeline, ...] = ()


def check(
    config: Config,
    *,
    paths: Sequence[str] | None = None,
    select: str | None = None,
    jobs: int = 1,
) -> Report:
    """Check a working tree and report. What `rowless check` calls."""
    return run(config, paths=paths, select=select, jobs=jobs).report


def build_graph(config: Config) -> LineageGraph:
    """The column graph for a working tree. What `rowless diff` calls per side."""
    return run(config).graph


def run(
    config: Config,
    *,
    paths: Sequence[str] | None = None,
    select: str | None = None,
    jobs: int = 1,
) -> Outcome:
    """Do the work once, and return every view of it.

    Sources are resolved here, in the parent process, before any sandbox
    exists: a source is the one part of rowless that is allowed to touch the
    network, and the dry run itself must not need to.

    Dry runs then go out in parallel when `jobs > 1`. They are subprocesses,
    so threads are enough to overlap them, and the parent stays single-
    threaded for everything that touches the cache.
    """
    started = time.perf_counter()
    roots = [str(root) for root in (paths or config.resolved_roots())]
    pipelines = [
        pipeline
        for pipeline in discover(roots, finders=config.finders or None, exclude=config.exclude)
        if _wanted(pipeline, config, select)
    ]
    if not pipelines:
        return Outcome(report=_nothing_found(roots))

    cache = SourceCache(str(config.cache_path))
    diagnostics: list[Diagnostic] = []
    runnable: list[tuple[Pipeline, dict[str, Schema]]] = []
    skipped = 0
    for pipeline in pipelines:
        inputs, failures = _resolve_inputs(pipeline, config, cache)
        if failures:
            diagnostics.extend(failures)
            skipped += 1
            continue
        runnable.append((pipeline, inputs))

    results = _dry_run_all(runnable, config, jobs=jobs)

    graph = LineageGraph()
    coverage = Coverage(confidence=Confidence.EXACT)
    for (pipeline, inputs), result in zip(runnable, results, strict=True):
        diagnostics.extend(_located(result.diagnostics, pipeline))
        coverage = coverage.merge(result.coverage)
        _record(graph, pipeline, inputs, result)

    if skipped:
        # An aggregate score computed only over the pipelines that worked
        # would improve every time rowless failed to check something.
        coverage = coverage.degraded(f"{skipped} pipeline(s) were not checked")
    report = Report(
        diagnostics=tuple(_apply_policy(diagnostics, config)),
        pipelines_checked=len(results),
        pipelines_skipped=skipped,
        coverage=coverage,
        duration_ms=(time.perf_counter() - started) * 1000,
    )
    return Outcome(
        report=_enforce_coverage(report, config),
        graph=graph,
        results=tuple(results),
        pipelines=tuple(pipelines),
    )


def _dry_run_all(
    runnable: Sequence[tuple[Pipeline, dict[str, Schema]]],
    config: Config,
    *,
    jobs: int,
) -> list[DryRunResult]:
    """One dry run per pipeline, in order, optionally overlapped."""
    requests = [_request_for(pipeline, inputs, config) for pipeline, inputs in runnable]
    workers = (os.cpu_count() or 1) if jobs == 0 else max(jobs, 1)
    if workers == 1 or len(requests) < 2:
        return [dry_run(request) for request in requests]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(dry_run, requests))


def _request_for(
    pipeline: Pipeline,
    inputs: dict[str, Schema],
    config: Config,
) -> DryRunRequest:
    """The request one pipeline's dry run needs."""
    return DryRunRequest(
        target=pipeline.target,
        inputs=inputs,
        engine=pipeline.engine,
        options=config.engine_options.get(pipeline.engine or "", {}),
        # The URI as the pipeline wrote it, not the rewritten one: a
        # module-level `rowless.source("lake://orders")` looks itself up by
        # the string in the source, and `[sources]` rewrites are config's
        # business rather than the pipeline's.
        uris={parameter: uri for parameter, uri in pipeline.inputs if uri is not None},
        sys_path=_sys_path_for(pipeline),
        probe_udfs=config.probe_udfs,
    )


def _wanted(pipeline: Pipeline, config: Config, select: str | None) -> bool:
    """Filter by `--select`, by configured engines, and by the exclude globs."""
    if pipeline.file is not None and config.is_excluded(Path(pipeline.file)):
        return False
    if config.engines and pipeline.engine is not None and pipeline.engine not in config.engines:
        return False
    if select is None:
        return True
    return any(
        fnmatch.fnmatch(candidate, select)
        for candidate in (pipeline.target, pipeline.name, pipeline.file or "")
    )


def _resolve_inputs(
    pipeline: Pipeline,
    config: Config,
    cache: SourceCache,
) -> tuple[dict[str, Schema], list[Diagnostic]]:
    """Resolve every declared input, or explain why the pipeline is skipped.

    A pipeline rowless cannot feed is reported, never quietly dropped: a
    silent skip is indistinguishable from a pass, which is the failure mode
    this project is built to avoid.
    """
    origin = Origin(kind="python", ref=pipeline.target, file=pipeline.file, line=pipeline.line)
    schemas: dict[str, Schema] = {}
    failures: list[Diagnostic] = []
    for parameter, uri in pipeline.inputs:
        if uri is None:
            failures.append(
                Diagnostic(
                    code=Code.UNRESOLVED_SOURCE,
                    level=Level.UNVERIFIED,
                    message=f"no source for input {parameter!r} of {pipeline.name}",
                    origin=origin,
                    detail=(
                        "rowless could not work out which table this parameter reads. "
                        "Declare it with @rowless.pipeline(inputs={...}) or a [sources] rewrite."
                    ),
                )
            )
            continue
        resolved = config.resolve_source(uri)
        try:
            schemas[parameter] = resolve_source(resolved, cache=cache, base_dir=config.root_dir)
        except RowlessError as exc:
            failures.append(
                Diagnostic(
                    code=exc.code,
                    level=Level.UNVERIFIED,
                    message=f"cannot resolve {uri} for {pipeline.name}",
                    origin=origin,
                    detail=str(exc),
                )
            )
    return schemas, failures


def _sys_path_for(pipeline: Pipeline) -> tuple[str, ...]:
    """Import roots the worker needs to import this pipeline.

    A repository under check is usually not installed, so the directory that
    makes its module path importable travels with the request instead of being
    guessed in the child.
    """
    if pipeline.file is None:
        return ()
    return (str(import_root_for(pipeline.file)),)


def _record(
    graph: LineageGraph,
    pipeline: Pipeline,
    inputs: dict[str, Schema],
    result: DryRunResult,
) -> None:
    """Add one pipeline and its inputs to the column graph."""
    bindings: dict[str, str] = {}
    for parameter, uri in pipeline.inputs:
        schema = inputs.get(parameter)
        if schema is None:
            continue
        node_id = uri or parameter
        graph.add_node(
            Node(
                id=node_id,
                kind=NodeKind.SOURCE,
                schema=schema,
                origin=Origin(kind="source", ref=node_id),
            )
        )
        bindings[parameter] = node_id

    if result.output is None:
        # A failed dry run is still a node: an opaque one. Leaving it out
        # would make the graph look complete and the coverage look better
        # than it is.
        graph.add_node(
            Node(
                id=pipeline.target,
                kind=NodeKind.PYTHON,
                schema=Schema.empty(name=pipeline.name),
                engine=pipeline.engine,
                origin=Origin(
                    kind="python",
                    ref=pipeline.target,
                    file=pipeline.file,
                    line=pipeline.line,
                ),
                opaque=True,
            )
        )
        return

    traced = result.coverage.traced_ratio
    graph.add_node(
        Node(
            id=pipeline.target,
            kind=NodeKind.PYTHON,
            schema=result.output.schema,
            engine=result.output.engine,
            origin=Origin(
                kind="python",
                ref=pipeline.target,
                file=pipeline.file,
                line=pipeline.line,
            ),
            opaque=result.coverage.confidence is Confidence.UNKNOWN or traced == 0.0,
        )
    )
    graph.link_by_lineage(pipeline.target, bindings)
    _link_by_name(graph, pipeline.target, bindings, result)


def _link_by_name(
    graph: LineageGraph,
    node_id: str,
    bindings: dict[str, str],
    result: DryRunResult,
) -> None:
    """Fall back to name matching for columns the adapter did not trace.

    Labelled `NAME_MATCH`, which is excluded from breaking-change claims by
    default: a column that kept its name probably passed through, but
    "probably" is not what a CI gate should act on.
    """
    if result.output is None:
        return
    for column in result.output.schema.fields:
        if column.derived_from:
            continue
        for source_node in bindings.values():
            upstream = graph.nodes[source_node]
            if column.name in upstream.schema:
                graph.add_edge(
                    Edge(
                        source=ColumnRef(source_node, column.name),
                        target=ColumnRef(node_id, column.name),
                        evidence=Evidence.NAME_MATCH,
                    )
                )


def _located(diagnostics: Sequence[Diagnostic], pipeline: Pipeline) -> list[Diagnostic]:
    """Fill in a file and line from the pipeline when a finding has none.

    The runtime points at the user's frame when a traceback reaches it. When
    it does not - a schema that resolved to something wrong rather than
    raising - the pipeline's own definition is the next best place to put the
    squiggle, and a finding with no location cannot be suppressed or clicked.
    """
    located: list[Diagnostic] = []
    for diagnostic in diagnostics:
        origin = diagnostic.origin
        if origin is not None and origin.file:
            located.append(diagnostic)
            continue
        located.append(
            Diagnostic(
                code=diagnostic.code,
                level=diagnostic.level,
                message=diagnostic.message,
                origin=Origin(
                    kind="python",
                    ref=pipeline.target,
                    file=pipeline.file,
                    line=pipeline.line,
                ),
                related=diagnostic.related,
                detail=diagnostic.detail,
                suggestion=diagnostic.suggestion,
            )
        )
    return located


def _apply_policy(diagnostics: Sequence[Diagnostic], config: Config) -> list[Diagnostic]:
    """Apply configuration to raw findings: suppression and levels.

    The runtime decides *what* it found; configuration decides how loudly that
    is reported. Keeping the two apart is what lets `fail_on_unverified` and
    an inline ignore comment work without the runtime knowing they exist.
    """
    suppressed = _inline_suppressions(diagnostics)
    kept: list[Diagnostic] = []
    for diagnostic in diagnostics:
        level = config.level_for(diagnostic.code)
        if level is None:
            continue
        if _is_suppressed(diagnostic, suppressed):
            continue
        kept.append(diagnostic if level is diagnostic.level else _relevelled(diagnostic, level))
    return kept


def _is_suppressed(
    diagnostic: Diagnostic,
    suppressed: Mapping[tuple[str, int], tuple[Code, ...]],
) -> bool:
    """True when an inline comment turned this exact finding off."""
    origin = diagnostic.origin
    if origin is None or origin.file is None or origin.line is None:
        return False
    return diagnostic.code in suppressed.get((origin.file, origin.line), ())


def _relevelled(diagnostic: Diagnostic, level: Level) -> Diagnostic:
    """The same finding at a configured level."""
    return Diagnostic(
        code=diagnostic.code,
        level=level,
        message=diagnostic.message,
        origin=diagnostic.origin,
        related=diagnostic.related,
        detail=diagnostic.detail,
        suggestion=diagnostic.suggestion,
    )


def _inline_suppressions(
    diagnostics: Sequence[Diagnostic],
) -> dict[tuple[str, int], tuple[Code, ...]]:
    """Read `# rowless: ignore[...]` comments from the files that were flagged.

    Only the flagged files are read, so a clean repository pays nothing for
    the feature.
    """
    files = {
        diagnostic.origin.file
        for diagnostic in diagnostics
        if diagnostic.origin is not None and diagnostic.origin.file
    }
    found: dict[tuple[str, int], tuple[Code, ...]] = {}
    for file in files:
        try:
            for line, codes in inline_ignores(file).items():
                found[(file, line)] = tuple(codes)
        except RowlessError:
            continue
    return found


def _enforce_coverage(report: Report, config: Config) -> Report:
    """Add the coverage finding when typed coverage is under the floor."""
    if config.min_coverage <= 0.0 or not report.pipelines_checked:
        return report
    ratio = report.coverage.typed_ratio
    if ratio >= config.min_coverage:
        return report
    level = config.level_for(Code.COVERAGE_BELOW_THRESHOLD) or Level.WARNING
    finding = Diagnostic(
        code=Code.COVERAGE_BELOW_THRESHOLD,
        level=level,
        message=f"typed coverage {ratio:.0%} is below the configured floor "
        f"{config.min_coverage:.0%}",
        detail="\n".join(report.coverage.reasons) or None,
    )
    return Report(
        diagnostics=(*report.diagnostics, finding),
        pipelines_checked=report.pipelines_checked,
        pipelines_skipped=report.pipelines_skipped,
        coverage=report.coverage,
        duration_ms=report.duration_ms,
    )


def _nothing_found(roots: Sequence[str]) -> Report:
    """The report for a repository rowless found no pipelines in.

    Reported rather than silently exiting 0: "no findings" and "nothing
    checked" look identical on a terminal and mean opposite things.
    """
    where = ", ".join(roots) or "."
    return Report(
        diagnostics=(
            Diagnostic(
                code=Code.NO_PIPELINES_FOUND,
                level=Level.WARNING,
                message=f"no pipelines found under {where}",
                detail=(
                    "rowless looks for annotated functions, @rowless.pipeline markers and "
                    "framework assets. Run `rowless list` to see what it found."
                ),
            ),
        ),
    )

"""The dry run: hand rowless frames to real user code, collect the schema.

This is the orchestration layer. It does not know about any engine; it asks the
registry for an adapter, asks the adapter to materialize inputs, calls the
user's function in a sandboxed subprocess, and asks the adapter what came back.
"""

from __future__ import annotations

import importlib
import inspect
import json
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

from rowless.adapters.base import DryRunContext, SupportsColumnLineage
from rowless.adapters.registry import engine as get_engine
from rowless.adapters.registry import engines
from rowless.diagnostics.codes import Code
from rowless.diagnostics.models import Diagnostic, Level
from rowless.errors import AdapterNotFound, RowlessError, UnknownSchemaError
from rowless.ir.confidence import Confidence, Coverage
from rowless.ir.schema import Field, Origin, Relation, Schema
from rowless.ir.serde import coverage_from_dict, relation_from_dict, schema_to_dict
from rowless.runtime import rowdep
from rowless.runtime.worker import WorkerLimits, spawn


@dataclass(frozen=True, slots=True)
class DryRunRequest:
    """One pipeline entry point plus the schemas to feed it."""

    target: str
    """Importable reference: `pipelines.revenue:daily_revenue`."""

    inputs: Mapping[str, Schema]
    """Parameter name to input schema. Resolved from sources beforehand, so the
    subprocess never has to reach the network to find out what a table holds."""

    engine: str | None = None
    """Adapter name. Inferred from annotations when omitted."""

    options: Mapping[str, Any] = field(default_factory=dict)

    uris: Mapping[str, str] = field(default_factory=dict)
    """Parameter name to the source URI it was resolved from, so a module-level
    `rowless.source("lake://orders")` in the pipeline's own module resolves to
    the schema the parent already fetched."""

    sys_path: tuple[str, ...] = ()
    """Directories the worker adds to `sys.path` before importing the target.
    A repository under check is usually not installed, and guessing the import
    root in the child would duplicate what discovery already worked out."""

    probe_udfs: bool = False
    """Passed through from config: calling a UDF with invented values is
    opt-in, because it runs user code with made-up input. The adapter reads it
    as `ctx.options["probe_udfs"]`, since probing is something only an adapter
    can actually do."""

    def to_json(self) -> str:
        """The wire format the worker reads on stdin."""
        return json.dumps(
            {
                "target": self.target,
                "engine": self.engine,
                "options": dict(self.options),
                "uris": dict(self.uris),
                "sys_path": list(self.sys_path),
                "probe_udfs": self.probe_udfs,
                "inputs": {name: schema_to_dict(schema) for name, schema in self.inputs.items()},
            }
        )


@dataclass(frozen=True, slots=True)
class DryRunResult:
    """What one dry run produced, successful or not."""

    target: str
    output: Relation | None
    coverage: Coverage
    diagnostics: tuple[Any, ...] = ()
    """`rowless.diagnostics.Diagnostic` values. Typed loosely here to keep the
    import graph acyclic."""

    duration_ms: float = 0.0
    """Kept because "at linter speed" is a promise in the proposal, and a
    regression here is a bug worth failing CI over."""

    side_effects: tuple[str, ...] = ()
    """Writes the pipeline attempted, which are its output tables."""

    @property
    def ok(self) -> bool:
        """True when a schema was resolved and nothing error-level was raised."""
        if self.output is None or self.coverage.confidence is Confidence.UNKNOWN:
            return False
        return not any(getattr(d, "fails_run", False) for d in self.diagnostics)

    def to_json(self) -> str:
        """The wire format the worker writes on stdout."""
        from rowless.ir.serde import coverage_to_dict, relation_to_dict

        return json.dumps(
            {
                "target": self.target,
                "output": relation_to_dict(self.output) if self.output is not None else None,
                "coverage": coverage_to_dict(self.coverage),
                "duration_ms": self.duration_ms,
                "side_effects": list(self.side_effects),
                "diagnostics": [_diagnostic_to_wire(d) for d in self.diagnostics],
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> DryRunResult:
        """Parse a worker reply."""
        data = json.loads(payload)
        output = data.get("output")
        return cls(
            target=data["target"],
            output=relation_from_dict(output) if output else None,
            coverage=coverage_from_dict(data.get("coverage", {})),
            diagnostics=tuple(_diagnostic_from_wire(d) for d in data.get("diagnostics", ())),
            duration_ms=data.get("duration_ms", 0.0),
            side_effects=tuple(data.get("side_effects", ())),
        )


def dry_run(request: DryRunRequest, *, limits: WorkerLimits | None = None) -> DryRunResult:
    """Execute one pipeline against rowless inputs.

    Steps, each with its own failure mode worth reporting separately:

    1. Pick the adapter (explicit `engine`, else annotations, else registry).
    2. Materialize an input frame per parameter.
    3. Run the target in a sandboxed subprocess (`rowless.runtime.worker`).
    4. Describe the returned object.
    5. Attach lineage and compute coverage.

    Steps 1 to 5 all happen in the child. The parent only moves JSON, which is
    what keeps a segfaulting engine from taking the CLI down with it.
    """
    started = time.perf_counter()
    reply = spawn(request.to_json(), limits or WorkerLimits())
    try:
        result = DryRunResult.from_json(reply)
    except (json.JSONDecodeError, KeyError) as exc:
        return _crashed(request.target, f"unreadable worker reply: {exc}")
    elapsed = (time.perf_counter() - started) * 1000
    return _with_duration(result, elapsed)


def dry_run_callable(
    fn: Callable[..., Any],
    inputs: Mapping[str, Schema],
    *,
    engine: str | None = None,
    options: Mapping[str, Any] | None = None,
    uris: Mapping[str, str] | None = None,
) -> DryRunResult:
    """In-process dry run of an already-imported function.

    No subprocess and no sandbox, so it is only for callers that already trust
    the code: the pytest plugin and rowless's own tests. Everything reached
    from the CLI goes through `dry_run`.
    """
    target = target_name(fn)
    started = time.perf_counter()
    try:
        adapter = resolve_adapter(engine, fn=fn)
        with dry_run_scope(adapter, inputs, uris=uris, options=options) as scope:
            return scope.execute(fn, started=started)
    except RowlessError as exc:
        return _failed(target, exc, started)


def dry_run_target(
    target: str,
    inputs: Mapping[str, Schema],
    *,
    engine: str | None = None,
    options: Mapping[str, Any] | None = None,
    uris: Mapping[str, str] | None = None,
) -> DryRunResult:
    """Import `target` and dry-run it, reporting every failure as a finding.

    The path the worker takes. The import is inside the dry-run session
    whenever the engine is already known, because a pipeline module declares
    its inputs at import time with `rowless.source(...)` and that call needs a
    live session to return a real frame.

    Nothing escapes as an exception: a module that will not import, an engine
    that cannot be chosen and a pipeline that raises are all findings, and the
    caller is a subprocess whose job is to report them.
    """
    started = time.perf_counter()
    try:
        if engine:
            adapter = resolve_adapter(engine)
            with dry_run_scope(adapter, inputs, uris=uris, options=options) as scope:
                return scope.execute(import_target(target), started=started)
        function = import_target(target)
        adapter = resolve_adapter(None, fn=function)
        with dry_run_scope(adapter, inputs, uris=uris, options=options) as scope:
            return scope.execute(function, started=started)
    except RowlessError as exc:
        return _failed(target, exc, started)
    except Exception as exc:
        # An import that blows up is the pipeline's problem, not rowless's:
        # reporting it as an internal error would blame the wrong party.
        return _raised(target, exc, started)


def target_name(fn: Callable[..., Any]) -> str:
    """`module:function`, the way every rowless surface names a pipeline."""
    return f"{getattr(fn, '__module__', '?')}:{getattr(fn, '__qualname__', '?')}"


def resolve_adapter(engine: str | None, *, fn: Callable[..., Any] | None = None) -> Any:
    """The adapter to run with: a named one, else inferred, else the only one.

    `fn` is optional because the worker has to choose before it imports the
    user's module: a module-level `rowless.source()` runs at import time and
    needs a live session to come back as a real frame. When the engine is
    named - which discovery manages whenever the annotations say so - the
    session is open before the import. Otherwise the import happens first and
    the engine is inferred from the function, and a module-level
    `rowless.source()` in that repository reports that it needs an engine.
    """
    if engine:
        return get_engine(engine)
    if fn is not None:
        return _infer_adapter(fn)
    available = engines()
    if len(available) == 1:
        return next(iter(available.values()))
    installed = ", ".join(sorted(available)) or "none"
    raise AdapterNotFound(
        f"cannot choose an engine before importing the target; active adapters: {installed}"
    )


@dataclass(frozen=True, slots=True)
class DryRunScope:
    """An open dry run: an adapter, a scratch context and materialized inputs.

    Exists so the worker can hold a session open across the *import* of the
    user's module, not only across the call. Pipeline modules declare their
    inputs at import time, and that declaration has to resolve.
    """

    adapter: Any
    ctx: DryRunContext
    frames: dict[str, Any]

    def execute(self, fn: Callable[..., Any], *, started: float | None = None) -> DryRunResult:
        """Run the pipeline and score what came back."""
        began = time.perf_counter() if started is None else started
        target = target_name(fn)
        with rowdep.recording() as observed:
            try:
                produced = _call(fn, self.frames)
                relation = self.adapter.describe(produced)
            except RowlessError as exc:
                return _failed(target, exc, began, observed=observed)
            except Exception as exc:
                # User code raised something no adapter translated. It is a
                # failure in the pipeline, not in rowless, and it is reported
                # as such rather than as an internal error.
                return _raised(target, exc, began, observed=observed)

        relation = _attach_lineage(self.adapter, produced, relation)
        relation = _with_notes(relation, self.ctx.notes)
        reasons = (*self.ctx.degradations, *observed.reasons)
        coverage = Coverage.score(
            relation.schema,
            confidence=Confidence.UNKNOWN if reasons else None,
            reasons=reasons,
        )
        return DryRunResult(
            target=target,
            output=relation,
            coverage=coverage,
            diagnostics=_confidence_diagnostics(target, relation, observed, self.ctx),
            duration_ms=(time.perf_counter() - began) * 1000,
        )


@contextmanager
def dry_run_scope(
    adapter: Any,
    inputs: Mapping[str, Schema],
    *,
    uris: Mapping[str, str] | None = None,
    options: Mapping[str, Any] | None = None,
) -> Iterator[DryRunScope]:
    """Materialize the inputs and mark the process as running a dry run."""
    from rowless.api import dry_run_session

    with tempfile.TemporaryDirectory(prefix="rowless-") as workdir:
        ctx = DryRunContext(
            run_id=uuid.uuid4().hex[:12],
            workdir=workdir,
            options=dict(options or {}),
        )
        frames = {name: adapter.materialize(schema, ctx) for name, schema in inputs.items()}
        by_uri = {uri: inputs[name] for name, uri in (uris or {}).items() if name in inputs}
        with dry_run_session(adapter, ctx, by_uri):
            yield DryRunScope(adapter=adapter, ctx=ctx, frames=frames)


def _call(fn: Callable[..., Any], frames: Mapping[str, Any]) -> Any:
    """Call the pipeline with the frames bound to its parameter names.

    Bound by name where the signature allows it, so a function whose
    parameters are in a different order than the config still works.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*frames.values())
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    if accepted >= set(frames):
        return fn(**{name: frame for name, frame in frames.items() if name in accepted})
    return fn(*frames.values())


def _infer_adapter(fn: Callable[..., Any]) -> Any:
    """Pick an adapter from the function's annotations.

    The return annotation decides, then the parameters, then - only when
    exactly one adapter is installed - that one. The last rule is what makes
    `rowless check` work in a Polars-only repository with no annotations, and
    it is deliberately not a guess between two installed engines.
    """
    from rowless.discovery.plain import FRAME_ANNOTATIONS

    available = engines()
    annotations = dict(getattr(fn, "__annotations__", {}))
    # The return annotation first: a function taking Polars and returning
    # pandas is a conversion, and it belongs to the engine it produces.
    hints = [
        annotations.get("return"),
        *(value for key, value in annotations.items() if key != "return"),
    ]
    for hint in hints:
        if not isinstance(hint, type):
            continue
        qualified = f"{hint.__module__}.{hint.__qualname__}"
        name = FRAME_ANNOTATIONS.get(qualified)
        if name is None:
            root = hint.__module__.split(".")[0]
            name = FRAME_ANNOTATIONS.get(f"{root}.{hint.__qualname__}") or (
                root if root in available else None
            )
        if name is not None and name in available:
            return available[name]
    if len(available) == 1:
        return next(iter(available.values()))
    installed = ", ".join(sorted(available)) or "none"
    raise AdapterNotFound(
        f"cannot tell which engine {getattr(fn, '__qualname__', fn)!r} uses from its "
        f"annotations; active adapters: {installed}. Name one with "
        "@rowless.pipeline(engine=...)"
    )


def _attach_lineage(adapter: Any, produced: Any, relation: Relation) -> Relation:
    """Fill in `derived_from` when the adapter can report column lineage."""
    if not isinstance(adapter, SupportsColumnLineage):
        return relation
    try:
        lineage = adapter.column_lineage(produced)
    except Exception:
        # Lineage is a capability, not a requirement. Losing it lowers the
        # traced ratio, which is exactly how that is meant to be reported.
        return relation
    if not lineage:
        return relation
    fields = tuple(
        column
        if column.derived_from or column.name not in lineage
        else Field(
            name=column.name,
            dtype=column.dtype,
            nullable=column.nullable,
            origin=column.origin,
            description=column.description,
            derived_from=tuple(lineage[column.name]),
        )
        for column in relation.schema.fields
    )
    return Relation(
        schema=Schema(fields=fields, name=relation.schema.name),
        engine=relation.engine,
        origin=relation.origin,
        sort_keys=relation.sort_keys,
        notes=relation.notes,
    )


def _with_notes(relation: Relation, notes: list[str]) -> Relation:
    """Carry adapter commentary onto the relation, for `rowless explain`."""
    if not notes:
        return relation
    return Relation(
        schema=relation.schema,
        engine=relation.engine,
        origin=relation.origin,
        sort_keys=relation.sort_keys,
        notes=relation.notes + tuple(notes),
    )


def _confidence_diagnostics(
    target: str,
    relation: Relation,
    observed: rowdep.RowDependencyLog,
    ctx: DryRunContext,
) -> tuple[Diagnostic, ...]:
    """Findings about how much of the result was actually verified."""
    diagnostics: list[Diagnostic] = []
    origin = Origin(kind="python", ref=target)
    for entry in observed.observed:
        diagnostics.append(
            Diagnostic(
                code=Code.ROW_DEPENDENT_CONTROL_FLOW,
                level=Level.UNVERIFIED,
                message=f"row-dependent call {entry.api} in {target}",
                origin=Origin(kind="python", ref=target, file=entry.file, line=entry.line),
                detail=entry.explanation,
            )
        )
    any_columns = [f.name for f in relation.schema.fields if f.dtype.is_any]
    if any_columns:
        diagnostics.append(
            Diagnostic(
                code=Code.ANY_COLUMN_IN_OUTPUT,
                level=Level.WARNING,
                message=f"{len(any_columns)} column(s) of unknown type in {target}",
                origin=origin,
                detail=", ".join(any_columns),
            )
        )
    for reason in ctx.degradations:
        diagnostics.append(
            Diagnostic(
                code=Code.UNRESOLVED_SOURCE,
                level=Level.UNVERIFIED,
                message=f"{target} lost confidence",
                origin=origin,
                detail=reason,
            )
        )
    return tuple(diagnostics)


def _failed(
    target: str,
    exc: RowlessError,
    started: float,
    *,
    observed: rowdep.RowDependencyLog | None = None,
) -> DryRunResult:
    """A rowless-classified failure: reported under the error's own code."""
    level = Level.UNVERIFIED if isinstance(exc, UnknownSchemaError) else Level.ERROR
    suggestion = getattr(exc, "suggestion", None)
    reasons = observed.reasons if observed is not None else ()
    origin = _origin_of(exc, target)
    return DryRunResult(
        target=target,
        output=None,
        coverage=Coverage(
            confidence=Confidence.UNKNOWN,
            reasons=(str(exc), *reasons),
        ),
        diagnostics=(
            Diagnostic(
                code=exc.code,
                level=level,
                message=str(exc),
                origin=origin,
                suggestion=f"did you mean {suggestion!r}?" if suggestion else None,
            ),
        ),
        duration_ms=(time.perf_counter() - started) * 1000,
    )


def _raised(
    target: str,
    exc: BaseException,
    started: float,
    *,
    observed: rowdep.RowDependencyLog | None = None,
) -> DryRunResult:
    """An exception from user code that no adapter recognised."""
    reasons = observed.reasons if observed is not None else ()
    detail = f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"
    return DryRunResult(
        target=target,
        output=None,
        coverage=Coverage(confidence=Confidence.UNKNOWN, reasons=(detail, *reasons)),
        diagnostics=(
            Diagnostic(
                code=Code.PIPELINE_RAISED,
                level=Level.ERROR,
                message=f"{target} raised during the dry run",
                origin=_origin_of(exc, target),
                detail=detail,
            ),
        ),
        duration_ms=(time.perf_counter() - started) * 1000,
    )


def _origin_of(exc: BaseException, target: str) -> Origin:
    """Where to point a diagnostic raised by an exception.

    The user's own line when the traceback reaches one, so the finding is
    clickable in an editor; the pipeline's name otherwise.
    """
    file, line = rowdep.user_frame_from_traceback(exc.__traceback__)
    return Origin(kind="python", ref=target, file=file, line=line)


def _crashed(target: str, detail: str) -> DryRunResult:
    """The worker did not come back with a usable reply."""
    return DryRunResult(
        target=target,
        output=None,
        coverage=Coverage(confidence=Confidence.UNKNOWN, reasons=(detail,)),
        diagnostics=(
            Diagnostic(
                code=Code.WORKER_CRASHED,
                level=Level.ERROR,
                message=f"dry run of {target} did not complete",
                origin=Origin(kind="python", ref=target),
                detail=detail,
            ),
        ),
    )


def _with_duration(result: DryRunResult, elapsed: float) -> DryRunResult:
    """Record the wall-clock cost the parent saw, including process startup."""
    return DryRunResult(
        target=result.target,
        output=result.output,
        coverage=result.coverage,
        diagnostics=result.diagnostics,
        duration_ms=elapsed,
        side_effects=result.side_effects,
    )


def import_target(target: str) -> Callable[..., Any]:
    """Import `module:function` and return the callable.

    Used by the worker, and by the pytest plugin when a test names a pipeline
    by string instead of importing it.
    """
    module_path, separator, attribute = target.partition(":")
    if not separator:
        module_path, _, attribute = target.rpartition(".")
    module = importlib.import_module(module_path)
    found: object = getattr(module, attribute)
    if not callable(found):
        raise AdapterNotFound(f"{target} is not callable")
    return cast("Callable[..., Any]", found)


def _diagnostic_to_wire(diagnostic: Any) -> dict[str, Any]:
    """Diagnostic to JSON, for the worker boundary."""
    from rowless.ir.serde import origin_to_dict

    return {
        "code": diagnostic.code.value,
        "level": diagnostic.level.value,
        "message": diagnostic.message,
        "origin": origin_to_dict(diagnostic.origin) if diagnostic.origin else None,
        "related": [origin_to_dict(origin) for origin in diagnostic.related],
        "detail": diagnostic.detail,
        "suggestion": diagnostic.suggestion,
    }


def _diagnostic_from_wire(payload: Mapping[str, Any]) -> Diagnostic:
    """Diagnostic from JSON."""
    from rowless.ir.serde import origin_from_dict

    return Diagnostic(
        code=Code(payload["code"]),
        level=Level(payload["level"]),
        message=payload["message"],
        origin=origin_from_dict(payload.get("origin")),
        related=tuple(
            origin_from_dict(item) or Origin("synthetic", "") for item in payload.get("related", ())
        ),
        detail=payload.get("detail"),
        suggestion=payload.get("suggestion"),
    )

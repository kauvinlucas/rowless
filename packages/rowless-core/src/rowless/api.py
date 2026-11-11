"""The user-facing API: what `import rowless` gives you.

Small on purpose. Most users only ever touch `rowless.source` in a pipeline
module and `assert_schema` in a test; everything else is the CLI's business.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, overload

from rowless.adapters.base import DryRunContext
from rowless.adapters.registry import adapter_for
from rowless.errors import RowlessError, UnknownSchemaError
from rowless.ir.confidence import Confidence
from rowless.ir.diff import diff_schemas
from rowless.ir.schema import Schema

#: Attribute the `@pipeline` decorator leaves on a function. Read by
#: discovery, by `assert_schema` and by the pytest plugin, so a user declares
#: their inputs once.
MARKER = "__rowless_pipeline__"


@dataclass(frozen=True, slots=True)
class PipelineMarker:
    """What `@rowless.pipeline(...)` recorded about a function."""

    inputs: Mapping[str, str] = field(default_factory=dict)
    engine: str | None = None
    expect: Schema | None = None


@dataclass(frozen=True, slots=True)
class _Session:
    """The dry run currently executing in this process.

    `rowless.source()` needs it: a module-level `orders = rowless.source(...)`
    runs at import time, inside the worker, and has to come back as a real
    engine frame with the schema the parent already resolved.
    """

    adapter: Any
    ctx: DryRunContext
    schemas: Mapping[str, Schema]
    """Resolved input schemas, keyed by source URI."""


_ACTIVE: ContextVar[_Session | None] = ContextVar("rowless_session", default=None)


@contextmanager
def dry_run_session(
    adapter: Any,
    ctx: DryRunContext,
    schemas: Mapping[str, Schema],
) -> Iterator[None]:
    """Mark this process as executing a dry run. Used by the runtime only."""
    token = _ACTIVE.set(_Session(adapter=adapter, ctx=ctx, schemas=schemas))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def source(uri: str, *, engine: str | None = None) -> Any:
    """Declare a rowless input: full schema, zero rows.

        orders = rowless.source("lake://stg_orders")

    Under a dry run this returns a real frame of the resolved engine type, so
    user code sees a genuine `pl.LazyFrame` and not a proxy.

    Outside a dry run it raises, rather than quietly reading the table. A
    pipeline module that imports cleanly in production but would silently load
    a terabyte if a flag flipped is not a trade worth making; production code
    reads its inputs the way it always did.
    """
    session = _ACTIVE.get()
    if session is None:
        raise RowlessError(
            f"rowless.source({uri!r}) only works inside a dry run. Production code should "
            "read its inputs directly; rowless substitutes the frame when it runs the check."
        )
    schema = session.schemas.get(uri)
    if schema is None:
        known = ", ".join(sorted(session.schemas)) or "none"
        raise UnknownSchemaError(
            f"no resolved schema for {uri!r} in this dry run; resolved inputs: {known}"
        )
    adapter = session.adapter
    if engine is not None and getattr(adapter, "name", None) != engine:
        from rowless.adapters.registry import engine as get_engine

        adapter = get_engine(engine)
    return adapter.materialize(schema, session.ctx)


@overload
def pipeline[F: Callable[..., Any]](fn: F) -> F: ...


@overload
def pipeline[F: Callable[..., Any]](
    *,
    inputs: Mapping[str, str] | None = ...,
    engine: str | None = ...,
    expect: Schema | None = ...,
) -> Callable[[F], F]: ...


def pipeline[F: Callable[..., Any]](
    fn: F | None = None,
    *,
    inputs: Mapping[str, str] | None = None,
    engine: str | None = None,
    expect: Schema | None = None,
) -> Callable[[F], F] | F:
    """Mark a function as a pipeline entry point.

    Only needed when discovery cannot work it out: unannotated parameters, or
    inputs that are not bound by a framework. `expect` turns the function into
    a contract that `rowless check` enforces.

        @rowless.pipeline(inputs={"orders": "lake://stg_orders"})
        def daily_revenue(orders): ...

    The function is returned unchanged. A decorator that wrapped it would
    change what production runs, and rowless is a checker, not a runtime.
    """
    marker = PipelineMarker(inputs=dict(inputs or {}), engine=engine, expect=expect)

    def decorate(target: F) -> F:
        setattr(target, MARKER, marker)
        return target

    if fn is not None:
        return decorate(fn)
    return decorate


def marker_of(fn: object) -> PipelineMarker | None:
    """The `@pipeline` metadata on a function, if it has any."""
    found = getattr(fn, MARKER, None)
    return found if isinstance(found, PipelineMarker) else None


def schema_of(obj: object) -> Schema:
    """The schema of an engine object, via its adapter.

    Useful in tests and notebooks; the same call the runtime makes internally.
    """
    return adapter_for(obj).describe(obj).schema


def assert_schema(
    target: Callable[..., Any],
    expected: Schema | Mapping[str, str],
    *,
    inputs: Mapping[str, Schema | Mapping[str, str] | str] | None = None,
    exact: bool = True,
) -> None:
    """Contract-test a pipeline in milliseconds. Backs the pytest plugin.

        def test_revenue():
            rowless.assert_schema(daily_revenue, {"day": "date", "amount": "int64"})

    With `exact=False`, extra output columns are allowed and only the named
    ones are checked, which is what makes this usable on wide marts.

    Raises `AssertionError` with a schema diff, and never passes on an
    unverified result: a dry run that lost confidence fails the test, since a
    contract test that silently stops testing is worse than no test.
    """
    from rowless.runtime.dryrun import dry_run_callable

    wanted = as_schema(expected)
    resolved = resolve_inputs(target, inputs)
    marker = marker_of(target)
    result = dry_run_callable(
        target,
        resolved.schemas,
        engine=marker.engine if marker else None,
        uris=resolved.uris,
    )

    if result.output is None:
        reasons = "; ".join(result.coverage.reasons) or "no schema was produced"
        raise AssertionError(f"rowless could not resolve {result.target}: {reasons}")
    if result.coverage.confidence is Confidence.UNKNOWN:
        reasons = "; ".join(result.coverage.reasons) or "confidence was lost"
        raise AssertionError(
            f"rowless did not verify {result.target}, so this test proves nothing: {reasons}"
        )

    actual = result.output.schema
    problems = compare_schemas(wanted, actual, exact=exact)
    if problems:
        raise AssertionError(
            f"{result.target} does not match the expected schema:\n  " + "\n  ".join(problems)
        )


def compare_schemas(wanted: Schema, actual: Schema, *, exact: bool = True) -> list[str]:
    """Differences between a declared contract and a computed schema.

    One line per problem, empty when the contract holds. Public because the
    pytest plugin reports the same failures with its own framing around them,
    and a second implementation of this comparison would mean a test could
    pass under `rowless_check` and fail under `rowless.assert_schema`.

    An ANY or UNKNOWN column fails even against a matching declaration: the
    dtype algebra lets ANY match anything, which is right for inference and
    wrong for a test that is supposed to prove something.
    """
    problems: list[str] = []
    for column in wanted.fields:
        if column.name not in actual:
            problems.append(f"missing column {column.name!r} ({column.dtype})")
            continue
        found = actual[column.name].dtype
        if not found.is_known or found.is_any:
            problems.append(
                f"{column.name}: rowless resolved {found}, which is not evidence of {column.dtype}"
            )
        elif not column.dtype.matches(found):
            problems.append(f"{column.name}: expected {column.dtype}, got {found}")
    if exact:
        extra = [name for name in actual if name not in wanted]
        if extra:
            problems.append(f"unexpected column(s): {', '.join(extra)}")
        diff = diff_schemas(wanted, actual, detect_renames=False)
        order_changed = any(change.kind.value == "order_changed" for change in diff.changes)
        if order_changed:
            problems.append(
                f"column order differs: expected {', '.join(wanted)}, got {', '.join(actual)}"
            )
    return problems


def as_schema(expected: Schema | Mapping[str, str]) -> Schema:
    """Accept a `Schema` or the `{"day": "date"}` shorthand.

    The shorthand is what tests are written in, so the plugin and
    `assert_schema` parse it the same way.
    """
    if isinstance(expected, Schema):
        return expected
    from rowless.sources.literal import schema_from_mapping

    return schema_from_mapping(dict(expected))


@dataclass(frozen=True, slots=True)
class ResolvedInputs:
    """A test's inputs, resolved: a schema per parameter, and where each came from.

    The URIs travel next to the schemas because a pipeline may call
    `rowless.source()` in its body rather than taking the frame as a
    parameter, and that call looks itself up by the URI string the pipeline
    wrote.
    """

    schemas: Mapping[str, Schema]
    uris: Mapping[str, str] = field(default_factory=dict)


def resolve_inputs(
    target: Callable[..., Any],
    inputs: Mapping[str, Schema | Mapping[str, str] | str] | None = None,
    *,
    resolver: Callable[[str], Schema] | None = None,
) -> ResolvedInputs:
    """Resolve a test's inputs: schemas as given, URIs through a resolver.

    A value is a `Schema`, the `{"day": "date"}` shorthand, or a source URI.
    With no `inputs` at all the `@rowless.pipeline(inputs=...)` declaration is
    used, so a pipeline says where its inputs come from once and every test of
    it inherits that.

    `resolver` is how a URI becomes a schema, and defaults to the source
    registry with no cache and no config. The pytest plugin passes its own,
    which applies the `[sources]` rewrites in `rowless.toml` and the on-disk
    schema cache - a test suite resolving a URI differently from
    `rowless check` would be two tools with one name.

    Raises `AssertionError` rather than a `RowlessError`: both callers are a
    test, and the message is the one a test author needs to read.
    """
    from rowless.sources.base import resolve as resolve_source

    resolve = resolver if resolver is not None else resolve_source
    declared: Mapping[str, Schema | Mapping[str, str] | str]
    if inputs is not None:
        declared = inputs
    else:
        marker = marker_of(target)
        if marker is None or not marker.inputs:
            raise AssertionError(
                f"{getattr(target, '__qualname__', target)} has no declared inputs; pass "
                "inputs={...} or mark it with @rowless.pipeline(inputs={...})"
            )
        declared = dict(marker.inputs)

    schemas: dict[str, Schema] = {}
    uris: dict[str, str] = {}
    for parameter, value in declared.items():
        if isinstance(value, Schema):
            schemas[parameter] = value
        elif isinstance(value, Mapping):
            schemas[parameter] = as_schema(value)
        else:
            uri = str(value)
            uris[parameter] = uri
            schemas[parameter] = resolve(uri)
    return ResolvedInputs(schemas=schemas, uris=uris)

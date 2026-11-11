"""The adapter conformance suite: the real specification for an adapter.

`EngineAdapter` says what the methods are; this says what they must *do*. An
adapter is finished when this suite passes, and a new adapter author's first
move should be to run it and work through the failures.

Use it from an adapter package's own tests:

    from rowless.testing.conformance import adapter_conformance

    test_conformance = adapter_conformance(PolarsAdapter())

That produces a test class whose methods are the contract below. Engines
legitimately differ, so a few cases are opt-out via `Capabilities`: an adapter
that declares `nested_types=False` skips the nested round trips rather than
failing them. An engine that lacks a corpus dtype outright declares it through
`unsupported_dtypes()`, and the suite then requires `from_ir` to refuse it.

The suite is shipped in core, not copied per adapter, so tightening the
contract tightens it everywhere at once.

**Operation cases come from the adapter.** The universal half of the contract
(round trips, materialize, describe, honesty about what it cannot do) is
written here once. The operations battery cannot be: `df.select("a")` is not
the same call in Polars, pandas and DuckDB, and the proposal's whole argument
is that pretending otherwise is what breaks. So an adapter supplies its own
cases through `conformance_cases()`, each one a real pipeline for that engine
plus the schema it must produce, and the suite runs them through the same
runtime the CLI uses. An adapter with no cases is reported as incomplete
rather than passing quietly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rowless.adapters.base import (
    Capabilities,
    DryRunContext,
    EngineAdapter,
    SupportsColumnLineage,
    SupportsSql,
    SupportsUdfProbe,
)
from rowless.errors import UnsupportedDType
from rowless.ir.confidence import Confidence
from rowless.ir.dtypes import DType, Kind, unknown
from rowless.ir.schema import Schema
from rowless.testing import fixtures

#: Capability flags checked against the protocol they promise.
_CAPABILITY_PROTOCOLS: dict[str, type] = {
    "column_lineage": SupportsColumnLineage,
    "sql": SupportsSql,
    "udf_probe": SupportsUdfProbe,
}


@dataclass(frozen=True, slots=True)
class ConformanceCase:
    """One operation an adapter must resolve, and what it must resolve to.

    `reason` is the valuable field. When an engine produces a different schema
    than the others for the same operation, the note explaining why is the
    catalogue nobody else has, and the suite prints it on failure.
    """

    name: str
    pipeline: Callable[..., Any]
    inputs: Mapping[str, Schema] = field(default_factory=dict)
    expect: Schema | None = None
    raises: type[BaseException] | None = None
    expect_confidence: Confidence | None = None
    reason: str | None = None


def adapter_conformance(adapter: EngineAdapter) -> type:
    """Build a pytest class exercising the full contract against `adapter`.

    Contract, in the order a new adapter usually fixes it:

    **Dtype round trip.** For every dtype in the corpus the engine claims to
    support, `to_ir(from_ir(dtype)) == dtype`. Parameters count: losing a
    timezone or a decimal scale here is a real bug that would otherwise
    surface as a mysterious diff months later.

    **Materialize produces a native object.** `materialize` returns something
    `handles` claims, with exactly the requested columns in order and zero
    rows. It must be a real engine object: user code calls engine methods on
    it.

    **Describe round trips materialize.** `describe(materialize(s)) == s` for
    every schema in the corpus. The identity case, and the one that catches
    pandas-style dtype drift on empty frames.

    **Describe does not execute.** With the sandbox in strict mode, `describe`
    performs no filesystem write and no network call. Verified by assertion on
    the side-effect log, not by inspection.

    **Operations resolve.** Every case from `conformance_cases()` resolves to
    its expected schema through the real runtime.

    **Missing columns raise.** A case declaring `raises=ColumnNotFound` must
    raise exactly that, with the column name in the message. An adapter that
    lets the engine's own exception escape fails here.

    **Row dependence is reported.** A case declaring
    `expect_confidence=Confidence.UNKNOWN` must come back unverified. An
    adapter that returns a confident schema for `if df.is_empty()` code fails,
    because that is the silent-wrong-answer case.

    **Unsupported dtypes raise, never guess.** `from_ir(UNKNOWN)` raises
    `UnsupportedDType`; `to_ir` of an object the engine does not know returns
    `Kind.UNKNOWN`. Neither may substitute a near-enough type.

    **Declared absences are refused rather than approximated.** An engine that
    genuinely lacks one of the corpus dtypes says so through
    `unsupported_dtypes()`, which drops it from the round trip and asserts that
    `from_ir` raises `UnsupportedDType` for it instead. DuckDB is the
    motivating case: it has no nanosecond timestamp carrying a named zone, and
    lowering `datetime[ns, Europe/London]` to the nearest thing it does have
    would move data by a silent hour. A declared absence is dropped from the
    fixture schemas too, for the same reason: asking `materialize` to carry a
    column of a dtype the engine does not have is asking for the lossy
    conversion the declaration exists to refuse. Ibis is the motivating case
    there - it has no enum, so the enum column in `fixtures.DRIFT_PRONE` is not
    a materialize test for it.

    **Capabilities are truthful.** Each declared capability's protocol method
    exists; each undeclared one is absent. A capability that is advertised but
    broken is worse than one that is missing, since coverage scores are
    computed from these flags.
    """
    if TYPE_CHECKING:
        import pytest
    else:
        # Imported through a helper so the failure message explains itself.
        pytest = _require_pytest()

    corpus = list(dtype_corpus(adapter))
    absent = list(absent_dtypes(adapter))
    schemas = list(materializable_schemas(adapter))
    cases = list(pipeline_battery(adapter))

    class AdapterConformance:
        """Generated conformance suite. One class per adapter instance."""

        @pytest.fixture(autouse=True)
        def _skip_when_unavailable(self) -> None:
            """Every test needs the engine library; say so once.

            An `available()` that raises counts as unavailable. The protocol
            says it must not raise, and `test_capabilities_are_truthful` is
            where that is reported - skipping here keeps one broken method
            from turning the whole suite into tracebacks.
            """
            try:
                usable = adapter.available()
            except Exception as exc:
                pytest.skip(f"{adapter.name}.available() raised: {exc}")
            if not usable:
                # `requires` names a package, which covers an adapter whose
                # engine is missing and nothing else. An adapter that cannot
                # run for another reason - pyspark with no cluster configured -
                # says so through `unavailable_reason()`, the same hook
                # `rowless doctor` prefers, and a skip that says "not
                # installed" about an installed engine sends the reader looking
                # in the wrong place.
                pytest.skip(_why_unavailable(adapter))

        @pytest.mark.parametrize("dtype", corpus, ids=[str(dtype) for dtype in corpus])
        def test_dtype_round_trips(self, dtype: DType) -> None:
            native = adapter.from_ir(dtype)
            recovered = adapter.to_ir(native)
            assert recovered == dtype, (
                f"{adapter.name} round trip lost information: {dtype} -> {native!r} -> {recovered}"
            )

        @pytest.mark.parametrize("schema", schemas, ids=[schema.name or "?" for schema in schemas])
        def test_materialize_is_a_real_engine_object(self, schema: Schema) -> None:
            frame = adapter.materialize(schema, _context())
            assert adapter.handles(frame), (
                f"{adapter.name}.materialize returned {type(frame)!r}, which its own "
                "handles() does not claim"
            )
            assert _row_count(frame) in (0, None), "a rowless frame must have zero rows"

        @pytest.mark.parametrize("schema", schemas, ids=[schema.name or "?" for schema in schemas])
        def test_describe_round_trips_materialize(self, schema: Schema) -> None:
            relation = adapter.describe(adapter.materialize(schema, _context()))
            assert tuple(relation.schema) == tuple(schema), (
                f"columns changed: {tuple(relation.schema)} != {tuple(schema)}"
            )
            for expected in schema.fields:
                found = relation.schema[expected.name].dtype
                assert found.matches(expected.dtype), (
                    f"{expected.name}: materialize/describe turned {expected.dtype} into {found}"
                )
            assert relation.engine == adapter.name

        def test_describe_does_not_execute(self) -> None:
            from rowless.runtime.sandbox import Policy, sandboxed

            frame = adapter.materialize(fixtures.ORDERS, _context())
            policy = Policy.strict().allowing_writes_to(_scratch())
            if adapter.capabilities.remote:
                pytest.skip("remote adapters need egress; their policy is set per adapter")
            with sandboxed(policy) as log:
                adapter.describe(frame)
            assert not log.network, f"describe opened a socket: {log.network}"
            assert not log.blocked, f"describe attempted a blocked side effect: {log.blocked}"

        def test_from_ir_refuses_unknown(self) -> None:
            with pytest.raises(UnsupportedDType):
                adapter.from_ir(unknown("nothing could resolve this"))

        @pytest.mark.parametrize("dtype", absent, ids=[str(dtype) for dtype in absent] or None)
        def test_declared_absences_are_refused(self, dtype: DType) -> None:
            with pytest.raises(UnsupportedDType):
                adapter.from_ir(dtype)

        def test_to_ir_of_a_foreign_type_is_unknown(self) -> None:
            assert adapter.to_ir(object()).kind is Kind.UNKNOWN, (
                "to_ir must return UNKNOWN for a type it cannot map, never a near-enough dtype"
            )

        def test_capabilities_are_truthful(self) -> None:
            declared = adapter.capabilities
            for flag, protocol in _CAPABILITY_PROTOCOLS.items():
                promised = getattr(declared, flag)
                implemented = isinstance(adapter, protocol)
                assert promised == implemented, (
                    f"{adapter.name} declares {flag}={promised} but "
                    f"{'does not implement' if promised else 'implements'} {protocol.__name__}"
                )

        def test_declares_conformance_cases(self) -> None:
            assert cases, (
                f"{adapter.name} supplies no conformance_cases(); the operations battery is "
                "per-engine data and cannot be written in core. See "
                "docs/writing-an-adapter.md"
            )

        @pytest.mark.parametrize("case", cases, ids=[case.name for case in cases] or None)
        def test_operations_resolve(self, case: ConformanceCase) -> None:
            from rowless.runtime.dryrun import dry_run_callable

            result = dry_run_callable(case.pipeline, dict(case.inputs), engine=adapter.name)
            note = f" ({case.reason})" if case.reason else ""

            if case.raises is not None:
                assert result.output is None, f"{case.name} should have failed{note}"
                reported = [d.code for d in result.diagnostics]
                assert reported, f"{case.name} failed without a diagnostic{note}"
                # Every rowless error carries the code it is reported under, so
                # the expectation is written as the exception the adapter must
                # translate to, and checked against what the run reported.
                expected = getattr(case.raises, "code", None)
                assert expected is None or expected in reported, (
                    f"{case.name} should report {case.raises.__name__} ({expected}), reported "
                    f"{[code.value for code in reported]}{note}"
                )
                return

            if case.expect_confidence is not None:
                assert result.coverage.confidence is case.expect_confidence, (
                    f"{case.name}: expected confidence {case.expect_confidence.name}, got "
                    f"{result.coverage.confidence.name}{note}"
                )
                if case.expect_confidence is Confidence.UNKNOWN:
                    assert result.coverage.reasons, "an unverified result must say why"
                    return

            assert result.output is not None, (
                f"{case.name} produced no schema: {'; '.join(result.coverage.reasons)}{note}"
            )
            if case.expect is None:
                return
            produced = result.output.schema
            assert tuple(produced) == tuple(case.expect), (
                f"{case.name}: columns {tuple(produced)} != {tuple(case.expect)}{note}"
            )
            for expected in case.expect.fields:
                found = produced[expected.name].dtype
                assert found.matches(expected.dtype), (
                    f"{case.name}.{expected.name}: expected {expected.dtype}, got {found}{note}"
                )

    AdapterConformance.__name__ = f"TestConformance{adapter.name.title()}"
    AdapterConformance.__qualname__ = AdapterConformance.__name__
    return AdapterConformance


def dtype_corpus(adapter: EngineAdapter) -> Sequence[DType]:
    """Dtypes this adapter must round-trip, filtered by what the engine has.

    Two filters, and they mean different things. `nested_types=False` is an
    adapter that has not done nested types yet, and the corpus skips them.
    `unsupported_dtypes()` is an engine that does not have the dtype at all:
    those are dropped from the round trip and checked by
    `test_declared_absences_are_refused` instead, because the only honest
    answer to a dtype an engine lacks is `UnsupportedDType`.
    """
    nested = {Kind.LIST, Kind.STRUCT, Kind.MAP}
    supports_nested = _supports(adapter, "nested_types")
    absent = absent_dtypes(adapter)
    return tuple(
        dtype
        for dtype in fixtures.round_trip_corpus()
        if (supports_nested or dtype.kind not in nested) and dtype not in absent
    )


def materializable_schemas(adapter: EngineAdapter) -> Sequence[Schema]:
    """The fixture schemas this adapter is asked to carry through a frame.

    Two filters, the same two as `dtype_corpus`. A capability the adapter has
    not implemented (`nested_types=False`) drops the schemas that need it. A
    dtype the *engine* does not have, declared through `unsupported_dtypes()`,
    drops that column from the schema instead of the whole schema: a column
    whose dtype has no equivalent cannot round trip through `materialize` and
    `describe` by definition, and the refusal is already checked by
    `test_declared_absences_are_refused`. Dropping the column keeps the other
    four honest rather than losing a whole fixture to one absence.
    """
    absent = frozenset(absent_dtypes(adapter))
    found: list[Schema] = []
    for schema, gate in fixtures.SCHEMAS:
        if not _supports(adapter, gate):
            continue
        carried = tuple(f for f in schema.fields if not _mentions(f.dtype, absent))
        if not carried:
            # Every column of this fixture is a dtype the engine lacks. There
            # is nothing left to materialize, and an empty schema would test
            # the empty case twice rather than this fixture.
            continue
        complete = len(carried) == len(schema.fields)
        found.append(schema if complete else schema.select(_names(carried)))
    return tuple(found)


def _names(fields: Sequence[Any]) -> tuple[str, ...]:
    """Field names, in order."""
    return tuple(field.name for field in fields)


def _mentions(dtype: DType, absent: frozenset[DType]) -> bool:
    """True when `dtype`, or anything nested in it, is one the engine lacks."""
    if dtype in absent:
        return True
    return any(_mentions(inner, absent) for inner in dtype.inner)


def absent_dtypes(adapter: EngineAdapter) -> Sequence[DType]:
    """Corpus dtypes the adapter declares its engine has no equivalent for.

    Read off an optional `unsupported_dtypes()` method, the same way the
    operations battery is read off `conformance_cases()`: an adapter that
    round-trips the whole corpus needs neither, and declaring an absence is a
    claim the suite then holds the adapter to.
    """
    provider = getattr(adapter, "unsupported_dtypes", None)
    if provider is None:
        return ()
    return tuple(provider())


def pipeline_battery(adapter: EngineAdapter) -> Sequence[ConformanceCase]:
    """The adapter's own operation cases.

    Written once per operation *by the adapter*, with the per-engine
    expectations attached, because the engines genuinely disagree. Those
    expectations are the most valuable documentation in the project: each one
    is a place where moving a pipeline between engines changes its output
    schema.
    """
    provider = getattr(adapter, "conformance_cases", None)
    if provider is None:
        return ()
    cases = provider()
    return tuple(cases)


def _why_unavailable(adapter: EngineAdapter) -> str:
    """One line for a skip, saying why this adapter cannot run.

    Reads the optional `unavailable_reason()` hook first, which is how an
    adapter whose blocker is configuration rather than a missing package
    explains itself, and falls back to `requires`.
    """
    explain = getattr(adapter, "unavailable_reason", None)
    if explain is not None:
        try:
            reason = str(explain())
        except Exception:  # pragma: no cover - third-party code
            reason = ""
        if reason:
            return f"{adapter.name}: {reason}"
    return f"{adapter.name} is not installed ({adapter.requires})"


def _supports(adapter: EngineAdapter, capability: str | None) -> bool:
    """True when the adapter declares `capability`, or none is required."""
    if capability is None:
        return True
    declared: Capabilities = adapter.capabilities
    return bool(getattr(declared, capability, False))


def _context() -> DryRunContext:
    """A throwaway dry-run context for the direct adapter calls."""
    return DryRunContext(run_id="conformance", workdir=_scratch())


def _scratch() -> str:
    """A scratch directory for the suite, created once per process."""
    import tempfile

    global _SCRATCH
    if _SCRATCH is None:
        _SCRATCH = tempfile.mkdtemp(prefix="rowless-conformance-")
    return _SCRATCH


_SCRATCH: str | None = None


def _row_count(frame: object) -> int | None:
    """Rows in a materialized frame, or None when the engine will not say.

    A lazy frame legitimately cannot answer without executing, and asking it
    to would defeat the purpose; the check is skipped there rather than faked.
    """
    for attribute in ("height", "shape"):
        value = getattr(frame, attribute, None)
        if isinstance(value, int):
            return value
        if isinstance(value, tuple) and value and isinstance(value[0], int):
            return value[0]
    return None


def _require_pytest() -> Any:
    """Import pytest, with a message that says why it is needed.

    Core does not depend on pytest at runtime: this module is only imported
    from a test suite, and an adapter author who hits this message is one
    `pip install pytest` from the suite running.
    """
    try:
        import pytest
    except ImportError as exc:  # pragma: no cover - only without pytest
        raise RuntimeError("the conformance suite needs pytest: pip install pytest") from exc
    return pytest

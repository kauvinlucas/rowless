"""The adapter contract.

One adapter per engine, each in its own distribution, each owned by somebody
who uses that engine daily. Core never imports an engine library; it only ever
talks to these four required methods:

    materialize   schema  -> a rowless frame the engine accepts as input
    describe      engine object -> the schema the engine computed
    to_ir         engine dtype -> rowless dtype
    from_ir       rowless dtype -> engine dtype

Everything else is an optional capability. An adapter that implements none of
them still works; it just reports lower coverage, which is exactly the trade
gradual typing is supposed to make visible.

Write a new adapter by subclassing nothing: these are `Protocol`s, so an
adapter is any object with the right methods. Register it under the
`rowless.engines` entry-point group and run the conformance suite from
`rowless.testing.conformance`, which is the real specification.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from rowless.ir.dtypes import DType
from rowless.ir.schema import Relation, Schema


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What an adapter can actually do, reported by `rowless adapters`.

    Users need this to interpret a coverage score: "73% traced" means something
    different on an engine that cannot report lineage at all.
    """

    column_lineage: bool = False
    """Can attribute each output column to its inputs."""

    sql: bool = False
    """Can execute SQL strings against rowless relations."""

    udf_probe: bool = False
    """Can infer a UDF return dtype by probing with one synthetic row."""

    lazy: bool = False
    """Resolves schemas without executing anything (Polars, Spark, Ibis).
    Eager engines need the empty-frame treatment and its dtype-drift fixes."""

    nested_types: bool = False
    """Supports LIST, STRUCT and MAP in the IR round trip."""

    remote: bool = False
    """Needs a live connection (Spark Connect, a warehouse). The sandbox allows
    the adapter's own egress and nothing else."""


@dataclass(frozen=True, slots=True)
class DryRunContext:
    """Per-run state handed to an adapter.

    Created by the runtime inside the sandboxed subprocess, so an adapter never
    has to know how isolation was arranged.
    """

    run_id: str
    workdir: str
    """Scratch directory. The only path the sandbox permits writes to."""

    options: Mapping[str, Any] = field(default_factory=dict)
    """Adapter options from `rowless.toml`, under `[engine.<name>]`."""

    notes: list[str] = field(default_factory=list, compare=False)
    """Adapter commentary collected during the run. A mutable list inside a
    frozen record: the context's identity does not change when an adapter
    talks, and the runtime reads this back after the call returns."""

    degradations: list[str] = field(default_factory=list, compare=False)
    """Reasons confidence was lost. Non-empty means the result is not EXACT,
    whatever the schema looks like."""

    def note(self, message: str) -> None:
        """Record adapter commentary for `rowless explain`."""
        self.notes.append(message)

    def degrade(self, reason: str) -> None:
        """Flag that this run lost confidence, and why."""
        self.degradations.append(reason)


@runtime_checkable
class EngineAdapter(Protocol):
    """The four methods every adapter must provide."""

    name: ClassVar[str]
    """Short engine name: polars, pandas, duckdb, sql, pyspark, ibis."""

    requires: ClassVar[str]
    """PEP 508 requirement for the engine library, e.g. `polars>=1.0`. Used by
    `rowless doctor` to explain why an adapter is inactive."""

    capabilities: ClassVar[Capabilities]

    def available(self) -> bool:
        """True when the engine library is importable at a supported version.

        Must not raise, and must not import the engine at module scope: a user
        with only pandas installed still runs the pandas adapter.
        """
        ...

    def handles(self, obj: object) -> bool:
        """True when `obj` is one of this engine's frame or expression types.

        Checked by isinstance against lazily imported types. The registry calls
        this to route a discovered pipeline to the right adapter.
        """
        ...

    def materialize(self, schema: Schema, ctx: DryRunContext) -> Any:
        """Build a rowless frame: full schema, zero rows.

        The object returned is handed straight to user code, so it must be a
        genuine engine object (a real `pl.LazyFrame`, a real `pd.DataFrame`),
        not a proxy. Proxies are what make source-parsing tools fail on real
        code, and avoiding them is the whole premise of the project.
        """
        ...

    def describe(self, obj: object) -> Relation:
        """Read the schema the engine computed, without executing the query.

        Polars resolves the plan via `collect_schema`, DuckDB answers
        `DESCRIBE`, Spark Connect returns the analyzed plan. Raise
        `UnknownSchemaError` rather than guessing.
        """
        ...

    def to_ir(self, native: Any) -> DType:
        """Engine dtype to IR. Return `Kind.UNKNOWN` for anything unmappable."""
        ...

    def from_ir(self, dtype: DType) -> Any:
        """IR to engine dtype. Raise `UnsupportedDType` when there is no mapping."""
        ...


@runtime_checkable
class SupportsColumnLineage(Protocol):
    """Adapters that can attribute output columns to input columns."""

    def column_lineage(self, obj: object) -> Mapping[str, tuple[str, ...]]:
        """Output column name to the input columns it was computed from.

        Polars walks the resolved plan; sqlglot qualifies and resolves. An
        adapter that can only do this for some columns returns what it has:
        partial lineage is useful, as long as it is not reported as complete.
        """
        ...


@runtime_checkable
class SupportsSql(Protocol):
    """Adapters that can run a SQL string against rowless relations.

    This is the Python-and-SQL seam the proposal is built around: the DuckDB
    adapter registers rowless frames as empty tables, so a SQL model and a
    Polars function land in one graph.
    """

    def run_sql(self, sql: str, inputs: Mapping[str, object], ctx: DryRunContext) -> Relation:
        """Resolve the output schema of `sql` given named input relations."""
        ...


@runtime_checkable
class SupportsUdfProbe(Protocol):
    """Adapters that can determine a UDF return dtype empirically."""

    def probe_udf(
        self,
        fn: Callable[..., Any],
        arg_schema: Schema,
        ctx: DryRunContext,
    ) -> DType:
        """Infer the return dtype of `fn`.

        Order of preference, per hard problem 05: an explicit `return_dtype=`,
        then type annotations, then one synthetic row through the function
        inside the sandbox. The last option is opt-in, because it runs user
        code with made-up values.
        """
        ...


@runtime_checkable
class SupportsEmptyFrameFixups(Protocol):
    """Eager engines whose dtypes drift on zero rows.

    pandas is the motivating case: an empty frame silently becomes `object` or
    `float64` in places a populated frame would not. The adapter declares the
    fixups it applies so `rowless explain` can show them, rather than having
    the corrections buried in the adapter.
    """

    def dtype_drift_fixups(self) -> Sequence[str]:
        """Human-readable descriptions of the corrections this adapter applies."""
        ...

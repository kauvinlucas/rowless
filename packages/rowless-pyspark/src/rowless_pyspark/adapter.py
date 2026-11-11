"""The PySpark adapter.

Last on the roadmap, because the setup cost is the hard part, not the schema
resolution.

`df.schema` triggers analysis only: Catalyst resolves and type-checks the plan
without launching a job. So the schema is exact, and it is the answer Spark
itself would give, including implicit cast rules that differ from every other
engine here in ways that matter.

Spark Connect is what makes this practical. A classic local SparkSession takes
tens of seconds to start, which breaks the linter-speed promise outright. With
Connect, analysis happens on an already-running server and a dry run costs a
round trip. The trade is that this is the only adapter needing network egress,
so the sandbox opens exactly the configured endpoint and nothing else, and
`remote=True` tells the user this adapter talks to something.

Local mode stays supported for people without a cluster, behind
`[engine.pyspark] mode = "local"`, with the session reused across every dry run
in a check so the startup cost is paid once.

How the four required methods are satisfied:

* `materialize` calls `spark.createDataFrame([], schema)`: a real DataFrame
  carrying the full schema, nullability included, over a local relation with no
  rows. User code gets a genuine `DataFrame`, so real pipelines run unmodified.
* `describe` reads `df.schema`, which analyzes and does not execute.
* `to_ir` and `from_ir` are in `rowless_pyspark.dtypes`, where the dtype-level
  disagreements are written down.
* the session itself is in `rowless_pyspark.spark`, the per-run state in
  `rowless_pyspark.session`, and the statement guard, the writer recorders, the
  row dependence and the lineage recording are in
  `rowless_pyspark.interception`, which explains what it patches and why.

## The thing this adapter has to get right that the others do not

Every other adapter in rowless runs against a library in the same process. This
one runs against a **cluster that also serves production**, over a socket the
sandbox was told to allow. So the interception here is not only about better
findings: `spark.sql("INSERT OVERWRITE TABLE orders ...")` in a pipeline would
reach a real table, and the audit hook cannot tell that call from the analysis
round trips it is deliberately permitting. Non-query statements are recorded
and not sent, and every writer entry point is a recorder. That is the one part
of this adapter where a bug is a data-loss bug rather than a wrong schema, and
it is why `rowless_pyspark.interception` is worth reading before trusting this
against a cluster you care about.

## Where Spark disagrees with the other engines

The catalogue this adapter exists to write down. Each line is a place where
moving a pipeline between engines changes its output schema, and the
conformance cases in `rowless_pyspark.cases` assert every one of them.

* `groupBy("status").agg(F.sum("amount"))` names the column `sum(amount)` and
  keeps it `bigint`. DuckDB calls it `sum(amount)` too but widens to a 128-bit
  integer; Polars keeps the input name `amount`; ibis calls it `Sum(amount)`.
  Four engines, four answers, one operation.
* `F.sum` of a `decimal(18,2)` is `decimal(28,2)`: Spark adds ten digits of
  headroom to a decimal sum. Polars keeps `decimal(18,2)`, so a contract that
  passes on one fails on the other for arithmetic nobody changed.
* `F.avg` of an integer is `double`, and `F.count` is `bigint` whatever it
  counts.
* `col("amount") / 2` is `double`. Spark divides like Python here, as Polars
  and DuckDB do; the engines that keep an integer for `/` are the older SQL
  dialects.
* a join on a *condition* keeps **both** key columns and does not rename a
  collision, so joining two frames that both have `created_at` produces two
  columns of that name. Spark is content with that and rowless cannot be: a
  schema addressed by name has no way to describe it, so it is reported as a
  finding. Polars coalesces the key and suffixes the collision `_right`;
  pandas suffixes `_x` and `_y`; DuckDB does what Spark does.
* a join on a shared *name* (`on="created_at"`) coalesces it, which is the one
  join shape where all the engines agree.
* `union` is **positional** and `unionByName` is not. A pandas `concat` or a
  Polars `concat` aligns on names, so a pipeline ported to Spark with `union`
  silently pairs the columns in whatever order each frame happens to have them.
  This is the most expensive disagreement in this file, because it produces no
  error at all - just a frame whose values are in the wrong columns.
* `F.size(col("tags"))` is `int32`, named `size(tags)`. Polars reports `uint32`
  under the original column name: a dtype change *and* a name change for one
  call.
* **nullability is real, and it propagates.** Spark tracks it per field through
  analysis, so a contract that declares a column non-nullable can actually be
  checked here - against Polars, which does not model it at all, the same
  contract is unenforceable.
* `TimestampType` carries **no timezone of its own**: it is an instant rendered
  in the session's timezone. The adapter pins the sessions it creates to UTC,
  which is what makes `datetime[us, UTC]` round trip, and reads the zone back
  from a session it adopted rather than assuming one. See
  `rowless_pyspark.dtypes`.
* a UDF with no `returnType` is **not inferred**: Spark types the column
  `StringType` by default. That is a default dressed as an answer, so the
  adapter degrades the run and names the function. Polars calls the UDF to find
  out, which rowless declines to do without `probe_udfs`; Spark does not even
  offer that, which is why this adapter does not claim `udf_probe`.
* `CHAR(n)`, `VARCHAR(n)` and a collated string are all `string` in the IR. The
  length limit and the collation are real to Spark and invisible to a contract,
  which is worth knowing when one passes here and fails on an engine that
  models the width.
* `collect()` costs confidence here, where the Polars adapter exempts it. A
  Polars pipeline *ends* in `collect()`; a Spark pipeline ends by returning the
  DataFrame, and `collect()` means pulling rows into the driver. The same word
  means different things, so it is treated differently.

## Inputs the IR can describe and Spark cannot

`materialize` has to produce a frame for whatever schema a source resolved. An
input column whose dtype is UNKNOWN, ANY, or a dtype Spark genuinely lacks
(`uint32`, `enum{...}`, `datetime[ms]`) is materialized as `NullType` and the
run is degraded, with the column named in the reason. The alternative -
refusing to check the pipeline at all - loses the other fifteen columns to one
unresolved source, and a degraded run cannot be reported as a pass, so nothing
is claimed that was not earned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from rowless.adapters.base import Capabilities, DryRunContext, Egress
from rowless.errors import DuplicateColumn, UnknownSchemaError, UnsupportedDType
from rowless.ir.dtypes import DType
from rowless.ir.schema import Field, Relation, Schema

from rowless_pyspark import dtypes as dtype_map
from rowless_pyspark import errors, interception, lineage
from rowless_pyspark import session as run
from rowless_pyspark import spark as cluster

#: The oldest PySpark this adapter supports, and a *tested* floor: CI runs the
#: suite against it as well as against the current release. Everything newer
#: that is used is behind a feature check rather than a version check - the
#: `TIME` type (Spark 4.1) is declared absent when it is missing, and the
#: Connect classes are patched separately from the classic ones because on 3.5
#: they do not subclass them.
MINIMUM_PYSPARK: tuple[int, int] = cluster.MINIMUM_PYSPARK


class PySparkAdapter:
    """PySpark, via the Catalyst analyzed plan over Spark Connect."""

    name: ClassVar[str] = "pyspark"
    requires: ClassVar[str] = "pyspark>=3.5"
    capabilities: ClassVar[Capabilities] = Capabilities(
        column_lineage=True,
        sql=True,
        lazy=True,
        nested_types=True,
        remote=True,
    )

    def available(self) -> bool:
        """True when pyspark imports and Connect or local mode is configured.

        More than an import check, and deliberately: pyspark installs happily
        with no cluster to talk to and no Java new enough to start one, and
        reporting the adapter as active in that state would turn every Spark
        pipeline in a repository into a traceback. `unavailable_reason()` is
        where the actual blocker is named.
        """
        try:
            return cluster.usable()
        except Exception:
            # The protocol says this method may not raise. A half-installed
            # PySpark, or an unreadable JDK layout, counts as unavailable.
            return False

    def unavailable_reason(self) -> str:
        """One line for `rowless doctor` saying what to configure.

        The optional hook the registry prefers over `requires`, and this
        adapter is the case it exists for: the usual reason a Spark dry run
        cannot run is configuration, which a PEP 508 requirement cannot
        describe.
        """
        try:
            return cluster.unavailable_reason()
        except Exception:  # pragma: no cover - a doctor line is never fatal
            return "requires pyspark>=3.5 and a Spark Connect endpoint or a local JVM"

    def handles(self, obj: object) -> bool:
        """True for pyspark.sql DataFrame and Column, Connect or classic."""
        claimed = _frame_types() + _column_types()
        return bool(claimed) and isinstance(obj, claimed)

    def egress(self, options: Mapping[str, Any] | None = None) -> Egress:
        """What the sandbox has to permit for this adapter to reach Spark.

        Read by the worker before the sandbox is installed, which is why it
        takes the options rather than a context: there is no open run yet.
        Connect asks for exactly the configured endpoint. Local mode asks for
        loopback and for a subprocess, because starting a JVM is starting a
        process, and an adapter that needs that has to say so rather than have
        the audit hook discover it mid-run.
        """
        hosts = cluster.egress_hosts(options)
        return Egress(
            hosts=hosts,
            subprocesses=cluster.mode(options) == "local",
            reason=(
                "Spark Connect analysis happens on the server"
                if cluster.mode(options) != "local"
                else "a local Spark session runs in a JVM child process"
            ),
        )

    def materialize(self, schema: Schema, ctx: DryRunContext) -> Any:
        """`spark.createDataFrame([], schema)` from the IR schema.

        Also the point where this adapter installs its interception and gets
        hold of a session: the runtime calls `materialize` before it imports or
        calls user code, so the wrappers are in place for whatever the pipeline
        does next - and a pipeline module whose own `getOrCreate()` runs at
        import time gets this session rather than starting a second one.
        """
        interception.install(ctx)
        spark = cluster.session()
        fields = tuple(
            Field(
                name=field.name,
                dtype=self._input_dtype(field, ctx),
                nullable=field.nullable,
            )
            for field in schema.fields
        )
        with run.own_work():
            frame = spark.createDataFrame([], dtype_map.struct_type(fields))
        lineage.TRACKER.seed(frame, [field.name for field in fields])
        return frame

    def describe(self, obj: object) -> Relation:
        """Read `df.schema`, which runs analysis but no job."""
        if not isinstance(obj, _frame_types() or ()):
            raise UnknownSchemaError(_cannot_describe(obj))
        with run.own_work():
            try:
                native = obj.schema  # type: ignore[attr-defined]
            except Exception as exc:
                errors.raise_translated(exc, available=_columns_of(obj))
        resolved = dtype_map.to_schema(native)
        _refuse_duplicates(resolved)
        return Relation(schema=resolved, engine=self.name)

    def to_ir(self, native: Any) -> DType:
        """Spark DataType to IR, including nested and decimal types."""
        return dtype_map.to_ir(native)

    def from_ir(self, dtype: DType) -> Any:
        """IR to a Spark DataType, raising `UnsupportedDType` where none exists."""
        return dtype_map.from_ir(dtype)

    def unsupported_dtypes(self) -> Sequence[DType]:
        """Corpus dtypes Spark has no equivalent for, so the suite checks the refusal.

        Unsigned integers and enums, because Spark has neither; the millisecond
        and nanosecond timestamps, because Spark timestamps are microseconds;
        and `time[us]` as well on a PySpark older than 4.1, which has no `TIME`
        type at all. See `rowless_pyspark.dtypes`.
        """
        return dtype_map.unsupported_dtypes()

    def run_sql(
        self,
        sql: str,
        inputs: Mapping[str, object],
        ctx: DryRunContext,
    ) -> Relation:
        """Register temp views for the inputs, then analyze the query.

        The seam, in the one mode Spark offers without touching data:
        `CREATE OR REPLACE TEMPORARY VIEW` is session-scoped, so registering an
        input writes nothing to a metastore and leaves nothing behind on the
        cluster. `inputs` maps each name the SQL uses to a DataFrame or a
        `Schema` rowless has already resolved.

        Two honest limits, both documented rather than worked around. A
        qualified name - the three-part relation dbt compiles `ref()` into -
        cannot be a temporary view, so an input is registered under its leaf
        name and a reference to the full name resolves against the cluster's
        own catalog if it happens to exist there. And the columns come back
        **untraced**: Spark exposes no column lineage for a parsed query, and
        attributing them by name would be a guess. `rowless-sql` and
        `rowless-ibis` trace SQL with sqlglot, and a dbt model routed through
        either of those gets lineage this one cannot provide.
        """
        interception.install(ctx)
        statement = _strip_terminator(sql)
        if not interception.is_query(statement):
            kind = interception.statement_kind(statement) or "empty"
            raise UnknownSchemaError(
                f"run_sql was given a {kind} statement, and rowless resolves queries: a dry run "
                "does not send anything that could change a table"
            )
        spark = cluster.session()
        with run.own_work():
            for name, value in inputs.items():
                self._register(str(name), value, ctx)
            try:
                frame = spark.sql(statement)
                native = frame.schema
            except Exception as exc:
                errors.raise_translated(exc)
        resolved = dtype_map.to_schema(native)
        _refuse_duplicates(resolved)
        if resolved.fields:
            ctx.note(
                "rowless resolved this query's schema with Spark's own analyzer but cannot "
                "attribute its columns: Spark reports no column lineage for a query, so they are "
                "reported as untraced rather than guessed"
            )
        return Relation(schema=resolved, engine=self.name)

    def column_lineage(self, obj: object) -> Mapping[str, tuple[str, ...]]:
        """Attribute each output column to the input columns it was computed from.

        Recorded while the plan was built, by the wrappers in
        `rowless_pyspark.interception`. A frame produced by an operation those
        wrappers do not cover, or built while
        `[engine.pyspark] validate_each_call` was off, has no entry - so its
        columns are reported untraced rather than attributed to a guess. See
        `rowless_pyspark.lineage` for why this is recorded rather than read off
        the analyzed plan's expression IDs.
        """
        return lineage.TRACKER.of(obj) or {}

    def conformance_cases(self) -> Sequence[Any]:
        """The operations battery for Spark, with the per-engine expectations."""
        from rowless_pyspark.cases import conformance_cases

        return conformance_cases()

    def _input_dtype(self, field: Field, ctx: DryRunContext) -> DType:
        """The IR dtype to materialize one input column as, degrading when Spark has none.

        Returns an IR dtype rather than a Spark one so that `materialize` keeps
        a single conversion point: the degradation decision is here, and
        `dtypes.struct_type` does the lowering.
        """
        from rowless.ir.dtypes import null

        if not field.dtype.is_known:
            ctx.degrade(
                f"input column {field.name!r} has an unresolved dtype "
                f"({field.dtype.reason or 'no reason given'}); it was materialized as NullType"
            )
            return null()
        if field.dtype.is_any:
            ctx.note(f"input column {field.name!r} is typed ANY; it was materialized as NullType")
            return null()
        try:
            dtype_map.from_ir(field.dtype)
        except UnsupportedDType as exc:
            ctx.degrade(f"input column {field.name!r}: {exc}; it was materialized as NullType")
            return null()
        return field.dtype

    def _register(self, name: str, value: object, ctx: DryRunContext) -> None:
        """Register one `run_sql` input as a temporary view under the name given.

        A `Schema` is accepted as well as a DataFrame, since a caller that has
        one has no reason to build a frame first - and `rowless-dbt` has one for
        every upstream model in a manifest.
        """
        frame: Any = self.materialize(value, ctx) if isinstance(value, Schema) else value
        if not self.handles(frame):
            qualified = f"{type(value).__module__}.{type(value).__qualname__}"
            raise UnknownSchemaError(
                f"the pyspark adapter cannot register {qualified} as the SQL input {name!r}; "
                "pass a Spark DataFrame or a rowless Schema"
            )
        leaf = _leaf_name(name)
        if leaf != name:
            ctx.note(
                f"the SQL input {name!r} was registered as the temporary view {leaf!r}: a Spark "
                "temporary view cannot carry a qualified name, so a query that spells the full "
                "name resolves against the cluster's own catalog instead"
            )
        frame.createOrReplaceTempView(leaf)


# --- Helpers -----------------------------------------------------------------


def _frame_types() -> tuple[type, ...]:
    """The DataFrame classes this adapter claims, classic and Connect.

    Both are listed because on PySpark 3.5 the Connect DataFrame is a separate
    class, and from Spark 4 it subclasses the classic one. `isinstance` against
    the pair is correct either way.
    """
    return _classes(
        ("pyspark.sql.dataframe", "DataFrame"),
        ("pyspark.sql.connect.dataframe", "DataFrame"),
    )


def _column_types() -> tuple[type, ...]:
    """The Column classes this adapter claims, classic and Connect."""
    return _classes(
        ("pyspark.sql.column", "Column"),
        ("pyspark.sql.connect.column", "Column"),
    )


def _classes(*targets: tuple[str, str]) -> tuple[type, ...]:
    """Import the named classes, skipping the ones this install does not have.

    Never at module scope: `available()` has to work for a user who has no
    pyspark, and the Connect classes need the client extras.
    """
    found: list[type] = []
    for module_name, class_name in targets:
        try:
            module = __import__(module_name, fromlist=[class_name])
        except Exception:
            continue
        candidate = getattr(module, class_name, None)
        if isinstance(candidate, type) and candidate not in found:
            found.append(candidate)
    return tuple(found)


def _cannot_describe(obj: object) -> str:
    """The message for an object this adapter cannot read a schema off.

    A `Column` gets its own sentence, because it is the near miss: Spark cannot
    type a column expression without a frame to resolve it against, so a
    pipeline that returns one has to say which frame it belongs to.
    """
    qualified = f"{type(obj).__module__}.{type(obj).__qualname__}"
    columns = _column_types()
    if columns and isinstance(obj, columns):
        return (
            f"the pyspark adapter cannot describe {qualified}: a Spark Column has no dtype until "
            "it is resolved against a frame, so a pipeline has to return the DataFrame - "
            "df.select(that_column) is the one-line fix"
        )
    return (
        f"the pyspark adapter cannot describe {qualified}; a pipeline has to return a Spark "
        "DataFrame"
    )


def _columns_of(frame: object) -> tuple[str, ...]:
    """The column names of a frame, or nothing when it will not say."""
    try:
        return tuple(str(name) for name in frame.columns)  # type: ignore[attr-defined]
    except Exception:
        return ()


def _refuse_duplicates(schema: Schema) -> None:
    """Report two output columns of the same name, which Spark allows and rowless cannot.

    A join on a condition keeps both sides whole, so two frames that share a
    column name produce that name twice. Spark is content with it - a later
    reference is simply ambiguous, and it says so when one is made - but a
    rowless schema is addressed by name, and so is everything downstream of it:
    a contract, a diff, a lineage edge. Reporting the duplicate is the only
    honest option, and the message names the fix.
    """
    seen: set[str] = set()
    for field in schema.fields:
        if field.name in seen:
            raise DuplicateColumn(
                f"this plan produces two columns named {field.name!r}. Spark keeps both columns "
                "of a colliding name after a join rather than renaming one, so nothing "
                "downstream can refer to either; alias one of them, or join on the shared name "
                "so the key is coalesced"
            )
        seen.add(field.name)


def _leaf_name(name: str) -> str:
    """The last part of a possibly-qualified relation name, unquoted."""
    parts = [part.strip().strip('`"') for part in name.split(".") if part.strip()]
    return parts[-1] if parts else name


def _strip_terminator(sql: str) -> str:
    """Drop a trailing semicolon, which Spark refuses in a single statement."""
    return sql.strip().rstrip(";").strip()

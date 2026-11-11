"""Library-boundary interception: the Spark calls a dry run has to notice.

This is layer 2 of the sandbox described in `rowless.runtime.sandbox`, the one
that has to live in an adapter because only an adapter knows which calls of its
engine matter. It matters more here than in any other adapter in the project,
because this is the only engine where a dry run is pointed at a **live
cluster**: the sandbox's audit hook is deliberately holding the Connect
endpoint open, so a `spark.sql("INSERT OVERWRITE ...")` in a pipeline would
reach production over a socket rowless itself allowed. The audit hook cannot
tell that call from the analysis round trips, and these wrappers are what can.

Five jobs, deliberately in one place so that the full list of what rowless does
to PySpark can be read in one siting:

1. **Statements that are not queries.** `spark.sql` is the one call where a
   string decides whether Spark reads or writes. Anything that is not a query -
   `INSERT`, `CREATE`, `DROP`, `MERGE`, `REFRESH`, `GRANT` - is recorded as an
   output of the pipeline and **not sent**, and the empty frame Spark itself
   returns for a command is returned in its place. A query is passed straight
   through, because analyzing one moves no data.
2. **Writers.** `DataFrameWriter.save`, `saveAsTable`, `insertInto`, the V2
   writer's `append` and `overwrite`, `DataStreamWriter.start` and
   `Catalog.createTable` are replaced with recorders. "This pipeline writes to
   warehouse.orders" is information worth keeping, and it is information the
   audit hook cannot produce once the endpoint is allowlisted.
3. **Row dependence.** Everything that turns a DataFrame into Python values -
   `collect()`, `count()`, `first()`, `head()`, `isEmpty()`, `toPandas()` -
   answers from zero rows, so the schema rowless computed belongs to a code
   path production would not take. Each one is reported through
   `rowless.runtime.rowdep.observe`, which downgrades the result to
   `Confidence.UNKNOWN`.

   This adapter does **not** follow the Polars adapter in exempting
   `collect()`, and the difference is worth stating because it looks like an
   inconsistency. A Polars pipeline ends in `collect()`: it is how the plan is
   run, and the schema it returns is the schema the plan resolved to. A Spark
   pipeline ends by returning the DataFrame, and `collect()` means something
   else entirely - it pulls rows into the driver, which is exactly the
   row-dependent read the proposal is worried about. Treating the same word the
   same way in both engines would be the wrong answer in one of them.
4. **Analysis at the line that broke it.** Under Connect a plan is not analyzed
   until something asks, so `df.select("nope")` would resolve successfully and
   fail at the end of the dry run, pointing the finding at the wrong line. Each
   wrapped builder analyzes its result immediately, which costs one round trip
   and buys a finding that points at the user's own call. Analysis launches no
   job. `[engine.pyspark] validate_each_call = false` turns it off for a
   pipeline where the round trips are not worth it, at the price of findings
   that point at the end of the run and of lineage, which needs the resolved
   names.
5. **Column lineage.** The same wrappers record which input column each output
   column came from. See `rowless_pyspark.lineage`.

Two rules the patching follows. It is **installed once per process and gated
per run**: the wrappers check `session.tracking()` and do nothing outside a dry
run, so importing this module does not change how PySpark behaves for anybody
else in the process. And every wrapper **calls the original**, with the one
documented exception of the writers and the non-query statements, which are the
side effects the whole sandbox exists to stop.
"""

from __future__ import annotations

import functools
import re
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any

from rowless.adapters.base import DryRunContext
from rowless.runtime.rowdep import observe

from rowless_pyspark import errors, lineage
from rowless_pyspark import session as run
from rowless_pyspark import spark as cluster

#: Accessors whose answer depends on there being rows. The canonical list of
#: what rowless knows about lives in `rowless.runtime.rowdep.ROW_DEPENDENT_APIS`;
#: this is the Spark spelling of it, restricted to the calls that produce Python
#: values. Everything on `Column` and `GroupedData` builds a plan instead, which
#: is why a Spark pipeline comes out unverified far less often than a pandas one.
ROW_DEPENDENT: dict[str, tuple[str, ...]] = {
    "DataFrame": (
        "collect",
        "count",
        "first",
        "head",
        "take",
        "tail",
        "isEmpty",
        "toPandas",
        "toArrow",
        "toLocalIterator",
        "foreach",
        "foreachPartition",
        "show",
    ),
}

#: Writers, replaced with recorders. The streaming writer is in here because
#: `start()` would launch a query on the cluster that outlives the dry run.
WRITERS: dict[str, tuple[str, ...]] = {
    "DataFrameWriter": (
        "save",
        "saveAsTable",
        "insertInto",
        "parquet",
        "csv",
        "json",
        "orc",
        "text",
        "jdbc",
        "xml",
    ),
    "DataFrameWriterV2": (
        "create",
        "replace",
        "createOrReplace",
        "append",
        "overwrite",
        "overwritePartitions",
    ),
    "DataStreamWriter": ("start", "toTable"),
    "Catalog": ("createTable", "createExternalTable"),
}

#: Builders that keep their column names: lineage carries straight through, and
#: a column they invent is left untraced rather than attributed to a name that
#: happens to collide.
CARRY_THROUGH: tuple[str, ...] = (
    "filter",
    "where",
    "sort",
    "orderBy",
    "sortWithinPartitions",
    "limit",
    "offset",
    "distinct",
    "dropDuplicates",
    "drop_duplicates",
    "dropDuplicatesWithinWatermark",
    "drop",
    "dropna",
    "fillna",
    "replace",
    "sample",
    "repartition",
    "repartitionByRange",
    "coalesce",
    "cache",
    "persist",
    "unpersist",
    "localCheckpoint",
    "hint",
    "alias",
    "withWatermark",
    "intersect",
    "intersectAll",
    "exceptAll",
    "subtract",
)

#: Builders that produce one output column per expression, in order. The
#: positional rule in `rowless_pyspark.lineage` is what reads them.
PROJECTIONS: tuple[str, ...] = ("select", "selectExpr", "toDF")

#: Grouping entry points, remembered so the `agg` that follows knows the frame
#: and the keys it was built from.
GROUPINGS: tuple[str, ...] = ("groupBy", "groupby", "rollup", "cube")

#: Set operations: a column comes from every side that has one.
SET_OPERATIONS: tuple[str, ...] = ("union", "unionAll", "unionByName")

#: Statements `spark.sql` may send, because resolving one moves no data.
QUERY_KEYWORDS: frozenset[str] = frozenset(
    {"SELECT", "WITH", "VALUES", "TABLE", "FROM", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"}
)

#: `-- line comments` and `/* block comments */`, stripped before the first
#: keyword of a statement is read.
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)

#: Where each owner class lives, classic first and Connect second. Both are
#: patched: from Spark 4 the Connect classes subclass the classic ones, and on
#: 3.5 they do not, so an adapter that patched only one would silently miss a
#: whole mode.
_OWNER_MODULES: dict[str, tuple[tuple[str, str], ...]] = {
    "DataFrame": (
        ("pyspark.sql.dataframe", "DataFrame"),
        ("pyspark.sql.connect.dataframe", "DataFrame"),
    ),
    "DataFrameWriter": (
        ("pyspark.sql.readwriter", "DataFrameWriter"),
        ("pyspark.sql.connect.readwriter", "DataFrameWriter"),
    ),
    "DataFrameWriterV2": (
        ("pyspark.sql.readwriter", "DataFrameWriterV2"),
        ("pyspark.sql.connect.readwriter", "DataFrameWriterV2"),
    ),
    "DataStreamWriter": (
        ("pyspark.sql.streaming.readwriter", "DataStreamWriter"),
        ("pyspark.sql.connect.streaming.readwriter", "DataStreamWriter"),
    ),
    "GroupedData": (
        ("pyspark.sql.group", "GroupedData"),
        ("pyspark.sql.connect.group", "GroupedData"),
    ),
    "SparkSession": (
        ("pyspark.sql.session", "SparkSession"),
        ("pyspark.sql.connect.session", "SparkSession"),
    ),
    "Catalog": (
        ("pyspark.sql.catalog", "Catalog"),
        ("pyspark.sql.connect.catalog", "Catalog"),
    ),
}

#: Modules whose `udf` is wrapped, so a UDF with no declared return type
#: degrades the run instead of silently becoming a string column.
_FUNCTION_MODULES: tuple[str, ...] = (
    "pyspark.sql.functions.builtin",
    "pyspark.sql.functions",
    "pyspark.sql.connect.functions.builtin",
    "pyspark.sql.connect.functions",
)

_INSTALLED = False


@dataclass(frozen=True, slots=True)
class _Grouping:
    """The frame and key expressions a `GroupedData` was built from.

    Spark does not expose either on the grouped object in a way both modes
    agree on, so the grouping call is intercepted to remember them for the
    `agg` that follows.
    """

    frame: Any
    keys: tuple[Any, ...]


#: `id(GroupedData)` to the grouping it came from, purged when it is collected.
_GROUPINGS: dict[int, _Grouping] = {}
_GROUPINGS_ALIVE: dict[int, weakref.ref[Any]] = {}


def install(ctx: DryRunContext) -> None:
    """Patch the PySpark entry points, and bind the wrappers to this dry run.

    Called from `materialize`, which the runtime always calls before user code.
    Patching happens once per process; the per-run caches are reset whenever a
    new run starts.
    """
    global _INSTALLED

    run.begin(ctx)
    if _INSTALLED:
        return
    _install_patches()
    _INSTALLED = True


def installed() -> bool:
    """True once the patches are in place. For tests."""
    return _INSTALLED


def validating() -> bool:
    """Whether each builder call analyzes its result. See the module docstring."""
    configured = run.option("validate_each_call", True)
    return bool(configured)


def statement_kind(sql: str) -> str:
    """The first keyword of a statement, upper-cased, with comments stripped.

    What decides whether `spark.sql` is a read or a write. Deliberately a
    keyword check rather than a parse: rowless has no business reimplementing
    Spark's grammar here, and the question is only whether this string could
    possibly change anything.
    """
    text = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql or ""))
    text = text.strip().lstrip("(").strip()
    for token in text.split():
        cleaned = token.strip("(;").upper()
        if cleaned:
            return cleaned
    return ""


def is_query(sql: str) -> bool:
    """True when a statement only reads, so analyzing it is safe."""
    return statement_kind(sql) in QUERY_KEYWORDS


# --- Patching ----------------------------------------------------------------


def _install_patches() -> None:
    """Wrap every PySpark entry point this adapter needs to see."""
    for owner_name, methods in ROW_DEPENDENT.items():
        for method in methods:
            _patch_all(owner_name, method, _row_dependent_wrapper)
    for owner_name, methods in WRITERS.items():
        for method in methods:
            _patch_all(owner_name, method, _writer_wrapper)
    for method in CARRY_THROUGH:
        _patch_all("DataFrame", method, _builder(rule=_carry_rule))
    for method in PROJECTIONS:
        _patch_all("DataFrame", method, _builder(rule=_projection_rule))
    for method in SET_OPERATIONS:
        _patch_all("DataFrame", method, _builder(rule=_union_rule))
    for method in ("withColumn", "withColumns"):
        _patch_all("DataFrame", method, _builder(rule=_added_rule))
    for method in ("withColumnRenamed", "withColumnsRenamed"):
        _patch_all("DataFrame", method, _builder(rule=_rename_rule))
    for method in ("join", "crossJoin", "joinWith"):
        _patch_all("DataFrame", method, _builder(rule=_join_rule))
    for method in ("unpivot", "melt", "transpose"):
        # These invent their own output columns, which nothing in the call
        # names, so they are validated and left untraced.
        _patch_all("DataFrame", method, _builder(rule=_untraced_rule))
    for method in GROUPINGS:
        _patch_all("DataFrame", method, _grouping_wrapper)
    _patch_all("GroupedData", "agg", _builder(rule=_aggregate_rule, grouped=True))
    for method in ("count", "sum", "min", "max", "mean", "avg", "pivot", "apply", "applyInPandas"):
        _patch_all("GroupedData", method, _builder(rule=_untraced_rule, grouped=True))
    _patch_all("SparkSession", "sql", _sql_wrapper)
    _patch_udf()


def _patch_all(owner_name: str, method: str, factory: Callable[[str, Any], Any]) -> None:
    """Patch one method on every class that spells it, classic and Connect."""
    for module_name, class_name in _OWNER_MODULES.get(owner_name, ()):
        owner = _load_class(module_name, class_name)
        if owner is None:
            continue
        _patch(owner, owner_name, method, factory)


def _load_class(module_name: str, class_name: str) -> type | None:
    """Import one owner class, or None when this install does not have it.

    The Connect classes need the client extras, and an engine release may not
    have a method - or a whole class - this adapter knows about. Neither is an
    error: a class that is not there cannot be called.
    """
    try:
        module = __import__(module_name, fromlist=[class_name])
    except Exception:
        return None
    found = getattr(module, class_name, None)
    return found if isinstance(found, type) else None


def _patch(
    owner: type,
    owner_name: str,
    method: str,
    factory: Callable[[str, Any], Any],
) -> None:
    """Replace one method with a wrapper, skipping what this PySpark lacks.

    Only a method the class spells *itself* is wrapped. From Spark 4 the
    Connect classes subclass the classic ones, so an inherited method is
    already wrapped on the class that defines it and wrapping it again would
    report every call twice.
    """
    original = owner.__dict__.get(method)
    if original is None or getattr(original, "__rowless_wrapped__", False):
        return
    if not callable(original):
        return
    wrapper = factory(f"pyspark.{owner_name}.{method}", original)
    wrapper.__rowless_wrapped__ = True
    setattr(owner, method, wrapper)


def _patch_udf() -> None:
    """Wrap `udf` so an undeclared return type is reported rather than defaulted.

    Spark does not infer a UDF's return type: with no `returnType` it uses
    `StringType`, silently. That is a default dressed as an answer, and a
    column rowless reported as a string on that basis would be a schema nobody
    checked. `pandas_udf` is not wrapped, because it requires a return type.
    """
    for module_name in _FUNCTION_MODULES:
        try:
            # Typed as Any deliberately: this is a module object being
            # rebound, and the whole point of the call is that the attribute
            # is not the one the type checker knows about any more.
            module: Any = __import__(module_name, fromlist=["udf"])
        except Exception:
            continue
        original = getattr(module, "udf", None)
        if original is None or getattr(original, "__rowless_wrapped__", False):
            continue
        module.udf = _udf_wrapper(f"{module_name}.udf", original)


def _builder(*, rule: _Rule, grouped: bool = False) -> Callable[[str, Any], Any]:
    """A `_patch` factory for one builder method, bound to its lineage rule."""
    return functools.partial(_builder_wrapper, rule=rule, grouped=grouped)


# --- Wrappers ----------------------------------------------------------------


def _row_dependent_wrapper(api: str, original: Any) -> Any:
    """Report a row-count-sensitive call, then hand back what Spark returned."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not run.tracking():
            return original(self, *args, **kwargs)
        with _translating(self):
            result = original(self, *args, **kwargs)
        observe(api, returned=_summarise(result))
        return result

    return wrapper


def _writer_wrapper(api: str, original: Any) -> Any:
    """Record a write instead of performing it, and say so in the notes."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not run.tracking():
            return original(self, *args, **kwargs)
        target = _write_target(args, kwargs) or _writer_target(self) or "an unnamed destination"
        run.note(
            f"{api} would write to {target}; a dry run resolves schemas and writes nothing",
            once_for=f"{api}:{target}",
        )
        return None

    return wrapper


def _sql_wrapper(api: str, original: Any) -> Any:
    """Pass a query through; record anything that could change something.

    The most important wrapper in the adapter. The dry run holds a socket to a
    real cluster, so this is the one place where a string in a pipeline decides
    whether rowless reads a catalog or rewrites a table.
    """

    @functools.wraps(original)
    def wrapper(self: Any, sqlQuery: Any = "", *args: Any, **kwargs: Any) -> Any:
        statement = sqlQuery if isinstance(sqlQuery, str) else ""
        if not run.tracking():
            return original(self, sqlQuery, *args, **kwargs)
        if is_query(statement):
            with _translating(self):
                return original(self, sqlQuery, *args, **kwargs)
        kind = statement_kind(statement)
        run.note(
            f"{api} would run a {kind} statement, which is not a query: it was recorded as an "
            "output of this pipeline and not sent. A dry run resolves schemas and changes "
            "nothing",
            once_for=f"{api}:{kind}:{statement[:120]}",
        )
        with run.own_work():
            return _command_result(self)

    return wrapper


def _udf_wrapper(api: str, original: Any) -> Any:
    """Degrade the run when a UDF declares no return type."""

    @functools.wraps(original)
    def wrapper(f: Any = None, returnType: Any = None, *args: Any, **kwargs: Any) -> Any:
        declared = returnType if returnType is not None else kwargs.get("returnType")
        if run.tracking() and declared is None:
            qualified = getattr(f, "__qualname__", None) or "a UDF"
            run.degrade(
                f"{qualified} declares no return type, so Spark types its column as a string by "
                "default rather than by inference; pass returnType= to udf() so the column has a "
                "dtype rowless can check"
            )
            run.note(f"{api} was called without returnType", once_for=f"{api}:{qualified}")
        if returnType is None:
            return original(f, *args, **kwargs) if f is not None else original(*args, **kwargs)
        return original(f, returnType, *args, **kwargs)

    return wrapper


@dataclass(frozen=True, slots=True)
class _Plan:
    """One intercepted builder call, with its result already analyzed.

    Everything a lineage rule needs, so the rules stay pure functions of the
    call and the wrapper below stays the only place that knows about tracking,
    reentrancy and analysis.
    """

    receiver: Any
    """What the method was called on: a `DataFrame`, or a `GroupedData`."""

    source: Any
    """The frame the result derives from. The receiver, except for a grouped
    call, where it is the frame the `GroupedData` was built from."""

    result: Any
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    columns: tuple[str, ...]
    """Resolved output column names, in order."""

    @property
    def parent(self) -> lineage.Lineage | None:
        """Lineage of the source frame, or None when it was never tracked."""
        return lineage.TRACKER.of(self.source)


#: Maps one intercepted call onto the lineage of the frame it produced, or None
#: when there is nothing to record.
_Rule = Callable[[_Plan], "lineage.Lineage | None"]


def _builder_wrapper(api: str, original: Any, *, rule: _Rule, grouped: bool = False) -> Any:
    """Wrap a builder method: analyze its result, then record its lineage.

    The shared half of every plan-building interception. Analyzing the result is
    the validation - a plan that cannot be analyzed raises here, inside the user
    call, rather than at the end of the dry run - and `rule` is the per-method
    part that says where each output column came from.
    """
    _ = api

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not run.tracking():
            return original(self, *args, **kwargs)
        with _translating(self):
            result = original(self, *args, **kwargs)
        if not _is_dataframe(result) or not validating():
            return result
        with run.own_work():
            source = _grouped_source(self) if grouped else self
            columns = _analyze(result, parent=source)
            if columns is None:
                return result
            recorded = rule(
                _Plan(
                    receiver=self,
                    source=source,
                    result=result,
                    args=args,
                    kwargs=kwargs,
                    columns=columns,
                )
            )
            if recorded is not None:
                lineage.TRACKER.record(result, recorded)
        return result

    return wrapper


def _grouping_wrapper(api: str, original: Any) -> Any:
    """Remember the frame and keys behind a `GroupedData`, for the `agg` to come."""
    _ = api

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not run.tracking():
            return original(self, *args, **kwargs)
        with _translating(self):
            group = original(self, *args, **kwargs)
        with run.own_work():
            _remember_grouping(group, self, args, kwargs)
        return group

    return wrapper


# --- Lineage rules -----------------------------------------------------------


def _carry_rule(plan: _Plan) -> lineage.Lineage | None:
    """`filter`, `sort`, `drop`, `limit`: the surviving names keep their lineage."""
    parent = plan.parent
    return None if parent is None else lineage.carry(parent, plan.columns)


def _projection_rule(plan: _Plan) -> lineage.Lineage | None:
    """`select` and `selectExpr`: one output column per expression, in order."""
    parent = plan.parent
    if parent is None:
        return None
    return lineage.positional(parent, plan.columns, lineage.terms(plan.args, plan.kwargs))


def _added_rule(plan: _Plan) -> lineage.Lineage | None:
    """`withColumn` and `withColumns`: Spark states the output name itself."""
    parent = plan.parent
    if parent is None:
        return None
    return lineage.added(parent, plan.columns, _additions(plan.args, plan.kwargs))


def _rename_rule(plan: _Plan) -> lineage.Lineage | None:
    """`withColumnRenamed`: the new name inherits what the old name had."""
    parent = plan.parent
    if parent is None:
        return None
    return lineage.renamed(parent, plan.columns, _renames(plan.args, plan.kwargs))


def _join_rule(plan: _Plan) -> lineage.Lineage | None:
    """A join: each output column comes from the side that produced it."""
    left = plan.parent
    other = plan.args[0] if plan.args else plan.kwargs.get("other")
    right = lineage.TRACKER.of(other) if other is not None else None
    if left is None and right is None:
        return None
    return lineage.joined(left or {}, right, plan.columns)


def _union_rule(plan: _Plan) -> lineage.Lineage | None:
    """`union` and `unionByName`: a column comes from every side that has one."""
    other = plan.args[0] if plan.args else plan.kwargs.get("other")
    parents = [plan.parent, lineage.TRACKER.of(other) if other is not None else None]
    if all(parent is None for parent in parents):
        return None
    return lineage.merged(parents, plan.columns)


def _aggregate_rule(plan: _Plan) -> lineage.Lineage | None:
    """`agg`: the group keys first, then the aggregates, which is Spark's own order."""
    parent = plan.parent
    grouping = _GROUPINGS.get(id(plan.receiver))
    if parent is None or grouping is None:
        return None
    terms = [*grouping.keys, *lineage.terms(plan.args, plan.kwargs)]
    return lineage.positional(parent, plan.columns, terms)


def _untraced_rule(plan: _Plan) -> lineage.Lineage | None:
    """A call whose output columns nothing in it names: validated, not traced.

    `unpivot` invents `variable` and `value`; `GroupedData.pivot` invents one
    column per distinct value, which it has to *read the data* to find. Those
    columns have no expression to attribute them to, and attributing them to a
    name that happens to collide would be a guess.
    """
    _ = plan
    return {}


def _grouped_source(group: Any) -> Any:
    """The frame a `GroupedData` was built from, or None when it was not seen."""
    grouping = _GROUPINGS.get(id(group))
    return grouping.frame if grouping is not None else None


# --- Helpers -----------------------------------------------------------------


@contextmanager
def _translating(receiver: Any) -> Iterator[None]:
    """Run an engine call as this adapter's own work, translating a Spark failure.

    Two jobs in one place. `own_work()` keeps the call from being reported as
    something the pipeline did, and the translation is what makes a finding
    point at the user's line - which matters more than it looks, because the
    two modes fail at different times. A classic session analyzes **eagerly**,
    so `df.select("nope")` raises inside the builder call itself; under Connect
    the same plan resolves and fails later, when `_analyze` reads the schema.
    Wrapping both means one finding either way instead of a translated error in
    one mode and a raw traceback in the other.
    """
    try:
        with run.own_work():
            yield
    except Exception as exc:
        errors.raise_translated(exc, available=_columns_of(receiver))


def _is_dataframe(obj: object) -> bool:
    """True when `obj` is a Spark DataFrame, the only thing the builders track."""
    for module_name, class_name in _OWNER_MODULES["DataFrame"]:
        owner = _load_class(module_name, class_name)
        if owner is not None and isinstance(obj, owner):
            return True
    return False


def _analyze(frame: Any, *, parent: Any = None) -> tuple[str, ...] | None:
    """Analyze the plan now, translating a failure into a finding.

    Returns the output column names, or None when Spark failed in a way this
    adapter cannot classify. None means "leave it for `describe`": analyzing
    the same plan again at the end of the dry run reports the engine message
    through `PIPELINE_RAISED`, which is better than guessing at a code here.

    `df.schema` runs Catalyst analysis and launches no job, which is the whole
    reason this adapter can afford to do it on every builder call. Under
    Connect it is one round trip to a server that is already running.

    `parent` is the frame the operation was called on. Its columns are the
    fallback for a missing-column finding's near miss, for the rare condition
    where Spark does not compute a proposal of its own.
    """
    try:
        schema = frame.schema
    except Exception as exc:
        translated = errors.translate(exc, available=_columns_of(parent))
        if translated is not None:
            raise translated from exc
        return None
    return tuple(str(field.name) for field in getattr(schema, "fields", ()))


def _columns_of(frame: Any) -> tuple[str, ...]:
    """The column names of a frame, or nothing when it will not say."""
    if frame is None:
        return ()
    try:
        return tuple(str(name) for name in frame.columns)
    except Exception:
        # The parent plan is broken too, so it has no column list to offer. The
        # finding still names the column that was asked for.
        return ()


def _command_result(spark: Any) -> Any:
    """The empty frame Spark returns for a statement that is not a query.

    Spark answers a command with a DataFrame of no columns and no rows, so that
    is what the recorder hands back: a pipeline that returns the result of its
    own `INSERT` gets the same shape it would have got, without the insert.
    """
    import pyspark.sql.types as t

    builders: tuple[Callable[[], Any], ...] = (
        lambda: spark.createDataFrame([], t.StructType([])),
        lambda: spark.range(0).select(),
        lambda: spark.range(0).limit(0),
    )
    for build in builders:
        try:
            return build()
        except Exception:
            continue
    # Nothing could be built, which means the session itself is unusable. The
    # caller is better served by Spark's own failure than by a None.
    return spark.sql("SELECT 1 WHERE false")


def _summarise(result: object) -> object:
    """A short value for the diagnostic: what the call handed back on zero rows."""
    if isinstance(result, bool | int | float | str) or result is None:
        return result
    if isinstance(result, list | tuple):
        return f"{len(result)} rows"
    shape = getattr(result, "shape", None)
    if isinstance(shape, tuple) and shape and isinstance(shape[0], int):
        return f"{shape[0]} rows"
    return None


def _write_target(args: Sequence[Any], kwargs: Mapping[str, Any]) -> str | None:
    """Where a writer was pointed, from its arguments."""
    import os

    candidates: list[Any] = [*args]
    for key in ("path", "name", "tableName", "table", "url", "source"):
        if key in kwargs:
            candidates.append(kwargs[key])
    for candidate in candidates:
        if isinstance(candidate, str | os.PathLike):
            return str(candidate)[:200]
    return None


def _writer_target(writer: Any) -> str | None:
    """Where a writer was pointed before the call, when it carries it.

    `df.write.format("delta").option("path", ...)` puts the destination on the
    writer rather than in the call, and the V2 writer keeps the table name it
    was constructed with.
    """
    for attribute in ("_table_name", "_table", "_source", "_format"):
        found = getattr(writer, attribute, None)
        if isinstance(found, str) and found:
            return found[:200]
    options = getattr(writer, "_options", None)
    if isinstance(options, Mapping):
        for key in ("path", "table", "dbtable"):
            found = options.get(key)
            if isinstance(found, str) and found:
                return found[:200]
    return None


def _additions(args: Sequence[Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The name-to-expression pairs a `withColumn`/`withColumns` call carries."""
    if args and isinstance(args[0], Mapping):
        return {str(name): value for name, value in args[0].items()}
    if len(args) >= 2 and isinstance(args[0], str):
        return {args[0]: args[1]}
    name = kwargs.get("colName") or kwargs.get("existing")
    column = kwargs.get("col")
    if isinstance(name, str) and column is not None:
        return {name: column}
    mapping = kwargs.get("colsMap")
    if isinstance(mapping, Mapping):
        return {str(key): value for key, value in mapping.items()}
    return {}


def _renames(args: Sequence[Any], kwargs: Mapping[str, Any]) -> dict[str, str]:
    """The old-to-new mapping of a rename call, however it was spelled."""
    if args and isinstance(args[0], Mapping):
        return {str(old): str(new) for old, new in args[0].items()}
    if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str):
        return {args[0]: args[1]}
    mapping = kwargs.get("colsMap")
    if isinstance(mapping, Mapping):
        return {str(old): str(new) for old, new in mapping.items()}
    existing, new = kwargs.get("existing"), kwargs.get("new")
    if isinstance(existing, str) and isinstance(new, str):
        return {existing: new}
    return {}


def _remember_grouping(
    group: Any,
    frame: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> None:
    """Record the frame and keys a `GroupedData` was built from."""
    key = id(group)
    _GROUPINGS[key] = _Grouping(frame=frame, keys=tuple(lineage.terms(args, kwargs)))

    def purge(_reference: weakref.ref[Any], key: int = key) -> None:
        """Drop the grouping once the `GroupedData` is gone."""
        _forget_grouping(key)

    with suppress(TypeError):
        _GROUPINGS_ALIVE[key] = weakref.ref(group, purge)


def _forget_grouping(key: int) -> None:
    """Drop a collected `GroupedData` entry."""
    _GROUPINGS.pop(key, None)
    _GROUPINGS_ALIVE.pop(key, None)


def _reset() -> None:
    """Drop what was cached for the previous run."""
    _GROUPINGS.clear()
    _GROUPINGS_ALIVE.clear()
    cluster.forget_if_stopped()


run.on_reset(_reset)

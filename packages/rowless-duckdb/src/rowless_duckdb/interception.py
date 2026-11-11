"""Library-boundary interception: the DuckDB calls a dry run has to notice.

This is layer 2 of the sandbox described in `rowless.runtime.sandbox`, the one
that has to live in an adapter because only an adapter knows which calls of its
engine matter. Four jobs, deliberately in one place so that the full list of
what rowless does to DuckDB can be read in one sitting:

1. **Findings at the line that caused them.** DuckDB binds eagerly, so
   `rel.project("amount_cents")` raises inside the user's call rather than at
   the end of the dry run. That is a gift - the traceback already points at the
   right line - but the exception is a `BinderException`, and CI needs
   `ColumnNotFound` with the column named and a near miss suggested. Every
   builder call is wrapped so the translation in `rowless_duckdb.errors`
   happens where the error was raised.
2. **Row dependence.** Everything that turns a relation into Python values -
   `fetchall()`, `df()`, `len(rel)`, `rel.shape` - answers from zero rows, and
   each one is reported through `rowless.runtime.rowdep.observe`, which
   downgrades the result to `Confidence.UNKNOWN`: production would take the
   other branch.

   DuckDB needs no equivalent of the Polars `collect()` carve-out, and the
   reason is a genuine difference between the engines. `rel.max("amount")`
   returns a *relation* in DuckDB, not a number, so the aggregate methods that
   cost confidence on a Polars `Series` cost nothing here; they are query
   building. Only the handful of accessors that hand back Python objects are
   row-dependent, and a pipeline that ends in one has genuinely stopped
   resolving a schema and started reading data.
3. **Writers.** `write_parquet`, `to_table`, `insert_into` and the rest are
   replaced with recorders, and a `COPY ... TO` in raw SQL is recognised by
   statement type and skipped. The audit hook cannot help here: DuckDB writes
   through its own C++ filesystem and never calls Python's `open`. The scratch
   database closes the same hole from the other side by refusing file access
   outright, and `duckdb.connect` to a real database file is forced read-only.
4. **Column lineage.** The same wrappers record which input column each output
   column came from. See `rowless_duckdb.lineage`.

Two rules the patching follows. It is **installed once per process and gated
per run**: the wrappers check `rowless_duckdb.session.tracking()` and do
nothing outside a dry run, so importing this module does not change how DuckDB
behaves for anybody else in the process. And every wrapper **calls the
original**, with the one documented exception of the writers, which are the
side effects the whole sandbox exists to stop.

Methods that a given DuckDB does not have are skipped rather than failing the
install: `with_columns`, `rename`, `to_dicts` and `iter_rows` arrived after the
declared floor of 1.0, and the adapter supports the whole range.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from rowless.adapters.base import DryRunContext
from rowless.runtime.rowdep import observe

from rowless_duckdb import database, errors, lineage, session

#: Relation accessors that hand back Python values, and so answer from the
#: zero rows a rowless relation has. The canonical list of what rowless knows
#: about lives in `rowless.runtime.rowdep.ROW_DEPENDENT_APIS`; this is the
#: DuckDB spelling of it.
ROW_DEPENDENT: tuple[str, ...] = (
    "fetchall",
    "fetchone",
    "fetchmany",
    "fetchnumpy",
    "fetchdf",
    "df",
    "to_df",
    "pl",
    "arrow",
    "to_arrow_table",
    "fetch_arrow_table",
    "to_arrow_reader",
    "fetch_arrow_reader",
    "fetch_record_batch",
    "torch",
    "tf",
    "to_dicts",
    "iter_rows",
    "show",
    "preview",
    "__len__",
)

#: Row-dependent properties, which need wrapping as properties rather than as
#: methods. `shape` runs a count, and the count is zero.
ROW_DEPENDENT_PROPERTIES: tuple[str, ...] = ("shape",)

#: Connection accessors that hand back Python values, for a pipeline that uses
#: the DBAPI surface instead of relations.
CONNECTION_ROW_DEPENDENT: tuple[str, ...] = (
    "fetchall",
    "fetchone",
    "fetchmany",
    "fetchnumpy",
    "fetchdf",
    "fetch_df",
    "fetch_df_chunk",
    "fetch_arrow_table",
    "df",
    "arrow",
    "pl",
    "torch",
    "tf",
)

#: Relation methods replaced with recorders. Each one returns None already, so
#: recording instead of writing does not change what the call evaluates to.
WRITERS: tuple[str, ...] = (
    "write_parquet",
    "write_csv",
    "to_parquet",
    "to_csv",
    "to_table",
    "create",
    "insert",
    "insert_into",
    "update",
)

#: Statement kinds that write something outside the database. Recognised in SQL
#: text by DuckDB's own parser and skipped, the way the writer methods are.
WRITING_STATEMENTS: frozenset[str] = frozenset({"COPY", "EXPORT", "COPY_DATABASE"})

#: Connection methods that build or resolve a query, wrapped so a binder error
#: becomes a finding at the user's line.
CONNECTION_QUERIES: tuple[str, ...] = (
    "sql",
    "query",
    "from_query",
    "table",
    "view",
    "values",
    "execute",
    "executemany",
)

#: Module-level shorthands for the same calls on the default connection. Wrapped
#: for translation only: there are no rowless relations in a connection this
#: adapter never made, so there is nothing to attribute.
MODULE_QUERIES: tuple[str, ...] = ("sql", "query", "from_query", "table", "view", "execute")

#: Builder methods whose output column names are unchanged by the operation.
CARRY_THROUGH: tuple[str, ...] = (
    "filter",
    "order",
    "sort",
    "limit",
    "head",
    "distinct",
    "sample",
    "set_alias",
)

#: Builder methods that take a select list, matched to the output positionally.
PROJECTING: tuple[str, ...] = ("project", "select", "aggregate")

#: Set operations: a column comes from every side that has one.
SET_OPERATIONS: tuple[str, ...] = ("union", "union_by_name", "except_", "intersect")

#: Builder methods that are wrapped for error translation only. They return a
#: relation whose columns this adapter cannot attribute - DuckDB names them,
#: reshapes them, or invents them - so those columns are reported untraced
#: rather than attributed to a guess.
TRANSLATED: tuple[str, ...] = (
    "apply",
    "count",
    "cross",
    "describe",
    "map",
    "max",
    "mean",
    "median",
    "min",
    "pivot",
    "query",
    "select_dtypes",
    "select_types",
    "sum",
    "unique",
    "unnest",
    "unpivot",
    "value_counts",
)

_INSTALLED = False


def install(ctx: DryRunContext) -> None:
    """Patch the DuckDB entry points, and bind the wrappers to this dry run.

    Called from `materialize`, which the runtime always calls before user code.
    Patching happens once per process; the per-run caches are reset through
    `rowless_duckdb.session` whenever a new run starts.
    """
    global _INSTALLED

    session.begin(ctx)
    if _INSTALLED:
        return
    _install_patches()
    _INSTALLED = True


# --- Patching ----------------------------------------------------------------


def _install_patches() -> None:
    """Wrap every DuckDB entry point this adapter needs to see."""
    import duckdb

    relation = duckdb.DuckDBPyRelation
    connection = duckdb.DuckDBPyConnection

    for method in ROW_DEPENDENT:
        _patch(relation, "DuckDBPyRelation", method, _row_dependent_wrapper)
    for name in ROW_DEPENDENT_PROPERTIES:
        _patch_property(relation, "DuckDBPyRelation", name)
    for method in WRITERS:
        _patch(relation, "DuckDBPyRelation", method, _writer_wrapper)
    for method in CARRY_THROUGH:
        _patch(relation, "DuckDBPyRelation", method, _builder(rule=_carry_rule))
    for method in PROJECTING:
        _patch(relation, "DuckDBPyRelation", method, _builder(rule=_projection_rule))
    for method in SET_OPERATIONS:
        _patch(relation, "DuckDBPyRelation", method, _builder(rule=_set_rule))
    _patch(relation, "DuckDBPyRelation", "join", _builder(rule=_join_rule))
    _patch(relation, "DuckDBPyRelation", "with_columns", _builder(rule=_extend_rule))
    _patch(relation, "DuckDBPyRelation", "rename", _builder(rule=_rename_rule))
    for method in TRANSLATED:
        _patch(relation, "DuckDBPyRelation", method, _builder(rule=None))

    for method in CONNECTION_ROW_DEPENDENT:
        _patch(connection, "DuckDBPyConnection", method, _row_dependent_wrapper)
    for method in CONNECTION_QUERIES:
        _patch(connection, "DuckDBPyConnection", method, _query_wrapper)
    _patch_module(duckdb)


def _patch(
    owner: type,
    owner_name: str,
    method: str,
    factory: Callable[[str, Any], Any],
) -> None:
    """Replace one method with a wrapper, skipping what this DuckDB lacks.

    Releases differ in which methods exist, and a missing one is not an error:
    the adapter supports a range of DuckDB releases, and a method that is not
    there cannot be called. The class is patched rather than the instances
    because DuckDB builds every relation in C++ and hands back a new object
    from each call.
    """
    original = owner.__dict__.get(method)
    if original is None or getattr(original, "__rowless_wrapped__", False):
        return
    if not callable(original):
        return
    wrapper = factory(f"duckdb.{owner_name}.{method}", original)
    wrapper.__rowless_wrapped__ = True
    try:
        setattr(owner, method, wrapper)
    except TypeError:  # pragma: no cover - a DuckDB whose types refuse patching
        # Nothing to fall back on: the call simply goes unobserved, and saying
        # so is better than a dry run that quietly reports less than it should.
        session.note(
            f"this DuckDB does not allow {owner_name}.{method} to be wrapped, so rowless "
            "cannot report what it does",
            once_for=f"unpatchable:{owner_name}.{method}",
        )


def _patch_property(owner: type, owner_name: str, name: str) -> None:
    """Wrap a row-dependent property, keeping it a property."""
    original = owner.__dict__.get(name)
    getter = getattr(original, "fget", None)
    if getter is None or getattr(getter, "__rowless_wrapped__", False):
        return
    api = f"duckdb.{owner_name}.{name}"

    def wrapper(instance: Any) -> Any:
        """Report the access, then hand back what DuckDB computed."""
        if not session.tracking():
            return getter(instance)
        with session.own_work():
            value = getter(instance)
        observe(api, returned=_summarise(value))
        return value

    # The marker goes on the getter, not on the property: a `property` object
    # takes no attributes, and the getter is what the next install reads.
    wrapper.__rowless_wrapped__ = True  # type: ignore[attr-defined]
    with suppress(TypeError):  # pragma: no cover - a DuckDB that refuses patching
        setattr(owner, name, property(wrapper))


def _patch_module(duckdb: Any) -> None:
    """Wrap the module-level query shorthands for error translation."""
    for name in MODULE_QUERIES:
        original = getattr(duckdb, name, None)
        if original is None or getattr(original, "__rowless_wrapped__", False):
            continue
        wrapper = _translating_wrapper(f"duckdb.{name}", original)
        wrapper.__rowless_wrapped__ = True
        with suppress(Exception):  # pragma: no cover - a module takes an attribute
            setattr(duckdb, name, wrapper)
    _patch_connect(duckdb)


def _patch_connect(duckdb: Any) -> None:
    """Make `duckdb.connect` safe to call from inside a dry run.

    A pipeline that opens its own database is the normal case, not an exotic
    one, and the file it opens is usually the warehouse. Reads are fine - the
    sandbox says so, and resolving a schema is a read - so the connection is
    opened read-only rather than refused. Creating a database file is a write,
    and that is refused, because a dry run that leaves a new file behind has
    already failed at its one job.
    """
    original = getattr(duckdb, "connect", None)
    if original is None or getattr(original, "__rowless_wrapped__", False):  # pragma: no cover
        return

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Open a connection, read-only when it points at a real database."""
        if not session.tracking():
            return original(*args, **kwargs)
        target = _connect_target(args, kwargs)
        if _is_in_memory(target):
            return original(*args, **kwargs)
        from pathlib import Path

        from rowless.errors import SideEffectBlocked

        if not Path(target).exists():
            raise SideEffectBlocked(
                f"duckdb.connect({target!r}) would create a database file during a dry run. "
                "rowless resolves schemas; it does not create databases. Declare the input as "
                "a rowless source instead"
            )
        session.note(
            f"duckdb.connect({target!r}) was opened read-only for the dry run, so the pipeline "
            "can resolve schemas from it but cannot write to it",
            once_for=f"connect:{target}",
        )
        kwargs = {**kwargs, "read_only": True}
        return original(*args, **kwargs)

    wrapper.__rowless_wrapped__ = True  # type: ignore[attr-defined]
    duckdb.connect = wrapper


# --- Wrappers ----------------------------------------------------------------


def _row_dependent_wrapper(api: str, original: Any) -> Any:
    """Report a row-count-sensitive call, then hand back what DuckDB returned."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not session.tracking():
            return original(self, *args, **kwargs)
        with session.own_work():
            result = _called(api, original, self, args, kwargs)
        observe(api, returned=_summarise(result))
        return result

    return wrapper


def _writer_wrapper(api: str, original: Any) -> Any:
    """Record a write instead of performing it, and say so in the notes."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        target = _write_target(args, kwargs)
        if not session.tracking() or target is None:
            return original(self, *args, **kwargs)
        session.note(
            f"{api} would write to {target}; a dry run resolves schemas and writes nothing",
            once_for=f"{api}:{target}",
        )
        return None

    return wrapper


def _translating_wrapper(api: str, original: Any) -> Any:
    """Translate a DuckDB error into a finding, and otherwise stay out of the way."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not session.tracking():
            return original(*args, **kwargs)
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            errors.raise_translated(exc)

    _ = api
    return wrapper


def _query_wrapper(api: str, original: Any) -> Any:
    """Wrap a connection entry point: skip writes, translate, record lineage.

    This is where raw SQL joins the graph. A query that reads registered
    relations directly has its projection attributed to their columns; a query
    with a subquery in the way is left untraced, for the reason in
    `rowless_duckdb.lineage`.
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not session.tracking():
            return original(self, *args, **kwargs)
        skipped = _skip_write(api, args, kwargs)
        if skipped:
            return None
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            errors.raise_translated(exc)
        if not _is_relation(result):
            return result
        with session.own_work():
            _record_query_lineage(result)
        return result

    return wrapper


@dataclass(frozen=True, slots=True)
class _Call:
    """One intercepted builder call, with its result already resolved.

    Everything a lineage rule needs, so the rules stay pure functions of the
    call and the wrapper stays the only place that knows about tracking,
    reentrancy and error translation.
    """

    receiver: Any
    """The relation the method was called on."""

    result: Any
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    columns: tuple[str, ...]
    """Resolved output column names, in order."""

    @property
    def parent(self) -> lineage.Lineage | None:
        """Lineage of the receiver, or None when it was never tracked."""
        return lineage.TRACKER.of(self.receiver)


#: Maps one intercepted call onto the lineage of the relation it produced, or
#: None when there is nothing to record.
_Rule = Callable[[_Call], "lineage.Lineage | None"]


def _builder(*, rule: _Rule | None) -> Callable[[str, Any], Any]:
    """A `_patch` factory for one builder method, bound to its lineage rule."""
    return functools.partial(_builder_wrapper, rule=rule)


def _builder_wrapper(api: str, original: Any, *, rule: _Rule | None) -> Any:
    """Wrap a builder method: translate what it raises, record what it produced.

    The shared half of every relational interception. DuckDB binds while the
    relation is built, so the `try` here is the whole of the plan validation
    the Polars adapter has to arrange for itself.
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not session.tracking():
            return original(self, *args, **kwargs)
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            with session.own_work():
                available = tuple(database.columns_of(self))
            errors.raise_translated(exc, available=available)
        if rule is None or not _is_relation(result):
            return result
        with session.own_work():
            columns = _columns_of(result)
            if columns is None:
                return result
            recorded = rule(
                _Call(receiver=self, result=result, args=args, kwargs=kwargs, columns=columns)
            )
            if recorded is not None:
                lineage.TRACKER.record(result, recorded)
        return result

    _ = api
    return wrapper


# --- Lineage rules -----------------------------------------------------------


def _carry_rule(call: _Call) -> lineage.Lineage | None:
    """`filter`, `order`, `limit`: the surviving names keep their lineage."""
    parent = call.parent
    return None if parent is None else lineage.carry(parent, call.columns)


def _projection_rule(call: _Call) -> lineage.Lineage | None:
    """`project`, `select`, `aggregate`: each output comes from its expression.

    `aggregate` takes the group keys as a second argument, which contribute no
    output column of their own when they are passed that way: the select list
    is the first argument, and that is what the output columns line up with.
    """
    parent = call.parent
    if parent is None:
        return None
    fragments = [lineage.expression_text(argument) for argument in call.args]
    parsed = lineage.parse_expressions(fragments)
    if not parsed.resolvable:
        # A `*` or a parse failure. The names did not necessarily change, so
        # carrying what matches is better than nothing and cannot invent a
        # parent: a name only carries if the input really had it.
        return lineage.carry(parent, call.columns)
    return lineage.positional(parent, call.columns, parsed)


def _extend_rule(call: _Call) -> lineage.Lineage | None:
    """`with_columns`: keyword names, so the attribution needs no position."""
    parent = call.parent
    if parent is None:
        return None
    added = {
        name: lineage.parse_expressions([lineage.expression_text(value)])
        for name, value in call.kwargs.items()
    }
    return lineage.extended(parent, call.columns, added)


def _rename_rule(call: _Call) -> lineage.Lineage | None:
    """`rename`: the new name inherits what the old name had."""
    parent = call.parent
    if parent is None:
        return None
    mapping = {str(old): str(new) for old, new in call.kwargs.items()}
    return lineage.renamed(parent, call.columns, mapping)


def _join_rule(call: _Call) -> lineage.Lineage | None:
    """A join: each output column comes from the side that produced it."""
    left = call.parent
    other = call.args[0] if call.args else call.kwargs.get("other_rel")
    right = lineage.TRACKER.of(other) if other is not None else None
    if left is None and right is None:
        return None
    suffix = call.kwargs.get("suffix") or ""
    return lineage.joined(left or {}, right, call.columns, suffix=str(suffix))


def _set_rule(call: _Call) -> lineage.Lineage | None:
    """`union` and friends: a column comes from every side that has one."""
    other = call.args[0] if call.args else call.kwargs.get("union_rel")
    parents = [call.parent, lineage.TRACKER.of(other) if other is not None else None]
    if all(parent is None for parent in parents):
        return None
    return lineage.merged(parents, call.columns)


# --- Helpers -----------------------------------------------------------------


def _record_query_lineage(relation: Any) -> None:
    """Attribute the projection of a raw query, when its shape allows it."""
    columns = _columns_of(relation)
    sql = database.relation_sql(relation)
    if not columns or sql is None:
        return
    resolved = lineage.of_query(sql, columns)
    if resolved:
        lineage.TRACKER.record(relation, resolved)


def _skip_write(api: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> bool:
    """True when the SQL in this call writes outside the database, so it is skipped.

    Recognised by statement type rather than by pattern: DuckDB's parser knows
    what a `COPY ... TO` is, and a regular expression over SQL text would not.
    """
    sql = args[0] if args else kwargs.get("query") or kwargs.get("sql_query")
    if not isinstance(sql, str):
        return False
    kinds = [kind for kind in database.statement_types(sql) if kind in WRITING_STATEMENTS]
    if not kinds:
        return False
    session.note(
        f"{api} would run a {kinds[0]} statement, which writes outside the database; a dry run "
        "resolves schemas and writes nothing",
        once_for=f"{api}:{kinds[0]}:{sql[:80]}",
    )
    return True


def _called(api: str, original: Any, receiver: Any, args: Any, kwargs: Any) -> Any:
    """Call a row-dependent accessor, translating what it raises.

    Some of these execute, and executing binds: a `fetchall` on a relation that
    does not resolve raises the binder error here rather than when it was
    built.
    """
    try:
        return original(receiver, *args, **kwargs)
    except Exception as exc:
        _ = api
        errors.raise_translated(exc)


def _is_relation(obj: object) -> bool:
    """True when `obj` is a DuckDB relation, the only thing lineage tracks."""
    try:
        import duckdb
    except ImportError:  # pragma: no cover - the wrappers only exist with duckdb
        return False
    return isinstance(obj, duckdb.DuckDBPyRelation)


def _columns_of(relation: Any) -> tuple[str, ...] | None:
    """The output column names of a relation, or None when it will not say."""
    try:
        return tuple(str(name) for name in relation.columns)
    except Exception:
        # The relation is bound - it was built without raising - so this is a
        # DuckDB that will not report columns for it. Nothing to attribute.
        return None


def _summarise(result: object) -> object:
    """A short value for the diagnostic: what the call handed back on zero rows."""
    if isinstance(result, bool | int | float | str) or result is None:
        return result
    if isinstance(result, tuple | list):
        return f"{len(result)} rows"
    shape = getattr(result, "shape", None)
    if isinstance(shape, tuple) and shape and isinstance(shape[0], int):
        return f"{shape[0]} rows"
    return None


def _write_target(args: Sequence[Any], kwargs: Mapping[str, Any]) -> str | None:
    """Where a writer was pointed, or None when it was not pointed anywhere."""
    import os

    candidates: list[Any] = [*args]
    for key in ("file_name", "table_name", "view_name", "path", "name"):
        if key in kwargs:
            candidates.append(kwargs[key])
    for candidate in candidates:
        if isinstance(candidate, str | os.PathLike):
            return str(candidate)[:200]
        if candidate is not None and not isinstance(candidate, bool | int | float):
            return f"a {type(candidate).__qualname__}"
    return None


def _connect_target(args: Sequence[Any], kwargs: Mapping[str, Any]) -> str:
    """The database `duckdb.connect` was pointed at, defaulting to in-memory."""
    if args and isinstance(args[0], str):
        return args[0]
    found = kwargs.get("database")
    return found if isinstance(found, str) else ":memory:"


def _is_in_memory(target: str) -> bool:
    """True when this database lives in memory, so opening it writes nothing."""
    return target in {"", ":memory:"} or target.startswith(":memory:")

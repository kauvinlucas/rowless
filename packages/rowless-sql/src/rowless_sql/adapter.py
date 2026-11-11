"""The SQL adapter: schemas from SQL text, no database required.

`sqlglot` parses, qualifies and annotates types given the input schemas, so a
SQL model resolves with no engine running at all. That makes this the adapter
that works in a credential-free CI job, which is the property that makes
rowless adoptable and is worth having an adapter of its own.

It is also the lower-confidence option, and it says so. Type annotation in
sqlglot is dialect-aware but not complete: some functions have no annotated
return type, and the honest result is an ANY column with a note naming the
function. Where a real engine is available the DuckDB binder is the better
answer, and `rowless-dbt` prefers this one only because a dbt check that needs
a warehouse connection is a dbt check that will not run in CI.

How the four required methods are satisfied:

* `materialize` registers the schema in the run's catalogue and returns a real
  `sqlglot.exp.Table` naming it. That is this engine's frame: a query is built
  from a table, `str()` of it renders the table name, and an f-string SQL
  model therefore refers to the right input without the pipeline having to
  know anything about rowless.
* `describe` resolves whatever came back - a `sqlglot` expression, or a SQL
  string - against the catalogue, and reads the projection off the annotated
  query. See `rowless_sql.resolve`.
* `to_ir` and `from_ir` are in `rowless_sql.dtypes`, which is the one
  sqlglot-to-IR mapping in the workspace: `rowless_dbt.dtypes` parses a
  manifest's type strings and delegates the mapping there.
* the catalogue is in `rowless_sql.catalog`, the error translation in
  `rowless_sql.errors`, the column lineage in `rowless_sql.lineage` and the
  dialect resolution in `rowless_sql.dialects`, each with a docstring saying
  what it decides and why.

## Why there is no interception

Every other adapter in this repository patches its engine: to notice a
pipeline reading data, to translate an exception at the line that caused it, or
to stop a statement that would change something. None of that applies here.
Building a sqlglot expression touches nothing, resolving one executes nothing,
and there is no API on an expression that could read a row - so a SQL pipeline
is never reported as row-dependent, and a dry run on this engine cannot have a
side effect to record. The flip side is that a pipeline which *does* reach for
data has left this engine to do it, and whichever adapter it reached for
reports it.

## Where sqlglot disagrees with the other engines

The catalogue this adapter exists to write down. Each line is a place where
moving a pipeline between engines changes its output schema, and the
conformance cases in `rowless_sql.cases` assert every one of them.

* **There is no binder, and that is the trade.** `SELECT amount + status`
  annotates as `BIGINT` and nothing complains; DuckDB and Ibis both refuse the
  expression. sqlglot resolves *names*, and types what it recognises - so this
  adapter finds the rename in the proposal, the missing column, the ambiguous
  reference and the duplicate output name, and it does not find a type error.
  A conformance case asserts the gap rather than hiding it, because a user
  choosing between this adapter and DuckDB needs to know what they are buying.
* **An aggregate keeps its operand's type.** `sum(amount)` is `BIGINT`, which
  agrees with Polars and Ibis and disagrees with DuckDB, where the same
  aggregate widens to a 128-bit integer. `count(*)` is `BIGINT` everywhere.
  `avg` is `DOUBLE`.
* **`amount / 2` is `DOUBLE`.** SQL engines are often said to do integer
  division; sqlglot's annotator does not, and neither does DuckDB.
* **An unaliased projection has no name rowless will report.** sqlglot calls
  it `_col_0`, which is sqlglot's placeholder rather than any warehouse's
  answer: DuckDB would call that column `sum(amount)`, Spark `_col0` and
  Postgres `sum`. Reporting a name no engine uses is worse than reporting
  none, so the query is refused as unverified with the fix in the message -
  alias the expression.
* **A join keeps both key columns and does not rename the collision**, so
  `SELECT *` over two tables that both have `created_at` produces that name
  twice. rowless reports the duplicate, because a name-addressed schema cannot
  describe it and neither can anything downstream. Polars coalesces the key
  and suffixes the collision `_right`; pandas suffixes `_x` and `_y`; DuckDB
  agrees with SQL and keeps both.
* **An unqualified column two sources have is ambiguous, not missing.**
  sqlglot reports both with the same message on the oldest release this
  adapter supports, so the catalogue is asked which tables have the name and
  the finding says "qualify this" rather than sending the reader to look for a
  rename.
* **Nullability is not modelled at all.** A resolved query's columns are all
  reported nullable, because SQL carries no nullability through a projection
  and sqlglot does not infer it. DuckDB tracks it and Ibis propagates it, so a
  rowless contract that declares a column non-nullable is enforceable on those
  two and unenforceable here - which the run says once, in a note.
* **A set operation is unified rather than taken from the left.** sqlglot
  reports a union's columns as the left-hand side's, types included; rowless
  combines the two sides through its own dtype lattice, so `int32` unioned
  with `int64` widens and `VARCHAR` unioned with `BIGINT` comes back
  unresolved instead of reading as a string. A set operation whose sides have
  different column counts resolves in sqlglot and is reported here, since no
  engine would run it.
* **Unquoted identifiers are folded the way the dialect folds them** - lower
  for Postgres and generic SQL, upper for Snowflake. An input column declared
  `CustomerID` is reported under its declared spelling rather than the folded
  one, because a case change nobody made would be reported downstream as a
  rename; an alias the query itself wrote keeps whatever the fold produced,
  since that is genuinely what the warehouse will call the column.
* **Lineage resolves through CTEs and subqueries**, which no other adapter
  here manages: sqlglot's scopes are the data structure, so a column selected
  from a CTE over a join is attributed to the base table column it came from.
  The DuckDB adapter refuses that case on purpose rather than guess across a
  scope it cannot see through.

## Inputs the IR can describe and SQL cannot

`materialize` has to register a schema for whatever a source resolved. An input
column whose dtype is UNKNOWN, ANY, or one SQL genuinely lacks
(`datetime[ns, Europe/London]`, `float16`, `duration[ms]`) is registered as the
SQL null type and the run is degraded, with the column named in the reason. The
alternative - refusing to check the model at all - loses the other fifteen
columns to one unresolved source, and a degraded run cannot be reported as a
pass, so nothing is claimed that was not earned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from rowless.adapters.base import Capabilities, DryRunContext
from rowless.errors import UnknownSchemaError, UnsupportedDType
from rowless.ir.dtypes import DType, null
from rowless.ir.schema import Field, Relation, Schema

from rowless_sql import catalog as catalogue
from rowless_sql import dialects, lineage, resolve, session
from rowless_sql import dtypes as dtype_map

#: The oldest sqlglot this adapter supports, and it is a *tested* floor: CI
#: runs the suite against it. 25.7 rather than the 25.0 the skeleton guessed
#: at, because `annotate_types` only takes a `dialect` from 25.7 on, and an
#: adapter that could not name the dialect would type a Snowflake model with
#: generic SQL's rules while claiming to have read it as Snowflake.
MINIMUM_SQLGLOT: tuple[int, int] = (25, 7)


@dataclass(frozen=True, slots=True)
class _Resolution:
    """The last query this adapter resolved, kept so lineage is not re-resolved.

    `rowless.runtime.dryrun` calls `describe` and then `column_lineage` on the
    same object, and qualifying a query twice to answer two questions about it
    is pure waste. One entry is enough for that sequence, and the source object
    is held rather than its `id()`, so a recycled address cannot be mistaken
    for a cache hit.
    """

    source: object
    resolved: Any
    columns: tuple[resolve.Column, ...]
    dialect: str


_LAST: _Resolution | None = None


def _forget() -> None:
    """Drop the cached resolution. Registered as a per-run reset."""
    global _LAST

    _LAST = None


session.on_reset(_forget)


class SqlAdapter:
    """SQL strings, via sqlglot qualification and type annotation."""

    name: ClassVar[str] = "sql"
    requires: ClassVar[str] = "sqlglot>=25.7"
    capabilities: ClassVar[Capabilities] = Capabilities(
        column_lineage=True,
        sql=True,
        lazy=True,
        nested_types=True,
    )

    def available(self) -> bool:
        """True when sqlglot imports at >= 25.7."""
        try:
            import sqlglot
        except Exception:
            # An engine that fails to import is simply not available. The
            # protocol says this method may not raise, and a half-installed
            # sqlglot - a missing `sqlglotrs` extension - raises on import.
            return False
        return _at_least(str(getattr(sqlglot, "__version__", "0")), MINIMUM_SQLGLOT)

    def handles(self, obj: object) -> bool:
        """True for sqlglot expressions, and for `str` while a dry run is open.

        A SQL model is a string, which is the one frame type that collides with
        every other use of a string in Python. So a `str` is claimed only in
        SQL mode - while this adapter has a run open, where a pipeline
        returning a string is returning a query. Outside one,
        `rowless.adapters.registry.adapter_for` cannot route an arbitrary
        string here. See `rowless_sql.session.sql_mode`.
        """
        try:
            from sqlglot import exp
        except Exception:
            return False
        if isinstance(obj, str):
            return session.sql_mode()
        return isinstance(obj, exp.Expression)

    def materialize(self, schema: Schema, ctx: DryRunContext) -> Any:
        """Register the schema in the run's catalogue and return a table for it.

        The returned `sqlglot.exp.Table` is a real engine object: a query is
        built from a table, and `str()` of it renders the name the catalogue
        registered - so `f"SELECT status FROM {orders}"` refers to this input,
        and so does `exp.select("status").from_(orders)`.
        """
        from sqlglot import exp

        session.begin(ctx)
        columns = {field.name: self._input_type(field, ctx) for field in schema.fields}
        table = session.catalog().register(schema, columns)
        return exp.table_(exp.to_identifier(table.name))

    def describe(self, obj: object) -> Relation:
        """Resolve what a pipeline returned, without executing anything.

        A table is the input itself, so its declared schema is the answer -
        read back through the type mapping, so a dtype SQL could not express
        shows up as what sqlglot will actually answer with rather than as what
        rowless hoped for. Anything else is a query: it is qualified against
        the catalogue and its projection is read off the annotated expression.
        """
        from sqlglot import exp

        if isinstance(obj, exp.Table):
            return self._described_table(obj)
        expression = self._as_query(obj)
        dialect = self._dialect()
        cat = session.catalog()
        resolved = resolve.resolved_expression(expression, cat, dialect=dialect)
        columns = resolve.columns_of(resolved, cat)
        _remember(obj, resolved, columns, dialect)
        return self._relation(columns, traced={})

    def to_ir(self, native: Any) -> DType:
        """Sqlglot DataType to IR, including nested types and enum variants."""
        return dtype_map.to_ir(native)

    def from_ir(self, dtype: DType) -> Any:
        """IR to a sqlglot DataType, raising `UnsupportedDType` where none exists."""
        return dtype_map.from_ir(dtype)

    def unsupported_dtypes(self) -> Sequence[DType]:
        """Corpus dtypes SQL has no equivalent for, so the suite checks the refusal.

        One entry, and it is the same one DuckDB declares for the same reason:
        a timestamp carrying a named zone. SQL's `TIMESTAMP WITH TIME ZONE` is
        an instant, so `datetime[ns, Europe/London]` would read back as
        `datetime[ns, UTC]` with every value silently moved.
        """
        return dtype_map.UNSUPPORTED

    def run_sql(
        self,
        sql: str,
        inputs: Mapping[str, object],
        ctx: DryRunContext,
    ) -> Relation:
        """Resolve the output schema of a query against named input schemas.

        The credential-free half of the Python-to-SQL seam. `inputs` maps each
        name the SQL might use - a bare table name, or the three-part quoted
        relation dbt compiles `ref()` into - to a table this adapter
        materialized or a `Schema` rowless has already resolved. Each one is
        registered under the final identifier of its name, which is what makes
        `FROM "analytics"."staging"."stg_orders"` and `FROM stg_orders` both
        resolve against one entry.

        Column lineage comes back attached, resolved through whatever CTEs and
        subqueries the model is built from, so a dbt model's columns join the
        same graph as a Python pipeline's.
        """
        session.begin(ctx)
        dialect = self._dialect()
        cat = catalogue.Catalog()
        for name, value in inputs.items():
            self._register(cat, str(name), value, ctx)
        expression = resolve.parse(sql, dialect=dialect)
        resolved = resolve.resolved_expression(expression, cat, dialect=dialect)
        columns = resolve.columns_of(resolved, cat)
        traced = self._lineage_of(resolved, columns, cat, dialect)
        if not traced and columns:
            session.note(
                "rowless resolved this query's schema but could not attribute its columns to "
                "input columns, so its lineage is reported as untraced rather than guessed",
                once_for="untraced-query",
            )
        return self._relation(columns, traced=traced)

    def column_lineage(self, obj: object) -> Mapping[str, tuple[str, ...]]:
        """Attribute each output column to the input columns it was computed from.

        Read off the same resolved query `describe` produced, which is why a
        pipeline pays for one qualification rather than two. A column computed
        from no input at all - a literal, `CURRENT_DATE` - has no entry, so it
        is reported untraced rather than attributed to a guess. See
        `rowless_sql.lineage`.
        """
        from sqlglot import exp

        if isinstance(obj, exp.Table):
            table = session.catalog().get(str(obj.name))
            if table is None:
                return {}
            # The one place a column may be attributed to itself: these really
            # are the input relation's columns, which is what the lineage graph
            # resolves `derived_from` against.
            return {column: (column,) for column in table.schema}
        cached = _LAST if _LAST is not None and _LAST.source is obj else None
        if cached is None:
            if not self.handles(obj):
                return {}
            dialect = self._dialect()
            cat = session.catalog()
            resolved = resolve.resolved_expression(self._as_query(obj), cat, dialect=dialect)
            columns = resolve.columns_of(resolved, cat)
            cached = _Resolution(obj, resolved, columns, dialect)
        return self._lineage_of(
            cached.resolved,
            cached.columns,
            session.catalog(),
            cached.dialect,
        )

    def conformance_cases(self) -> Sequence[Any]:
        """The operations battery for SQL, with the per-engine expectations."""
        from rowless_sql.cases import conformance_cases

        return conformance_cases()

    def _described_table(self, table: Any) -> Relation:
        """The schema of a materialized input, read back through the type mapping."""
        registered = session.catalog().get(str(table.name))
        if registered is None:
            known = ", ".join(session.catalog().names()) or "nothing"
            raise UnknownSchemaError(
                f"the sql adapter has no schema for the table {table.name!r}; this run "
                f"registered: {known}. A pipeline has to return a query over the inputs it was "
                "given, or one of those inputs unchanged"
            )
        fields = tuple(
            Field(
                name=field.name,
                dtype=dtype_map.to_ir(registered.columns[field.name]),
                nullable=field.nullable,
                derived_from=(field.name,),
            )
            for field in registered.schema.fields
        )
        return Relation(schema=Schema(fields=fields, name=registered.schema.name), engine=self.name)

    def _as_query(self, obj: object) -> Any:
        """The sqlglot query behind whatever a pipeline returned.

        A string is parsed, which is the SQL-model case; an expression is taken
        as it is. Anything else cannot be resolved, and saying so names what a
        pipeline on this engine may return.
        """
        from sqlglot import exp

        if isinstance(obj, str):
            if not session.sql_mode():  # pragma: no cover - handles() gates this
                raise UnknownSchemaError(
                    "the sql adapter was handed a string outside a dry run, where it cannot "
                    "tell SQL from any other string"
                )
            return resolve.parse(obj, dialect=self._dialect())
        if isinstance(obj, exp.Expression):
            return obj
        qualified = f"{type(obj).__module__}.{type(obj).__qualname__}"
        raise UnknownSchemaError(
            f"the sql adapter cannot describe {qualified}; a pipeline has to return a SQL "
            "string, a sqlglot expression, or one of the tables it was given"
        )

    def _relation(
        self,
        columns: Sequence[resolve.Column],
        *,
        traced: Mapping[str, tuple[str, ...]],
    ) -> Relation:
        """Build the relation for a resolved query, with its notes recorded."""
        for column in columns:
            if column.note is None:
                continue
            if column.dtype.is_known:
                # ANY is the gradual-typing escape hatch: it lowers the typed
                # ratio and warns. A column nobody could type at all is a loss
                # of confidence, and the run has to say so rather than report
                # a schema with a hole in it as verified.
                session.note(column.note, once_for=column.note)
            else:
                session.degrade(column.note)
        if any(column.restored for column in columns):
            session.note(
                "this dialect folds unquoted identifiers, so rowless reported at least one "
                "column under the spelling its input declares rather than the folded one; an "
                "alias the query wrote keeps the fold, which is what the warehouse will call it",
                once_for="identifier-folding",
            )
        if columns:
            session.note(
                "SQL carries no nullability through a projection and sqlglot infers none, so "
                "every column of a resolved query is reported nullable. rowless-duckdb reports "
                "the real answer where a binder is available",
                once_for="nullability",
            )
        return Relation(
            schema=Schema(
                fields=tuple(
                    Field(
                        name=column.name,
                        dtype=column.dtype,
                        nullable=True,
                        derived_from=tuple(traced.get(column.resolved_name, ())),
                    )
                    for column in columns
                )
            ),
            engine=self.name,
        )

    def _lineage_of(
        self,
        resolved: Any,
        columns: Sequence[resolve.Column],
        cat: catalogue.Catalog,
        dialect: str,
    ) -> dict[str, tuple[str, ...]]:
        """Trace a resolved query's columns, keyed by the name they resolved under."""
        return lineage.of_resolved(
            resolved,
            [column.resolved_name for column in columns],
            schema=cat.mapping(dialect),
            dialect=dialect,
            spellings=resolve.declared_spellings(cat),
        )

    def _register(
        self,
        cat: catalogue.Catalog,
        name: str,
        value: object,
        ctx: DryRunContext,
    ) -> None:
        """Record one `run_sql` input in `cat`, under the name given.

        A table this adapter materialized is looked up in the run's catalogue
        rather than re-derived, so the degradations its schema already produced
        are not reported twice. A `Schema` is accepted as well, since a caller
        that has one has no reason to build a table first - and `rowless-dbt`
        has one for every upstream model in a manifest.
        """
        from sqlglot import exp

        if isinstance(value, Schema):
            schema = value
            columns = {field.name: self._input_type(field, ctx) for field in value.fields}
        elif isinstance(value, exp.Table):
            registered = session.catalog().get(str(value.name))
            if registered is None:
                raise UnknownSchemaError(
                    f"the sql adapter has no schema for the table {value.name!r}, offered as "
                    f"the SQL input {name!r}; pass the table this adapter materialized, or a "
                    "rowless Schema"
                )
            schema, columns = registered.schema, dict(registered.columns)
        else:
            qualified = f"{type(value).__module__}.{type(value).__qualname__}"
            raise UnknownSchemaError(
                f"the sql adapter cannot register {qualified} as the SQL input {name!r}; pass a "
                "sqlglot table or a rowless Schema"
            )
        collision = cat.add(name, schema, columns)
        if collision is not None:
            ctx.degrade(collision)

    def _input_type(self, field: Field, ctx: DryRunContext) -> Any:
        """The sqlglot type for one input column, degrading when there is none."""
        if not field.dtype.is_known:
            ctx.degrade(
                f"input column {field.name!r} has an unresolved dtype "
                f"({field.dtype.reason or 'no reason given'}); it was registered as the SQL "
                "null type"
            )
            return dtype_map.from_ir(null())
        if field.dtype.is_any:
            ctx.note(
                f"input column {field.name!r} is typed ANY; it was registered as the SQL null type"
            )
            return dtype_map.from_ir(null())
        try:
            return dtype_map.from_ir(field.dtype)
        except UnsupportedDType as exc:
            ctx.degrade(
                f"input column {field.name!r}: {exc}; it was registered as the SQL null type"
            )
            return dtype_map.from_ir(null())

    def _dialect(self) -> str:
        """The dialect this run reads SQL in. See `rowless_sql.dialects`."""
        ctx = session.context()
        return dialects.for_options(ctx.options if ctx is not None else {})


def _remember(
    source: object,
    resolved: Any,
    columns: tuple[resolve.Column, ...],
    dialect: str,
) -> None:
    """Keep the resolution so `column_lineage` does not redo the work."""
    global _LAST

    _LAST = _Resolution(source=source, resolved=resolved, columns=columns, dialect=dialect)


def _at_least(version: str, minimum: tuple[int, int]) -> bool:
    """True when a version string is at least `minimum`, ignoring any suffix."""
    parts: list[int] = []
    for component in version.split(".")[:2]:
        digits = ""
        for character in component:
            if not character.isdigit():
                break
            digits += character
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    return (parts[0], parts[1]) >= minimum

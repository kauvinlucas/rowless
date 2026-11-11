"""The Ibis adapter. Structurally the cleanest fit in the project.

Ibis already separates expression building from execution: `ibis.table(schema)`
makes an unbound table, and `expr.schema()` resolves the output type without a
backend. That is precisely what rowless needs, which makes this the adapter
with the least engine-specific trickery - and the one where every answer comes
from the engine's own type rules rather than from anything rowless arranged.

How the four required methods are satisfied:

* `materialize` builds `ibis.table(schema)`: a real unbound table carrying the
  full schema, nullability included, and no backend. User code gets a genuine
  `ibis.Table`, so real pipelines run unmodified.
* `describe` calls `expr.schema()`, which resolves through ibis's type system
  and touches nothing.
* `to_ir` and `from_ir` are in `rowless_ibis.dtypes`, where the dtype-level
  disagreements are written down.
* the row dependence, the error translation and the raw-SQL seam are in
  `rowless_ibis.interception`, which explains what it patches and why, and the
  column lineage is a plain walk of the expression tree in
  `rowless_ibis.lineage`.

Two capabilities beyond the contract, both of which fall out of what ibis is:

* **The SQL seam with no database.** `run_sql` resolves a query through
  `ibis.parse_sql` against the input schemas - sqlglot parses, ibis types the
  result - so a dbt model's compiled SQL resolves with no warehouse and no
  credentials, and comes back as an ibis expression whose columns are traced.
  See `rowless_ibis.sql`.
* **Portability findings.** The same expression can resolve to different
  schemas on different backends, because their type systems differ.
  `backend_disagreements` lowers a resolved schema through ibis's per-backend
  type mappers - pure Python, nothing installed, nothing connected - and
  reports where the answers diverge. No other adapter here can produce that
  finding. See `rowless_ibis.portability`.

## Where Ibis disagrees with the other engines

The catalogue this adapter exists to write down. Each line is a place where
moving a pipeline between engines changes its output schema, and the
conformance cases in `rowless_ibis.cases` assert every one of them.

* `group_by("status").agg(t.amount.sum())` names the column `Sum(amount)`.
  Polars keeps the input name `amount`, DuckDB calls it `sum(amount)`. All
  three keep a different name for the same operation, and all three break the
  same declared contract differently. Ibis takes a keyword
  (`agg(total=...)`), which is the only spelling of the three that is stable
  across engines.
* `amount.sum()` stays `int64`. DuckDB widens the same aggregate to a 128-bit
  integer; Polars agrees with ibis and does not widen. An aggregate that
  overflows on ibis and does not on DuckDB is a difference in the engine, and
  rowless reports what each one says rather than averaging them.
* `amount / 2` is `float64`, as it is in Polars and in DuckDB. The engines that
  keep an integer for `/` are the older SQL dialects.
* `join(customers, t.order_id == c.customer_id)` keeps **both** key columns and
  suffixes a colliding name `_right`. Polars coalesces the key away and
  suffixes the collision the same way; DuckDB keeps both columns *and* both
  names, which rowless has to report as a duplicate.
* `join(customers, "created_at")` - an *inner* join on a shared name -
  coalesces it, so the key appears once. This is the one join shape where all
  three engines agree.
* the same join with `how="left"` does **not** coalesce: ibis keeps
  `created_at` and `created_at_right`, because an outer join can leave the
  right key null where the left one is not, so the two columns genuinely hold
  different values. Polars coalesces a left join anyway, so a pipeline moved
  between the two engines gains or loses a column here. This is the kind of
  difference that only shows up on the row that did not match.
* `tags.length()` is `int64` and the column is named `ArrayLength(tags)`.
  Polars reports `uint32` under the original column name: a dtype change *and*
  a name change between two engines for one call.
* `union` requires the schemas to be equal and says which columns are missing
  from which side. Polars requires the same and DuckDB coerces, which is the
  difference between a finding in CI and a surprise in production.
* **Nullability is real, and it propagates.** Ibis carries nullability on the
  dtype itself, so a `!int64` input is still `!int64` after a projection, a
  filter and a union. Of the engines rowless targets, only DuckDB tracks it at
  all, and only through a projection - against Polars, which does not model it,
  the same contract is unenforceable.
* **An unbound expression cannot be executed at all.** Where Polars will
  happily `collect()` an empty frame and pandas will hand back an empty one,
  ibis refuses, and so does this adapter: `execute()` in a dry run is reported
  as row dependence and the result is marked unverified. The flip side is that
  the aggregate *methods* cost nothing here - `t.amount.max()` builds an
  expression rather than reading a value - so far fewer pipelines come out
  unverified than on the dataframe engines.

## Inputs the IR can describe and Ibis cannot

`materialize` has to produce a table for whatever schema a source resolved. An
input column whose dtype is UNKNOWN, ANY, or a dtype ibis genuinely lacks
(`enum{...}`, `int128`, `time[ms]`) is materialized as the ibis null type and
the run is degraded, with the column named in the reason. The alternative -
refusing to check the pipeline at all - loses the other fifteen columns to one
unresolved source, and a degraded run cannot be reported as a pass, so nothing
is claimed that was not earned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from rowless.adapters.base import Capabilities, DryRunContext
from rowless.errors import UnknownSchemaError, UnsupportedDType
from rowless.ir.dtypes import DType
from rowless.ir.schema import Field, Relation, Schema

from rowless_ibis import dtypes as dtype_map
from rowless_ibis import errors, interception, lineage, portability, session
from rowless_ibis import sql as sql_seam

#: The oldest Ibis this adapter supports, and it is a *tested* floor: the
#: suite runs against it in CI, which is why it is 10.0 rather than the 9.0
#: that would also import. The 9.0 relations rework is what makes
#: `rowless_ibis.lineage` a walk rather than a reconstruction - every relation
#: node has carried a `values` mapping since then - but 9.x has a weaker SQL
#: parser behind `ibis.parse_sql` (`count(*)` does not resolve), no `to_list`
#: on a column, and a set-operation error that does not name the columns that
#: differ. Declaring 9.0 would be claiming a seam that is not there.
MINIMUM_IBIS: tuple[int, int] = (10, 0)


class IbisAdapter:
    """Ibis, via unbound tables and expression schema resolution."""

    name: ClassVar[str] = "ibis"
    requires: ClassVar[str] = "ibis-framework>=10.0"
    capabilities: ClassVar[Capabilities] = Capabilities(
        column_lineage=True,
        sql=True,
        lazy=True,
        nested_types=True,
    )

    def available(self) -> bool:
        """True when ibis-framework imports at >= 10.0."""
        try:
            import ibis
        except Exception:
            # An engine that fails to import is simply not available. The
            # protocol says this method may not raise, and a half-installed
            # ibis (a missing sqlglot, say) raises on import.
            return False
        return _at_least(str(getattr(ibis, "__version__", "0")), MINIMUM_IBIS)

    def handles(self, obj: object) -> bool:
        """True for Table, Column and Scalar expressions."""
        try:
            import ibis.expr.types as ir
        except Exception:
            return False
        return isinstance(obj, ir.Table | ir.Column | ir.Scalar)

    def materialize(self, schema: Schema, ctx: DryRunContext) -> Any:
        """`ibis.table()` with the IR schema: unbound, so nothing executes.

        Also the point where this adapter installs its interception: the
        runtime calls `materialize` before it imports or calls user code, so
        the wrappers are in place for whatever the pipeline does next.
        """
        import ibis

        interception.install(ctx)
        columns = {
            field.name: dtype_map.with_nullability(
                self._input_dtype(field, ctx), nullable=field.nullable
            )
            for field in schema.fields
        }
        return ibis.table(ibis.schema(columns), name=session.unique_name(schema.name))

    def describe(self, obj: object) -> Relation:
        """Read the schema ibis resolved, with no backend bound.

        A table answers for every column; a single column or scalar expression
        answers for itself, which is what a pipeline that returns one aggregate
        leaves behind.
        """
        import ibis.expr.types as ir

        with session.own_work():
            if isinstance(obj, ir.Table):
                fields = self._table_fields(obj)
            elif isinstance(obj, ir.Column | ir.Scalar):
                fields = self._value_fields(obj)
            else:
                qualified = f"{type(obj).__module__}.{type(obj).__qualname__}"
                raise UnknownSchemaError(
                    f"the ibis adapter cannot describe {qualified}; a pipeline has to return an "
                    "ibis table, column or scalar expression"
                )
        return Relation(schema=Schema(fields=fields), engine=self.name)

    def to_ir(self, native: Any) -> DType:
        """Ibis DataType to IR, keeping scale, timezone and decimal precision."""
        return dtype_map.to_ir(native)

    def from_ir(self, dtype: DType) -> Any:
        """IR to an Ibis DataType, raising `UnsupportedDType` where none exists."""
        return dtype_map.from_ir(dtype)

    def unsupported_dtypes(self) -> Sequence[DType]:
        """Corpus dtypes ibis has no equivalent for, so the suite checks the refusal.

        One entry, and it is a surprising one: ibis has an `Enum` dtype that is
        an *alias for* `String`. It carries no variant set, so lowering
        `enum{a,b,c}` to it would read back as a plain string and be reported
        downstream as a dtype change nobody made.
        """
        return dtype_map.UNSUPPORTED

    def run_sql(
        self,
        sql: str,
        inputs: Mapping[str, object],
        ctx: DryRunContext,
    ) -> Relation:
        """Resolve raw SQL against the input schemas, with no backend at all.

        The seam, and the credential-free half of it. `inputs` maps each name
        the SQL might use - a bare table name, or the three-part relation dbt
        compiles `ref()` into - to a table or schema rowless has already
        resolved. `ibis.parse_sql` reads the query against those schemas and
        returns an ordinary ibis expression, so the output schema comes from
        ibis's own type rules and the column lineage is the same walk a Python
        pipeline gets.

        A query ibis's parser cannot read is reported as unverified rather than
        as a broken model: the SQL may well be valid for its warehouse, and
        rowless has no way to know what it returns.
        """
        interception.install(ctx)
        schemas = {
            str(name): self._sql_input(str(name), value, ctx) for name, value in inputs.items()
        }
        catalog, collisions = sql_seam.catalog_for(schemas)
        for collision in collisions:
            ctx.degrade(collision)
        with session.own_work():
            expr = sql_seam.resolve(
                sql_seam.strip_terminator(sql),
                catalog,
                dialect=_dialect(ctx),
            )
            relation = self.describe(expr)
            traced = lineage.of_expression(expr)
        if not traced and relation.schema.fields:
            ctx.note(
                "rowless resolved this query's schema but could not attribute its columns to "
                "input columns, so its lineage is reported as untraced rather than guessed"
            )
        return Relation(
            schema=Schema(
                fields=tuple(
                    Field(
                        name=field.name,
                        dtype=field.dtype,
                        nullable=field.nullable,
                        derived_from=tuple(traced.get(field.name, ())),
                    )
                    for field in relation.schema.fields
                )
            ),
            engine=self.name,
        )

    def column_lineage(self, obj: object) -> Mapping[str, tuple[str, ...]]:
        """Attribute each output column to the input columns it was computed from.

        Read straight off the expression tree, which carries explicit column
        references: no interception, no parsing, and nothing recorded while the
        pipeline ran. A column computed from no input at all - a literal, or
        anything behind a subquery the walk stops at - has no entry, so it is
        reported untraced rather than attributed to a guess. See
        `rowless_ibis.lineage`.
        """
        with session.own_work():
            return lineage.of_expression(obj)

    def backend_disagreements(
        self,
        obj: object,
        backends: Sequence[str] = portability.DEFAULT_BACKENDS,
    ) -> dict[str, Schema]:
        """Resolve against several backends and report where the schemas differ.

        A portability finding no other adapter can produce: one expression is
        not guaranteed to have the same output type on DuckDB and on BigQuery.
        Each named backend's own ibis type mapper lowers the resolved schema and
        reads it back, so nothing has to be installed and nothing is connected
        to.

        Returns one entry per backend whose answer differs from the schema ibis
        resolved; an empty mapping means the expression is portable across the
        backends asked about. See `rowless_ibis.portability` for what the
        defaults catch.
        """
        resolved = self.describe(obj)
        with session.own_work():
            return portability.disagreements(resolved.schema, tuple(backends))

    def conformance_cases(self) -> Sequence[Any]:
        """The operations battery for Ibis, with the per-engine expectations."""
        from rowless_ibis.cases import conformance_cases

        return conformance_cases()

    def _table_fields(self, table: Any) -> tuple[Field, ...]:
        """Read a table's columns, translating a resolution failure into a finding."""
        try:
            resolved = table.schema()
        except Exception as exc:
            errors.raise_translated(exc, relation=_name_of(table))
        return tuple(
            Field(
                name=str(name),
                dtype=dtype_map.to_ir(dtype),
                # Ibis keeps nullability on the dtype and propagates it, so
                # this is the engine's answer rather than a default.
                nullable=dtype_map.is_nullable(dtype),
            )
            for name, dtype in resolved.items()
        )

    def _value_fields(self, value: Any) -> tuple[Field, ...]:
        """Read a single column or scalar expression as a one-column schema."""
        try:
            dtype = value.type()
        except Exception as exc:
            errors.raise_translated(exc)
        return (
            Field(
                name=_value_name(value),
                dtype=dtype_map.to_ir(dtype),
                nullable=dtype_map.is_nullable(dtype),
            ),
        )

    def _sql_input(self, name: str, value: object, ctx: DryRunContext) -> Any:
        """The ibis schema for one `run_sql` input, under the name given.

        A rowless `Schema` is accepted as well as a table, since a caller that
        has one has no reason to build a table first - and `rowless-dbt` has
        one for every upstream model in a manifest.
        """
        import ibis

        if isinstance(value, Schema):
            return ibis.schema(
                {
                    field.name: dtype_map.with_nullability(
                        self._input_dtype(field, ctx), nullable=field.nullable
                    )
                    for field in value.fields
                }
            )
        return sql_seam.require_schema(value, name)

    def _input_dtype(self, field: Field, ctx: DryRunContext) -> Any:
        """The ibis dtype for one input column, degrading when there is none."""
        import ibis.expr.datatypes as dt

        if not field.dtype.is_known:
            ctx.degrade(
                f"input column {field.name!r} has an unresolved dtype "
                f"({field.dtype.reason or 'no reason given'}); it was materialized as the ibis "
                "null type"
            )
            return dt.Null()
        if field.dtype.is_any:
            ctx.note(
                f"input column {field.name!r} is typed ANY; it was materialized as the ibis "
                "null type"
            )
            return dt.Null()
        try:
            return dtype_map.from_ir(field.dtype)
        except UnsupportedDType as exc:
            ctx.degrade(
                f"input column {field.name!r}: {exc}; it was materialized as the ibis null type"
            )
            return dt.Null()


def _dialect(ctx: DryRunContext) -> str | None:
    """The SQL dialect to parse with, from `[engine.ibis] dialect` in config.

    dbt compiles for a specific warehouse, and a Snowflake model is not
    portable SQL. Naming the dialect lets sqlglot read what dbt wrote; leaving
    it out parses the query as generic SQL, which is right for most models and
    wrong for the ones that use a dialect's own functions.
    """
    dialect = ctx.options.get("dialect")
    return str(dialect) if dialect else None


def _name_of(table: Any) -> str | None:
    """The name of a table expression, when it has one."""
    name = getattr(table.op(), "name", None)
    return str(name) if isinstance(name, str) and name else None


def _value_name(value: Any) -> str:
    """The name ibis gives a column or scalar expression.

    Ibis names an unaliased expression after the operation that produced it
    (`Sum(amount)`), which is the name a user would see in the result, so it is
    the name rowless reports rather than something invented here.
    """
    try:
        return str(value.get_name())
    except Exception:
        return "value"


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

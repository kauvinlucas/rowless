"""The DuckDB adapter, and the Python-to-SQL seam.

Create tables with the declared schema and no rows, then ask the binder. DuckDB
binds and plans without scanning, so the answer comes from its own binder,
including its coercion rules for joins and set operations. The relational API
binds *eagerly*, which is the property that makes this adapter pleasant to
write: a missing column raises at the line that read it, so a finding points at
the user's code with no plan-resolution machinery in between.

This adapter matters more than the popularity of its engine suggests, because
it is how a SQL string becomes a node in the same graph as a Polars function:
the `SupportsSql` implementation here registers rowless relations as empty
tables and resolves arbitrary SQL against them. A dbt model and a `group_by`
land in one lineage graph, which is the gap the proposal is built around.

How the four required methods are satisfied:

* `materialize` creates a table with the declared schema, `NOT NULL` included,
  and returns `conn.table(...)`: a real `DuckDBPyRelation` over zero rows, so
  real pipelines run unmodified.
* `describe` reads `relation.columns` and `relation.types` - no type string is
  parsed on the way - and recovers nullability with `DESCRIBE`.
* `to_ir` and `from_ir` are in `rowless_duckdb.dtypes`, where the dtype-level
  disagreements are written down.
* the scratch database is in `rowless_duckdb.database`, and the row dependence,
  the error translation, the lineage and the writer recorders are in
  `rowless_duckdb.interception`, which explains what it patches and why.

## Where DuckDB disagrees with the other engines

The catalogue this adapter exists to write down. Each line is a place where
moving a pipeline between engines changes its output schema, and the
conformance cases in `rowless_duckdb.cases` assert every one of them.

* `aggregate("sum(amount)")` names the column `sum(amount)` and widens it to
  `HUGEINT`, a 128-bit integer. Polars keeps the name `amount` and the
  `int64`. Two differences in one call, and both break a declared contract.
* `amount / 2` is `DOUBLE`. SQL engines are often said to do integer division,
  and DuckDB does not: `/` is true division and `//` is the integer one, which
  stays `BIGINT`. The Polars answer is `float64`, so the two agree here and
  disagree with the folklore.
* a join on a condition keeps **both** key columns and does not rename the
  collision, so `SELECT *` over two tables that both have `created_at`
  produces two columns called `created_at`. Polars coalesces the key away and
  suffixes the collision `_right`; pandas suffixes `_x` and `_y`. rowless
  reports the duplicate as a finding, because a name-addressed schema cannot
  describe it and neither can anything downstream.
* a join on a shared *name* (`rel.join(other, "created_at")`) coalesces it
  instead, which is the one join shape where DuckDB and Polars agree.
* `COUNT(*)` is `BIGINT`, and `sum` of an integer is `HUGEINT`: DuckDB widens
  aggregates aggressively to avoid overflow, where Polars does not widen at
  all.
* the aggregate *methods* on a relation return relations, not values.
  `rel.max("amount")` is query building and costs no confidence; `df["x"].max()`
  in Polars or pandas reads data and costs all of it. An engine where the
  natural spelling of an aggregate is lazy is an engine where fewer pipelines
  come out unverified.
* nullability is real here. DuckDB tracks `NOT NULL` through a projection and
  reports it in `DESCRIBE`, so a rowless contract that declares a column
  non-nullable can actually be checked - against Polars, which does not model
  nullability at all, the same contract is unenforceable.

## Inputs the IR can describe and DuckDB cannot

`materialize` has to produce a relation for whatever schema a source resolved.
An input column whose dtype is UNKNOWN, ANY, or a dtype DuckDB genuinely lacks
(`datetime[ns, Europe/London]`, `float16`, `time[ms]`) is materialized as the
null type and the run is degraded, with the column named in the reason. The
alternative - refusing to check the pipeline at all - loses the other fifteen
columns to one unresolved source, and a degraded run cannot be reported as a
pass, so nothing is claimed that was not earned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from rowless.adapters.base import Capabilities, DryRunContext
from rowless.errors import DuplicateColumn, UnknownSchemaError, UnsupportedDType
from rowless.ir.dtypes import DType
from rowless.ir.schema import Field, Relation, Schema

from rowless_duckdb import database, errors, interception, lineage, session
from rowless_duckdb import dtypes as dtype_map

#: The oldest DuckDB this adapter supports. Everything it relies on -
#: `sql_query()`, `json_serialize_sql`, inline `ENUM` in a column definition,
#: the type parser - is present in 1.0.0, and the methods that are not
#: (`with_columns`, `rename`, `to_dicts`, `iter_rows`) are patched only when
#: the installed release has them.
MINIMUM_DUCKDB: tuple[int, int] = (1, 0)


class DuckDBAdapter:
    """DuckDB, via empty tables and its own binder."""

    name: ClassVar[str] = "duckdb"
    requires: ClassVar[str] = "duckdb>=1.0"
    capabilities: ClassVar[Capabilities] = Capabilities(
        column_lineage=True,
        sql=True,
        lazy=True,
        nested_types=True,
    )

    def available(self) -> bool:
        """True when duckdb imports at >= 1.0."""
        try:
            import duckdb
        except Exception:
            # An engine that fails to import is simply not available. The
            # protocol says this method may not raise, and a half-installed
            # DuckDB (a missing native extension module) raises on import.
            return False
        return _at_least(str(getattr(duckdb, "__version__", "0")), MINIMUM_DUCKDB)

    def handles(self, obj: object) -> bool:
        """True for DuckDBPyRelation and DuckDBPyConnection."""
        try:
            import duckdb
        except Exception:
            return False
        return isinstance(obj, duckdb.DuckDBPyRelation | duckdb.DuckDBPyConnection)

    def materialize(self, schema: Schema, ctx: DryRunContext) -> Any:
        """Create an empty table from the IR schema, and return it as a relation.

        Also the point where this adapter installs its interception: the
        runtime calls `materialize` before it imports or calls user code, so
        the wrappers are in place for whatever the pipeline does next.
        """
        interception.install(ctx)
        columns = [
            database.Column(
                name=field.name,
                dtype=self._input_type(field, ctx),
                nullable=field.nullable,
            )
            for field in schema.fields
        ]
        relation = database.empty_table(columns, name_hint=schema.name)
        lineage.TRACKER.seed(relation, [field.name for field in schema.fields])
        return relation

    def describe(self, obj: object) -> Relation:
        """Read the schema DuckDB computed, without scanning anything.

        A relation answers from its own bound plan. A connection answers from
        the description of its last result, which is what a pipeline written
        against the DBAPI surface leaves behind.
        """
        import duckdb

        if isinstance(obj, duckdb.DuckDBPyRelation):
            columns = self._described(obj)
        elif isinstance(obj, duckdb.DuckDBPyConnection):
            columns = _from_description(obj)
        else:
            qualified = f"{type(obj).__module__}.{type(obj).__qualname__}"
            raise UnknownSchemaError(
                f"the duckdb adapter cannot describe {qualified}; a pipeline has to return a "
                "DuckDBPyRelation or a connection that has executed a query"
            )
        _refuse_duplicates(columns)
        return Relation(
            schema=Schema(
                fields=tuple(
                    Field(
                        name=column.name,
                        dtype=dtype_map.to_ir(column.dtype),
                        nullable=column.nullable,
                    )
                    for column in columns
                )
            ),
            engine=self.name,
        )

    def to_ir(self, native: Any) -> DType:
        """DuckDB type to IR, including nested types and ENUM variants."""
        return dtype_map.to_ir(native)

    def from_ir(self, dtype: DType) -> Any:
        """IR to a DuckDB type, raising `UnsupportedDType` where none exists."""
        return dtype_map.from_ir(dtype)

    def unsupported_dtypes(self) -> Sequence[DType]:
        """Corpus dtypes DuckDB has no equivalent for, so the suite checks the refusal.

        One entry, and it is the interesting one: a timestamp carrying a named
        zone. `TIMESTAMP WITH TIME ZONE` stores UTC instants, so rowless can
        round-trip `datetime[us, UTC]` and nothing else - lowering
        `datetime[ns, Europe/London]` to it would shift every value and change
        the resolution at the same time.
        """
        return dtype_map.UNSUPPORTED

    def run_sql(
        self,
        sql: str,
        inputs: Mapping[str, object],
        ctx: DryRunContext,
    ) -> Relation:
        """Register the inputs as empty tables, then resolve the query against them.

        The seam. `inputs` maps each name the SQL might use - a bare table
        name, or the three-part quoted relation dbt compiles `ref()` into - to
        a relation or schema rowless has already resolved. Each one is
        re-created here as an empty table, including the catalog and schema the
        name asks for, and `DESCRIBE` then reports what the query produces.

        Column lineage comes back attached where the query shape allows it, so
        a SQL model's columns join the same graph as a Python pipeline's. A
        query whose projection sits behind a subquery or a CTE resolves its
        schema and reports its columns untraced, which is the honest outcome
        rather than a guessed attribution.
        """
        interception.install(ctx)
        for name, value in inputs.items():
            self._register(str(name), value)
        statement = database.strip_terminator(sql)
        try:
            columns = database.describe_query(statement)
        except Exception as exc:
            errors.raise_translated(exc)
        _refuse_duplicates(columns)
        names = [column.name for column in columns]
        traced = lineage.of_query(statement, names)
        if not traced and names:
            ctx.note(
                "rowless resolved this query's schema but could not attribute its columns: the "
                "projection does not read its tables directly, so its lineage is reported as "
                "untraced rather than guessed"
            )
        return Relation(
            schema=Schema(
                fields=tuple(
                    Field(
                        name=column.name,
                        dtype=dtype_map.to_ir(column.dtype),
                        nullable=column.nullable,
                        derived_from=tuple(traced.get(column.name, ())),
                    )
                    for column in columns
                )
            ),
            engine=self.name,
        )

    def column_lineage(self, obj: object) -> Mapping[str, tuple[str, ...]]:
        """Attribute each output column to the input columns it was computed from.

        Recorded while the relation was built, by the wrappers in
        `rowless_duckdb.interception`. A relation this adapter never saw being
        built - one that came out of `conn.sql(...)` - is read back off its own
        SQL instead, which resolves when the projection reads its tables
        directly. Anything else has no entry, so its columns are reported
        untraced rather than attributed to a guess.
        """
        recorded = lineage.TRACKER.of(obj)
        if recorded is not None:
            return recorded
        if not self.handles(obj):
            return {}
        with session.own_work():
            sql = database.relation_sql(obj)
            columns = [str(name) for name in getattr(obj, "columns", ())]
        if sql is None or not columns:
            return {}
        return lineage.of_query(sql, columns)

    def conformance_cases(self) -> Sequence[Any]:
        """The operations battery for DuckDB, with the per-engine expectations."""
        from rowless_duckdb.cases import conformance_cases

        return conformance_cases()

    def _described(self, relation: Any) -> tuple[database.Column, ...]:
        """Read a relation's columns, translating a binding failure into a finding."""
        try:
            return database.describe_relation(relation)
        except Exception as exc:
            errors.raise_translated(exc)

    def _register(self, name: str, value: object) -> None:
        """Create an empty table for one `run_sql` input, under the name given.

        The value is normally a relation this adapter materialized, and its
        schema is read back rather than its view registered: a relation belongs
        to the connection that made it, and one from somebody else's connection
        would register there instead of here. A `Schema` is accepted too, since
        a caller that has one has no reason to build a relation first.
        """
        if isinstance(value, Schema):
            columns = [
                database.Column(
                    name=field.name,
                    dtype=dtype_map.type_spec(field.dtype),
                    nullable=field.nullable,
                )
                for field in value.fields
            ]
        elif self.handles(value):
            columns = list(self._described(value))
        else:
            qualified = f"{type(value).__module__}.{type(value).__qualname__}"
            raise UnknownSchemaError(
                f"the duckdb adapter cannot register {qualified} as the SQL input {name!r}; "
                "pass a DuckDBPyRelation or a rowless Schema"
            )
        database.create_relation(name, columns)

    def _input_type(self, field: Field, ctx: DryRunContext) -> str:
        """The DuckDB type for one input column, degrading when there is none."""
        if not field.dtype.is_known:
            ctx.degrade(
                f"input column {field.name!r} has an unresolved dtype "
                f"({field.dtype.reason or 'no reason given'}); it was materialized as the "
                "DuckDB null type"
            )
            return '"NULL"'
        if field.dtype.is_any:
            ctx.note(
                f"input column {field.name!r} is typed ANY; it was materialized as the DuckDB "
                "null type"
            )
            return '"NULL"'
        try:
            return dtype_map.type_spec(field.dtype)
        except UnsupportedDType as exc:
            ctx.degrade(
                f"input column {field.name!r}: {exc}; it was materialized as the DuckDB null type"
            )
            return '"NULL"'


def _from_description(connection: Any) -> tuple[database.Column, ...]:
    """Read the schema of a connection's last result from its DBAPI description."""
    with session.own_work():
        described = getattr(connection, "description", None)
    if not described:
        raise UnknownSchemaError(
            "this DuckDB connection has no result to describe; a pipeline that returns a "
            "connection has to have executed a query on it first"
        )
    return tuple(
        # The description carries no nullability, so every column is reported
        # nullable rather than guessed at. A pipeline that returns the relation
        # instead gets the real answer.
        database.Column(name=str(row[0]), dtype=row[1], nullable=True)
        for row in described
    )


def _refuse_duplicates(columns: Sequence[database.Column]) -> None:
    """Report two output columns of the same name, which DuckDB allows and rowless cannot.

    `SELECT *` over a join whose tables share a column name produces the name
    twice. DuckDB is content with that - a later reference is simply ambiguous,
    and it says so when one is made - but a rowless schema is addressed by
    name, and so is everything downstream of it: a contract, a diff, a lineage
    edge. Reporting the duplicate is the only honest option, and the message
    names the fix.
    """
    seen: set[str] = set()
    for column in columns:
        if column.name in seen:
            raise DuplicateColumn(
                f"this query produces two columns named {column.name!r}. DuckDB keeps both "
                "columns of a colliding name after a join rather than renaming one, so nothing "
                "downstream can refer to either; alias one of them, or join on the shared name "
                "so the key is coalesced"
            )
        seen.add(column.name)


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

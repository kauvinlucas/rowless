# rowless-sql

SQL adapter for rowless: sqlglot qualification and type annotation.

## The trick

`sqlglot` qualifies identifiers and annotates types against the input schemas,
so SQL resolves with no database and no credentials. That makes this the
adapter that works in a CI job with no warehouse access, which is the property
that makes rowless adoptable on a dbt project.

Lower confidence than a real binder, and it reports that: a function sqlglot
has no return type for becomes an ANY column with a note naming the function,
never a guess. Where a real engine is available the DuckDB binder is the better
answer, and `rowless-dbt` prefers this one only because a check that needs a
warehouse connection is a check that will not run in CI.

## What a pipeline looks like

A rowless frame on this engine is a real `sqlglot.exp.Table`, and `str()` of
one renders the table name. So a SQL model is a string:

```python
@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="sql")
def daily_revenue(orders):
    return f"""
        SELECT CAST(created_at AS DATE) AS day, sum(amount) AS total
        FROM {orders}
        WHERE status = 'paid'
        GROUP BY 1
    """
```

and sqlglot's own builder API works on the same object:

```python
from sqlglot import exp


def paid_orders(orders):
    return exp.select("order_id", "amount").from_(orders).where("status = 'paid'")
```

A compiled dbt model goes through `run_sql` instead, which is what a `dbt://`
source uses to resolve a model that declares no columns.

## Status

Implemented: the conformance suite in `rowless.testing.conformance` passes,
including the operations battery in `rowless_sql.cases`.

| Capability | State |
| --- | --- |
| schemas without executing | `qualify` plus `annotate_types`; no engine, no connection |
| `sql` | `run_sql`, keyed on the final identifier so dbt's three-part names resolve |
| `column_lineage` | `sqlglot.lineage`, and it resolves **through CTEs and subqueries** |
| `nested_types` | `ARRAY`, `STRUCT` and a real `MAP` |
| `lazy` | yes: there is nothing here that could execute |
| type checking | **no.** sqlglot has no binder - see below |
| nullability | not modelled: every column of a resolved query is reported nullable |
| `udf_probe` | no. There is no runtime to probe a function with |

## What it cannot do, in one place

* **There is no binder.** `SELECT amount + status` annotates as `BIGINT` and
  nothing complains; DuckDB and Ibis both refuse the expression. This adapter
  resolves *names* and types what it recognises, so it finds renames, missing
  columns, ambiguous references and duplicate output names - and it does not
  find type errors. A conformance case asserts the gap rather than hiding it.
* **Nullability is absent.** SQL carries none through a projection and sqlglot
  infers none, so a contract declaring a column non-nullable is unenforceable
  here and enforceable on DuckDB. The run says so once, in a note.
* **An unannotated function is ANY.** A warehouse-specific function or a UDF
  has no annotated return type; the column comes back ANY with the function
  named, the typed-coverage score drops, and nothing is claimed.
* **An unaliased projection is refused.** sqlglot calls one `_col_0`, which is
  its own placeholder: DuckDB would call that column `sum(amount)`, Spark
  `_col0` and Postgres `sum`. Reporting a name no engine uses would put a
  column in a contract that cannot hold anywhere, so the query reads as
  unverified until the expression is aliased.

## What it does not need

An interception layer. Every other adapter here patches its engine - to notice
a pipeline reading data, to translate an exception at the line that caused it,
or to stop a statement that would change something. Building a sqlglot
expression touches nothing and resolving one executes nothing, and there is no
API on an expression that could read a row, so a SQL pipeline is never reported
as row-dependent and a dry run on this engine has no side effect to record.

The unverified case this adapter *does* produce is a set operation with no
common dtype, which is in the battery.

## Dialects

Dialects are configuration, not detection: guessing wrong turns a correct query
into a parse error, and a checker that reports valid SQL as broken is one
nobody runs twice. The answer comes from one of three places, in order:

```toml
[tool.rowless.engine.sql]
dialect = "snowflake"
```

then the dbt manifest, when `rowless-dbt` is installed and a project is
findable - dbt knows which warehouse it compiled for, and a project that has to
repeat it here is configuration nobody will keep in sync - and then generic
SQL, which reads the large majority of models.

A name sqlglot does not know is a configuration error, reported as one.
Falling back to generic SQL after a typo would turn `dialect = "snowflak"` into
a pile of parse failures a hundred lines away from the setting that caused them.

## Identifiers

Qualification folds unquoted identifiers the way the dialect does: lower for
Postgres and generic SQL, upper for Snowflake. Two consequences, both of which
this adapter handles rather than passes on:

* an input column declared `CustomerID` is reported under **its declared
  spelling**, because a case change nobody made would be reported downstream
  as a rename;
* an *alias the query wrote* keeps the fold, because that is genuinely what the
  warehouse will call the column.

## Versions

`requires = "sqlglot>=25.7"`, and the suite runs against 25.7.0 - the declared
floor - as well as the current release.

The floor is 25.7 rather than the 25.0 the skeleton guessed at, because
`annotate_types` only takes a `dialect` from that release on. Without it the
adapter would type a Snowflake model with generic SQL's rules while claiming to
have read it as Snowflake, which is the kind of quiet wrongness the project is
built to avoid.

Three differences inside the supported range, all of them sqlglot's own
annotations rather than anything the adapter chooses. The battery stays off
them on purpose:

* **`length(status)`** is `INT` on the current release and `BIGINT` at the
  floor.
* **`row_number() OVER (...)`** is `BIGINT` on the current release and
  unannotated at the floor, where it comes back as an ANY column with a note.
* **Arithmetic on a `DECIMAL(18, 2)`** keeps its parameters on the current
  release and loses them at the floor, where the result is reported as
  unresolved rather than as a decimal of invented width.

`sqlglot` is declared without extras. Installing
[`sqlglotc`](https://pypi.org/project/sqlglotc/) alongside it makes parsing
faster and changes no answer; the older `sqlglot[rs]` extra does the same job
and now warns that it is unmaintained.

## Where SQL disagrees with the other engines

The catalogue is in the module docstrings, next to the code that implements it:
`rowless_sql.adapter` for the operations, `rowless_sql.dtypes` for the types and
`rowless_sql.resolve` for what rowless adds on top of sqlglot's answers. Every
line of it is asserted by a case in `rowless_sql.cases`.

The headline ones:

| Operation | SQL (sqlglot) | Polars | DuckDB | Ibis |
| --- | --- | --- | --- | --- |
| `sum(amount)` on `int64` | `int64` | `int64` | `int128` | `int64` |
| unnamed `sum(amount)` | **refused**: no engine-neutral name | `amount` | `sum(amount)` | `Sum(amount)` |
| `amount / 2` | `float64` | `float64` | `float64` | `float64` |
| `amount + status` | `int64`, **no error** | refused | refused | refused |
| `SELECT *` over a join with a colliding name | both kept, duplicate reported | key coalesced, collision `_right` | both kept, duplicate reported | both kept, collision `_right` |
| unqualified column two sources have | **ambiguous**, tables named | n/a | ambiguous | n/a |
| `union` of `int32` and `int64` | `int64`, unified through the IR | refused | coerced | refused |
| `union` of `string` and `int64` | **unresolved**, and says why | refused | coerced | refused |
| nullability | not modelled | not modelled | tracked through projections | tracked everywhere |
| lineage through a CTE | **resolved** | n/a | refused rather than guessed | resolved |

The last line is the one this adapter wins outright: sqlglot's scopes are the
data structure, so a column selected from a CTE over a join is attributed to
the base table column it came from. The DuckDB adapter refuses that case on
purpose rather than guess across a scope it cannot see through.

## One type mapping, not two

`rowless_sql.dtypes` is the only sqlglot-to-IR mapping in the workspace, in
both directions. `rowless_dbt.dtypes` parses a manifest's warehouse type
strings - that part is dbt's - and delegates the mapping here, which is what
its docstring promised would happen once this adapter landed. A test asserts
the delegation, because a duplicate of a 200-line table is exactly the kind of
thing that drifts.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-sql
pytest packages/rowless-sql
```

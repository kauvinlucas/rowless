# rowless-duckdb

DuckDB adapter for rowless: empty tables plus DESCRIBE.

## The trick

Empty tables with the declared schema, then ask the binder. DuckDB binds and
plans without scanning, so the coercion rules are its own. The relational API
binds *eagerly*, so a missing column raises at the line that read it - this is
the one adapter that gets findings pointed at the right line for free.

It is also the bridge that puts SQL strings and Python frames into one lineage
graph: `run_sql` registers rowless relations as empty tables and resolves
arbitrary SQL against them, so a dbt model and a `project` land in the same
graph.

## Status

Implemented: the conformance suite in `rowless.testing.conformance` passes,
including the operations battery in `rowless_duckdb.cases`.

| Capability | State |
| --- | --- |
| schemas without executing | empty tables, `relation.columns`/`types`, `DESCRIBE` |
| `sql` | `run_sql` registers inputs as tables, including dbt's three-part names |
| `column_lineage` | composed through the relational API, read off single-level SQL |
| `nested_types` | `LIST`, `STRUCT`, a real `MAP`, and `ENUM` with its variants |
| `lazy` | yes: binding resolves a schema and scans nothing |
| nullability | **reported** on DuckDB 1.2+, which no other implemented adapter can do |
| `udf_probe` | no. A DuckDB Python function declares its return type up front |

## What it does to DuckDB

Four interceptions, all in `rowless_duckdb.interception` and all gated on an
open dry run, so DuckDB behaves normally for everybody else in the process:

* **findings at the user's line** - the binder raises while the relation is
  built, and the wrapper translates `BinderException` into `ColumnNotFound`
  with the column named and the near miss suggested, out of the candidate
  bindings DuckDB prints.
* **row dependence** - `fetchall()`, `df()`, `len(rel)` and `rel.shape` answer
  from zero rows, and each one marks the result unverified. The aggregate
  *methods* do not: `rel.max("amount")` returns a relation, so it is query
  building and costs nothing.
* **writers** - `write_parquet`, `to_table`, `insert_into` and the rest are
  recorded rather than performed, and a `COPY ... TO` in raw SQL is recognised
  by statement type and skipped.
* **column lineage** - each output column is attributed to the input columns it
  was computed from, through projections, aliases, aggregates, joins and
  unions.

## Why the sandbox needs help here

DuckDB writes through its own C++ filesystem and never calls Python's `open`,
so the audit hook in `rowless.runtime.sandbox` cannot see a `COPY ... TO`.
Layer 2 closes that from two sides:

* the scratch database is opened with `enable_external_access = false` and
  `lock_configuration = true`, so nothing in it can touch a file or a socket,
  and user code cannot turn that back on;
* `duckdb.connect` to a real database file is forced read-only, with a note
  saying so. Pointing it at a file that does not exist is refused outright: a
  dry run that leaves a new database behind has already failed at its one job.

## Versions

`requires = "duckdb>=1.0"`, and the suite is run across that range: 1.0.0, the
declared floor, 1.3.2 and the current release. Everything the adapter relies on
exists at the floor - `sql_query()`, `json_serialize_sql`, inline `ENUM` in a
column definition, the type parser - and the methods that arrived later
(`with_columns`, `rename`, `to_dicts`, `iter_rows`) are wrapped only when the
installed release has them.

Three differences inside the range are worth knowing about:

* **nullability arrived in 1.2.** Before that, `DESCRIBE` reports every column
  as nullable, including one declared `NOT NULL`. The adapter reads whatever
  the release reports and falls back to nullable, which is the safe direction:
  a contract that declares a column non-nullable is simply unenforceable on an
  older DuckDB, rather than wrongly enforced.

* **an all-null column changes type when projected.** `SELECT all_null FROM t`
  is the null type on 2.x and `INTEGER` on 1.0. Nothing can normalise that -
  an `INTEGER` column carries no record of having been null - so the battery
  asserts the stable case (reading the column through a filter, where the
  declared type survives on every release) and this is written down instead.
* **candidate bindings are qualified on 1.x** (`"orders.amount"`) and bare on
  2.x. The error reader strips the table prefix either way, so the near-miss
  suggestion does not depend on which release is installed.

## Where DuckDB disagrees with the other engines

The catalogue is in the module docstrings, where it stays next to the code that
implements it: `rowless_duckdb.adapter` for the operations (aggregate naming and
widening, true versus integer division, joins keeping both colliding columns,
real nullability) and `rowless_duckdb.dtypes` for the types (`TIMESTAMPTZ` as a
UTC instant rather than a zone, `INTERVAL` as a month/day/microsecond triple,
`DECIMAL` capped at 38 digits, `UUID` and `BIT` having no IR dtype at all).
Every line of it is asserted by a case in `rowless_duckdb.cases`.

The headline ones, against Polars:

| Operation | DuckDB | Polars |
| --- | --- | --- |
| `sum(amount)` on `int64` | `sum(amount)`, `int128` | `amount`, `int64` |
| `count(*)` | `int64` | `len`, `uint32` |
| `amount / 2` | `float64` | `float64` |
| `len(tags)` on a list | `int64` | `uint32` |
| join on a condition | both key columns, collisions duplicated | key coalesced, collision suffixed `_right` |
| join on a shared name | key coalesced | key coalesced |
| `sum` of `decimal(18,2)` | `decimal(38,2)` | `decimal(38,2)` |
| nullability | tracked through projections | not modelled |

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-duckdb
pytest packages/rowless-duckdb
```

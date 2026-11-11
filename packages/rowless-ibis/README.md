# rowless-ibis

Ibis adapter for rowless: unbound tables and expression schemas.

## The trick

`ibis.table(schema)` is already unbound and `expr.schema()` resolves with no
backend, so this adapter is mostly a dtype mapping. Ibis separates expression
building from execution by design, which makes it the adapter with the least
engine-specific trickery in the project - and the one where every answer is the
engine's own type rule rather than something rowless arranged.

It builds expressions *eagerly*, so a missing column raises at the line that
read it: like DuckDB, this adapter gets findings pointed at the right line for
free.

Two capabilities fall out of what ibis is, rather than having to be built:

* **A SQL seam with no database.** `run_sql` resolves a query through
  `ibis.parse_sql` - sqlglot parses, ibis types the result - so a dbt model's
  compiled SQL resolves with no warehouse, no credentials and nothing
  executed, and comes back as an ibis expression whose columns are traced.
* **Portability findings.** The same expression can resolve to different
  schemas on different backends, because their type systems differ.
  `backend_disagreements` lowers a resolved schema through ibis's per-backend
  type mappers - pure Python, nothing installed, nothing connected to - and
  reports where the answers diverge. No other adapter here can produce that
  finding.

## Status

Implemented: the conformance suite in `rowless.testing.conformance` passes,
including the operations battery in `rowless_ibis.cases`.

| Capability | State |
| --- | --- |
| schemas without executing | `ibis.table()` plus `expr.schema()`; nothing is bound |
| `sql` | `run_sql` through `ibis.parse_sql`, dbt's three-part names included |
| `column_lineage` | a walk of the expression tree: no interception, no parsing |
| `nested_types` | `Array`, `Struct` and a real `Map` |
| `lazy` | yes: an unbound expression cannot execute even by accident |
| nullability | **reported, and propagated** through projections, filters and unions |
| portability | `backend_disagreements`, which nothing else in the project has |
| `udf_probe` | no. An ibis UDF must declare its return type, so there is nothing to probe |

## What it does to Ibis

Four interceptions, all in `rowless_ibis.interception` and all gated on an open
dry run, so ibis behaves normally for everybody else in the process:

* **findings at the user's line** - ibis resolves column references while the
  expression is built, and the wrappers translate `IbisTypeError` into
  `ColumnNotFound` with the column named and the near miss suggested. The
  important one is `Table.__getattr__`: `orders.amount` is how an ibis pipeline
  reads a column, and a miss is a plain `AttributeError`. It becomes
  `MissingColumn`, which is a `ColumnNotFound` *and* an `AttributeError`, so the
  finding is the flagship one while `hasattr` keeps behaving.
* **row dependence** - `execute()`, `to_pandas()`, `to_pyarrow()`, `to_list()`
  and `cache()` are reported and then refused. There is nothing to hand back:
  an unbound expression has no backend, and inventing rows would be worse than
  stopping. The result reads as unverified rather than as a broken pipeline.
* **writers** - `to_parquet`, `to_csv`, `to_delta` and the rest are recorded
  rather than performed, which doubles as output lineage.
* **raw SQL inside a pipeline** - `Table.sql(...)` normally asks the backend
  what a query returns. The wrapper resolves it through the same seam `run_sql`
  uses and hands back an equivalent ibis expression, so a pipeline that mixes
  ibis and SQL resolves, and the columns the SQL produced are traced.

## What the sandbox does not need here

Nothing. This is the one adapter where the engine refuses to execute on
rowless's behalf: an unbound expression has no backend, so there is no scan to
prevent and no file to write. The interception above exists to produce better
*findings*, not to stop side effects.

The flip side is worth knowing when reading coverage numbers: the aggregate
methods cost nothing here. `t.amount.max()` builds an expression where the same
spelling on a Polars `Series` or a pandas column reads data and costs all of
the confidence, so far fewer ibis pipelines come out unverified.

## Versions

`requires = "ibis-framework>=10.0"`, and the suite runs against 10.0.0 - the
declared floor - as well as the current release.

The floor is 10.0 rather than 9.0, which would also import, because 9.x cannot
do what this adapter claims: `ibis.parse_sql` does not resolve `count(*)`, a
column has no `to_list`, and a mismatched set operation does not name the
columns that differ. Declaring 9.0 would be advertising a SQL seam that is not
there.

Two differences inside the supported range are worth knowing about, and both
are ibis's own answers rather than anything the adapter chooses:

* **`FLOAT` in a Snowflake cast.** `amount::float` resolves to `float64` on
  ibis 12 and `float32` on the earlier releases in the range. Snowflake's
  `FLOAT` is double precision, so 12 is right; the battery asserts that the
  dialect was read at all and leaves the width to the engine.
* **BigQuery `NUMERIC` read back.** ibis 12 reads it as `decimal(38, 9)`;
  earlier releases refuse to read it at all. Either answer is the same
  portability finding - a contract declaring `decimal(18, 2)` cannot hold on
  BigQuery - so that is what is asserted.
* **How much a mismatched `union` says.** The releases at the top of the range
  list the columns missing from each side; the floor says only that the schemas
  differ. Both are reported as the same finding, so the battery asserts the
  finding rather than the prose.

## Where Ibis disagrees with the other engines

The catalogue is in the module docstrings, where it stays next to the code that
implements it: `rowless_ibis.adapter` for the operations and `rowless_ibis.dtypes`
for the types (no enum, a unit-less `Time`, a timestamp *scale* instead of a
unit, calendar intervals that are not durations, no 128-bit integers). Every
line of it is asserted by a case in `rowless_ibis.cases`.

The headline ones:

| Operation | Ibis | Polars | DuckDB |
| --- | --- | --- | --- |
| unnamed `sum(amount)` on `int64` | `Sum(amount)`, `int64` | `amount`, `int64` | `sum(amount)`, `int128` |
| `amount / 2` | `float64` | `float64` | `float64` |
| length of a list column | `ArrayLength(tags)`, `int64` | `tags`, `uint32` | `int64` |
| join on a condition | both keys, collision suffixed `_right` | key coalesced, collision suffixed `_right` | both keys, collisions duplicated |
| inner join on a shared name | key coalesced | key coalesced | key coalesced |
| **left** join on a shared name | key kept **twice** | key coalesced | key coalesced |
| `union` of different schemas | refused | refused | coerced |
| nullability | tracked through projections, filters, unions | not modelled | tracked through projections |
| reading values (`execute`, `collect`) | impossible without a backend | empty frame, unverified | empty result, unverified |

The left-join line is the one to watch when moving a pipeline: ibis coalesces a
shared key only for an inner join, because an outer join can leave the right
key null where the left one is not, so the two columns genuinely hold different
values. Polars coalesces it anyway. A pipeline moved between the two gains or
loses a column, and only on the rows that did not match.

## Portability findings

```python
from rowless_ibis import IbisAdapter

adapter = IbisAdapter()
for backend, schema in adapter.backend_disagreements(expr).items():
    print(backend, [(f.name, str(f.dtype)) for f in schema.fields])
```

With the default backends (`duckdb`, `bigquery`, `snowflake`, `sqlite`), chosen
for how far apart their type systems are rather than for popularity:

* BigQuery has one integer type, so `int32` lands as `int64`, and its `NUMERIC`
  is fixed at (38, 9), so `decimal(18, 2)` cannot be held as declared.
* DuckDB's `TIMESTAMPTZ` stores UTC instants, so `datetime[ns, Europe/London]`
  comes back `datetime[us, UTC]`: a zone and a resolution gone in one hop.
* Snowflake loses an array's element type and has no unsigned integers.
* SQLite has no array type at all, so the column is reported unresolved rather
  than approximated.

A backend that agrees about every column is left out, so an empty result means
the expression is portable across the backends asked about.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-ibis
pytest packages/rowless-ibis
```

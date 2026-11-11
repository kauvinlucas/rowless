# rowless-polars

Polars adapter for rowless: resolve LazyFrame schemas without executing.

## The trick

`LazyFrame.collect_schema()` resolves the query plan using the type
rules of Polars itself and executes nothing. The resolved plan is also walkable,
so this is the one adapter that gets exact schemas and full column lineage from
the same call.

## Status

Implemented, and the reference adapter: the conformance suite in
`rowless.testing.conformance` passes, including the operations battery in
`rowless_polars.cases`.

| Capability | State |
| --- | --- |
| schemas without executing | `collect_schema()`, on every builder call |
| `column_lineage` | recorded while the plan is built, via `Expr.meta` |
| `udf_probe` | `return_dtype=`, then the return annotation, then an opt-in probe |
| `nested_types` | `List`, `Struct`, and maps via the Arrow `list<struct>` encoding |
| `lazy` | yes: nothing is materialized to find a schema |
| `sql` | no. `pl.SQLContext` is a candidate, not a promise |

## What it does to Polars

Four interceptions, all in `rowless_polars.interception` and all gated on an
open dry run, so Polars behaves normally for everybody else in the process:

* **row dependence** - `is_empty()`, `item()`, `len(df)`, `iter_rows()`,
  `Series.max()` and friends are reported, which marks the result unverified.
  Expression methods are not: `pl.col("amount").sum()` resolves to a dtype
  without touching data.
* **plan validation** - each builder call resolves its own result, so
  `ColumnNotFound` is raised at the line that read the missing column instead
  of at the end of the dry run.
* **column lineage** - each output column is attributed to the input columns it
  was computed from, through selects, renames, joins, aggregations and unions.
* **writers** - `write_parquet`, `sink_parquet`, `write_database` and the rest
  are recorded rather than performed.

`collect()` is the one call in `rowless.runtime.rowdep.ROW_DEPENDENT_APIS` this
adapter deliberately does not treat as row-dependent. It executes, but the
schema it returns is the schema the plan resolved to, and a Polars pipeline
that ends in `.collect()` is the normal way to write one; downgrading all of
them would fail CI on code rowless checked correctly. It is recorded as a note,
and the value accessors a collected frame offers - all intercepted - are what
cost confidence.

## Versions

`requires = "polars>=1.0"`, and the suite is run at both ends of that range:
Polars 1.0.0, the declared floor, and the current release. Two behaviours
changed inside it, and the adapter normalises both so that a finding does not
depend on which release is installed:

* **duplicate output names.** 2.x raises; 1.x silently keeps one of them. The
  adapter checks the expression names itself and reports `DuplicateColumn` on
  both.
* **a UDF with no `return_dtype`.** 2.x calls the function to find out and
  raises when that fails; 1.x types the column `Unknown`. Either way the run is
  reported unverified, with the UDF named.

Features that only exist further up the range - 128-bit integers, `remove`,
`raise_if_undetermined` on expression metadata - are behind feature checks, not
version checks, so they light up when the installed Polars has them.

## Where Polars disagrees with the other engines

The catalogue is in the module docstrings, where it stays next to the code that
implements it: `rowless_polars.adapter` for the operations (integer division,
grouped aggregate names, join coalescing, `list.len()` being `uint32`) and
`rowless_polars.dtypes` for the types (no `Map`, unit-less `Time`,
`Categorical` as a string, no nullability). Every line of it is asserted by a
case in `rowless_polars.cases`.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-polars
pytest packages/rowless-polars
```

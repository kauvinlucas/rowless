# rowless-dbt

dbt integration: manifest schemas, model discovery, and the Python-to-SQL link.

## The trick

Read `target/manifest.json` with the standard library and nothing else. No
dbt-core dependency and no warehouse credentials, so this works in the CI job
where it matters. Models whose columns are not declared get resolved by running
their compiled SQL through `rowless-sql`.

## Why this package carries the proposal

The dbt column-lineage tools list Python models as unsupported, and the Python
lineage tools stop at the SQL boundary. This package is the join: a dbt model
and a Polars function become nodes in the same column graph, so one rename
reports both.

That link works today, and it needs nothing dbt-aware on the Python side:

```python
# pipelines/revenue.py
orders = rowless.source("dbt://model.shop.stg_orders")


def daily_revenue(orders: pl.LazyFrame) -> pl.LazyFrame:
    return orders.select("order_id", "amount")
```

```
$ rowless check
pipelines/revenue.py:10  ColumnNotFound 'amount', did you mean 'amount_cents'?  RWL101
```

The dbt contract calls it `amount_cents`; the Polars pipeline still reads
`amount`; the finding lands in the PR that renamed it.
`tests/test_dbt_discovery.py` asserts exactly that.

## Status

Implemented: the manifest reader (v11 and v12), the `dbt://` schema source, the
warehouse-type translation and model discovery. Two things a dbt model still
cannot do, both waiting on something outside this package:

* **A SQL model cannot be dry-run.** The engine half of this is done:
  `rowless-sql` resolves a compiled query against its upstream schemas with no
  warehouse, which is what `?columns=sql` uses and what makes an uncontracted
  model resolvable in CI. What is missing is in core - the runtime has no path
  for a target that is not an importable function, and a model is named by its
  dbt node id. Models are discovered and reported as unchecked until then,
  which is the honest outcome and is covered by a test that will fail the day
  it changes.
* **A dbt Python model cannot be called.** dbt gives it the signature
  `model(dbt, session)` and supplies both arguments itself; the runtime binds
  inputs by parameter name and has no way for a finder to hand it objects. A
  `dbt` shim answering `ref()` out of `rowless.source` is a dozen lines here
  the moment core can pass one in. Python models are skipped rather than
  yielded as pipelines nothing can run.

## Where a schema comes from

`?columns=` picks the evidence; `auto` is the default and tries three of them
in order.

| value | behaviour |
| --- | --- |
| `auto` | an enforced contract, then `catalog.json`, then the compiled SQL |
| `contract` | only an enforced contract (dbt guarantees the column set) |
| `catalog` | only `target/catalog.json`, which has real warehouse types |
| `declared` | the `schema.yml` columns, contract or not. Opt-in |
| `sql` | only the compiled SQL, through an engine that can run it |

`declared` is deliberately not part of `auto`. dbt lets a project document
three columns of a twelve-column model, and nothing in the manifest says
whether a list is complete; feeding a short schema to a dry run turns every
pipeline that reads one of the missing columns into a false `ColumnNotFound`.
A project whose documentation is complete can opt in per URI, or for every
model at once:

```toml
[tool.rowless.sources]
"dbt://*" = "dbt://{name}?columns=declared"
```

When nothing can answer, the finding names all four ways to fix it rather than
returning a schema the tool cannot stand behind.

## Where the manifest comes from

A `?manifest=` option on the URI, then `DBT_TARGET_PATH` or `DBT_PROJECT_DIR`
(dbt's own variables, which a CI job that runs dbt has already set), then the
nearest `dbt_project.yml` walking up from the working directory, honouring its
`target-path:`. It is parsed once per run and cached by mtime.

## Warehouse types

Column types are whatever the warehouse calls them - `number`,
`varchar(16777216)`, `timestamp with time zone`, `array<struct<a int64>>` - so
parsing is handed to sqlglot, which knows each dialect and fills in its
defaults. The dbt adapter name picks the dialect. The mapping onto the IR, and
the judgement calls in it (an aware timestamp is UTC, a precision becomes a
time unit, `json` and `variant` are ANY, a bare `decimal` is UNKNOWN rather
than a guess) are documented in `rowless_dbt.dtypes` and asserted in
`tests/test_dbt_dtypes.py`.

## Versions

Manifest schema versions `v11` (dbt 1.7) and `v12` (dbt 1.8+), each with a
committed fixture. Anything else is refused with a message naming what is
supported: half-parsing a format nobody tested produces findings nobody can
trust. Adding a version means committing a fixture for it and extending
`SUPPORTED_VERSIONS`, in that order.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-dbt
pytest packages/rowless-dbt
```

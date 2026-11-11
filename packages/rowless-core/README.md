# rowless

A dry-run type checker for data pipelines. Schemas without data.

```bash
pip install rowless rowless-polars
rowless check
```

```
pipelines/revenue.py:9  ColumnNotFound amount
  upstream models/stg_orders.sql:14 renamed it to amount_cents (this branch)
```

## What this package contains

Core only: the schema IR, the dry-run runtime and sandbox, source resolution,
pipeline discovery, the lineage graph, diagnostics and the CLI. It has no
dataframe dependency at all, and never imports an engine library.

Install an adapter for each engine you use:

| Package | Engine |
| --- | --- |
| `rowless-polars` | Polars |
| `rowless-pandas` | pandas |
| `rowless-duckdb` | DuckDB |
| `rowless-sql` | SQL strings, via sqlglot |
| `rowless-dbt` | dbt manifests and models |
| `rowless-pyspark` | PySpark |
| `rowless-ibis` | Ibis |
| `rowless-iceberg` | Iceberg schema source |

Or `pip install rowless-all` for everything.

## Status

Implemented and tested: the IR, the runtime, the sandbox, sources, discovery,
lineage, diagnostics and the CLI. Every engine adapter in the table above is
implemented and passes the conformance suite, so `rowless check` discovers your
pipelines, resolves their inputs, runs them on rowless frames and reports what
the engine itself computed. Core still has no path for a pipeline that is not
an importable function, which is what a discovered dbt SQL model is: those are
reported as unchecked rather than passed. See the
[roadmap](https://github.com/kauvinlucas/rowless/blob/main/docs/roadmap.md).

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)). Core covers the
schema IR, the runtime and sandbox, source resolution, discovery, lineage,
diagnostics, the CLI and the adapter contract.

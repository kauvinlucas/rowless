# rowless

Your pipeline, minus the data.

A dry-run type checker for Python data pipelines. It catches broken pipelines in
the editor and in CI before any data moves, by running your real code against
frames that have a schema and no rows.

```
$ rowless check
pipelines/revenue.py:9  ColumnNotFound amount
  upstream models/stg_orders.sql:14 renamed it to amount_cents (this branch)

$ rowless diff main
BREAKING  stg_orders.amount -> amount_cents
  |- pipelines/revenue.py::daily_revenue    (Python, Polars)
  |- marts/finance_kpis.sql                 (SQL, DuckDB)
```

> **Status: core implemented, no engine adapters yet.** The `rowless`
> distribution is complete and tested: the schema IR, the sandboxed dry-run
> runtime, discovery, sources, the lineage graph, the diagnostics and the CLI.
> Every engine adapter (`rowless-polars` and the rest) is still a skeleton
> whose methods raise `NotImplementedError`, so `rowless check` finds your
> pipelines, resolves their inputs and then reports honestly that it has no
> engine to run them on. The adapter contract is settled and the conformance
> suite is the specification: see
> [docs/writing-an-adapter.md](docs/writing-an-adapter.md). Read the
> [proposal](docs/rowless-proposal.html) for the pitch and
> [docs/roadmap.md](docs/roadmap.md) for what is being built when.

## How it works

rowless does not re-implement pandas or Polars semantics. It runs your actual
code on *rowless* inputs that carry a full schema and zero rows, and lets each
engine work out the output schema with its own rules.

| Engine | How the schema comes out without touching data |
| --- | --- |
| Polars | `LazyFrame.collect_schema()` resolves the plan, executes nothing |
| DuckDB | empty tables with the declared schema, then `DESCRIBE` |
| pandas | empty typed frames, with the dtypes that drift on empty data fixed |
| SQL, dbt | `sqlglot` qualification and type annotation, no database |
| PySpark | the Catalyst analyzed plan, via Spark Connect |
| Ibis | unbound tables and `expr.schema()` |

## Layout

```
packages/
  rowless-core/          the `rowless` distribution: IR, runtime, sandbox, CLI
  rowless-polars/        one distribution per engine adapter
  rowless-pandas/
  rowless-duckdb/
  rowless-sql/           SQL strings via sqlglot
  rowless-dbt/           manifests, model discovery, the Python-to-SQL link
  rowless-pyspark/
  rowless-ibis/
  rowless-iceberg/       schema source
  rowless-pytest/        pytest plugin: millisecond contract tests
  rowless-openlineage/   emit the column graph where catalogs already look
  rowless-lsp/           language server
  rowless-all/           meta-distribution
integrations/
  github-action/         PR comment, annotations, gate
docs/                    architecture, the schema IR, adapter guide, ADRs
examples/                small broken repositories, used as integration tests
benchmarks/              findings on real repositories, and speed
```

## Develop

```bash
uv sync --all-packages --all-extras
uv run nox -s lint types tests
uv run nox -s conformance          # every adapter against the shared suite
```

Working on one adapter needs only that adapter:

```bash
uv sync --package rowless-polars
uv run pytest packages/rowless-polars
```

## Contributing

An adapter is the self-contained way in: pick the engine you already use daily,
and the conformance suite in `rowless.testing.conformance` tells you what is
left to do. See [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/writing-an-adapter.md](docs/writing-an-adapter.md).

One rule matters more than the rest: **rowless must never report a pass it has
not earned.** When a schema cannot be determined, say so. See
[docs/confidence.md](docs/confidence.md).

## Name

Working name. Check PyPI availability before committing to it.

## Licence

Apache-2.0.

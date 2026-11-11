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

> **Status: core implemented, Polars works end to end.** The `rowless`
> distribution is complete and tested: the schema IR, the sandboxed dry-run
> runtime, discovery, sources, the lineage graph, the diagnostics and the CLI.
> `rowless-polars` is implemented and passes the conformance suite, so
> `rowless check` finds your Polars pipelines, resolves their inputs, runs
> them on rowless frames and reports what Polars itself computed - with column
> lineage, and with row-dependent logic flagged rather than guessed at.
> `rowless-dbt` is implemented too, so `dbt://model.shop.stg_orders` resolves
> out of a committed manifest with no credentials and a rename in a dbt
> contract breaks the Python pipeline that reads it. `rowless-duckdb` is
> implemented as well, which closes the Python-to-SQL seam: it resolves a SQL
> string against rowless relations with DuckDB's own binder - dbt's three-part
> relation names included - and reports column lineage for both halves, so a
> SQL model and a Python pipeline land in one graph. `rowless-ibis` is
> implemented too, and it adds two things the others cannot: it resolves a SQL
> string with no database at all - sqlglot parses it, ibis types it, so a dbt
> model resolves in CI without a warehouse - and it reports where backends
> *disagree*, because the same expression does not have the same output schema
> on DuckDB and on BigQuery. `rowless-iceberg` is implemented as well, and it
> is the one source whose renames are facts: Iceberg keeps a column's field ID
> across a rename, so `rowless diff` reports `amount -> amount_cents` out of
> the table metadata instead of pairing a removal with an addition by
> position - and it resolves a historical snapshot, so the base side of a diff
> needs no git checkout. `rowless-lsp` is implemented too, which closes the
> first clause of the pitch: the findings arrive in the editor and not only in
> CI. It publishes what `rowless check` publishes, because it calls the same
> function - a full dry run on save, and between saves the checks the buffer
> alone supports, so a column the source renamed gets a squiggle while it is
> being typed, with completion, hover and inline schema hints served from the
> last check's schemas. `rowless-openlineage` is implemented as well, which
> answers where the graph goes: nowhere new. `rowless emit` publishes the
> cross-language column graph as OpenLineage static lineage - `DatasetEvent`
> and `JobEvent`, never a run event, because nothing ran - into the catalog a
> team already has, with the coverage score on every event so nothing there
> mistakes partly-inferred lineage for verified lineage, and with name-matched
> edges withheld rather than drawn as facts. `rowless-pandas` is implemented as
> well, which is the adapter most repositories actually need and the one the
> proposal expected to be hardest: pandas has no plan to resolve, so the frames
> are real and zero-row, and the places where pandas computes a *different*
> dtype from zero rows than it would from data are a published table rather
> than a correction buried in an adapter - each entry either corrected, with
> the correction shown in `rowless explain`, or reported as unverified when
> nothing could derive the populated answer. `rowless-pyspark` is implemented
> as well, which closes the last engine in the pitch and is the first whose
> engine is not in the process: `df.schema` runs Catalyst analysis and launches
> no job, and Spark Connect makes that affordable because the analysis happens
> on a server that is already running. It is also the adapter where the
> sandbox has to let something out, so an adapter that declares `remote` is now
> asked which hosts to open and nothing wider - and where the interception
> stops being about better findings and starts being the thing that keeps a dry
> run from reaching a production table, since a statement that is not a query
> is recorded and never sent. `rowless-sql` is implemented as well, which is
> the last engine adapter in the plan and the one with no engine behind it:
> sqlglot qualifies a query against the input schemas and annotates its types,
> so a SQL model resolves with no database, no credentials and nothing
> installed but a parser - and its column lineage resolves *through* CTEs and
> subqueries, which is the one thing the DuckDB adapter refuses to guess at.
> It is also the adapter that has to be clearest about what it does not know:
> there is no binder, so `amount + status` resolves where DuckDB refuses it, a
> function sqlglot has no return type for becomes an ANY column with the
> function named, nullability is not modelled at all, and an unaliased
> aggregate is reported rather than named `_col_0` by a tool when no warehouse
> would call it that. With it, every engine in the pitch has an adapter, and
> what is left is in core rather than in one: a dbt SQL model is discovered and
> reported as unchecked, because its target is a node id rather than an
> importable function and the runtime has no path yet for a pipeline that is a
> compiled query. `rowless-pytest` is implemented as well, and it is the last
> package in the plan and the one that asks a team for least: a test file, no CI
> stage, no new command, no credentials. It runs the comparison
> `rowless.assert_schema` runs and resolves its inputs through the same
> `rowless.toml`, so a contract test cannot hold a different opinion from
> `rowless check` about a column - what it adds is the half only pytest can do.
> The engine a test needs is decided at collection time, so a contributor who
> has pandas and not Polars gets a skip naming the missing engine rather than an
> error; one source cache serves the whole suite; an unverified result fails the
> test instead of passing quietly, with a named per-assertion opt-out the summary
> counts; and typed coverage prints next to the passes, where
> `--rowless-min-coverage` can fail a run in which every assertion passed,
> because forty green tests over a schema rowless typed half of is not the same
> state of the world as forty over one it typed all of. With it every package in
> the layout below is implemented, and what is left is outside them: the GitHub
> Action, Airflow and dlt discovery, and the core runtime path named above. The
> adapter contract is settled and the conformance suite is the specification: see
> [docs/writing-an-adapter.md](docs/writing-an-adapter.md), with
> [rowless-polars](packages/rowless-polars/) as the worked example. Read the
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
| Ibis | unbound tables and `expr.schema()`, and the same query on four backends |

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
  rowless-ibis/           unbound tables, SQL with no database, portability findings
  rowless-iceberg/       schema source: exact renames from field IDs
  rowless-pytest/        pytest plugin: millisecond contract tests
  rowless-openlineage/   emit the column graph where catalogs already look
  rowless-lsp/           language server: the same findings, in the editor
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

## Licence

Apache-2.0.

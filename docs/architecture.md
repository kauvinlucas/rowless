# Architecture

## The one idea

Do not re-implement engine semantics. Run the real pipeline code against frames
that have a full schema and zero rows, and let each engine compute the output
schema with its own rules.

Everything in the repository layout follows from that: if the engines do the
semantic work, the code that remains is a schema vocabulary, a way to run user
code safely, and one adapter per engine to translate.

## The pieces

```
rowless check
    |
    +-- check           the one run: everything below, in order
    |
    |-- config          rowless.toml, or [tool.rowless] in pyproject.toml
    |-- discovery       find pipelines: Dagster, Airflow, dlt, dbt, plain functions
    |-- sources         resolve input schemas: Parquet, Iceberg, dbt, catalog, contracts
    |
    |-- runtime         per pipeline:
    |     |-- worker        one subprocess
    |     |-- sandbox       audit hooks, blocked writes and sockets
    |     |-- adapter       materialize inputs, run code, describe output
    |     |-- rowdep        catch logic that needs rows
    |
    |-- lineage         build the column graph across Python and SQL
    |-- diagnostics     codes, one record shape, many renderers
    |
    +-- exit code       0 clean, 1 findings, 2 rowless itself failed
```

`rowless.check` is that pipeline as a function, returning a report and a
lineage graph and printing nothing. The CLI picks a renderer and an exit code;
the pytest plugin, the language server and `rowless diff`'s base-revision
build all enter at the same place, so there is one answer to "what does
rowless do" rather than four.

Core never imports an engine library. That is enforced by the CI job that runs
core tests in an environment with no engine installed: if that job starts
needing pandas, something has leaked.

## Why a workspace of many distributions

One repository, twelve distributions. The alternative, one package with optional
extras, was rejected for three reasons:

1. **Dependency isolation.** A user who wants Polars should not resolve against
   a PySpark pin. With separate distributions, `pip install rowless
   rowless-polars` pulls in Polars and nothing else.
2. **Ownership.** The proposal splits the work into adapter workstreams with
   their own owners. Separate packages make that real: CODEOWNERS grants review
   rights per package, and an adapter owner can release without the core team.
3. **Release independence after v1.0.** An adapter fixing an engine
   incompatibility should not wait on a core release. Before v1.0 everything
   releases together, because the adapter contract is still moving.

The cost is ceremony: twelve `pyproject.toml` files and a matrix build. uv
workspaces make the development experience single-repository anyway.

## Plugin registries

Four entry-point groups, all discovered the same way:

| Group | Protocol | Lives in |
| --- | --- | --- |
| `rowless.engines` | `EngineAdapter` | one distribution per engine |
| `rowless.sources` | `SchemaSource` | core, plus dbt and Iceberg |
| `rowless.discovery` | `PipelineFinder` | core, plus dbt |
| `rowless.emitters` | lineage emitters | `rowless-openlineage` |

Entry points are listed without being imported, so `rowless adapters` can show
an inactive adapter and a reason rather than crashing on a broken plugin.

## The data flow of one dry run

1. **Resolve inputs.** Sources run in the parent process, before any sandbox
   exists, and results are cached under `.rowless/`. A re-check is offline.
2. **Spawn a worker.** The request crosses as IR JSON, never a pickle: a pickled
   LazyFrame would need Polars in the parent, and it is not there.
3. **Install the sandbox.** Audit hooks are permanent once installed, which is
   exactly why this happens in the child.
4. **Materialize.** The adapter builds real engine objects with the declared
   schema and zero rows. Real objects, not proxies: proxies are what make
   source-parsing tools fail on real code.
5. **Run the user code.** Row-dependent accessors are recorded as they fire.
6. **Describe the result.** The adapter asks the engine what the schema is.
7. **Score it.** Coverage and confidence, with every reason for a downgrade
   recorded.

Steps 4 through 6 are the only engine-specific ones, and they are the adapter
contract.

## Two revisions, for diff

`rowless diff` needs the base revision too. The lineage graph serialises, so the
base is built once in a `git worktree` and cached by commit sha. A PR with ten
pushes builds the base graph once.

## What is deliberately not here

**Static schema inference from Python source.** The proposal is explicit that
parsing Python to guess schemas breaks on real code, and the existing proofs of
concept bear that out. The AST pass in `rowless.analysis` only flags places
where a dry-run result deserves less trust, which is a weaker claim and a
reliable one.

**A UI.** Column lineage is already an OpenLineage facet that Marquez, DataHub
and OpenMetadata read. rowless emits there instead.

**A schema registry.** Sources read whatever a team already has. One more
registry to keep in sync is the problem, not the solution.

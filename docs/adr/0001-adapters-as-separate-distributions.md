# 0001: One distribution per adapter

Status: accepted
Date: project start

## Context

rowless needs adapters for Polars, pandas, DuckDB, sqlglot, dbt, PySpark and
Ibis, plus integrations for pytest, OpenLineage and an LSP. The alternatives
were one package with optional extras, or one distribution per adapter in a
workspace.

## Decision

One distribution per adapter, in a uv workspace, all releasing together until
v1.0.

## Why

**Dependency isolation.** A user wanting Polars should not resolve against a
PySpark pin. `pip install rowless rowless-polars` installs Polars and nothing
else.

**Ownership is the organising principle of the project.** The proposal splits
work into adapter workstreams with their own owners. Separate packages make that
concrete through CODEOWNERS and independent releases, rather than leaving it as
an aspiration in a README.

**Independent releases after v1.0.** An adapter fixing an engine
incompatibility should not wait for a core release.

## Costs

Twelve `pyproject.toml` files, a matrix build, and a release job per package.
Mitigated by the workspace: development stays single-repository.

Also a risk of contract drift, since adapters can be released separately. The
mitigation is that the conformance suite ships in core, so tightening the
contract tightens it everywhere, and the CI conformance job runs every adapter
against the current suite.

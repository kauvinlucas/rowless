"""Task runner. `nox -l` lists sessions; CI calls the same ones."""

from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "types", "tests"]

PYTHONS = ["3.12", "3.13"]

#: Every adapter distribution, keyed by the extra that pulls in its engine.
ADAPTERS = [
    "polars",
    "pandas",
    "duckdb",
    "sql",
    "dbt",
    "pyspark",
    "ibis",
    "iceberg",
]


@nox.session
def lint(session: nox.Session) -> None:
    """Ruff check and format, the same invocation CI uses."""
    session.install("ruff")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session
def types(session: nox.Session) -> None:
    """Type-check every package with mypy strict."""
    session.run_install("uv", "sync", "--all-packages", "--all-extras", external=True)
    session.run("mypy", "packages")


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    """Core tests only: no engine libraries installed, so import guards get exercised."""
    session.run_install("uv", "sync", "--package", "rowless", external=True)
    session.run("pytest", "packages/rowless-core", *session.posargs)


@nox.session(python=PYTHONS)
@nox.parametrize("adapter", ADAPTERS)
def adapter(session: nox.Session, adapter: str) -> None:
    """One isolated environment per adapter, so engine deps never cross-contaminate."""
    session.run_install("uv", "sync", "--package", f"rowless-{adapter}", external=True)
    session.run("pytest", f"packages/rowless-{adapter}", "-m", "not slow", *session.posargs)


@nox.session
def conformance(session: nox.Session) -> None:
    """Run the shared adapter conformance suite against every installed engine."""
    session.run_install("uv", "sync", "--all-packages", "--all-extras", external=True)
    session.run("pytest", "-m", "conformance", *session.posargs)


@nox.session
def docs(session: nox.Session) -> None:
    """Build the docs site, failing on a broken link."""
    session.run_install("uv", "sync", "--group", "docs", external=True)
    session.run("mkdocs", "build", "--strict")

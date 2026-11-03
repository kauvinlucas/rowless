"""Reading `target/manifest.json`, with no dbt installed.

Deliberately stdlib-only. dbt-core is a heavy dependency with its own
dependency opinions, and requiring it would put rowless out of reach of exactly
the CI job that needs it. The manifest is a documented JSON artifact; reading it
directly keeps `rowless check` credential-free and fast.

The cost is version drift: the manifest schema changes between dbt versions. So
the reader is explicit about `metadata.dbt_schema_version`, supports the
versions it has been tested against, and refuses the rest with a clear message
instead of half-parsing a format it does not know.
"""

from __future__ import annotations

from dataclasses import dataclass

from rowless.ir.schema import Schema

#: Manifest schema versions this reader handles. Extend only with a test
#: fixture for the new version committed alongside.
SUPPORTED_VERSIONS: tuple[str, ...] = ("v11", "v12")


@dataclass(frozen=True, slots=True)
class ManifestNode:
    """One dbt node: a model, source, seed or snapshot."""

    unique_id: str
    name: str
    resource_type: str
    schema: Schema
    """Columns from the node contract when it has one, else from the catalog,
    else empty. Which of the three it was decides the confidence of everything
    downstream, so the reader records it."""

    compiled_sql: str | None = None
    depends_on: tuple[str, ...] = ()
    path: str | None = None
    has_contract: bool = False


@dataclass(frozen=True, slots=True)
class Manifest:
    """A parsed dbt manifest."""

    version: str
    project_name: str
    nodes: dict[str, ManifestNode]
    adapter_type: str = "duckdb"
    """dbt adapter name, which decides the sqlglot dialect."""

    @classmethod
    def load(cls, path: str) -> Manifest:
        """Parse a manifest, raising `ConfigError` on an unsupported version."""
        raise NotImplementedError("TODO(dbt)")

    def models(self) -> tuple[ManifestNode, ...]:
        """Just the models, in dependency order."""
        raise NotImplementedError("TODO(dbt)")

    def catalog_schema(self, unique_id: str) -> Schema | None:
        """Columns from `target/catalog.json` when it has been generated.

        Richer than the manifest, since it has real warehouse types, but it
        requires a `dbt docs generate` against a live warehouse, so it is
        treated as optional enrichment rather than a requirement.
        """
        raise NotImplementedError("TODO(dbt)")

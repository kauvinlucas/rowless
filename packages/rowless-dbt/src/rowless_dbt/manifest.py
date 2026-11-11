"""Reading `target/manifest.json`, with no dbt installed.

Deliberately stdlib-only. dbt-core is a heavy dependency with its own
dependency opinions, and requiring it would put rowless out of reach of exactly
the CI job that needs it. The manifest is a documented JSON artifact; reading it
directly keeps `rowless check` credential-free and fast.

The cost is version drift: the manifest schema changes between dbt versions. So
the reader is explicit about `metadata.dbt_schema_version`, supports the
versions it has been tested against, and refuses the rest with a clear message
instead of half-parsing a format it does not know.

## Where a node's columns come from

Three sources of truth, and which one answered decides what every finding
downstream is worth, so the reader records it on the node as `schema_source`:

1. **An enforced contract** (`contract: {enforced: true}` with `data_type` on
   every column). dbt itself guarantees the column set and the types, having
   failed the build if the model did not match. This is the only evidence that
   a declared column list is *complete*.
2. **`target/catalog.json`**, which carries the real warehouse types for every
   column of every relation. It needs `dbt docs generate` against a live
   warehouse, so it is treated as optional enrichment rather than a
   requirement - but when it is there it is ground truth.
3. **Nothing.** A model with neither is left with an empty schema and
   `schema_source="none"`. Its columns may still be *declared* in `schema.yml`
   and are kept on the node, but `declared_schema` is opt-in: dbt lets a
   project document three columns of a twelve-column model, and a schema that
   is quietly missing nine columns turns every correct pipeline that reads one
   of them into a false ColumnNotFound. `rowless_dbt.source` explains the
   opt-in, and resolving the compiled SQL is the better answer when an engine
   that can run it is installed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rowless.errors import ConfigError
from rowless.ir.schema import Field, Origin, Schema

from rowless_dbt.dtypes import dtype_from_sql, sqlglot_dialect

#: Manifest schema versions this reader handles. Extend only with a test
#: fixture for the new version committed alongside.
SUPPORTED_VERSIONS: tuple[str, ...] = ("v11", "v12")

#: `https://schemas.getdbt.com/dbt/manifest/v12.json` -> `v12`.
_VERSION = re.compile(r"/(v\d+)\.json")

#: Node kinds worth a schema. Tests, analyses and operations produce no
#: relation anything can read, so they are skipped rather than carried.
RESOURCE_TYPES: frozenset[str] = frozenset({"model", "seed", "snapshot", "source"})

#: `shop://models/schema.yml` -> `models/schema.yml`.
_PACKAGE_PREFIX = re.compile(r"^[^:]+://")


@dataclass(frozen=True, slots=True)
class DbtColumn:
    """One column as `schema.yml` declares it.

    Kept raw alongside the resolved schema, because the declaration carries
    things the IR does not: the description a warehouse never stores, and
    whether the type was declared at all.
    """

    name: str
    data_type: str | None = None
    description: str | None = None
    not_null: bool = False
    """True when the column carries a `not_null` constraint in its contract."""


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

    schema_source: str = "none"
    """Which evidence answered: `contract`, `catalog` or `none`."""

    language: str = "sql"
    """`sql` or `python`. A dbt Python model is a Python pipeline, and is
    discovered as one rather than as a query."""

    columns: tuple[DbtColumn, ...] = ()
    """Every column `schema.yml` declares, typed or not."""

    relation_name: str | None = None
    """The quoted `"database"."schema"."table"`, which is how compiled SQL
    refers to this node."""

    @property
    def is_model(self) -> bool:
        """True for a model, as opposed to a source, seed or snapshot."""
        return self.resource_type == "model"


@dataclass(frozen=True, slots=True)
class Manifest:
    """A parsed dbt manifest."""

    version: str
    project_name: str
    nodes: dict[str, ManifestNode]
    adapter_type: str = "duckdb"
    """dbt adapter name, which decides the sqlglot dialect."""

    invocation_id: str = ""
    """The run that wrote this manifest. Half of the source fingerprint."""

    path: str = ""
    """Where the manifest was read from, so callers can find the project."""

    catalog: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    """`unique_id` to column-name-to-warehouse-type, from `catalog.json`."""

    @property
    def dialect(self) -> str:
        """The sqlglot dialect for this project's adapter."""
        return sqlglot_dialect(self.adapter_type)

    @property
    def project_dir(self) -> str:
        """The dbt project directory: the parent of `target/`."""
        return str(Path(self.path).resolve().parent.parent) if self.path else "."

    @classmethod
    def load(cls, path: str, *, catalog: str | None = None) -> Manifest:
        """Parse a manifest, raising `ConfigError` on an unsupported version.

        Args:
            path: path to `target/manifest.json`.
            catalog: path to `target/catalog.json`. Defaults to the sibling of
                the manifest, and a missing one is not an error: most projects
                have never run `dbt docs generate`.
        """
        payload = _read_json(path)
        metadata = _mapping(payload.get("metadata"))
        version = _version_of(metadata, path)
        catalog_path = catalog or str(Path(path).with_name("catalog.json"))
        catalog_columns = _read_catalog(catalog_path)
        adapter_type = str(metadata.get("adapter_type") or "")
        dialect = sqlglot_dialect(adapter_type)

        nodes: dict[str, ManifestNode] = {}
        for section in ("nodes", "sources"):
            for unique_id, raw in _mapping(payload.get(section)).items():
                node = _node(
                    str(unique_id),
                    _mapping(raw),
                    catalog=catalog_columns.get(str(unique_id), {}),
                    dialect=dialect,
                    project_dir=str(Path(path).resolve().parent.parent),
                )
                if node is not None:
                    nodes[node.unique_id] = node

        return cls(
            version=version,
            project_name=str(metadata.get("project_name") or ""),
            nodes=nodes,
            adapter_type=adapter_type or "duckdb",
            invocation_id=str(metadata.get("invocation_id") or ""),
            path=str(path),
            catalog=catalog_columns,
        )

    def models(self) -> tuple[ManifestNode, ...]:
        """Just the models, in dependency order.

        Upstream before downstream, so a caller resolving schemas in one pass
        always has a model's inputs before it needs them. Ties are broken by
        `unique_id`, which keeps the order stable between runs, and a cycle -
        which dbt refuses to build, but which a hand-edited manifest can
        contain - is appended rather than dropped.
        """
        models = {node.unique_id: node for node in self.nodes.values() if node.is_model}
        return tuple(models[unique_id] for unique_id in _topological(models))

    def node_for(self, reference: str) -> ManifestNode | None:
        """Find a node by `unique_id`, or by the name a human would write.

        `model.shop.stg_orders` is the exact form. `stg_orders` is accepted
        too, because nobody wants to type the package name into every URI, and
        it resolves against models first: a source and a model may share a
        name, and the model is what a pipeline reads.
        """
        found = self.nodes.get(reference)
        if found is not None:
            return found
        for prefix in ("model", "seed", "snapshot", "source"):
            for node in self.nodes.values():
                if node.resource_type == prefix and node.name == reference:
                    return node
        return None

    def catalog_schema(self, unique_id: str) -> Schema | None:
        """Columns from `target/catalog.json` when it has been generated.

        Richer than the manifest, since it has real warehouse types, but it
        requires a `dbt docs generate` against a live warehouse, so it is
        treated as optional enrichment rather than a requirement.
        """
        columns = self.catalog.get(unique_id)
        if not columns:
            return None
        node = self.nodes.get(unique_id)
        origin = Origin(kind="source", ref=unique_id, file=node.path if node else None)
        return Schema(
            fields=tuple(
                Field(
                    name=name,
                    dtype=dtype_from_sql(spelling, dialect=self.dialect),
                    origin=origin,
                )
                for name, spelling in columns.items()
            ),
            name=node.name if node else unique_id.rpartition(".")[2],
        )

    def declared_schema(self, unique_id: str) -> Schema | None:
        """Columns as `schema.yml` declares them, contract or no contract.

        Opt-in, and not what `nodes[...].schema` reports, because dbt does not
        require the declaration to be complete. See the module docstring.
        """
        node = self.nodes.get(unique_id)
        if node is None or not node.columns:
            return None
        origin = Origin(kind="source", ref=unique_id, file=node.path)
        return Schema(
            fields=tuple(_field(column, self.dialect, origin) for column in node.columns),
            name=node.name,
        )


def find_manifest(start: str = ".", *, target: str | None = None) -> str | None:
    """Locate `target/manifest.json` for the project `start` sits in.

    Order of preference, so that an explicit answer always beats a guess:

    1. `target`, when the caller was given one (a `?manifest=` URI option).
    2. `DBT_TARGET_PATH` or `DBT_PROJECT_DIR`, dbt's own environment variables,
       which a CI job that runs dbt has already set.
    3. The nearest `dbt_project.yml` walking up from `start`, and the target
       directory beside it.

    Returns None when there is no manifest to read, which is how the source
    and the finder decide they are not applicable rather than failing.
    """
    import os

    if target:
        candidate = Path(target)
        found = candidate if candidate.is_file() else candidate / "manifest.json"
        return str(found) if found.is_file() else None

    for variable, suffix in (("DBT_TARGET_PATH", ""), ("DBT_PROJECT_DIR", "target")):
        value = os.environ.get(variable)
        if not value:
            continue
        candidate = Path(value, suffix, "manifest.json") if suffix else Path(value, "manifest.json")
        if candidate.is_file():
            return str(candidate)

    here = Path(start).resolve()
    if here.is_file():
        here = here.parent
    for directory in (here, *here.parents):
        if not (directory / "dbt_project.yml").is_file():
            continue
        for target_dir in _target_dirs(directory):
            candidate = directory / target_dir / "manifest.json"
            if candidate.is_file():
                return str(candidate)
        return None
    return None


def project_name_of(project_dir: str) -> str:
    """The `name:` from `dbt_project.yml`, read without a YAML parser.

    One shallow key off the top of a small file, matched as text. Pulling in a
    YAML dependency to read it would cost every user of `rowless-dbt` a wheel
    they do not otherwise need, and a project whose name rowless cannot read
    still works: the name is only used to spell a URI more conveniently.
    """
    try:
        text = Path(project_dir, "dbt_project.yml").read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        if not line.startswith("name:"):
            continue
        return line.partition(":")[2].strip().strip("\"'")
    return ""


def _target_dirs(project_dir: Path) -> tuple[str, ...]:
    """Target directories to look in: the configured one, then the default."""
    configured = _target_path_of(project_dir)
    return (configured, "target") if configured and configured != "target" else ("target",)


def _target_path_of(project_dir: Path) -> str:
    """The `target-path:` from `dbt_project.yml`, when it sets one."""
    try:
        text = Path(project_dir, "dbt_project.yml").read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        if not line.startswith("target-path:"):
            continue
        return line.partition(":")[2].strip().strip("\"'")
    return ""


def _read_json(path: str) -> Mapping[str, Any]:
    """Read a JSON object, reporting the file in every failure."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read dbt manifest {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{path}: expected a JSON object, found {type(payload).__name__}")
    return payload


def _read_catalog(path: str) -> dict[str, dict[str, str]]:
    """Column types per node from `catalog.json`, or nothing when absent.

    A missing or unreadable catalog is not an error: it is the normal state of
    a project that has not run `dbt docs generate`, and the manifest alone is
    enough to work with.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    found: dict[str, dict[str, str]] = {}
    for section in ("nodes", "sources"):
        for unique_id, raw in _mapping(payload.get(section)).items():
            columns = _mapping(_mapping(raw).get("columns"))
            typed = {
                str(_mapping(column).get("name") or name): str(_mapping(column).get("type") or "")
                for name, column in columns.items()
            }
            if typed:
                found[str(unique_id)] = typed
    return found


def _version_of(metadata: Mapping[str, Any], path: str) -> str:
    """The manifest schema version, refusing one this reader has not been tested on."""
    declared = str(metadata.get("dbt_schema_version") or "")
    found = _VERSION.search(declared)
    version = found.group(1) if found else ""
    if version not in SUPPORTED_VERSIONS:
        supported = ", ".join(SUPPORTED_VERSIONS)
        raise ConfigError(
            f"{path}: manifest schema version {declared or 'unknown'!r} is not supported; "
            f"rowless-dbt reads {supported}. Half-parsing a format it does not know would "
            "produce findings nobody could trust"
        )
    return version


def _node(
    unique_id: str,
    raw: Mapping[str, Any],
    *,
    catalog: Mapping[str, str],
    dialect: str,
    project_dir: str,
) -> ManifestNode | None:
    """Build one node, or None for a kind that produces no relation."""
    resource_type = str(raw.get("resource_type") or "")
    if resource_type not in RESOURCE_TYPES:
        return None

    name = str(raw.get("name") or unique_id.rpartition(".")[2])
    columns = _columns(raw)
    enforced = _contract_enforced(raw)
    path = _node_path(raw, project_dir)
    schema, schema_source = _schema_of(
        unique_id,
        name=name,
        columns=columns,
        enforced=enforced,
        catalog=catalog,
        dialect=dialect,
        path=path,
        patch_path=_patch_path(raw, project_dir),
    )
    return ManifestNode(
        unique_id=unique_id,
        name=name,
        resource_type=resource_type,
        schema=schema,
        compiled_sql=_compiled_sql(raw),
        depends_on=_depends_on(raw),
        path=path,
        has_contract=enforced,
        schema_source=schema_source,
        language=str(raw.get("language") or "sql"),
        columns=columns,
        relation_name=_optional_str(raw.get("relation_name")),
    )


def _schema_of(
    unique_id: str,
    *,
    name: str,
    columns: Sequence[DbtColumn],
    enforced: bool,
    catalog: Mapping[str, str],
    dialect: str,
    path: str | None,
    patch_path: str | None,
) -> tuple[Schema, str]:
    """The node schema and where it came from: contract, catalog or nothing."""
    if enforced and columns and all(column.data_type for column in columns):
        origin = Origin(kind="contract", ref=unique_id, file=patch_path or path)
        fields = tuple(_field(column, dialect, origin) for column in columns)
        return Schema(fields=fields, name=name), "contract"
    if catalog:
        origin = Origin(kind="source", ref=unique_id, file=path)
        fields = tuple(
            Field(name=column, dtype=dtype_from_sql(spelling, dialect=dialect), origin=origin)
            for column, spelling in catalog.items()
        )
        return Schema(fields=fields, name=name), "catalog"
    return Schema.empty(name=name), "none"


def _field(column: DbtColumn, dialect: str, origin: Origin) -> Field:
    """One IR field from a declared dbt column."""
    return Field(
        name=column.name,
        dtype=dtype_from_sql(column.data_type, dialect=dialect),
        nullable=not column.not_null,
        origin=origin,
        description=column.description,
    )


def _columns(raw: Mapping[str, Any]) -> tuple[DbtColumn, ...]:
    """Declared columns, in declaration order."""
    found: list[DbtColumn] = []
    for key, value in _mapping(raw.get("columns")).items():
        column = _mapping(value)
        found.append(
            DbtColumn(
                name=str(column.get("name") or key),
                data_type=_optional_str(column.get("data_type")),
                description=_optional_str(column.get("description")) or None,
                not_null=_has_not_null(column.get("constraints")),
            )
        )
    return tuple(found)


def _has_not_null(constraints: object) -> bool:
    """True when a column's contract constraints include `not_null`."""
    if not isinstance(constraints, Iterable) or isinstance(constraints, str | bytes):
        return False
    return any(str(_mapping(entry).get("type") or "") == "not_null" for entry in constraints)


def _contract_enforced(raw: Mapping[str, Any]) -> bool:
    """True when dbt enforced a contract on this node.

    Read from both places dbt writes it: the node's own `contract` block, and
    the `config.contract` the project set. Either one being enforced is dbt
    guaranteeing the column set, which is the only reason this reader trusts a
    declared column list to be complete.
    """
    for container in (raw, _mapping(raw.get("config"))):
        contract = _mapping(container.get("contract"))
        if bool(contract.get("enforced")):
            return True
    return False


def _compiled_sql(raw: Mapping[str, Any]) -> str | None:
    """The compiled query, when dbt has compiled this node."""
    for key in ("compiled_code", "compiled_sql"):
        value = _optional_str(raw.get(key))
        if value:
            return value
    return None


def _depends_on(raw: Mapping[str, Any]) -> tuple[str, ...]:
    """The nodes this one reads, as `unique_id`s, macros excluded."""
    nodes = _mapping(raw.get("depends_on")).get("nodes")
    if not isinstance(nodes, Iterable) or isinstance(nodes, str | bytes):
        return ()
    return tuple(str(entry) for entry in nodes)


def _node_path(raw: Mapping[str, Any], project_dir: str) -> str | None:
    """Absolute path to the file that defines this node, when it has one."""
    relative = _optional_str(raw.get("original_file_path")) or _optional_str(raw.get("path"))
    if not relative:
        return None
    return str(Path(project_dir, relative))


def _patch_path(raw: Mapping[str, Any], project_dir: str) -> str | None:
    """Absolute path to the `schema.yml` that declares this node's columns.

    dbt writes it as `package://models/schema.yml`; the package prefix is
    dropped, because a finding has to point at a file on this disk.
    """
    declared = _optional_str(raw.get("patch_path"))
    if not declared:
        return None
    return str(Path(project_dir, _PACKAGE_PREFIX.sub("", declared)))


def _topological(models: Mapping[str, ManifestNode]) -> Iterator[str]:
    """Model ids, upstream first, ties and cycles in `unique_id` order."""
    pending = {
        unique_id: {parent for parent in node.depends_on if parent in models}
        for unique_id, node in models.items()
    }
    emitted: set[str] = set()
    while pending:
        ready = sorted(unique_id for unique_id, parents in pending.items() if parents <= emitted)
        if not ready:
            # A cycle, which dbt would have refused to build. Emitting the rest
            # in a stable order beats dropping models nobody can see.
            yield from sorted(pending)
            return
        for unique_id in ready:
            emitted.add(unique_id)
            del pending[unique_id]
            yield unique_id


def _mapping(value: object) -> Mapping[str, Any]:
    """`value` as a mapping, or an empty one: manifests have optional sections."""
    return value if isinstance(value, Mapping) else {}


def _optional_str(value: object) -> str | None:
    """A non-empty string, or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None

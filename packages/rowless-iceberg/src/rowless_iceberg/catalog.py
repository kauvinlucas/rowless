"""Finding the table a `iceberg://` URI names, with or without a catalog.

Two ways in, and the second one is the one that matters in CI:

* **Through a catalog.** `iceberg://prod.sales.orders` loads catalog `prod` and
  table `sales.orders` through PyIceberg's own configuration - `.pyiceberg.yaml`
  or `PYICEBERG_*` environment variables. Catalog configuration is never read
  out of `rowless.toml`: a REST catalog URI and its token are credentials, and
  the same reasoning keeps the warehouse DSN out of config in core.
* **Through a metadata file.** `iceberg://orders?metadata=.../v3.metadata.json`
  reads the table metadata directly, with no catalog and no credentials. An
  Iceberg table is self-describing, so this is the whole schema rather than a
  subset of it - which makes it the Iceberg equivalent of the committed catalog
  snapshot core's warehouse source leans on, and the reason `rowless check` can
  run on an Iceberg-backed repository in a CI job with no warehouse access.

## How a path splits into a catalog and an identifier

`iceberg://catalog.db.table`, and the rule has to be stated because Iceberg
namespaces nest arbitrarily and no URI can be unambiguous about it:

| URI | catalog | table |
| --- | --- | --- |
| `iceberg://prod.sales.orders` | `prod` | `sales.orders` |
| `iceberg://sales.orders` | the default | `sales.orders` |
| `iceberg://sales.orders?catalog=prod` | `prod` | `sales.orders` |

So the first segment is the catalog name whenever there are three or more of
them, `?catalog=` overrides that and leaves the whole path as the identifier,
and `ROWLESS_ICEBERG_CATALOG` names a default for a repository where every URI
would otherwise have to repeat it. A single segment is refused: Iceberg tables
live in a namespace, and guessing one would resolve the wrong table silently.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rowless.errors import BadSourceRef, SchemaNotFound
from rowless.sources.base import SourceRef

#: Catalog to use when a URI names none. PyIceberg has its own notion of a
#: default catalog; this is for a repository whose URIs are written without one
#: but whose catalog is not called `default`.
CATALOG_ENV = "ROWLESS_ICEBERG_CATALOG"

#: Loaded tables, keyed by what identifies them. A `rowless check` run resolves
#: one URI per pipeline input, and a REST catalog round trip for each of them is
#: the difference between a linter and a chore. It is also what makes
#: `fingerprint` and `resolve` agree: both see one metadata version per run, so
#: a commit landing mid-run cannot produce a schema from one version cached
#: under the fingerprint of another.
_LOADED: dict[str, tuple[float, Any]] = {}


@dataclass(frozen=True, slots=True)
class TableRef:
    """A `iceberg://` URI resolved into something loadable."""

    identifier: str
    """Dotted table identifier, e.g. `sales.orders`."""

    catalog: str | None = None
    """Catalog name, or None for PyIceberg's default."""

    metadata: str | None = None
    """Metadata file location, when the URI gave one. Bypasses the catalog."""

    @property
    def key(self) -> str:
        """Cache key: what distinguishes one loadable table from another."""
        if self.metadata is not None:
            return f"metadata:{self.metadata}"
        return f"catalog:{self.catalog or ''}:{self.identifier}"


def parse_table_ref(ref: SourceRef) -> TableRef:
    """Split a source URI into a catalog, an identifier and a metadata location.

    Raises:
        BadSourceRef: when the URI names no table, or names one with no
            namespace and no metadata file to read it out of.
    """
    metadata = _option(ref, "metadata")
    catalog = _option(ref, "catalog") or os.environ.get(CATALOG_ENV) or None
    segments = [segment for segment in ref.path.split(".") if segment]
    if not segments:
        raise BadSourceRef(f"source URI {ref.uri!r} names no Iceberg table")

    if metadata is not None:
        # The metadata file is the address, so the path is only a label. Reading
        # a catalog and an identifier out of it as well would let the two
        # disagree about which table this is.
        return TableRef(identifier=".".join(segments), metadata=metadata)

    if catalog is None and len(segments) >= 3:
        catalog, segments = segments[0], segments[1:]
    if len(segments) < 2:
        raise BadSourceRef(
            f"source URI {ref.uri!r} names a table with no namespace. Iceberg tables live in "
            "one, so write iceberg://namespace.table, iceberg://catalog.namespace.table, or "
            "point at the metadata file with ?metadata=<path>.metadata.json"
        )
    return TableRef(identifier=".".join(segments), catalog=catalog)


def available() -> bool:
    """True when PyIceberg imports.

    Deliberately not "and a catalog is configured". The `?metadata=` route needs
    no catalog at all, and it is the route a CI job uses, so gating on catalog
    configuration would hide the credential-free path behind a message saying
    the source is not installed. Whether one particular table resolves is
    `load_table`'s problem, and it reports which of the two routes failed and
    how to fix it.

    Never raises. `find_spec` returns None for a module that is simply absent,
    but it propagates whatever a meta-path finder does - and a half-removed
    distribution leaves one that raises. This is called while *listing* sources,
    including by `rowless doctor`, which is the command a user runs precisely
    when their environment is broken.
    """
    try:
        return importlib.util.find_spec("pyiceberg") is not None
    except (ImportError, ValueError):
        return False


def load_table(ref: SourceRef) -> Any:
    """Load the PyIceberg table a URI names, cached for the run.

    Returns a `pyiceberg.table.Table`, typed loosely because PyIceberg ships no
    stubs and a precise annotation here would be a claim nothing checks.

    Raises:
        SchemaNotFound: when the table, the catalog or the metadata file cannot
            be read. The message names the ways to fix it rather than leaving a
            PyIceberg traceback as the user-facing error.
    """
    table_ref = parse_table_ref(ref)
    stamp = _stamp(table_ref)
    cached = _LOADED.get(table_ref.key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    table = _static(table_ref, ref) if table_ref.metadata is not None else _catalog(table_ref, ref)
    _LOADED[table_ref.key] = (stamp, table)
    return table


def clear_cache() -> None:
    """Forget every loaded table. For tests, and for a long-lived LSP process."""
    _LOADED.clear()


def _static(table_ref: TableRef, ref: SourceRef) -> Any:
    """Read a metadata file directly, with no catalog in the way."""
    from pyiceberg.table import StaticTable

    location = table_ref.metadata or ""
    try:
        return StaticTable.from_metadata(location)
    except Exception as exc:
        raise SchemaNotFound(
            f"{ref.uri}: cannot read the Iceberg metadata at {location}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _catalog(table_ref: TableRef, ref: SourceRef) -> Any:
    """Load the table through a configured PyIceberg catalog."""
    from pyiceberg.catalog import list_catalogs, load_catalog

    wanted = table_ref.catalog or "the default catalog"
    try:
        catalog = load_catalog(table_ref.catalog)
    except Exception as exc:
        known = ", ".join(sorted(list_catalogs())) or "none"
        raise SchemaNotFound(
            f"{ref.uri}: cannot load {wanted}: {type(exc).__name__}: {exc}. "
            f"Configured catalogs: {known}. Configure one in .pyiceberg.yaml or through "
            f"PYICEBERG_* variables, name it with ?catalog= or {CATALOG_ENV}, or skip the "
            "catalog entirely with ?metadata=<path>.metadata.json"
        ) from exc

    try:
        return catalog.load_table(table_ref.identifier)
    except Exception as exc:
        raise SchemaNotFound(
            f"{ref.uri}: {table_ref.identifier!r} is not in {wanted}: "
            f"{type(exc).__name__}: {exc}. {_reading_hint(ref, table_ref)}"
        ) from exc


def _reading_hint(ref: SourceRef, table_ref: TableRef) -> str:
    """Spell out how the URI was split, since that is the usual mistake.

    A three-segment URI is read as catalog, namespace, table. A user who meant
    a two-level namespace in the default catalog gets a `NoSuchTableError` for a
    table they can see in their catalog, and nothing in that message says why.
    """
    if _option(ref, "catalog") or table_ref.catalog is None:
        return "Check the namespace and the table name."
    return (
        f"rowless read {ref.uri!r} as catalog {table_ref.catalog!r} and table "
        f"{table_ref.identifier!r}; if the whole path is the table's, pass the catalog "
        "separately with ?catalog="
    )


def _stamp(table_ref: TableRef) -> float:
    """Cache validity token: a metadata file's mtime, or nothing for a catalog.

    A static table is a file, and anything that regenerates it rewrites it in
    place, so its mtime has to invalidate the entry. A catalog table has no such
    token, and one consistent view per process is the property that matters
    there.
    """
    if table_ref.metadata is None:
        return 0.0
    try:
        return Path(table_ref.metadata).stat().st_mtime_ns / 1e9
    except OSError:
        # Not a local file, or not readable. Either way the load is what
        # reports it, with a message naming the location.
        return 0.0


def _option(ref: SourceRef, key: str) -> str | None:
    """One URI option, or None when it is absent or empty."""
    value = ref.options.get(key)
    text = str(value).strip() if value is not None else ""
    return text or None

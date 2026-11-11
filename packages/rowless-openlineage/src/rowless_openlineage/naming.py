"""rowless node ids to OpenLineage dataset and job names.

This module decides whether rowless's graph lands *on* the datasets a catalog
already knows about, or beside them as a second disconnected copy of the
warehouse. A catalog merges lineage by dataset identity and nothing else, so
`s3://warehouse/stg_orders` has to come out as the namespace and name the
OpenLineage naming conventions give it - the same pair the Spark, Airflow and
dbt integrations emit - or the whole emission is decoration.

The conventions are at https://openlineage.io/docs/spec/naming, and the table
below is this package's reading of them for every URI scheme rowless resolves:

| rowless node id | namespace | name |
| --- | --- | --- |
| `s3://bucket/key` | `s3://bucket` | `key` |
| `gs://bucket/key` | `gs://bucket` | `key` |
| `az://container/key` | `az://container` | `key` |
| `file:///srv/data/orders` | `file` | `/srv/data/orders` |
| `file://host/data/orders` | `file://host` | `/data/orders` |
| `data/orders.parquet` | `file` | `data/orders.parquet` |
| `iceberg://catalog.db.table` | `iceberg://catalog` | `db.table` |
| `warehouse://db.schema.table` | `warehouse` | `db.schema.table` |
| `snapshot://db.schema.table` | `warehouse` | `db.schema.table` |
| `dbt://model.shop.stg_orders` | `dbt` | `model.shop.stg_orders` |
| `lake://stg_orders` | `lake` | `stg_orders` |
| `pipelines.revenue:daily_revenue` | the job namespace | the target, verbatim |

Three of those rows are compromises rather than conventions, and each is a
place where rowless does not hold what the convention wants:

* **Azure.** The canonical namespace is
  `abfss://container@account.dfs.core.windows.net`, and an `az://` URI carries
  no account. Inventing one would produce a dataset name that looks canonical
  and points nowhere, so the container alone is emitted.
* **A warehouse table.** The convention wants the datasource:
  `postgres://host:5432`. rowless deliberately keeps the DSN out of config and
  in an environment variable, because it is a credential, so a node id does
  not contain the host.
* **A bare path.** The convention wants an absolute path. rowless node ids are
  the URI the pipeline was written with, which is usually repo-relative, and a
  relative name is the same on every machine while an absolute one is not.

All three are why `ROWLESS_OPENLINEAGE_NAMESPACES` exists: it maps a scheme
onto the namespace a team's catalog actually uses, which is the one piece of
knowledge that cannot be derived from the repository.

    ROWLESS_OPENLINEAGE_NAMESPACES="warehouse=postgres://db:5432,lake=s3://warehouse"

A bare path has no scheme, so its override is written with an empty one -
`=file://nas1` - which is the row most likely to need it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from rowless.lineage.graph import Node, NodeKind

#: Namespace for rowless's own nodes: a Python pipeline's job, and the dataset
#: standing for its output when nothing downstream names a real table. Taken
#: from `OPENLINEAGE_NAMESPACE`, the variable every other integration reads.
DEFAULT_NAMESPACE = "rowless"

#: Environment variable for the job namespace, honoured so rowless's events
#: land in the same namespace as the rest of a team's pipelines.
ENV_NAMESPACE = "OPENLINEAGE_NAMESPACE"

#: Environment variable for the per-scheme namespace overrides described above.
ENV_NAMESPACE_MAP = "ROWLESS_OPENLINEAGE_NAMESPACES"

#: Object stores whose first path segment is the bucket, and whose namespace is
#: therefore `scheme://bucket`.
_BUCKET_SCHEMES: frozenset[str] = frozenset({"s3", "s3a", "gs", "gcs", "az", "abfs", "abfss"})

#: Schemes whose path is a dotted `catalog.database.table`, where the first
#: segment names the catalog and so belongs in the namespace.
_CATALOG_SCHEMES: frozenset[str] = frozenset({"iceberg"})

#: Schemes that name a table in a warehouse rowless has no host for. A live
#: read and a snapshot read of one warehouse share a namespace on purpose:
#: they are the same table, and two namespaces would split its lineage.
_WAREHOUSE_SCHEMES: Mapping[str, str] = {"warehouse": "warehouse", "snapshot": "warehouse"}

#: Kinds whose node id is a filesystem path rather than a Python target, when
#: it carries no scheme of its own.
_PATH_KINDS: frozenset[NodeKind] = frozenset({NodeKind.SOURCE, NodeKind.SINK})


@dataclass(frozen=True, slots=True)
class DatasetName:
    """An OpenLineage `(namespace, name)` pair."""

    namespace: str
    name: str

    def __str__(self) -> str:
        return f"{self.namespace}:{self.name}"


@dataclass(frozen=True, slots=True)
class Naming:
    """The mapping rules, resolved once per emission.

    A value rather than a set of functions because the job namespace and the
    overrides have to be identical for every event in one emission: half an
    emission in one namespace is worse in a catalog than none at all.
    """

    namespace: str = DEFAULT_NAMESPACE
    """Namespace for jobs, and for datasets that are not real tables."""

    overrides: Mapping[str, str] = field(default_factory=dict)
    """Scheme to namespace, from `ROWLESS_OPENLINEAGE_NAMESPACES`."""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Naming:
        """Read the namespace and the overrides out of the environment."""
        environ = os.environ if env is None else env
        return cls(
            namespace=environ.get(ENV_NAMESPACE) or DEFAULT_NAMESPACE,
            overrides=parse_overrides(environ.get(ENV_NAMESPACE_MAP, "")),
        )

    def dataset(self, node: Node) -> DatasetName:
        """The dataset a node is published as.

        A node id with a scheme is a real table wherever it appears, including
        when the node is a dbt model rowless ran rather than read: the point of
        the mapping is that both halves of a seam land on one dataset.
        """
        scheme, separator, remainder = node.id.partition("://")
        if not separator:
            if node.kind in _PATH_KINDS:
                return self._apply(scheme="", found=_path_name(node.id))
            return DatasetName(namespace=self.namespace, name=node.id)
        return self._apply(scheme=scheme, found=_with_scheme(scheme, remainder))

    def job(self, node: Node) -> DatasetName:
        """The job a non-source node is published as.

        The name is the rowless target, verbatim: `rowless explain`, the
        diagnostics and the catalog then all call the pipeline the same thing,
        and a reader can paste it back into the CLI.
        """
        return DatasetName(namespace=self.namespace, name=node.id)

    def _apply(self, *, scheme: str, found: DatasetName) -> DatasetName:
        """Replace the namespace when the environment overrides this scheme."""
        override = self.overrides.get(scheme)
        if override is None:
            return found
        return DatasetName(namespace=override, name=found.name)


def parse_overrides(raw: str) -> dict[str, str]:
    """Parse `scheme=namespace,scheme=namespace`, skipping anything malformed.

    An empty scheme is legal and means a bare filesystem path, which is the
    row that most often needs overriding; an empty namespace is not, because
    there is no namespace an entry could usefully map a scheme onto.

    Skipping rather than raising: this runs inside `available()`, which the
    plugin protocol says must not raise, and a typo in one entry must not stop
    the other entries - or the emission - from working. A dropped entry shows
    up as a dataset in the default namespace, which is visible in the catalog.
    """
    found: dict[str, str] = {}
    for entry in raw.split(","):
        scheme, separator, namespace = entry.partition("=")
        if not separator:
            continue
        value = namespace.strip()
        if value:
            found[scheme.strip()] = value
    return found


def _with_scheme(scheme: str, remainder: str) -> DatasetName:
    """Split a `scheme://remainder` node id per the naming conventions."""
    path = remainder.split("?", 1)[0]
    if scheme in _BUCKET_SCHEMES:
        bucket, _, key = path.partition("/")
        return DatasetName(namespace=f"{scheme}://{bucket}", name=key.strip("/") or bucket)
    if scheme == "file":
        host, _, rest = path.partition("/")
        if host:
            return DatasetName(namespace=f"file://{host}", name="/" + rest.strip("/"))
        return DatasetName(namespace="file", name="/" + rest.strip("/"))
    if scheme in _CATALOG_SCHEMES:
        catalog, separator, table = path.partition(".")
        if separator and table:
            return DatasetName(namespace=f"{scheme}://{catalog}", name=table)
        return DatasetName(namespace=scheme, name=path)
    if scheme in _WAREHOUSE_SCHEMES:
        return DatasetName(namespace=_WAREHOUSE_SCHEMES[scheme], name=path)
    return DatasetName(namespace=scheme, name=path.strip("/") or path)


def _path_name(node_id: str) -> DatasetName:
    """A filesystem node id, kept as rowless recorded it.

    Backslashes are normalised because a dataset emitted from a developer's
    Windows machine and the same dataset emitted from CI have to be one dataset
    in the catalog. Nothing else about the path is touched, for the reason in
    the module docstring: the node id is the spelling the pipeline used.
    """
    cleaned = node_id.replace("\\", "/").split("?", 1)[0]
    return DatasetName(namespace="file", name=cleaned)

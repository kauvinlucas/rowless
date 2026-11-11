"""dbt models as pipelines to check.

Each model is a node: its compiled SQL resolved against its upstream schemas.
This is the part that overlaps with existing tools, and the overlap is the
point. Once dbt models and Python pipelines are in one graph, `rowless diff`
produces the tree from the proposal, where a single rename lists both a Python
function and a SQL mart as affected.

Python models in dbt get the same treatment as any other Python pipeline, which
is the thing the dbt column-lineage tools list as unsupported.

## What a discovered model looks like

One `Pipeline` per SQL model, with:

* `target` set to the dbt `unique_id` (`model.shop.stg_orders`). A SQL model
  has no importable reference, and the unique id is the identity dbt itself
  uses, the one `dbt://` URIs are written with, and the one a SQL runner can
  resolve back to compiled SQL through the manifest.
* `inputs` naming each upstream node and the `dbt://` URI that resolves it, so
  a rename in an upstream model reaches this model through the ordinary source
  resolution rather than through anything dbt-specific.
* `engine` set to `sql`, because that is what resolves a compiled query.
* `file` pointing at the `.sql` file, so a finding is clickable.

## Two gaps, both in core rather than here

**A SQL model cannot be dry-run yet.** The runtime calls an importable target,
and it has no path for a non-importable one. Resolving a model means handing
its compiled SQL to an engine that implements `SupportsSql`, and two now do -
`rowless-duckdb`, with a real binder, and `rowless-ibis`, which parses the
query with sqlglot and types it with ibis's rules and so needs no database at
all. `dbt://` URIs resolve through whichever is installed, which is what
`rowless_dbt.source` uses them for, but the runtime still cannot *run* a
discovered model as a pipeline. Until it can, a discovered model is reported as
unchecked: `rowless check` resolves its inputs, finds no path to the SQL and
says so. That is the honest outcome, and `tests/test_discovery.py` asserts it
so that the day it changes, it changes visibly.

**A dbt Python model cannot be called yet.** dbt gives it the signature
`model(dbt, session)` and expects the framework to supply both arguments; the
runtime binds inputs by parameter name and has no way for a finder to say
"call this with these objects". A `dbt` shim that answered `ref()` out of
`rowless.source` would be a dozen lines here, and is worth writing the moment
core can pass it in. Python models are therefore skipped rather than yielded
as pipelines nothing can run.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import ClassVar

from rowless.discovery.base import Pipeline

from rowless_dbt.dtypes import sqlglot_dialect
from rowless_dbt.manifest import Manifest, ManifestNode, find_manifest

#: The engine that resolves a compiled dbt model.
SQL_ENGINE = "sql"


class DbtModelDiscovery:
    """Find dbt models, SQL and Python alike."""

    name: ClassVar[str] = "dbt"
    requires: ClassVar[str] = ""

    def available(self) -> bool:
        """True when a dbt project with a compiled manifest can be found.

        Looked for from the working directory, because that is all a finder is
        given: core asks whether a backend is applicable before it says which
        roots are being checked. So `rowless check analytics/` from outside the
        project skips this finder, while running it from anywhere inside the
        project - or with `DBT_PROJECT_DIR` set, which a CI job that runs dbt
        already has - finds it. `find` then searches the roots as well.
        """
        try:
            return find_manifest() is not None
        except Exception:
            # Discovery lists every finder before running any of them, and a
            # finder that raises while being asked is a finder that hides the
            # pipelines every other one found.
            return False

    def find(self, roots: Sequence[str]) -> Iterable[Pipeline]:
        """Yield one pipeline per model, with upstream refs as input URIs."""
        manifest = _manifest_for(roots)
        if manifest is None:
            return ()
        return tuple(
            _pipeline(manifest, node) for node in manifest.models() if node.language != "python"
        )

    def dialect_for(self, adapter_type: str) -> str:
        """Dbt adapter name to sqlglot dialect, e.g. `snowflake`, `bigquery`.

        The table is in `rowless_dbt.dtypes`, next to the type mapping that
        needs the same answer.
        """
        return sqlglot_dialect(adapter_type)


def _manifest_for(roots: Sequence[str]) -> Manifest | None:
    """The manifest for the project being checked, or None when there is none.

    Each root is tried before the working directory, so `rowless check
    analytics/` finds the dbt project inside `analytics/` rather than the one
    the shell happens to be sitting in.
    """
    for start in (*roots, "."):
        path = find_manifest(start)
        if path is None:
            continue
        try:
            return Manifest.load(path)
        except Exception:
            # An unreadable or unsupported manifest is reported by the source
            # when a pipeline actually needs it, with the version in the
            # message. Discovery is not the place to fail a whole run.
            continue
    return None


def _pipeline(manifest: Manifest, node: ManifestNode) -> Pipeline:
    """One discovered model."""
    inputs = tuple(
        (manifest.nodes[parent].name, f"dbt://{parent}")
        for parent in node.depends_on
        if parent in manifest.nodes
    )
    return Pipeline(
        target=node.unique_id,
        inputs=inputs,
        engine=SQL_ENGINE,
        framework="dbt",
        file=node.path,
        tags=_tags(manifest, node),
    )


def _tags(manifest: Manifest, node: ManifestNode) -> tuple[str, ...]:
    """Framework metadata, for `--select` filters and for `rowless explain`."""
    tags = ["dbt", f"dbt:{node.resource_type}", f"schema_source:{node.schema_source}"]
    if manifest.project_name:
        tags.append(f"package:{manifest.project_name}")
    if node.has_contract:
        tags.append("contract")
    return tuple(tags)

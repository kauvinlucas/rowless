"""dbt models as pipelines to check.

Each model is a node: its compiled SQL resolved against its upstream schemas.
This is the part that overlaps with existing tools, and the overlap is the
point. Once dbt models and Python pipelines are in one graph, `rowless diff`
produces the tree from the proposal, where a single rename lists both a Python
function and a SQL mart as affected.

Python models in dbt get the same treatment as any other Python pipeline, which
is the thing the dbt column-lineage tools list as unsupported.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import ClassVar

from rowless.discovery.base import Pipeline


class DbtModelDiscovery:
    """Find dbt models, SQL and Python alike."""

    name: ClassVar[str] = "dbt"
    requires: ClassVar[str] = ""

    def available(self) -> bool:
        """True when `dbt_project.yml` is present."""
        raise NotImplementedError("TODO(dbt)")

    def find(self, roots: Sequence[str]) -> Iterable[Pipeline]:
        """Yield one pipeline per model, with upstream refs as input URIs."""
        raise NotImplementedError("TODO(dbt)")

    def dialect_for(self, adapter_type: str) -> str:
        """Dbt adapter name to sqlglot dialect, e.g. `snowflake`, `bigquery`."""
        raise NotImplementedError("TODO(dbt)")

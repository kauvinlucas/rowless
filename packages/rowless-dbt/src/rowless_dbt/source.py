"""Schemas from a dbt manifest: `dbt://model.shop.stg_orders`.

The highest-value source in the project, because it is what connects the two
halves of the gap the proposal describes. A Python pipeline reading
`dbt://model.shop.stg_orders` gets the column that a SQL model declares, so
renaming it in SQL breaks the Python check in the same PR.
"""

from __future__ import annotations

from typing import ClassVar

from rowless.ir.schema import Schema
from rowless.sources.base import SourceRef


class DbtManifestSource:
    """Resolve `dbt://` URIs against `target/manifest.json`."""

    name: ClassVar[str] = "dbt"
    schemes: ClassVar[tuple[str, ...]] = ("dbt",)
    requires: ClassVar[str] = ""

    def available(self) -> bool:
        """True when a manifest is found at the configured path."""
        raise NotImplementedError("TODO(dbt)")

    def resolve(self, ref: SourceRef) -> Schema:
        """Prefer a declared contract, then the catalog, then compiled SQL.

        Falling back to resolving the compiled SQL through the sql adapter is
        what makes this work on projects with no contracts and no catalog,
        which is most projects.
        """
        raise NotImplementedError("TODO(dbt)")

    def fingerprint(self, ref: SourceRef) -> str:
        """`metadata.invocation_id` plus the manifest mtime."""
        raise NotImplementedError("TODO(dbt)")

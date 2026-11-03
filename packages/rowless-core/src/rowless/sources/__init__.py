"""Schema sources: metadata in, IR out.

Built in: Parquet footers, data contracts, warehouse snapshots, hand-written
JSON. Iceberg metadata lives in `rowless-iceberg` and dbt manifests in
`rowless-dbt`, because both drag in heavy dependencies.
"""

from rowless.sources.base import SchemaSource, SourceCache, SourceRef, resolve, source_for

__all__ = ["SchemaSource", "SourceCache", "SourceRef", "resolve", "source_for"]

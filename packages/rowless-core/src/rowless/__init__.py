"""rowless: a dry-run type checker for data pipelines.

Run your real pipeline code against frames that have a full schema and zero
rows, let each engine compute the output schema with its own semantics, and
report broken column references before any data moves.

    import rowless

    orders = rowless.source("lake://stg_orders")

See `docs/architecture.md` for how the pieces fit, and `rowless --help` for the
command line.
"""

from importlib.metadata import PackageNotFoundError, version

from rowless.api import assert_schema, pipeline, schema_of, source
from rowless.errors import (
    ColumnNotFound,
    DTypeMismatch,
    RowlessError,
    SchemaError,
    UnknownSchemaError,
)
from rowless.ir import Confidence, Coverage, DType, Field, Schema

try:
    __version__ = version("rowless")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"

__all__ = [
    "ColumnNotFound",
    "Confidence",
    "Coverage",
    "DType",
    "DTypeMismatch",
    "Field",
    "RowlessError",
    "Schema",
    "SchemaError",
    "UnknownSchemaError",
    "__version__",
    "assert_schema",
    "pipeline",
    "schema_of",
    "source",
]

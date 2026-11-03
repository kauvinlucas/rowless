"""The schema intermediate representation: dtypes, schemas, confidence, diffs.

Everything else in rowless speaks this vocabulary. Adapters translate their
engine into it, sources parse external metadata into it, the lineage graph
links it, and the CLI renders it.
"""

from rowless.ir.confidence import Confidence, Coverage
from rowless.ir.diff import Change, ChangeKind, SchemaDiff, Severity, diff_schemas
from rowless.ir.dtypes import DType, Kind, TimeUnit, unify
from rowless.ir.schema import Field, Origin, Relation, Schema

__all__ = [
    "Change",
    "ChangeKind",
    "Confidence",
    "Coverage",
    "DType",
    "Field",
    "Kind",
    "Origin",
    "Relation",
    "Schema",
    "SchemaDiff",
    "Severity",
    "TimeUnit",
    "diff_schemas",
    "unify",
]

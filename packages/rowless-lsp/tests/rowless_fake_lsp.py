"""A minimal in-process adapter, so the server can be tested with no engine.

The same reasoning as `rowless-core`'s `rowless_fake`: this package depends on
`rowless` and `pygls` and on no engine at all, which is what its CI cell
installs. But a language server with nothing to resolve has nothing to serve,
so the end-to-end tests need *an* adapter.

Deliberately not a copy of core's fake: this one carries only the operations
the fixture pipelines use, and `col` exists so a fixture can write a column
reference that looks like the engines' - `col("amount")` rather than a method
call - which is what the document index has to recognise.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from rowless.adapters.base import Capabilities, DryRunContext
from rowless.errors import UnknownSchemaError, UnsupportedDType
from rowless.ir.dtypes import DType, Kind, unknown
from rowless.ir.schema import Field, Relation, Schema


@dataclass(frozen=True)
class FakeExpr:
    """A column reference. `col("amount")` raises here if there is no column."""

    name: str


def col(name: str) -> FakeExpr:
    """A column expression, the way every engine spells one."""
    return FakeExpr(name)


@dataclass(frozen=True)
class FakeFrame:
    """A frame with a schema and no rows."""

    schema: Schema
    lineage: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def select(self, *names: str | FakeExpr) -> FakeFrame:
        """Project. Raises `ColumnNotFound` for a name that is not there."""
        wanted = tuple(name.name if isinstance(name, FakeExpr) else name for name in names)
        return FakeFrame(self.schema.select(wanted), dict(self.lineage))

    def rename(self, mapping: Mapping[str, str]) -> FakeFrame:
        """Rename, tracking lineage the way a plan-walking adapter would."""
        traced = {new: (old,) for old, new in mapping.items()}
        return FakeFrame(self.schema.rename(mapping), {**self.lineage, **traced})

    def with_columns(self, **columns: FakeExpr) -> FakeFrame:
        """Add a column per keyword, taking the dtype of the column it reads."""
        schema = self.schema
        lineage = dict(self.lineage)
        for name, expression in columns.items():
            source = schema[expression.name]
            schema = schema.with_fields(Field(name=name, dtype=source.dtype))
            lineage[name] = (expression.name,)
        return FakeFrame(schema, lineage)

    def drop(self, *names: str) -> FakeFrame:
        """Remove columns, erroring on names that are not present."""
        return FakeFrame(self.schema.drop(tuple(names)), dict(self.lineage))


class FakeAdapter:
    """The adapter for `FakeFrame`. Implements the full required contract."""

    name: ClassVar[str] = "fake"
    requires: ClassVar[str] = ""
    capabilities: ClassVar[Capabilities] = Capabilities(column_lineage=True, lazy=True)

    def available(self) -> bool:
        """Always: the engine is this module."""
        return True

    def handles(self, obj: object) -> bool:
        """True for `FakeFrame`."""
        return isinstance(obj, FakeFrame)

    def materialize(self, schema: Schema, ctx: DryRunContext) -> Any:
        """A frame with the declared schema and no rows."""
        _ = ctx
        return FakeFrame(schema)

    def describe(self, obj: object) -> Relation:
        """Read the schema back, refusing anything that is not a frame."""
        if not isinstance(obj, FakeFrame):
            raise UnknownSchemaError(f"fake adapter cannot describe {type(obj).__name__}")
        return Relation(schema=obj.schema, engine=self.name)

    def to_ir(self, native: Any) -> DType:
        """The fake engine's dtypes *are* IR dtypes; anything else is UNKNOWN."""
        if isinstance(native, DType):
            return native
        return unknown(f"fake adapter has no mapping for {type(native).__name__}")

    def from_ir(self, dtype: DType) -> Any:
        """Identity, except that UNKNOWN cannot be lowered."""
        if dtype.kind is Kind.UNKNOWN:
            raise UnsupportedDType("the fake engine has no unknown type")
        return dtype

    def column_lineage(self, obj: object) -> Mapping[str, tuple[str, ...]]:
        """Whatever the frame recorded as it was built."""
        if not isinstance(obj, FakeFrame):
            return {}
        return obj.lineage


def install() -> FakeAdapter:
    """Register the fake adapter for this process and return it."""
    from rowless.adapters.registry import register

    adapter = FakeAdapter()
    register("rowless.engines", adapter.name, adapter)
    return adapter

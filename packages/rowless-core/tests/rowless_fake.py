"""A minimal in-process adapter, so core can be tested with no engine installed.

Core tests run in an environment with no Polars, no pandas and no DuckDB, on
purpose: it is the environment a first-time contributor has, and it keeps core
honest about never importing an engine. But the runtime, the registry and the
conformance suite all need *an* adapter to talk to.

The module is named `rowless_fake` to imitate a real adapter distribution
(`rowless_polars`, `rowless_duckdb`): `rowless.runtime.rowdep` walks out of
adapter frames to find the user's line, and a test double that is not named
like an adapter would not exercise that.

`FakeFrame` is a real object with real methods, not a mock. It resolves schemas
the way a lazy engine does (by applying the operation to the schema), and it
routes its row-dependent accessors through `rowless.runtime.rowdep`, which is
exactly what a true adapter has to do.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from rowless.adapters.base import Capabilities, DryRunContext
from rowless.errors import ColumnNotFound, UnknownSchemaError, UnsupportedDType
from rowless.ir.confidence import Confidence
from rowless.ir.dtypes import DType, Kind, int_, string, unknown
from rowless.ir.schema import Field, Relation, Schema
from rowless.runtime.rowdep import observe
from rowless.testing.conformance import ConformanceCase


@dataclass(frozen=True)
class FakeFrame:
    """A frame with a schema and no rows, with just enough engine surface."""

    schema: Schema
    lineage: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def select(self, *names: str) -> FakeFrame:
        """Project. Raises `ColumnNotFound` for a name that is not there."""
        return FakeFrame(self.schema.select(tuple(names)), dict(self.lineage))

    def rename(self, mapping: Mapping[str, str]) -> FakeFrame:
        """Rename, tracking lineage the way a plan-walking adapter would."""
        traced = {new: (old,) for old, new in mapping.items()}
        return FakeFrame(self.schema.rename(mapping), {**self.lineage, **traced})

    def derive(self, name: str, dtype: DType, *, sources: Sequence[str] = ()) -> FakeFrame:
        """Add a computed column, with its inputs recorded."""
        for source in sources:
            _ = self.schema[source]
        return FakeFrame(
            self.schema.with_fields(Field(name=name, dtype=dtype)),
            {**self.lineage, name: tuple(sources)},
        )

    def is_empty(self) -> bool:
        """Row-dependent: always False here, and always reported as such."""
        observe("fake.FakeFrame.is_empty", returned=False)
        return False

    def item(self) -> Any:
        """Row-dependent: there is no first row to read.

        Returns None, the way an engine does on an empty frame, which is the
        value that makes downstream logic take the wrong branch.
        """
        observe("fake.FakeFrame.item", returned=None)
        return None


class FakeAdapter:
    """The adapter for `FakeFrame`. Implements the full required contract."""

    name: ClassVar[str] = "fake"
    requires: ClassVar[str] = ""
    capabilities: ClassVar[Capabilities] = Capabilities(
        column_lineage=True,
        lazy=True,
        nested_types=True,
    )

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
        fields = tuple(
            Field(
                name=column.name,
                dtype=column.dtype,
                nullable=column.nullable,
                origin=column.origin,
                derived_from=obj.lineage.get(column.name, column.derived_from),
            )
            for column in obj.schema.fields
        )
        return Relation(schema=Schema(fields=fields, name=obj.schema.name), engine=self.name)

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

    def conformance_cases(self) -> Sequence[ConformanceCase]:
        """The operations battery for this engine, as every adapter supplies."""
        from rowless.testing import fixtures

        orders = {"orders": fixtures.ORDERS}
        return (
            ConformanceCase(
                name="select",
                pipeline=lambda orders: orders.select("order_id", "amount"),
                inputs=orders,
                expect=Schema.from_mapping({"order_id": int_(64), "amount": int_(64)}),
            ),
            ConformanceCase(
                name="rename",
                pipeline=lambda orders: orders.rename({"amount": "amount_cents"}),
                inputs=orders,
                expect=Schema.from_mapping(
                    {
                        "order_id": int_(64),
                        "status": string(),
                        "amount_cents": int_(64),
                        "created_at": fixtures.ORDERS["created_at"].dtype,
                    }
                ),
            ),
            ConformanceCase(
                name="missing_column",
                pipeline=lambda orders: orders.select("amount_cents"),
                inputs=orders,
                raises=ColumnNotFound,
                reason="the schema knows the column set, so the adapter never sees the engine fail",
            ),
            ConformanceCase(
                name="row_dependent_branch",
                pipeline=lambda orders: (
                    orders.select("order_id") if not orders.is_empty() else orders
                ),
                inputs=orders,
                expect_confidence=Confidence.UNKNOWN,
                reason="zero rows takes the branch production would not",
            ),
        )


def install() -> FakeAdapter:
    """Register the fake adapter for this process and return it."""
    from rowless.adapters.registry import register

    adapter = FakeAdapter()
    register("rowless.engines", adapter.name, adapter)
    return adapter

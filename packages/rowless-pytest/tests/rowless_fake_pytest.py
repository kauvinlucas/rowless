"""A minimal in-process adapter, so the plugin can be tested with no engine.

The same reasoning as `rowless-core`'s `rowless_fake` and `rowless-lsp`'s
`rowless_fake_lsp`: this distribution depends on `rowless` and `pytest` and on
no engine at all, which is what its CI cell installs - but a plugin whose
whole job is to resolve a schema needs *an* adapter to resolve one with.

Carries only what the fixture pipelines below use, plus the two things the
plugin's rules are about: `is_empty`, which goes through
`rowless.runtime.rowdep` and is what makes a result unverified, and `derive`,
which can add a column whose dtype is ANY - the case a contract test has to
fail even though the declared dtype "matches".

`FakeUnavailable` is the other half of the skip story: an adapter that is
registered and whose engine is not installed, which is what a user with
`rowless-polars` and no Polars has.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from rowless.adapters.base import Capabilities, DryRunContext
from rowless.errors import UnknownSchemaError, UnsupportedDType
from rowless.ir.dtypes import DType, Kind, unknown
from rowless.ir.schema import Field, Relation, Schema
from rowless.runtime.rowdep import observe

ENGINE = "fake"
"""Name the fake adapter registers under."""

UNAVAILABLE_ENGINE = "fake_absent"
"""Name of the adapter that is installed but whose engine is not."""


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


class FakeAdapter:
    """The adapter for `FakeFrame`. Implements the full required contract."""

    name: ClassVar[str] = ENGINE
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


class FakeUnavailable(FakeAdapter):
    """A registered adapter whose engine library is not installed.

    What the collection hook skips on, and the reason it reports the
    registry's own message: "install rowless-fake[engine]" is the only part of
    a skip reason a user can act on.
    """

    name: ClassVar[str] = UNAVAILABLE_ENGINE
    requires: ClassVar[str] = "fake-engine>=1.0"

    def available(self) -> bool:
        """Never. The point of this one is to be inactive."""
        return False


def install() -> FakeAdapter:
    """Register both fake adapters for this process, and return the usable one."""
    from rowless.adapters.registry import register

    adapter = FakeAdapter()
    register("rowless.engines", adapter.name, adapter)
    register("rowless.engines", UNAVAILABLE_ENGINE, FakeUnavailable())
    return adapter

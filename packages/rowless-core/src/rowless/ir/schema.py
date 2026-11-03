"""Schemas: ordered, named columns plus where each one came from."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace

from rowless.ir.dtypes import DType


@dataclass(frozen=True, slots=True)
class Origin:
    """Where a column was defined, for diagnostics and lineage.

    `rowless diff` is only useful because every column can be traced back to a
    file and line, across both Python and SQL.
    """

    kind: str
    """One of: source, python, sql, contract, synthetic."""

    ref: str
    """Table URI, module path, dbt model name or file path."""

    file: str | None = None
    line: int | None = None
    column: int | None = None

    def __str__(self) -> str:
        """`file:line` when known, else the ref: what a terminal line shows."""
        if self.file is None:
            return self.ref
        if self.line is None:
            return self.file
        return f"{self.file}:{self.line}"


@dataclass(frozen=True, slots=True)
class Field:
    """One column."""

    name: str
    dtype: DType
    nullable: bool = True
    origin: Origin | None = None
    description: str | None = None

    #: Columns this one was computed from. Populated by adapters that can report
    #: it (Polars plan walking, sqlglot qualification) and left empty by the
    #: rest, which is what drops the coverage score.
    derived_from: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Schema(Mapping[str, Field]):
    """An ordered, name-addressed set of fields.

    Immutable: every operation returns a new schema, so a dry run can keep the
    input schema around to diff against the output.
    """

    fields: tuple[Field, ...] = ()
    name: str | None = None

    # --- Mapping protocol ---

    def __getitem__(self, key: str) -> Field:
        """Look a column up by name, raising `ColumnNotFound` with a near miss.

        A linear scan: schemas are tens of columns, not millions, and an index
        would have to be cached on a frozen dataclass to be worth anything.
        """
        for existing in self.fields:
            if existing.name == key:
                return existing
        # Imported here, not at module scope: `rowless.errors` reaches the
        # diagnostic codes, which reach back into the IR.
        from rowless.errors import ColumnNotFound

        raise ColumnNotFound(key, available=tuple(f.name for f in self.fields), relation=self.name)

    def __iter__(self) -> Iterator[str]:
        """Column names, in schema order."""
        return (existing.name for existing in self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    # --- Construction ---

    @classmethod
    def from_mapping(cls, columns: Mapping[str, DType], *, name: str | None = None) -> Schema:
        """Build from a plain name-to-dtype mapping. The quick path for tests."""
        return cls(
            fields=tuple(Field(name=column, dtype=dtype) for column, dtype in columns.items()),
            name=name,
        )

    @classmethod
    def empty(cls, *, name: str | None = None) -> Schema:
        """Zero columns. Legal, and distinct from an unknown schema."""
        return cls(fields=(), name=name)

    # --- Derivation ---

    def select(self, names: tuple[str, ...]) -> Schema:
        """Project, preserving the order given."""
        return replace(self, fields=tuple(self[column] for column in names))

    def drop(self, names: tuple[str, ...]) -> Schema:
        """Remove columns, erroring on names that are not present."""
        for column in names:
            _ = self[column]
        dropped = set(names)
        return replace(self, fields=tuple(f for f in self.fields if f.name not in dropped))

    def rename(self, mapping: Mapping[str, str]) -> Schema:
        """Rename in place, keeping position."""
        for column in mapping:
            _ = self[column]
        return replace(
            self,
            fields=tuple(
                replace(f, name=mapping[f.name]) if f.name in mapping else f for f in self.fields
            ),
        )

    def with_fields(self, *fields: Field) -> Schema:
        """Append, or replace by name."""
        merged = list(self.fields)
        positions = {f.name: index for index, f in enumerate(merged)}
        for incoming in fields:
            if incoming.name in positions:
                merged[positions[incoming.name]] = incoming
            else:
                positions[incoming.name] = len(merged)
                merged.append(incoming)
        return replace(self, fields=tuple(merged))

    def relabel(self, name: str) -> Schema:
        """Copy under a new schema name."""
        return replace(self, name=name)


@dataclass(frozen=True, slots=True)
class Relation:
    """A schema plus the metadata rowless tracks alongside it.

    Adapters return this, not a bare `Schema`, because the interesting part of
    a dry run is often what the engine *could not* tell us.
    """

    schema: Schema
    engine: str
    """Adapter name that produced it: polars, pandas, duckdb, sql, pyspark, ibis."""

    origin: Origin | None = None

    sort_keys: tuple[str, ...] = ()
    """Engines that report ordering fill this in; used to flag order-dependent logic."""

    notes: tuple[str, ...] = ()
    """Adapter commentary surfaced by `rowless explain`."""

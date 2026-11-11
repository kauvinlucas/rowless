"""The Iceberg-to-IR type mapping, read through real table metadata.

The table below is the specification for this adapter's half of the IR: every
primitive and nested type the Iceberg spec defines, mapped once. It is asserted
against a committed metadata file rather than against hand-built PyIceberg
objects, because what has to be true is that the mapping survives the format -
a type spelled one way in the JSON and another way in Python is exactly the
kind of gap that produces a schema nobody can reproduce.
"""

from __future__ import annotations

from pathlib import Path

import iceberg_tables
import pytest
from rowless.ir.dtypes import (
    DType,
    Kind,
    TimeUnit,
    any_,
    binary,
    boolean,
    date,
    datetime,
    decimal,
    float_,
    int_,
    list_,
    map_,
    null,
    string,
    struct,
    time,
)
from rowless.sources.base import resolve
from rowless_iceberg.dtypes import dtype_from_iceberg

#: Column name in `types_v3.metadata.json` to the IR dtype it must resolve to.
EXPECTED: dict[str, DType] = {
    "flag": boolean(),
    # Iceberg has two integer widths and no unsigned type.
    "small": int_(32),
    "big": int_(64),
    "single": float_(32),
    "wide": float_(64),
    "money": decimal(18, 2),
    "day": date(),
    # One time type in the spec, and it is microseconds.
    "clock": time(TimeUnit.MICROSECOND),
    "naive": datetime(TimeUnit.MICROSECOND),
    # `timestamptz` is a UTC instant by definition, not a normalisation rowless
    # chose: this is the distinction that breaks pipelines.
    "aware": datetime(TimeUnit.MICROSECOND, "UTC"),
    "naive_nanos": datetime(TimeUnit.NANOSECOND),
    "aware_nanos": datetime(TimeUnit.NANOSECOND, "UTC"),
    "label": string(),
    "blob": binary(),
    # The length is dropped so this agrees with the Parquet footer for the same
    # file, and no Iceberg promotion can change it.
    "digest": binary(),
    # No IR equivalent: ANY lowers the coverage score instead of claiming a
    # type, which `string` would, or refusing forever, which UNKNOWN would.
    "key": any_(),
    # v3 `unknown` is a column of nulls, which is the IR's NULL exactly, and
    # the opposite of the IR's UNKNOWN.
    "nothing": null(),
    "shape": any_(),
    "region": any_(),
    "tags": list_(string()),
    "counters": map_(string(), int_(64)),
    "address": struct({"street": string(), "zip": int_(32)}),
    "lines": list_(struct({"sku": string(), "qty": int_(32)})),
}


def _corpus(tmp_path: Path) -> dict[str, DType]:
    """Every column of the type corpus, resolved through the source."""
    schema = resolve(iceberg_tables.uri(tmp_path, iceberg_tables.TYPES, table="corpus.every_type"))
    return {field.name: field.dtype for field in schema.fields}


@pytest.mark.parametrize("column", sorted(EXPECTED))
def test_every_iceberg_type_maps_to_one_ir_dtype(column: str, tmp_path: Path) -> None:
    """One parameter per column, so a failure names the type that moved."""
    assert _corpus(tmp_path)[column] == EXPECTED[column]


def test_the_corpus_covers_the_whole_fixture(tmp_path: Path) -> None:
    """A type added to the fixture has to be given an expectation.

    Without this the fixture can grow a column that nothing asserts on, which
    is how an unmapped type ships as UNKNOWN and nobody notices.
    """
    assert sorted(_corpus(tmp_path)) == sorted(EXPECTED)


def test_an_opaque_type_says_which_one_it_was(tmp_path: Path) -> None:
    """ANY carries its reason, which is what `rowless explain` prints."""
    resolved = _corpus(tmp_path)
    assert resolved["key"].kind is Kind.ANY
    assert "uuid" in (resolved["key"].reason or "")
    assert "geometry" in (resolved["shape"].reason or "")


def test_nullability_comes_from_the_required_flag(tmp_path: Path) -> None:
    """Iceberg `required` is the IR's `nullable`, inverted."""
    schema = resolve(iceberg_tables.uri(tmp_path, iceberg_tables.TYPES, table="corpus.every_type"))
    assert schema["flag"].nullable is False
    assert schema["label"].nullable is True


class _FutureType:
    """A primitive this PyIceberg does not have. Named by its spec spelling."""

    def __init__(self, spelling: str) -> None:
        self._spelling = spelling

    def __str__(self) -> str:
        return self._spelling


def test_an_unmapped_type_is_unknown_with_its_spelling() -> None:
    """A spec addition rowless has not mapped must poison, not pass.

    The mapping dispatches on the spelling rather than on a class, so a
    primitive added to a later Iceberg version is reported by name instead of
    failing to import. That is only worth anything if what comes back is
    UNKNOWN: `matches` refuses UNKNOWN against itself, so a diff over this
    column reads UNVERIFIED and fails the gate.
    """
    dtype = dtype_from_iceberg(_FutureType("variant"))

    assert dtype.kind is Kind.UNKNOWN
    assert "variant" in (dtype.reason or "")
    assert dtype.matches(dtype) is False


def test_a_decimal_wider_than_its_precision_is_refused() -> None:
    """An impossible decimal is UNKNOWN rather than a silently clamped one."""
    dtype = dtype_from_iceberg(_FutureType("decimal(2, 5)"))

    assert dtype.kind is Kind.UNKNOWN
    assert "scale larger than its precision" in (dtype.reason or "")


def test_a_parameterised_opaque_type_still_matches_on_its_family() -> None:
    """`geometry(OGC:CRS84)` is a geometry, CRS and all."""
    dtype = dtype_from_iceberg(_FutureType("geography(OGC:CRS84, spherical)"))

    assert dtype.kind is Kind.ANY
    assert "geography" in (dtype.reason or "")

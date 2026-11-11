"""Warehouse type strings to the IR, including the judgement calls."""

from __future__ import annotations

import pytest
from rowless.ir.dtypes import (
    DType,
    Kind,
    TimeUnit,
    binary,
    boolean,
    date,
    datetime,
    decimal,
    duration,
    enum,
    float_,
    int_,
    list_,
    map_,
    string,
    struct,
    time,
    uint,
)
from rowless_dbt.dtypes import dtype_from_sql, sqlglot_dialect


@pytest.mark.parametrize(
    ("spelling", "dialect", "expected"),
    [
        # Strings, however the warehouse spells them.
        ("varchar(100)", "postgres", string()),
        ("character varying", "postgres", string()),
        ("text", "", string()),
        ("string", "spark", string()),
        # Integers keep their declared width.
        ("smallint", "", int_(16)),
        ("int", "", int_(32)),
        ("bigint", "", int_(64)),
        ("int64", "bigquery", int_(64)),
        ("hugeint", "duckdb", int_(128)),
        ("int unsigned", "mysql", uint(32)),
        # Floats, after the dialect has had its say.
        ("float4", "postgres", float_(32)),
        ("double precision", "postgres", float_(64)),
        ("float", "snowflake", float_(64)),
        ("float64", "bigquery", float_(64)),
        # Decimals, including the dialect default sqlglot fills in.
        ("numeric(18,2)", "postgres", decimal(18, 2)),
        ("decimal(18)", "", decimal(18, 0)),
        ("number", "snowflake", decimal(38, 0)),
        # Temporal, where naive against aware is the distinction that matters.
        ("date", "", date()),
        ("time", "", time(TimeUnit.MICROSECOND)),
        ("timestamp", "postgres", datetime(TimeUnit.MICROSECOND)),
        ("timestamp without time zone", "postgres", datetime(TimeUnit.MICROSECOND)),
        ("timestamp with time zone", "postgres", datetime(TimeUnit.MICROSECOND, "UTC")),
        ("timestamptz", "postgres", datetime(TimeUnit.MICROSECOND, "UTC")),
        ("timestamp_ntz", "snowflake", datetime(TimeUnit.MICROSECOND)),
        ("timestamp_tz", "snowflake", datetime(TimeUnit.MICROSECOND, "UTC")),
        # BigQuery TIMESTAMP is UTC and DATETIME is naive, which is the kind of
        # thing this adapter must not get backwards.
        ("timestamp", "bigquery", datetime(TimeUnit.MICROSECOND, "UTC")),
        ("datetime", "bigquery", datetime(TimeUnit.MICROSECOND)),
        # A declared precision becomes a time unit.
        ("timestamp(0)", "", datetime(TimeUnit.SECOND)),
        ("timestamp(3)", "", datetime(TimeUnit.MILLISECOND)),
        ("timestamp(6)", "", datetime(TimeUnit.MICROSECOND)),
        ("timestamp(9)", "", datetime(TimeUnit.NANOSECOND)),
        ("timestamp_ms", "duckdb", datetime(TimeUnit.MILLISECOND)),
        ("interval", "postgres", duration(TimeUnit.MICROSECOND)),
        # The rest.
        ("boolean", "", boolean()),
        ("bytea", "postgres", binary()),
        ("array<int64>", "bigquery", list_(int_(64))),
        ("int[]", "postgres", list_(int_(32))),
        ("struct<a int64, b string>", "bigquery", struct({"a": int_(64), "b": string()})),
        ("map<string, int>", "spark", map_(string(), int_(32))),
        ("enum('paid','refunded')", "mysql", enum(("paid", "refunded"))),
    ],
)
def test_sql_types_map_onto_the_ir(spelling: str, dialect: str, expected: DType) -> None:
    assert dtype_from_sql(spelling, dialect=dialect) == expected


@pytest.mark.parametrize(
    ("spelling", "dialect"),
    [
        ("json", ""),
        ("jsonb", "postgres"),
        ("variant", "snowflake"),
        ("super", "redshift"),
        ("uuid", "postgres"),
        ("geography", "bigquery"),
        ("int4range", "postgres"),
        # A zone-aware time has no IR equivalent, and dropping the zone
        # silently is the bug the IR models timezones to avoid.
        ("timetz", "postgres"),
    ],
)
def test_types_with_no_ir_equivalent_are_any(spelling: str, dialect: str) -> None:
    """ANY lowers coverage and warns; it never fails a check or passes one."""
    dtype = dtype_from_sql(spelling, dialect=dialect)

    assert dtype.is_any
    assert dtype.reason is not None and spelling in dtype.reason


@pytest.mark.parametrize(
    ("spelling", "fragment"),
    [
        (None, "declares no data_type"),
        ("", "declares no data_type"),
        ("not a type at all", "could not parse"),
        ("decimal", "declares no precision"),
        ("numeric", "declares no precision"),
        ("bigdecimal", "no IR dtype"),
        ("array", "does not say what it holds"),
        ("map", "does not say what it maps"),
        ("struct", "does not say what fields"),
    ],
)
def test_types_rowless_cannot_read_are_unknown(spelling: str | None, fragment: str) -> None:
    """UNKNOWN poisons comparisons instead of passing them, and says why."""
    dtype = dtype_from_sql(spelling)

    assert dtype.kind is Kind.UNKNOWN
    assert dtype.reason is not None and fragment in dtype.reason


def test_an_enum_with_no_variants_widens_to_a_string() -> None:
    """The IR accepts an enum into a string slot, so this loses nothing."""
    assert dtype_from_sql("enum", dialect="mysql") == string()


def test_a_wrong_dialect_still_parses_standard_sql() -> None:
    """A dialect rowless was told wrongly must not take every column with it."""
    assert dtype_from_sql("varchar(10)", dialect="bigquery") == string()
    assert dtype_from_sql("bigint", dialect="not_a_dialect") == int_(64)


@pytest.mark.parametrize(
    ("adapter", "dialect"),
    [
        ("postgres", "postgres"),
        ("snowflake", "snowflake"),
        ("bigquery", "bigquery"),
        ("databricks", "databricks"),
        ("duckdb", "duckdb"),
        ("Redshift", "redshift"),
        ("sqlserver", "tsql"),
        ("synapse", "tsql"),
        ("glue", "spark"),
        # Not a dialect sqlglot knows: empty, so the default is used rather
        # than a guess that turns every query into a parse error.
        ("vertica", ""),
        ("", ""),
    ],
)
def test_dbt_adapters_map_to_sqlglot_dialects(adapter: str, dialect: str) -> None:
    assert sqlglot_dialect(adapter) == dialect

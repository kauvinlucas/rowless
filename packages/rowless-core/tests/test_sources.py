"""Schema sources: parsing, resolution and caching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rowless.errors import BadSourceRef, SchemaNotFound
from rowless.ir.dtypes import TimeUnit, datetime, int_, string
from rowless.ir.schema import Schema
from rowless.sources.base import SourceCache, SourceRef, resolve, source_for
from rowless.sources.information_schema import sql_type_to_ir
from rowless.sources.literal import INLINE_SCHEMAS, JsonSchemaSource, schema_from_mapping


def test_parses_a_uri_into_scheme_and_path() -> None:
    """`lake://stg_orders` and a bare path both have to work."""
    ref = SourceRef.parse("lake://stg_orders")
    assert (ref.scheme, ref.path, ref.table_name) == ("lake", "stg_orders", "stg_orders")

    bare = SourceRef.parse("data/orders.parquet")
    assert (bare.scheme, bare.table_name) == ("", "orders.parquet")

    dotted = SourceRef.parse("dbt://model.shop.stg_orders")
    assert dotted.table_name == "stg_orders"

    with_options = SourceRef.parse("warehouse://db.public.orders?strict=1")
    assert with_options.options == {"strict": "1"}
    assert with_options.uri == "warehouse://db.public.orders"


def test_rejects_an_unusable_uri() -> None:
    """A URI nobody can resolve fails at parse time with the reason."""
    with pytest.raises(BadSourceRef):
        SourceRef.parse("")
    with pytest.raises(BadSourceRef):
        SourceRef.parse("://orders")
    with pytest.raises(BadSourceRef):
        SourceRef.parse("lake://")


def test_unregistered_scheme_lists_what_is_registered() -> None:
    """An unknown scheme alone does not tell a user what to install."""
    with pytest.raises(BadSourceRef) as caught:
        source_for(SourceRef.parse("nosuchthing://orders"))

    assert "registered schemes" in str(caught.value)


def test_json_source_reads_both_accepted_shapes(tmp_path: Path) -> None:
    """A hand-written mapping and a generated schema file are the same thing."""
    plain = tmp_path / "orders.json"
    plain.write_text('{"order_id": "int64", "created_at": "datetime[us, UTC]"}', encoding="utf-8")

    schema = resolve(f"json://{plain.as_posix()}")

    assert tuple(schema) == ("order_id", "created_at")
    assert schema["created_at"].dtype == datetime(TimeUnit.MICROSECOND, "UTC")
    assert schema["order_id"].origin is not None
    assert schema.name == "orders.json"

    structured = tmp_path / "full.json"
    structured.write_text(
        json.dumps(
            {
                "name": "stg_orders",
                "fields": [
                    {"name": "order_id", "dtype": "int64", "nullable": False},
                    {"name": "amount", "dtype": "decimal(18,2)"},
                ],
            }
        ),
        encoding="utf-8",
    )

    full = resolve(f"json://{structured.as_posix()}")
    assert full.name == "stg_orders"
    assert full["order_id"].nullable is False


def test_inline_source_names_what_is_defined() -> None:
    """The no-dependency path: three schemas in config and a working check."""
    INLINE_SCHEMAS["orders"] = schema_from_mapping({"order_id": "int64"}, name="orders")
    try:
        assert tuple(resolve("inline://orders")) == ("order_id",)
        with pytest.raises(SchemaNotFound, match="orders"):
            resolve("inline://nope")
    finally:
        INLINE_SCHEMAS.clear()


def test_a_bad_dtype_spelling_names_the_column(tmp_path: Path) -> None:
    """The error has to say which column, or a 200-column file is unfixable."""
    path = tmp_path / "orders.json"
    path.write_text('{"order_id": "int64", "amount": "biginteger"}', encoding="utf-8")

    with pytest.raises(SchemaNotFound, match="amount"):
        resolve(f"json://{path.as_posix()}")


def test_cache_round_trips_and_invalidates(tmp_path: Path) -> None:
    """CI and the editor have to agree about what a table looks like."""
    cache = SourceCache(str(tmp_path / ".rowless"))
    ref = SourceRef.parse("lake://stg_orders")
    schema = Schema.from_mapping({"order_id": int_(64), "status": string()}, name="stg_orders")

    assert cache.get(ref, "v1") is None

    cache.put(ref, "v1", schema)

    assert cache.get(ref, "v1") == schema
    assert cache.get(ref, "v2") is None, "a changed fingerprint is a miss"


def test_resolve_reads_the_cache_before_the_source(tmp_path: Path) -> None:
    """A re-check costs one fingerprint check, not another metadata read.

    Proved by seeding the cache with a schema the file does not contain: if
    the cached entry comes back, the source was never consulted.
    """
    path = tmp_path / "orders.json"
    path.write_text('{"order_id": "int64"}', encoding="utf-8")
    cache = SourceCache(str(tmp_path / ".rowless"))
    uri = f"json://{path.as_posix()}"
    ref = SourceRef.parse(uri)

    resolve(uri, cache=cache)
    fingerprint = JsonSchemaSource().fingerprint(ref)
    cache.put(ref, fingerprint, Schema.from_mapping({"from_cache": int_(64)}))

    assert tuple(resolve(uri, cache=cache)) == ("from_cache",)

    path.write_text('{"order_id": "int64", "amount": "int64"}', encoding="utf-8")
    assert tuple(resolve(uri, cache=cache)) == ("order_id", "amount"), (
        "a changed file must invalidate the entry"
    )


def test_empty_schema_from_a_source_is_an_error(tmp_path: Path) -> None:
    """A source that resolves to nothing is reported, not treated as empty."""
    path = tmp_path / "orders.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(SchemaNotFound, match="no columns"):
        resolve(f"json://{path.as_posix()}")


def test_sql_types_map_without_guessing() -> None:
    """Warehouse spellings are a lookup table, and misses are UNKNOWN."""
    assert sql_type_to_ir("bigint") == int_(64)
    assert sql_type_to_ir("character varying") == string()
    assert sql_type_to_ir("numeric", precision=18, scale=2).scale == 2
    assert sql_type_to_ir("timestamp with time zone", datetime_precision=6).tz == "UTC"
    assert sql_type_to_ir("timestamp", datetime_precision=3).unit is TimeUnit.MILLISECOND
    assert not sql_type_to_ir("geography").is_known
    assert "geography" in str(sql_type_to_ir("geography").reason)

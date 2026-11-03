"""Shared schemas and pipelines used by the conformance suite and core tests.

Kept in one place so an adapter bug and a core bug are reproduced against
identical inputs, and so "the orders schema" means the same thing in every
test file in the workspace.
"""

from __future__ import annotations

from rowless.ir.dtypes import (
    DType,
    TimeUnit,
    boolean,
    datetime,
    decimal,
    enum,
    int_,
    list_,
    map_,
    null,
    string,
    struct,
)
from rowless.ir.schema import Field, Schema

#: The schema from the proposal's example. Every adapter's smoke test uses it.
ORDERS: Schema = Schema(
    fields=(
        Field(name="order_id", dtype=int_(64), nullable=False),
        Field(name="status", dtype=string()),
        Field(name="amount", dtype=int_(64)),
        Field(name="created_at", dtype=datetime(TimeUnit.MICROSECOND, "UTC")),
    ),
    name="stg_orders",
)

#: Join partner for the join battery, with a deliberate name collision on
#: `created_at` so post-join ambiguity gets exercised everywhere.
CUSTOMERS: Schema = Schema(
    fields=(
        Field(name="customer_id", dtype=int_(64), nullable=False),
        Field(name="country", dtype=string()),
        Field(name="created_at", dtype=datetime(TimeUnit.MICROSECOND, "UTC")),
    ),
    name="stg_customers",
)

#: Nested types, skipped by adapters that declare `nested_types=False`.
EVENTS: Schema = Schema(
    fields=(
        Field(name="event_id", dtype=int_(64), nullable=False),
        Field(
            name="payload",
            dtype=struct({"source": string(), "retries": int_(32)}),
        ),
        Field(name="tags", dtype=list_(string())),
        Field(name="attrs", dtype=map_(string(), string())),
    ),
    name="stg_events",
)

#: Dtypes that drift on empty frames in at least one engine. The regression
#: corpus for the pandas adapter, and useful pressure on the others.
DRIFT_PRONE: Schema = Schema(
    fields=(
        Field(name="all_null", dtype=null()),
        Field(name="nullable_bool", dtype=boolean(), nullable=True),
        Field(name="category", dtype=enum(("a", "b", "c"))),
        Field(name="naive_ts", dtype=datetime(TimeUnit.MICROSECOND)),
        Field(name="money", dtype=decimal(18, 2)),
    ),
    name="drift_prone",
)

#: Every schema the conformance suite materializes, with the capability that
#: gates it. Adapters that cannot do nested types skip those rather than fail.
SCHEMAS: tuple[tuple[Schema, str | None], ...] = (
    (ORDERS, None),
    (CUSTOMERS, None),
    (EVENTS, "nested_types"),
    (DRIFT_PRONE, None),
)


def round_trip_corpus() -> tuple[DType, ...]:
    """Every dtype an adapter is asked to round-trip, before capability filtering.

    Built from the fixture schemas plus the parameter combinations that are
    easy to lose in translation: a timezone, a decimal scale, a non-default
    time unit, and the two widths either side of the common default.
    """
    from rowless.ir.dtypes import binary, date, duration, float_, time, uint

    scalars = (
        boolean(),
        int_(8),
        int_(16),
        int_(32),
        int_(64),
        uint(8),
        uint(32),
        float_(32),
        float_(64),
        decimal(18, 2),
        decimal(38, 9),
        string(),
        binary(),
        date(),
        time(TimeUnit.MICROSECOND),
        datetime(TimeUnit.MILLISECOND),
        datetime(TimeUnit.MICROSECOND, "UTC"),
        datetime(TimeUnit.NANOSECOND, "Europe/London"),
        duration(TimeUnit.MICROSECOND),
    )
    nested = (
        list_(int_(64)),
        list_(string()),
        struct({"source": string(), "retries": int_(32)}),
        map_(string(), int_(64)),
        enum(("a", "b", "c")),
    )
    return scalars + nested

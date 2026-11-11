"""Graphs the emitter tests assert against.

One shape, built once, so a naming test and an event test are talking about the
same pipeline. It is the proposal's example: two sources, a Polars pipeline
that renames and aggregates, and a table it writes - plus, deliberately, one
column whose only evidence is a matching name, because what the emitter does
with weak evidence is the thing most worth testing.
"""

from __future__ import annotations

from rowless.ir.confidence import Confidence, Coverage
from rowless.ir.dtypes import TimeUnit, datetime, int_, string, struct
from rowless.ir.schema import Field, Origin, Schema
from rowless.lineage.graph import (
    ColumnRef,
    Edge,
    Evidence,
    LineageGraph,
    Node,
    NodeKind,
)

ORDERS_URI = "lake://stg_orders"
CUSTOMERS_URI = "s3://warehouse/customers/part-0.parquet"
PIPELINE_TARGET = "pipelines.revenue:daily_revenue"
SINK_URI = "s3://marts/revenue/"

ORDERS = Schema(
    fields=(
        Field(name="order_id", dtype=int_(64), nullable=False),
        Field(name="status", dtype=string()),
        Field(name="amount", dtype=int_(64)),
        Field(name="created_at", dtype=datetime(TimeUnit.MICROSECOND, "UTC")),
    ),
    name="stg_orders",
)

CUSTOMERS = Schema(
    fields=(
        Field(name="customer_id", dtype=int_(64), nullable=False),
        Field(name="country", dtype=string(), description="ISO 3166-1 alpha-2."),
        Field(
            name="address",
            dtype=struct({"city": string(), "postcode": string()}),
        ),
    ),
    name="stg_customers",
)

#: The pipeline output. `status` carries no `derived_from` on purpose: it is the
#: column that only a name match connects to anything.
REVENUE = Schema(
    fields=(
        Field(
            name="day",
            dtype=datetime(TimeUnit.MICROSECOND, "UTC"),
            derived_from=("created_at",),
        ),
        Field(name="amount_cents", dtype=int_(64), derived_from=("amount",)),
        Field(name="status", dtype=string()),
        Field(name="country", dtype=string(), derived_from=("country",)),
    ),
    name="daily_revenue",
)


def orders_graph() -> LineageGraph:
    """Two sources, one pipeline, one sink. Every evidence level represented."""
    graph = LineageGraph()
    graph.add_node(
        Node(
            id=ORDERS_URI,
            kind=NodeKind.SOURCE,
            schema=ORDERS,
            origin=Origin(kind="source", ref=ORDERS_URI),
        )
    )
    graph.add_node(
        Node(
            id=CUSTOMERS_URI,
            kind=NodeKind.SOURCE,
            schema=CUSTOMERS,
            origin=Origin(kind="source", ref=CUSTOMERS_URI),
        )
    )
    graph.add_node(
        Node(
            id=PIPELINE_TARGET,
            kind=NodeKind.PYTHON,
            schema=REVENUE,
            engine="polars",
            origin=Origin(
                kind="python",
                ref=PIPELINE_TARGET,
                file="pipelines/revenue.py",
                line=9,
            ),
        )
    )
    graph.add_node(
        Node(
            id=SINK_URI,
            kind=NodeKind.SINK,
            schema=REVENUE.relabel("revenue"),
            engine="polars",
            origin=Origin(kind="source", ref=SINK_URI),
        )
    )

    graph.add_edge(
        Edge(
            source=ColumnRef(ORDERS_URI, "created_at"),
            target=ColumnRef(PIPELINE_TARGET, "day"),
            evidence=Evidence.PLAN,
            transform="dt.date",
        )
    )
    graph.add_edge(
        Edge(
            source=ColumnRef(ORDERS_URI, "amount"),
            target=ColumnRef(PIPELINE_TARGET, "amount_cents"),
            evidence=Evidence.PLAN,
            transform="sum",
        )
    )
    graph.add_edge(
        Edge(
            source=ColumnRef(ORDERS_URI, "status"),
            target=ColumnRef(PIPELINE_TARGET, "status"),
            evidence=Evidence.NAME_MATCH,
        )
    )
    graph.add_edge(
        Edge(
            source=ColumnRef(CUSTOMERS_URI, "country"),
            target=ColumnRef(PIPELINE_TARGET, "country"),
            evidence=Evidence.SQL_QUALIFIED,
        )
    )
    for column in REVENUE.fields:
        graph.add_edge(
            Edge(
                source=ColumnRef(PIPELINE_TARGET, column.name),
                target=ColumnRef(SINK_URI, column.name),
                evidence=Evidence.PLAN,
            )
        )
    return graph


def opaque_graph() -> LineageGraph:
    """A pipeline whose dry run failed: a node, no schema, no edges."""
    graph = LineageGraph()
    graph.add_node(
        Node(
            id=ORDERS_URI,
            kind=NodeKind.SOURCE,
            schema=ORDERS,
            origin=Origin(kind="source", ref=ORDERS_URI),
        )
    )
    graph.add_node(
        Node(
            id=PIPELINE_TARGET,
            kind=NodeKind.PYTHON,
            schema=Schema.empty(name="daily_revenue"),
            engine="polars",
            origin=Origin(kind="python", ref=PIPELINE_TARGET, file="pipelines/revenue.py", line=9),
            opaque=True,
        )
    )
    return graph


def coverage(confidence: Confidence = Confidence.PARTIAL) -> Coverage:
    """A partial score with a reason, which is the interesting case."""
    return Coverage(
        columns_total=4,
        columns_typed=4,
        columns_traced=3,
        confidence=confidence,
        reasons=("status has no lineage evidence",),
    )

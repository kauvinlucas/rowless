"""The cross-language column graph, and the blast radius read off it."""

from __future__ import annotations

import pytest
from rowless.ir.diff import Severity
from rowless.ir.dtypes import int_, string, unknown
from rowless.ir.schema import Field, Schema
from rowless.lineage.blast_radius import compare
from rowless.lineage.graph import (
    ColumnRef,
    Edge,
    Evidence,
    LineageGraph,
    Node,
    NodeKind,
)

ORDERS = Schema(
    fields=(Field(name="amount", dtype=int_(64)), Field(name="status", dtype=string())),
    name="stg_orders",
)


def _graph(*, amount_name: str = "amount", opaque_mart: bool = False) -> LineageGraph:
    """A source, a Python pipeline and a SQL mart, linked column to column.

    Two languages on purpose: a column renamed upstream resolving to both the
    Python function and the SQL model is the thing the proposal argues nobody
    else does.
    """
    graph = LineageGraph()
    graph.add_node(
        Node(
            id="lake://stg_orders",
            kind=NodeKind.SOURCE,
            schema=ORDERS.rename({"amount": amount_name}),
        )
    )
    graph.add_node(
        Node(
            id="pipelines.revenue:daily_revenue",
            kind=NodeKind.PYTHON,
            schema=Schema(
                fields=(Field(name="revenue", dtype=int_(64), derived_from=(amount_name,)),)
            ),
            engine="polars",
        )
    )
    graph.add_node(
        Node(
            id="marts/finance_kpis.sql",
            kind=NodeKind.SQL,
            schema=Schema(fields=(Field(name="kpi", dtype=int_(64), derived_from=("revenue",)),)),
            engine="duckdb",
            opaque=opaque_mart,
        )
    )
    graph.add_edge(
        Edge(
            source=ColumnRef("lake://stg_orders", amount_name),
            target=ColumnRef("pipelines.revenue:daily_revenue", "revenue"),
            evidence=Evidence.PLAN,
            transform="sum",
        )
    )
    graph.add_edge(
        Edge(
            source=ColumnRef("pipelines.revenue:daily_revenue", "revenue"),
            target=ColumnRef("marts/finance_kpis.sql", "kpi"),
            evidence=Evidence.SQL_QUALIFIED,
        )
    )
    return graph


def test_downstream_is_nearest_first() -> None:
    """The first line of a blast radius should be the direct consumer."""
    graph = _graph()

    reached = list(graph.downstream(ColumnRef("lake://stg_orders", "amount")))

    assert reached == [
        ColumnRef("pipelines.revenue:daily_revenue", "revenue"),
        ColumnRef("marts/finance_kpis.sql", "kpi"),
    ]
    assert list(graph.downstream(ColumnRef("lake://stg_orders", "amount"), max_depth=1)) == [
        ColumnRef("pipelines.revenue:daily_revenue", "revenue")
    ]


def test_upstream_answers_where_did_this_come_from() -> None:
    """The debugging direction, with the evidence attached."""
    graph = _graph()

    assert list(graph.upstream(ColumnRef("marts/finance_kpis.sql", "kpi"))) == [
        ColumnRef("pipelines.revenue:daily_revenue", "revenue"),
        ColumnRef("lake://stg_orders", "amount"),
    ]
    edges = graph.edges_into(ColumnRef("pipelines.revenue:daily_revenue", "revenue"))
    assert [edge.evidence for edge in edges] == [Evidence.PLAN]
    assert edges[0].transform == "sum"


def test_an_edge_to_an_unknown_node_is_refused() -> None:
    """An edge nobody declared would make the radius wrong in either direction."""
    graph = _graph()

    with pytest.raises(KeyError, match="unknown node"):
        graph.add_edge(
            Edge(
                source=ColumnRef("nowhere", "x"),
                target=ColumnRef("marts/finance_kpis.sql", "kpi"),
                evidence=Evidence.NAME_MATCH,
            )
        )


def test_graph_round_trips_through_json() -> None:
    """`rowless diff` reads the base revision's graph instead of rebuilding it."""
    graph = _graph(opaque_mart=True)

    restored = LineageGraph.from_json(graph.to_json())

    assert set(restored.nodes) == set(graph.nodes)
    assert len(restored.edges) == len(graph.edges)
    assert restored.nodes["marts/finance_kpis.sql"].opaque
    assert restored.nodes["lake://stg_orders"].schema == graph.nodes["lake://stg_orders"].schema


def test_exports_are_renderable() -> None:
    """Dot and mermaid both exist because people paste them in different places."""
    graph = _graph()

    assert "digraph rowless" in graph.to_dot()
    assert "graph LR" in graph.to_mermaid()


def test_blast_radius_crosses_languages() -> None:
    """A rename upstream names both the Python function and the SQL mart."""
    base = _graph()
    head = _graph(amount_name="amount_cents")

    radius = compare(base, head, base_ref="main", head_ref="working tree")

    assert radius.severity is Severity.BREAKING
    assert radius.is_breaking
    renames = [impact for impact in radius.impacts if impact.change.renamed_to]
    assert len(renames) == 1
    affected = renames[0].change.affects
    assert any("daily_revenue" in entry and "polars" in entry for entry in affected)
    assert any("finance_kpis.sql" in entry and "duckdb" in entry for entry in affected)


def test_an_opaque_node_marks_the_radius_as_possibly_larger() -> None:
    """A small-looking impact must not read as a safe one."""
    base = _graph(opaque_mart=True)
    head = _graph(amount_name="amount_cents", opaque_mart=True)

    radius = compare(base, head, base_ref="main", head_ref="working tree")

    paths = [impact.unverified_paths for impact in radius.impacts if impact.unverified_paths]
    assert paths and "marts/finance_kpis.sql" in paths[0]


def test_unresolved_column_in_the_head_is_unverified() -> None:
    """Losing the trail between revisions fails the gate, by design."""
    base = _graph()
    head = _graph()
    head.add_node(
        Node(
            id="pipelines.revenue:daily_revenue",
            kind=NodeKind.PYTHON,
            schema=Schema(fields=(Field(name="revenue", dtype=unknown("adapter gave up")),)),
            engine="polars",
        )
    )

    radius = compare(base, head, base_ref="main", head_ref="working tree")

    assert radius.severity is Severity.UNVERIFIED
    assert radius.is_breaking


def test_a_removed_relation_is_breaking() -> None:
    """A whole pipeline disappearing is as breaking as a column does."""
    base = _graph()
    head = _graph()
    del head.nodes["marts/finance_kpis.sql"]

    radius = compare(base, head, base_ref="main", head_ref="working tree")

    assert any(diff.relation == "marts/finance_kpis.sql" for diff in radius.diffs)
    assert radius.is_breaking

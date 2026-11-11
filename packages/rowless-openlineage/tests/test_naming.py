"""Dataset naming: the test that decides whether a catalog merges or forks.

Every row here is a claim about the OpenLineage naming conventions. If one of
them is wrong, rowless publishes a dataset beside the one a team already has
and the lineage graph quietly forks in two.
"""

from __future__ import annotations

import pytest
from rowless.ir.schema import Schema
from rowless.lineage.graph import Node, NodeKind
from rowless_openlineage.naming import (
    DEFAULT_NAMESPACE,
    DatasetName,
    Naming,
    parse_overrides,
)


def node(node_id: str, kind: NodeKind = NodeKind.SOURCE) -> Node:
    return Node(id=node_id, kind=kind, schema=Schema.empty(name=node_id))


@pytest.mark.parametrize(
    ("node_id", "expected"),
    [
        ("s3://bucket/key/part-0.parquet", ("s3://bucket", "key/part-0.parquet")),
        ("s3a://bucket/key", ("s3a://bucket", "key")),
        ("gs://bucket/warehouse/orders", ("gs://bucket", "warehouse/orders")),
        ("gcs://bucket/orders", ("gcs://bucket", "orders")),
        ("az://container/orders", ("az://container", "orders")),
        ("abfss://container/orders", ("abfss://container", "orders")),
        ("file:///srv/data/orders.parquet", ("file", "/srv/data/orders.parquet")),
        ("file://nas1/data/orders.parquet", ("file://nas1", "/data/orders.parquet")),
        ("data/orders.parquet", ("file", "data/orders.parquet")),
        ("/srv/data/orders.parquet", ("file", "/srv/data/orders.parquet")),
        ("iceberg://prod.sales.orders", ("iceberg://prod", "sales.orders")),
        ("warehouse://analytics.public.orders", ("warehouse", "analytics.public.orders")),
        ("snapshot://analytics.public.orders", ("warehouse", "analytics.public.orders")),
        ("dbt://model.shop.stg_orders", ("dbt", "model.shop.stg_orders")),
        ("lake://stg_orders", ("lake", "stg_orders")),
    ],
)
def test_dataset_names_follow_the_conventions(node_id: str, expected: tuple[str, str]) -> None:
    """The table in `naming`'s docstring, asserted."""
    found = Naming().dataset(node(node_id))

    assert (found.namespace, found.name) == expected


def test_a_trailing_slash_does_not_make_a_second_dataset() -> None:
    """`s3://marts/revenue/` and `s3://marts/revenue` are one table."""
    with_slash = Naming().dataset(node("s3://marts/revenue/"))
    without = Naming().dataset(node("s3://marts/revenue"))

    assert with_slash == without


def test_source_options_are_not_part_of_the_name() -> None:
    """`?snapshot=` selects a point in history, not a different table.

    An Iceberg URI carrying a snapshot id has to land on the same dataset as
    the plain one, or a diff against a historical snapshot would publish a
    second copy of every table it touched.
    """
    pinned = Naming().dataset(node("iceberg://prod.sales.orders?snapshot=42"))

    assert (pinned.namespace, pinned.name) == ("iceberg://prod", "sales.orders")


def test_a_windows_path_is_the_same_dataset_as_the_posix_one() -> None:
    """A developer's machine and CI must not publish two datasets."""
    windows = Naming().dataset(node("data\\orders.parquet"))

    assert windows == DatasetName(namespace="file", name="data/orders.parquet")


def test_a_pipeline_is_named_after_its_target() -> None:
    """The name a user can paste back into `rowless explain`."""
    naming = Naming()
    pipeline = node("pipelines.revenue:daily_revenue", NodeKind.PYTHON)

    assert naming.job(pipeline) == DatasetName(DEFAULT_NAMESPACE, "pipelines.revenue:daily_revenue")
    assert naming.dataset(pipeline) == DatasetName(
        DEFAULT_NAMESPACE, "pipelines.revenue:daily_revenue"
    )


def test_a_dbt_model_node_is_still_the_dbt_dataset() -> None:
    """A model rowless *ran* and a model it *read* are one dataset.

    This is the Python-to-SQL seam. If a discovered dbt model were published
    in rowless's own namespace, the pipeline that reads `dbt://model.shop.x`
    and the model that produces it would be two unconnected datasets, which is
    precisely the fork this package exists to avoid.
    """
    model = node("dbt://model.shop.stg_orders", NodeKind.DBT_MODEL)

    assert Naming().dataset(model) == DatasetName("dbt", "model.shop.stg_orders")


def test_a_sink_path_is_a_file_dataset() -> None:
    """A recovered write target is a table, named like any other path."""
    sink = node("out/revenue.parquet", NodeKind.SINK)

    assert Naming().dataset(sink) == DatasetName("file", "out/revenue.parquet")


def test_the_job_namespace_comes_from_the_environment() -> None:
    """`OPENLINEAGE_NAMESPACE` is what every other integration reads."""
    naming = Naming.from_env({"OPENLINEAGE_NAMESPACE": "analytics"})

    assert naming.namespace == "analytics"
    assert naming.job(node("a:b", NodeKind.PYTHON)).namespace == "analytics"


def test_an_override_replaces_the_namespace_but_never_the_name() -> None:
    """The escape hatch for a datasource rowless cannot know the host of."""
    naming = Naming.from_env(
        {"ROWLESS_OPENLINEAGE_NAMESPACES": "warehouse=postgres://db:5432,lake=s3://warehouse"}
    )

    assert naming.dataset(node("warehouse://analytics.public.orders")) == DatasetName(
        "postgres://db:5432", "analytics.public.orders"
    )
    assert naming.dataset(node("lake://stg_orders")) == DatasetName("s3://warehouse", "stg_orders")
    # An unlisted scheme is untouched.
    assert naming.dataset(node("s3://bucket/key")).namespace == "s3://bucket"


def test_bare_paths_can_be_overridden_too() -> None:
    """A repo-relative path is the one name a catalog is least likely to share."""
    naming = Naming.from_env({"ROWLESS_OPENLINEAGE_NAMESPACES": "=file://nas1"})

    assert naming.dataset(node("data/orders.parquet")) == DatasetName(
        "file://nas1", "data/orders.parquet"
    )
    # A node with a scheme of its own is unaffected by the bare-path entry.
    assert naming.dataset(node("lake://stg_orders")) == DatasetName("lake", "stg_orders")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", {}),
        ("lake", {}),
        ("lake=", {}),
        ("=s3://bucket", {"": "s3://bucket"}),
        (" lake = s3://bucket ", {"lake": "s3://bucket"}),
        ("a=1,,b=2", {"a": "1", "b": "2"}),
    ],
)
def test_malformed_overrides_are_skipped_not_fatal(raw: str, expected: dict[str, str]) -> None:
    """`available()` must not raise, so a typo cannot be an exception.

    An entry with no `=` and an entry with no namespace are both dropped; an
    entry with no *scheme* is kept, because that is how a bare filesystem path
    is spelled.
    """
    assert parse_overrides(raw) == expected

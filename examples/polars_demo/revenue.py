"""The example from the proposal. Deliberately broken.

`rowless check` reports exactly one finding here:

    revenue.py  ColumnNotFound 'amount', did you mean 'amount_cents'?  RWL101

It points at the `.agg(...)` line below, the one that reads the renamed column
and the one to edit. The input schema is committed as
`schemas/stg_orders.json`, so the example resolves with no network access and
no credentials; running it needs `rowless-polars`, which is installed with
`uv sync --package rowless-polars`. `tests/test_examples.py` asserts the
finding, the line it lands on and that nothing else is reported.
"""

from __future__ import annotations

import polars as pl
import rowless

orders = rowless.source("lake://stg_orders")


def daily_revenue(orders: pl.LazyFrame) -> pl.LazyFrame:
    """Revenue per day. Reads a column that no longer exists."""
    return (
        orders.filter(pl.col("status") == "paid")
        .group_by(pl.col("created_at").dt.date().alias("day"))
        .agg(pl.col("amount").sum())
    )

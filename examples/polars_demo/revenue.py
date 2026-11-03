"""The example from the proposal. Deliberately broken.

`rowless check` should report one finding here:

    revenue.py:25  ColumnNotFound amount
      upstream lake://stg_orders has amount_cents

The input schema is committed as `schemas/stg_orders.json`, so the example
resolves with no network access and no credentials. Running it needs the
Polars adapter; until that exists, `rowless check` reports the missing engine
rather than a pass.
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

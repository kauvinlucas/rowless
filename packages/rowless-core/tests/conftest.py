"""Shared fixtures for core tests.

Core tests run with no engine installed, on purpose: it keeps core honest about
never importing an engine library, and it is the environment a first-time
contributor has. Anything that needs an adapter uses the one in
`rowless_fake.py`, which is a real object rather than a mock.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from rowless_fake import FakeAdapter, install


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    """A scratch directory standing in for a dry-run workdir."""
    return tmp_path


@pytest.fixture
def fake_engine() -> Iterator[FakeAdapter]:
    """The fake adapter, registered for the duration of one test."""
    from rowless.adapters.registry import clear_cache

    clear_cache()
    adapter = install()
    yield adapter
    clear_cache()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A tiny repository: one schema file and one pipeline that reads it.

    Shaped like the proposal's example, including the planted rename, so the
    end-to-end tests check the thing the tool exists to find.
    """
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "stg_orders.json").write_text(
        '{"order_id": "int64", "status": "string", "amount_cents": "int64"}',
        encoding="utf-8",
    )
    (tmp_path / "rowless.toml").write_text(
        "\n".join(
            [
                'roots = ["."]',
                "",
                "[sources]",
                '"lake://*" = "json://schemas/{name}.json"',
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "revenue.py").write_text(
        "\n".join(
            [
                '"""A pipeline that reads a column the source renamed."""',
                "",
                "import rowless",
                "from rowless_fake import FakeFrame",
                "",
                'orders = rowless.source("lake://stg_orders")',
                "",
                "",
                '@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="fake")',
                "def daily_revenue(orders: FakeFrame) -> FakeFrame:",
                '    return orders.select("order_id", "amount")',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return tmp_path

"""Manifest parsing, against committed fixtures per dbt version."""

import pytest

pytestmark = pytest.mark.skip(reason="TODO(dbt): commit manifest fixtures for v11 and v12")


def test_loads_v12_manifest() -> None:
    """A v12 manifest parses, and model schemas come from contracts first."""


def test_rejects_unknown_version() -> None:
    """An unsupported schema version raises rather than half-parsing."""


def test_resolves_model_without_contract() -> None:
    """A model with no contract falls back to compiled SQL via rowless-sql."""

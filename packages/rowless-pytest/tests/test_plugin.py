"""Plugin tests, using pytest's own pytester fixture."""

import pytest

pytest_plugins = ["pytester"]

pytestmark = pytest.mark.skip(reason="TODO(pytest): implement the plugin first")


def test_check_passes_on_matching_schema(pytester: pytest.Pytester) -> None:
    """A pipeline matching its contract passes."""


def test_check_fails_with_schema_diff(pytester: pytest.Pytester) -> None:
    """A mismatch fails, and the diff names the offending column."""


def test_unverified_result_fails_by_default(pytester: pytest.Pytester) -> None:
    """Row-dependent control flow fails the test instead of passing silently."""


def test_skips_when_engine_missing(pytester: pytest.Pytester) -> None:
    """A test needing an uninstalled engine is skipped, not errored."""

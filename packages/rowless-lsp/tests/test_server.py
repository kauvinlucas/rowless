"""Server tests: drive the handlers directly, no editor involved."""

import pytest

pytestmark = pytest.mark.skip(reason="TODO(lsp): implement create_server first")


def test_publishes_diagnostics_on_save() -> None:
    """A file with a bad column reference yields one diagnostic on the right line."""


def test_completion_offers_schema_columns() -> None:
    """Completion inside pl.col() returns the resolved input columns."""

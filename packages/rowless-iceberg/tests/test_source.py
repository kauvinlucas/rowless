"""Iceberg source tests, against a local catalog fixture."""

import pytest

pytestmark = pytest.mark.skip(reason="TODO(iceberg): build a local sqlite catalog fixture")


def test_resolves_current_schema() -> None:
    """A table resolves to its current schema with field IDs preserved."""


def test_renames_resolved_by_field_id() -> None:
    """A renamed column is reported as a rename, not as drop plus add."""

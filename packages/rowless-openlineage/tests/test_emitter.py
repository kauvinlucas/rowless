"""Emitter tests: assert on event payloads, never on a live transport."""

import pytest

pytestmark = pytest.mark.skip(reason="TODO(openlineage): implement to_events first")


def test_column_lineage_facet_shape() -> None:
    """The facet validates against the OpenLineage spec."""


def test_events_are_static_not_run_events() -> None:
    """No START or COMPLETE run events are emitted for a dry run."""


def test_coverage_facet_present() -> None:
    """Every event carries the coverage facet."""

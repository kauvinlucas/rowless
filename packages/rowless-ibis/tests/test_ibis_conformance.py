"""Conformance: the suite in core is the specification for this adapter."""

import pytest
from rowless.testing.conformance import adapter_conformance
from rowless_ibis.adapter import IbisAdapter

pytestmark = [pytest.mark.conformance, pytest.mark.requires_engine]

TestConformance = adapter_conformance(IbisAdapter())

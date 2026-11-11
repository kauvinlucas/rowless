"""Conformance: the suite in core is the specification for this adapter.

Every test in it needs a Spark to ask, and `available()` is what decides
whether there is one: a Connect endpoint, or a Java new enough to start a local
session. Where there is neither, the suite skips with the reason rather than
failing, because "no cluster configured" is not an adapter bug. CI runs it both
ways - a local session on Java 17, and the floor release - so the skip is never
the only thing that happens.
"""

import pytest
from rowless.testing.conformance import adapter_conformance
from rowless_pyspark.adapter import PySparkAdapter

pytestmark = [pytest.mark.conformance, pytest.mark.requires_engine]

TestConformance = adapter_conformance(PySparkAdapter())

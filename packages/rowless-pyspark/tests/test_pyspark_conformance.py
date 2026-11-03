"""Conformance: the suite in core is the specification for this adapter."""

import pytest
from rowless.testing.conformance import adapter_conformance
from rowless_pyspark.adapter import PySparkAdapter

pytestmark = [pytest.mark.conformance, pytest.mark.requires_engine]

#: Flip to True once this adapter's methods are implemented. The suite in core
#: is ready; an adapter that is still a skeleton would fail every case in it,
#: which says nothing a reader did not already know.
ADAPTER_IMPLEMENTED = False

if not ADAPTER_IMPLEMENTED:
    pytest.skip(
        "rowless-pyspark: the PySpark adapter is not implemented yet",
        allow_module_level=True,
    )

TestConformance = adapter_conformance(PySparkAdapter())

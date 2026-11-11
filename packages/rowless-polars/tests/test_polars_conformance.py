"""Conformance: the suite in core is the specification for this adapter."""

import pytest
from rowless.testing.conformance import adapter_conformance
from rowless_polars.adapter import PolarsAdapter

pytestmark = [
    pytest.mark.conformance,
    pytest.mark.requires_engine,
    # One case uses `map_elements` on purpose, to exercise UDF return-dtype
    # inference. Polars warns that it is slow, which is good advice about
    # production code and noise in a test about the inference.
    pytest.mark.filterwarnings("ignore::polars.exceptions.PolarsInefficientMapWarning"),
]

TestConformance = adapter_conformance(PolarsAdapter())

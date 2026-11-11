"""Conformance: the suite in core is the specification for this adapter.

Run twice, once per dtype table. Which table is in use decides whether five of
the corpus dtypes exist at all - pandas has no decimal, date, time, binary or
null dtype without pyarrow - so an adapter that only ever ran the arrow path
would be an adapter nobody had checked against the pandas most people have
installed.
"""

from typing import Any

import pytest
from rowless.ir.dtypes import DType
from rowless.testing.conformance import adapter_conformance
from rowless_pandas import dtypes as dtype_map
from rowless_pandas.adapter import PandasAdapter

pytestmark = [pytest.mark.conformance, pytest.mark.requires_engine]


class PinnedAdapter(PandasAdapter):
    """A pandas adapter pinned to one dtype table for the whole suite.

    `materialize` chooses the table from the run's options, which is right for
    a pipeline and wrong for a conformance class: the suite calls `from_ir` and
    `unsupported_dtypes` directly, before any run exists. Pinning per instance
    is what lets both tables be built into two independent suites in one
    process, without either of them depending on which ran last.
    """

    def __init__(self, backend: dtype_map.Backend) -> None:
        self.backend = backend

    def from_ir(self, dtype: DType) -> Any:
        """Lower through the pinned table rather than the configured one."""
        return dtype_map.from_ir(dtype, prefer=self.backend)

    def unsupported_dtypes(self) -> tuple[DType, ...]:
        """Answer for the pinned table."""
        return dtype_map.absent_dtypes(prefer=self.backend)

    def _choose_backend(self, ctx: object) -> None:
        """Keep the pinned table instead of reading the run's options."""
        dtype_map.use_backend(self.backend)


def _suite(backend: dtype_map.Backend) -> type:
    """Build the shared conformance suite against one dtype table."""
    dtype_map.use_backend(backend)
    return adapter_conformance(PinnedAdapter(backend))


if dtype_map.arrow_available():
    TestConformanceArrow = _suite("arrow")

TestConformanceNumpy = _suite("numpy")

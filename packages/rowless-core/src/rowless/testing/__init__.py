"""Test helpers rowless ships for adapter authors.

Importable from the `rowless` distribution, not a separate test-only package,
so an out-of-tree adapter gets the conformance suite by depending on rowless.
"""

from rowless.testing import fixtures
from rowless.testing.conformance import ConformanceCase, adapter_conformance

__all__ = ["ConformanceCase", "adapter_conformance", "fixtures"]

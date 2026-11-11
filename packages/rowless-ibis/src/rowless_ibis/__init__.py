"""Ibis adapter for rowless: unbound tables and expression schemas."""

from rowless_ibis.adapter import IbisAdapter
from rowless_ibis.portability import DEFAULT_BACKENDS, UnknownBackend, disagreements

__all__ = ["DEFAULT_BACKENDS", "IbisAdapter", "UnknownBackend", "disagreements"]

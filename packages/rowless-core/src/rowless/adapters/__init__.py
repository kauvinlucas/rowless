"""The adapter contract and the plugin registry.

See `docs/writing-an-adapter.md` for the walkthrough, and
`rowless.testing.conformance` for the suite a new adapter has to pass.
"""

from rowless.adapters.base import (
    Capabilities,
    DryRunContext,
    EngineAdapter,
    SupportsColumnLineage,
    SupportsEmptyFrameFixups,
    SupportsSql,
    SupportsUdfProbe,
)
from rowless.adapters.registry import adapter_for, engine, engines, iter_plugins

__all__ = [
    "Capabilities",
    "DryRunContext",
    "EngineAdapter",
    "SupportsColumnLineage",
    "SupportsEmptyFrameFixups",
    "SupportsSql",
    "SupportsUdfProbe",
    "adapter_for",
    "engine",
    "engines",
    "iter_plugins",
]

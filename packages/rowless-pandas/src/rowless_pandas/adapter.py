"""The pandas adapter.

The hardest of the dataframe adapters, and the most widely useful.

pandas has no lazy plan, so the only option is a real frame with zero rows. That
works, except where pandas dtypes depend on the data in the frame:

* An empty column of an unspecified dtype becomes `object` or `float64`.
* `groupby().agg()` on an empty frame can return different dtypes than on a
  populated one, especially for `mean` and `sum` over nullable integers.
* Datetime columns lose their timezone through some operations when empty.
* `concat` of empty frames upcasts in places a populated `concat` would not.

So this adapter maintains an explicit list of drift fixups, declares them
through `SupportsEmptyFrameFixups`, and has `rowless explain` show which ones
fired. Correcting dtypes silently would make rowless wrong in a way nobody could
debug.

Where available, pyarrow-backed dtypes (`dtype_backend="pyarrow"`) drift far
less, so the adapter prefers them and says so when it falls back. That is the
`[engine.pandas] arrow_dtypes` option.
"""

from __future__ import annotations

from typing import Any, ClassVar

from rowless.adapters.base import Capabilities, DryRunContext
from rowless.ir.dtypes import DType
from rowless.ir.schema import Relation, Schema


class PandasAdapter:
    """pandas, via empty typed frames plus explicit drift corrections."""

    name: ClassVar[str] = "pandas"
    requires: ClassVar[str] = "pandas>=2.2"
    capabilities: ClassVar[Capabilities] = Capabilities(
        udf_probe=True,
        nested_types=False,
    )

    def available(self) -> bool:
        """True when pandas imports at >= 2.2."""
        raise NotImplementedError("TODO(pandas)")

    def handles(self, obj: object) -> bool:
        """True for DataFrame and Series."""
        raise NotImplementedError("TODO(pandas)")

    def materialize(self, schema: Schema, ctx: DryRunContext) -> Any:
        """Build a zero-row frame, preferring pyarrow-backed dtypes."""
        raise NotImplementedError("TODO(pandas)")

    def describe(self, obj: object) -> Relation:
        """Read `obj.dtypes`, then apply and record the drift fixups."""
        raise NotImplementedError("TODO(pandas)")

    def to_ir(self, native: Any) -> DType:
        """numpy, pandas extension and pyarrow dtypes to IR."""
        raise NotImplementedError("TODO(pandas)")

    def from_ir(self, dtype: DType) -> Any:
        """IR to a pandas dtype, honouring the arrow_dtypes option."""
        raise NotImplementedError("TODO(pandas)")

    def dtype_drift_fixups(self) -> tuple[str, ...]:
        """The corrections this adapter applies, for `rowless explain`."""
        raise NotImplementedError("TODO(pandas)")

    def probe_udf(self, fn: Any, arg_schema: Schema, ctx: DryRunContext) -> DType:
        """Infer an `apply` return dtype from annotations, then by probing."""
        raise NotImplementedError("TODO(pandas)")

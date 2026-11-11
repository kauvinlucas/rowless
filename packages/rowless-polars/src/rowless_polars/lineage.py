"""Column lineage, recorded as the user builds the plan.

`rowless diff` can only report a blast radius for columns it can trace, so the
Polars adapter attributes every output column to the input columns it was
computed from. The attribution happens while the plan is being built rather
than by walking it afterwards, for two reasons:

* **The expression metadata is public and stable.** `Expr.meta.root_names()`
  and `Expr.meta.output_name()` are part of the Polars API; the serialised plan
  is an internal format whose JSON form is already deprecated. Reading the
  former means this file does not break when the latter changes.
* **It composes.** Each builder call maps the frame it was given onto the frame
  it returned, so a chain of fifteen operations is fourteen small mappings
  rather than one recursive plan interpreter.

The bookkeeping lives in `PlanLineage`, keyed by frame identity. A frame built
by an operation this module does not understand gets **no** entry rather than
an inherited one: those columns are then reported as untraced, which lowers the
coverage score. That is the honest outcome, and it is why `of()` distinguishes
"no lineage" (None) from "lineage that happens to be empty" ({}).

Lineage here is derivation, not dependency: `df.filter(pl.col("status") ==
"paid").select("amount")` attributes `amount` to `amount`, not to `status`. The
filter decides which rows survive, not what the column holds, and inflating
lineage with predicate columns would make `rowless diff` report renames that
never happened.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from typing import Any

#: Output column name to the input column names it was computed from.
Lineage = dict[str, tuple[str, ...]]


class PlanLineage:
    """Lineage for every frame one dry run built, keyed by frame identity.

    Polars frames are not hashable, so this is an identity map rather than a
    `WeakKeyDictionary`. A weak reference per entry purges it when the frame is
    collected, which stops a recycled `id()` from inheriting the lineage of a
    frame that no longer exists.
    """

    def __init__(self) -> None:
        self._by_frame: dict[int, Lineage] = {}
        self._alive: dict[int, weakref.ref[Any]] = {}

    def __len__(self) -> int:
        """How many frames are currently tracked. For tests and diagnostics."""
        return len(self._by_frame)

    def clear(self) -> None:
        """Forget everything. Called when a new dry run starts."""
        self._by_frame.clear()
        self._alive.clear()

    def of(self, frame: object) -> Lineage | None:
        """Lineage recorded for `frame`, or None when it was never tracked."""
        found = self._by_frame.get(id(frame))
        return dict(found) if found is not None else None

    def record(self, frame: object, lineage: Mapping[str, tuple[str, ...]]) -> None:
        """Attach `lineage` to `frame`, dropping it again when the frame dies."""
        key = id(frame)
        self._by_frame[key] = {name: tuple(parents) for name, parents in lineage.items()}

        def purge(_reference: weakref.ref[Any], key: int = key) -> None:
            """Drop the entry once the frame it describes is gone."""
            self._forget(key)

        # An object that cannot be weakly referenced keeps its entry until the
        # run ends, which is short. Every Polars frame can be.
        with suppress(TypeError):
            self._alive[key] = weakref.ref(frame, purge)

    def seed(self, frame: object, columns: Iterable[str]) -> None:
        """Record identity lineage for a materialized input frame.

        The one place where a column may be attributed to itself: these column
        names really are the input relation column names, which is what the
        lineage graph resolves `derived_from` against.
        """
        self.record(frame, {column: (column,) for column in columns})

    def _forget(self, key: int) -> None:
        """Drop a collected frame entry."""
        self._by_frame.pop(key, None)
        self._alive.pop(key, None)


#: The tracker for this process. One dry run runs per worker, and the adapter
#: clears it when a run starts.
TRACKER = PlanLineage()


# --- Rules -------------------------------------------------------------------
# Each rule maps the parent frame lineage onto the lineage of the frame an
# operation returned. They are pure functions of the mappings and the resolved
# output column names, so they can be unit-tested without an engine.


def carry(parent: Mapping[str, tuple[str, ...]], columns: Sequence[str]) -> Lineage:
    """Lineage for an operation that keeps column names: filter, sort, cast, head."""
    return {column: tuple(parent[column]) for column in columns if column in parent}


def from_expressions(
    parent: Mapping[str, tuple[str, ...]],
    columns: Sequence[str],
    expressions: Sequence[Any],
    *,
    carry_rest: bool,
) -> Lineage:
    """Lineage for `select`, `with_columns` and `agg`.

    Args:
        parent: lineage of the frame the operation was called on.
        columns: resolved output column names, in order.
        expressions: the expressions the call was given.
        carry_rest: when true, an output column no expression claimed keeps the
            lineage it had upstream. That is what makes `select(pl.all())` and
            the column selectors trace: they expand to names this adapter
            cannot read off the expression, but the names did not change.

    Returns:
        Output column name to input column names.
    """
    resolved: Lineage = carry(parent, columns) if carry_rest else {}
    for expression in expressions:
        name = output_name(expression)
        if name is None:
            continue
        parents = tuple(
            dict.fromkeys(
                source for root in root_names(expression) for source in parent.get(root, ())
            )
        )
        if parents:
            resolved[name] = parents
        else:
            # Either the expression reads no input column (`pl.len()`,
            # `pl.lit(0)`) or every column it reads is itself untraced. Both
            # mean no lineage rather than the column name standing in for one.
            resolved.pop(name, None)
    produced = set(columns)
    return {name: parents for name, parents in resolved.items() if name in produced}


def renamed(
    parent: Mapping[str, tuple[str, ...]],
    columns: Sequence[str],
    mapping: Mapping[str, str],
) -> Lineage:
    """Lineage for `rename`: the new name inherits what the old name had."""
    inverse = {new: old for old, new in mapping.items()}
    resolved: Lineage = {}
    for column in columns:
        source = inverse.get(column, column)
        if source in parent:
            resolved[column] = tuple(parent[source])
    return resolved


def joined(
    left: Mapping[str, tuple[str, ...]],
    right: Mapping[str, tuple[str, ...]] | None,
    columns: Sequence[str],
    *,
    suffix: str,
) -> Lineage:
    """Lineage for a join, including the suffixed right-hand collisions.

    Matched by name, which is what Polars itself does to the output schema: a
    coalesced join key stays under the left name, and a collision that is not a
    key comes back as `created_at_right`.
    """
    resolved: Lineage = {}
    for column in columns:
        if column in left:
            resolved[column] = tuple(left[column])
            continue
        if right is None:
            continue
        if column in right:
            resolved[column] = tuple(right[column])
            continue
        if suffix and column.endswith(suffix):
            base = column[: -len(suffix)]
            if base in right:
                resolved[column] = tuple(right[base])
    return resolved


def merged(
    parents: Sequence[Mapping[str, tuple[str, ...]] | None],
    columns: Sequence[str],
) -> Lineage:
    """Lineage for `pl.concat`: a column comes from every frame that has one."""
    resolved: Lineage = {}
    for column in columns:
        found = tuple(
            dict.fromkeys(
                source
                for parent in parents
                if parent is not None
                for source in parent.get(column, ())
            )
        )
        if found:
            resolved[column] = found
    return resolved


# --- Expression metadata -----------------------------------------------------


def expressions(args: Sequence[Any], kwargs: Mapping[str, Any]) -> list[Any]:
    """Normalise the arguments of a Polars builder call into a list of expressions.

    Strings become `pl.col(...)`, nested sequences are flattened, and keyword
    arguments are aliased to the keyword, which is what Polars does with them.
    Anything that is not a column reference - a literal, a `pl.Series`, a
    predicate kwarg like `how=` - contributes nothing and is dropped.
    """
    import polars as pl

    found: list[Any] = [expression for arg in args for expression in _flatten(arg)]
    for name, value in kwargs.items():
        for expression in _flatten(value):
            try:
                found.append(expression.alias(name))
            except Exception:
                found.append(pl.col(name))
    return found


def _flatten(value: Any) -> list[Any]:
    """Expressions inside one argument, which may be a string, an expr or a list."""
    import polars as pl

    if isinstance(value, pl.Expr):
        return [value]
    if isinstance(value, str):
        return [pl.col(value)]
    if isinstance(value, Mapping):
        return []
    if isinstance(value, Iterable) and not isinstance(value, bytes | pl.Series):
        return [expression for item in value for expression in _flatten(item)]
    return []


def output_name(expression: Any) -> str | None:
    """The column name an expression produces, or None when it is not one name.

    A selector or a wildcard expands to many columns, and Polars reports that
    by refusing to name one. Those columns are left to `carry_rest`.
    """
    meta = getattr(expression, "meta", None)
    if meta is None:
        return None
    try:
        return str(meta.output_name(raise_if_undetermined=False) or "") or None
    except TypeError:
        # Older Polars has no `raise_if_undetermined`; it raises instead.
        try:
            return str(meta.output_name())
        except Exception:
            return None
    except Exception:
        return None


def root_names(expression: Any) -> tuple[str, ...]:
    """The input column names an expression reads."""
    meta = getattr(expression, "meta", None)
    if meta is None:
        return ()
    try:
        return tuple(str(name) for name in meta.root_names())
    except Exception:
        return ()

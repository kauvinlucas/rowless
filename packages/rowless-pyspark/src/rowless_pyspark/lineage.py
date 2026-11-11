"""Column lineage, recorded as the user builds the plan.

`rowless diff` can only report a blast radius for columns it can trace, so this
adapter attributes every output column to the input columns it was computed
from. The attribution happens while the plan is being built rather than by
walking the analyzed plan afterwards, and that is a decision worth explaining,
because the analyzed plan is where Spark keeps the perfect answer.

Spark gives every analyzed attribute an expression ID, and following those IDs
back to the source relation would be exact. Reaching them is the problem:
in a classic session they live behind `df._jdf.queryExecution().analyzed()`,
which is a JVM-private plan whose shape changes between releases, and under
Spark Connect they are on the other side of an RPC that returns a schema and a
formatted string rather than a tree. An adapter built on either would be an
adapter that breaks on the next Spark release, or one that only works in one of
the two modes.

So lineage here is recorded per builder call, the way the Polars adapter does
it, from two things both modes expose:

* **the expressions the call was given.** Under Connect a `Column` carries a
  pure-Python expression tree (`ColumnReference`, `ColumnAlias`, ...) whose
  leaves name the columns they read. In a classic session the same information
  is one `collectLeaves()` call into Catalyst, which is a stable TreeNode API
  rather than a plan format.
* **the resolved output column names**, which the wrapper already has because
  it analyzed the result to validate it.

Where those two line up positionally - `select` and `agg` produce one output
column per expression, in order - the attribution is exact without reading a
single name. Where they do not, because the call contained a `*` or a selector
that expanded inside Spark, the names that survived keep the lineage they had
and the rest are left untraced.

A frame built by an operation this module does not understand gets **no** entry
rather than an inherited one: those columns are then reported as untraced,
which lowers the coverage score. That is the honest outcome, and it is why
`of()` distinguishes "no lineage" (None) from "lineage that happens to be
empty" ({}).

Lineage is derivation, not dependency: `df.filter(col("status") == "paid")
.select("amount")` attributes `amount` to `amount`, not to `status`. The filter
decides which rows survive, not what the column holds, and inflating lineage
with predicate columns would make `rowless diff` report renames that never
happened.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from typing import Any

from rowless_pyspark import session as run

#: Output column name to the input column names it was computed from.
Lineage = dict[str, tuple[str, ...]]

#: Catalyst leaf classes that name a column. Read by class name because the
#: classic path reaches them through py4j, where there is no Python class to
#: compare against.
_ATTRIBUTE_LEAVES: frozenset[str] = frozenset({"UnresolvedAttribute", "AttributeReference"})


class PlanLineage:
    """Lineage for every frame one dry run built, keyed by frame identity.

    Spark DataFrames are not hashable, so this is an identity map rather than a
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
        # run ends, which is short. Every Spark DataFrame can be.
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


#: The tracker for this process. One dry run runs per worker, and the run state
#: clears it when a new run starts.
TRACKER = PlanLineage()

run.on_reset(TRACKER.clear)


# --- Rules -------------------------------------------------------------------
# Each rule maps the parent frame lineage onto the lineage of the frame an
# operation returned. They are pure functions of the mappings and the resolved
# output column names, so they can be unit-tested without a Spark session.


def carry(parent: Mapping[str, tuple[str, ...]], columns: Sequence[str]) -> Lineage:
    """Lineage for an operation that keeps column names: filter, sort, limit, drop."""
    return {column: tuple(parent[column]) for column in columns if column in parent}


def positional(
    parent: Mapping[str, tuple[str, ...]],
    columns: Sequence[str],
    terms: Sequence[Any],
) -> Lineage:
    """Lineage for `select` and `agg`, matching each output column to its expression.

    Spark produces one output column per expression, in the order the
    expressions were given, so position is the attribution - no name parsing,
    and it works for an aggregate whose output name is `sum(amount)` as well as
    for a plain projection.

    Falls back to `carry` when the counts do not line up, which is what a `*` or
    a column regex in the call means: those expanded inside Spark, so the only
    thing this adapter knows about the extra columns is that their names did not
    change.
    """
    if len(terms) != len(columns) or any(is_star(term) for term in terms):
        return carry(parent, columns)
    resolved: Lineage = {}
    for column, term in zip(columns, terms, strict=True):
        parents = tuple(
            dict.fromkeys(source for root in root_names(term) for source in parent.get(root, ()))
        )
        if parents:
            resolved[column] = parents
        # Either the expression reads no input column (`lit(0)`, `count("*")`)
        # or every column it reads is itself untraced. Both mean no lineage
        # rather than the column name standing in for one.
    return resolved


def renamed(
    parent: Mapping[str, tuple[str, ...]],
    columns: Sequence[str],
    mapping: Mapping[str, str],
) -> Lineage:
    """Lineage for `withColumnRenamed`: the new name inherits what the old name had."""
    inverse = {new: old for old, new in mapping.items()}
    resolved: Lineage = {}
    for column in columns:
        source = inverse.get(column, column)
        if source in parent:
            resolved[column] = tuple(parent[source])
    return resolved


def added(
    parent: Mapping[str, tuple[str, ...]],
    columns: Sequence[str],
    additions: Mapping[str, Any],
) -> Lineage:
    """Lineage for `withColumn` and `withColumns`: the named column, plus the rest.

    The one builder family where Spark states the output name itself, so the
    attribution needs neither position nor a resolved name.
    """
    resolved = carry(parent, columns)
    for name, term in additions.items():
        parents = tuple(
            dict.fromkeys(source for root in root_names(term) for source in parent.get(root, ()))
        )
        if parents:
            resolved[name] = parents
        else:
            resolved.pop(name, None)
    return {name: found for name, found in resolved.items() if name in set(columns)}


def joined(
    left: Mapping[str, tuple[str, ...]],
    right: Mapping[str, tuple[str, ...]] | None,
    columns: Sequence[str],
) -> Lineage:
    """Lineage for a join, matched by name the way Spark builds the output.

    Spark keeps both sides whole - including both key columns when the join is
    on a condition - and does not rename a collision, so a name that appears on
    one side only is unambiguous and a name on both is a duplicate the adapter
    refuses in `describe` anyway.
    """
    resolved: Lineage = {}
    for column in columns:
        if column in left:
            resolved[column] = tuple(left[column])
        elif right is not None and column in right:
            resolved[column] = tuple(right[column])
    return resolved


def merged(
    parents: Sequence[Mapping[str, tuple[str, ...]] | None],
    columns: Sequence[str],
) -> Lineage:
    """Lineage for `union` and `unionByName`: a column comes from every side that has one."""
    resolved: Lineage = {}
    for index, column in enumerate(columns):
        found = tuple(
            dict.fromkeys(
                source
                for parent in parents
                if parent is not None
                for source in _union_sources(parent, column, index)
            )
        )
        if found:
            resolved[column] = found
    return resolved


def _union_sources(
    parent: Mapping[str, tuple[str, ...]],
    column: str,
    index: int,
) -> tuple[str, ...]:
    """What one side of a union contributes to an output column.

    `union` is positional in Spark and `unionByName` is not, which is a
    difference that bites: `union` on two frames whose columns are in different
    orders silently pairs the wrong ones. Both are covered by taking the name
    when the side has it and the column at that position when it does not, so
    the lineage reflects whichever one the pipeline used.
    """
    if column in parent:
        return tuple(parent[column])
    names = list(parent)
    if index < len(names):
        return tuple(parent[names[index]])
    return ()


# --- Reading the expressions a call was given --------------------------------


def terms(args: Sequence[Any], kwargs: Mapping[str, Any]) -> list[Any]:
    """Normalise the arguments of a Spark builder call into a flat list.

    Strings, `Column`s and lists of either are what `select`, `groupBy` and
    `agg` accept, and Spark flattens a single list argument itself. Keyword
    arguments are dropped: on the builders this adapter wraps they are options
    (`how=`, `on=`), not expressions.
    """
    _ = kwargs
    found: list[Any] = []
    for argument in args:
        found.extend(_flatten(argument))
    return found


def _flatten(value: Any) -> list[Any]:
    """The expressions inside one argument, which may be a list of them."""
    if isinstance(value, str):
        return [value]
    if _is_column(value):
        return [value]
    if isinstance(value, Mapping):
        return []
    if isinstance(value, Iterable) and not isinstance(value, bytes):
        return [item for element in value for item in _flatten(element)]
    return []


def root_names(term: Any) -> tuple[str, ...]:
    """The input column names an expression reads, as far as either mode can say.

    A string is the column it names. A `Column` is read from its Connect
    expression tree when there is one, and through Catalyst's `collectLeaves`
    when the session is classic. Anything neither path can read returns nothing,
    which is reported as untraced rather than guessed.
    """
    if isinstance(term, str):
        return () if _is_star_name(term) else (_root(term),)
    if not _is_column(term):
        return ()
    expression = _connect_expression(term)
    if expression is not None:
        found: list[str] = []
        _walk_connect(expression, found)
        return tuple(dict.fromkeys(found))
    return _catalyst_roots(term)


def output_name(term: Any) -> str | None:
    """The name a call gives a column, when the expression states one.

    Only an alias states it. Everything else is named by Spark after the
    expression that produced it (`sum(amount)`, `(amount / 2)`), which is a
    name this adapter reads off the resolved schema rather than reconstructs.
    """
    if isinstance(term, str):
        return None if _is_star_name(term) else term
    expression = _connect_expression(term)
    names = getattr(expression, "_alias", None) if expression is not None else None
    if isinstance(names, list | tuple) and len(names) == 1:
        return str(names[0])
    return _catalyst_alias(term)


def is_star(term: Any) -> bool:
    """True when an expression expands to more than one column inside Spark."""
    if isinstance(term, str):
        return _is_star_name(term)
    if not _is_column(term):
        return False
    expression = _connect_expression(term)
    if expression is not None:
        return type(expression).__name__ in {"UnresolvedStar", "UnresolvedRegex"}
    return _is_star_name(_column_text(term))


def _root(name: str) -> str:
    """The input column a possibly-qualified reference reads.

    `payload.source` reads the `payload` column, and that is the column a
    lineage edge should point at: the struct is what a rename or a drop
    upstream would break.
    """
    cleaned = name.strip().strip("`")
    return cleaned.split(".", maxsplit=1)[0].strip("`")


def _is_star_name(name: str) -> bool:
    """True for `*`, `df.*` and the regex form a column selector uses."""
    text = name.strip()
    return text.endswith("*") or (text.startswith("`") and text.endswith("`") and "*" in text)


def _column_class(module_name: str) -> type | None:
    """The `Column` class from one module, or None when this install lacks it."""
    try:
        module = __import__(module_name, fromlist=["Column"])
    except Exception:
        return None
    found = getattr(module, "Column", None)
    return found if isinstance(found, type) else None


def _is_column(value: Any) -> bool:
    """True when `value` is a Spark `Column`, in either mode.

    Checked against the classic class, which the Connect one subclasses from
    Spark 4 on, plus the Connect class itself for Spark 3.5 where it does not.
    """
    for module_name in ("pyspark.sql.column", "pyspark.sql.connect.column"):
        owner = _column_class(module_name)
        if owner is not None and isinstance(value, owner):
            return True
    return False


def _connect_expression(column: Any) -> Any:
    """The Connect expression behind a `Column`, or None for a classic one.

    Deliberately strict about *where* the object came from rather than duck
    typing on an attribute, and that is not fussiness: a classic `Column`
    answers `getattr` for **any** name by building a new `Column` for struct
    field access, so `getattr(col, "_expr")` on one returns a `Column` rather
    than None, and a reader that trusted it would walk into an object whose
    `__bool__` raises `CANNOT_CONVERT_COLUMN_INTO_BOOL`. Asking the class
    instead cannot be fooled.
    """
    owner = _column_class("pyspark.sql.connect.column")
    if owner is None or not isinstance(column, owner):
        return None
    expression = getattr(column, "_expr", None)
    if expression is None:
        return None
    # On Spark 4 a Connect `Column` subclasses the classic one, so confirm the
    # expression is a Connect node rather than something inherited.
    return expression if type(expression).__module__.startswith("pyspark.sql.connect") else None


def _walk_connect(expression: Any, found: list[str]) -> None:
    """Collect every column reference in a Connect expression tree.

    Pure Python and no round trip: the Connect client builds the expression
    locally and only sends it when something asks for a schema or a result.
    """
    identifier = getattr(expression, "_unparsed_identifier", None)
    if isinstance(identifier, str) and identifier:
        found.append(_root(identifier))
    parts = getattr(expression, "_name_parts", None)
    if isinstance(parts, list | tuple) and parts:
        found.append(_root(str(parts[0])))
    children = getattr(expression, "children", None)
    if not isinstance(children, list | tuple):
        return
    for child in children:
        with suppress(Exception):
            _walk_connect(child, found)


def _catalyst_roots(column: Any) -> tuple[str, ...]:
    """Column names read out of a classic `Column` through Catalyst.

    `collectLeaves()` is a `TreeNode` method rather than a plan format, so this
    survives the releases that move the analyzed plan around. A shape it cannot
    read - an expression whose leaves are not attributes, a py4j call that
    fails - reports nothing, and those columns come out untraced.
    """
    jc = _java_column(column)
    if jc is None:
        return ()
    found: list[str] = []
    try:
        leaves = jc.expr().collectLeaves()
        for index in range(int(leaves.size())):
            leaf = leaves.apply(index)
            if str(leaf.getClass().getSimpleName()) in _ATTRIBUTE_LEAVES:
                found.append(_root(str(leaf.name())))
    except Exception:
        return ()
    return tuple(dict.fromkeys(found))


def _java_column(column: Any) -> Any:
    """The JVM `Column` behind a classic one, or None when there is no JVM.

    Read off the instance dictionary rather than with `getattr`: a `Column`
    builds a new `Column` for any attribute it does not have, so `getattr` can
    never answer None here and the caller would get an object that raises on
    use. See `_connect_expression` for the same trap on the Connect side.
    """
    return vars(column).get("_jc")


def _catalyst_alias(column: Any) -> str | None:
    """The alias a classic `Column` carries, when it is an `Alias`."""
    jc = _java_column(column)
    if jc is None:
        return None
    try:
        expression = jc.expr()
        if str(expression.getClass().getSimpleName()) != "Alias":
            return None
        return str(expression.name())
    except Exception:
        return None


def _column_text(column: Any) -> str:
    """A `Column`'s own rendering, used only to recognise a star."""
    try:
        return str(column)
    except Exception:  # pragma: no cover - a column that will not render
        return ""

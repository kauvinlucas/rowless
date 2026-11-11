"""Column lineage, read straight off the ibis expression tree.

This is the one adapter that needs no bookkeeping to report lineage. Polars has
to record it while a plan is built and DuckDB has to parse SQL back, because
neither keeps an addressable record of which input column an output column came
from. Ibis does: every relation node carries a `values` mapping from output
column name to the expression that computes it, and every column reference in
that expression is an `ops.Field` naming its relation and its column. Lineage
is therefore a walk, with no interception, no parser and no second
implementation of anybody's semantics.

Two rules the walk follows, and both of them are about not inventing edges:

* **Stop at relation boundaries.** `node.find(ops.Field)` would be the obvious
  way to collect references, and it is wrong: an `ops.Field` holds its relation
  as a child, so a graph search from one field descends into the relation below
  and comes back with every column that relation mentions - including the ones
  in its `WHERE` clause. `_referenced_fields` walks the value expression only,
  treating a nested relation as the subquery it is.
* **A predicate is not a parent.** `filter(status == "paid")` decides which
  rows survive, not what a column holds, so `status` is not reported as a
  parent of anything the predicate did not compute. This falls out of walking
  `values` rather than the whole node.

A column computed from no input at all - `ibis.literal(1)`, or a correlated
subquery whose relation the walk stops at - gets no entry. Core reports those
columns as untraced, which lowers the coverage score, and that is the honest
outcome rather than a guessed attribution.

Parents are reported as bare column names, the way the Polars and DuckDB
adapters report them: `rowless.lineage.graph` attributes an unqualified parent
to the pipeline's input, and a qualified one would have to carry the pipeline's
*parameter* name, which an adapter never sees.
"""

from __future__ import annotations

from typing import Any

#: Output column name to the input column names it was computed from.
Lineage = dict[str, tuple[str, ...]]

#: How many relations deep the walk will resolve before giving up. A pipeline
#: with a thousand chained operations is pathological, and an unbounded walk on
#: a cyclic graph that ibis is not supposed to be able to build would hang the
#: dry run instead of reporting it.
MAX_DEPTH: int = 200


def of_expression(expr: Any) -> Lineage:
    """Attribute every column of an ibis expression to the inputs it came from.

    Returns an empty mapping for anything that is not an ibis expression, and
    leaves out the columns it cannot attribute rather than guessing at them.
    """
    import ibis.expr.operations as ops
    import ibis.expr.types as ir

    if not isinstance(expr, ir.Expr):
        return {}
    node = expr.op()
    if isinstance(node, ops.Relation):
        return _of_relation(node)
    if isinstance(node, ops.Value):
        return _of_value(expr, node)
    return {}


def _of_relation(node: Any) -> Lineage:
    """Lineage for a table expression, column by column."""
    traced: Lineage = {}
    for name in node.schema.names:
        parents = _sources(node, str(name), depth=0)
        if parents:
            traced[str(name)] = parents
    return traced


def _of_value(expr: Any, node: Any) -> Lineage:
    """Lineage for a single column or scalar expression."""
    parents: list[str] = []
    for field in _referenced_fields(node):
        parents.extend(_sources(field.rel, str(field.name), depth=0))
    if not parents:
        return {}
    name = _value_name(expr)
    return {name: _dedupe(parents)}


def _value_name(expr: Any) -> str:
    """The name ibis gives a column expression, falling back to a readable one."""
    try:
        return str(expr.get_name())
    except Exception:
        return "value"


def _sources(rel: Any, name: str, *, depth: int) -> tuple[str, ...]:
    """The input columns `rel.name` was computed from.

    Resolves one relation at a time rather than searching the whole graph: at
    each step the question is only "which column of the relation below does
    this come from", which an alias that shadows an input column name cannot
    confuse.
    """
    import ibis.expr.operations as ops

    if depth > MAX_DEPTH:  # pragma: no cover - needs a pathological expression
        return ()
    values = getattr(rel, "values", None) or {}
    if not values:
        # A leaf: an unbound table, a backend table, an in-memory table, or a
        # relation whose columns this walk cannot look inside (a SQL view). Its
        # own column name is the input column name.
        return (name,)
    if isinstance(rel, ops.Set):
        # A union takes its column from every side that has one, and ibis
        # requires the schemas to match, so the name is the same on both.
        return _dedupe(
            [
                *_sources(rel.left, name, depth=depth + 1),
                *_sources(rel.right, name, depth=depth + 1),
            ]
        )
    value = values.get(name)
    if value is None:
        return ()
    parents: list[str] = []
    for field in _referenced_fields(value):
        parents.extend(_sources(field.rel, str(field.name), depth=depth + 1))
    return _dedupe(parents)


def _referenced_fields(node: Any) -> list[Any]:
    """The column references in one value expression, not descending into relations.

    `ops.Field` carries its relation as a child, so a plain graph search from a
    value expression walks into the relation below it and collects every column
    that relation mentions anywhere - its predicates included. This walk stops
    at both: a field is a leaf, and a nested relation is a boundary.
    """
    import ibis.expr.operations as ops

    found: list[Any] = []
    seen: set[int] = set()
    stack: list[Any] = [node]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ops.Field):
            found.append(current)
            continue
        if isinstance(current, ops.Relation):
            continue
        stack.extend(_child_nodes(current))
    return found


def _child_nodes(node: Any) -> list[Any]:
    """The operation nodes directly below `node`, flattening ibis's containers."""
    import ibis.expr.operations as ops

    children: list[Any] = []
    pending: list[Any] = list(getattr(node, "args", ()) or ())
    while pending:
        argument = pending.pop()
        if isinstance(argument, ops.Node):
            children.append(argument)
        elif isinstance(argument, tuple | list):
            pending.extend(argument)
        elif isinstance(argument, dict):
            pending.extend(argument.values())
    return children


def _dedupe(names: list[str]) -> tuple[str, ...]:
    """Drop repeats while keeping the order the columns were first seen in."""
    return tuple(dict.fromkeys(names))

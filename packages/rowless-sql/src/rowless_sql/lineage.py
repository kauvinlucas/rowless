"""Column lineage, read out of sqlglot's own resolver.

`rowless diff` can only report a blast radius for columns it can trace, and
this is the adapter that traces them best. `sqlglot.lineage` walks a resolved
query from an output column down to the base table columns it came from, and it
walks *through* the things that stop every other adapter: a CTE, a derived
table, a chain of both. The DuckDB adapter has to refuse a projection behind a
subquery, because attributing names across a scope it cannot see through is how
a tool invents a rename that never happened. Here the scopes are the data
structure, so a column selected from a CTE over a join resolves to the column
in the base table.

Three decisions worth knowing about:

* **One qualification, not one per column.** `sqlglot.lineage` qualifies the
  query itself unless it is handed a scope, and a forty-column mart would pay
  for forty qualifications of the same query. The adapter has already
  qualified the query to read its schema, so the scope is built from that
  expression once and passed in. `trim_selects` is off for the same reason it
  matters here: it rewrites the expression it was given, and that expression is
  the one the schema was just read off.
* **Parents are bare column names.** sqlglot reports a leaf as
  `"orders"."amount"`, which carries the table or alias it was read through.
  `rowless.lineage.graph` attributes an unqualified parent to the pipeline's
  input, and a qualified one would have to carry the pipeline's *parameter*
  name - something an adapter never sees. The Polars, DuckDB and Ibis adapters
  all report bare names for the same reason, so a SQL model's lineage composes
  with theirs.
* **A column with no input behind it gets no entry.** `'web' AS source`,
  `CURRENT_DATE AS run_at` and anything behind a scope sqlglot could not
  resolve are left out rather than attributed to whatever the query happened
  to read. Core reports those columns as untraced, which lowers the coverage
  score, and that is the honest outcome.

Lineage here is derivation, not dependency: `WHERE status = 'paid'` decides
which rows survive, not what a column holds, so `status` is not a parent of
anything the predicate did not compute. That falls out of how sqlglot builds
the graph, which walks a projection rather than a whole query.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: Output column name to the input column names it was computed from.
Lineage = dict[str, tuple[str, ...]]

#: Characters a dialect may quote an identifier with. Qualification quotes
#: every identifier it rewrites, so a leaf arrives as `"orders"."amount"`.
_QUOTES = '"`[]'


def of_resolved(
    resolved: Any,
    columns: Sequence[str],
    *,
    schema: Any,
    dialect: str = "",
    spellings: Mapping[str, str] | None = None,
) -> Lineage:
    """Attribute each of `columns` to the input columns it was computed from.

    Args:
        resolved: the qualified, annotated query `rowless_sql.resolve`
            produced. Not modified.
        columns: the output column names to trace, as they appear in the
            resolved projection.
        schema: the `sqlglot.schema.MappingSchema` the query was resolved
            against, which lineage needs to walk into a base table.
        dialect: the dialect the query is written in.
        spellings: case-folded column name to the declared spelling of the
            input column that produced it, so a parent is named the way the
            input schema names it rather than the way the dialect folded it.

    Returns:
        Output column name to parent column names, leaving out every column
        that could not be attributed. An empty mapping means nothing was
        traced, which core reports as untraced columns rather than as an
        error.
    """
    from sqlglot.lineage import lineage as sqlglot_lineage
    from sqlglot.optimizer.scope import build_scope

    try:
        scope = build_scope(resolved)
    except Exception:
        # Lineage is a capability, not a requirement: an expression sqlglot
        # cannot scope costs the traced ratio and nothing else.
        return {}
    if scope is None:
        return {}

    declared = dict(spellings or {})
    traced: Lineage = {}
    for column in columns:
        try:
            node = sqlglot_lineage(
                column,
                resolved,
                schema=schema,
                dialect=dialect or None,
                scope=scope,
                trim_selects=False,
            )
        except Exception:
            # One column rowless cannot trace is one untraced column, not a
            # failed dry run. The usual cause is a projection whose name
            # sqlglot does not recognise as an output of this query.
            continue
        parents = _parents(node, declared)
        if parents:
            traced[column] = parents
    return traced


def _parents(node: Any, declared: Mapping[str, str]) -> tuple[str, ...]:
    """Base table columns a lineage node descends to, deduplicated, in order."""
    from sqlglot import exp

    found: list[str] = []
    for walked in node.walk():
        if not isinstance(walked.expression, exp.Table):
            continue
        name = _column_name(str(walked.name))
        if not name:
            continue
        found.append(declared.get(name.casefold(), name))
    return tuple(dict.fromkeys(found))


def _column_name(leaf: str) -> str:
    """The column half of a `"orders"."amount"` leaf name, unquoted."""
    parts = [part.strip().strip(_QUOTES) for part in leaf.split(".")]
    named = [part for part in parts if part]
    return named[-1] if named else ""

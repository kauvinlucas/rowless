"""Column lineage, read out of DuckDB's own parser.

`rowless diff` can only report a blast radius for columns it can trace, so this
adapter attributes every output column it can to the input columns it was
computed from. Where the Polars adapter reads expression metadata, DuckDB takes
SQL strings, so the attribution starts by parsing them - with
`json_serialize_sql`, which is DuckDB's parser rather than a second
implementation of its grammar. The engine that will bind the query is the one
that says what the query says, so the two cannot drift apart, and the adapter
takes no dependency on sqlglot to do it.

Two sources of lineage, and both are needed:

* **composed**, through the relational API. Each intercepted call maps the
  lineage of the relation it was given onto the relation it returned, so a
  chain of fifteen operations is fourteen small mappings rather than one
  recursive resolver. This is the accurate one: an alias that shadows an input
  column name cannot confuse it, because each step is resolved against the
  step below.
* **read off a query**, for a relation this adapter never saw being built -
  anything from `conn.sql(...)`, and the SQL handed to `run_sql`. Only a
  single-level query resolves this way: with a subquery or a CTE in the way,
  the names in the final projection belong to an intermediate scope rather
  than to a base table, and attributing them to inputs of the same name is
  how a tool invents a rename that never happened. Those queries get **no**
  lineage instead, which is reported as untraced and lowers the coverage
  score.

Matching output columns to select-list items is positional. The alternative is
to reproduce DuckDB's rules for naming an unaliased expression (`sum(amount)`,
`CAST(x AS INTEGER)`), and a mismatch there would attribute a column to the
wrong parent - the one failure mode worse than having no lineage at all. The
position of an item in a select list is the one thing that is not a matter of
interpretation, so a call whose item count does not match its output count is
left untraced.

Lineage here is derivation, not dependency: `WHERE status = 'paid'` decides
which rows survive, not what a column holds, so `status` is not a parent of
anything the predicate did not compute.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from rowless_duckdb import database, session

#: Output column name to the input column names it was computed from.
Lineage = dict[str, tuple[str, ...]]

#: AST node classes that stand for "every column of the input", which cannot be
#: matched to a select-list position.
STAR_CLASSES: frozenset[str] = frozenset({"STAR"})

#: The only `from_table` kinds a raw query can be resolved through. A subquery
#: or a CTE puts a scope between the projection and the base columns, and this
#: module will not guess across it. A table function is excluded for a
#: different reason: `read_parquet(...)` is not a relation rowless registered,
#: so its column names are not input column names, and core attributes an
#: unqualified parent to the single input when there is one - which would
#: invent an edge to a column that relation does not have.
RESOLVABLE_SOURCES: frozenset[str] = frozenset({"BASE_TABLE", "JOIN"})


class PlanLineage:
    """Lineage for every relation one dry run built, keyed by relation identity.

    DuckDB relations are not hashable, so this is an identity map rather than a
    `WeakKeyDictionary`. A weak reference per entry purges it when the relation
    is collected, which stops a recycled `id()` from inheriting the lineage of
    a relation that no longer exists.
    """

    def __init__(self) -> None:
        self._by_relation: dict[int, Lineage] = {}
        self._alive: dict[int, weakref.ref[Any]] = {}

    def __len__(self) -> int:
        """How many relations are currently tracked. For tests and diagnostics."""
        return len(self._by_relation)

    def clear(self) -> None:
        """Forget everything. Called when a new dry run starts."""
        self._by_relation.clear()
        self._alive.clear()

    def of(self, relation: object) -> Lineage | None:
        """Lineage recorded for `relation`, or None when it was never tracked."""
        found = self._by_relation.get(id(relation))
        return dict(found) if found is not None else None

    def record(self, relation: object, lineage: Mapping[str, tuple[str, ...]]) -> None:
        """Attach `lineage` to `relation`, dropping it when the relation dies."""
        key = id(relation)
        self._by_relation[key] = {name: tuple(parents) for name, parents in lineage.items()}

        def purge(_reference: weakref.ref[Any], key: int = key) -> None:
            """Drop the entry once the relation it describes is gone."""
            self._forget(key)

        # An object that cannot be weakly referenced keeps its entry until the
        # run ends, which is short. Every DuckDB relation can be.
        with suppress(TypeError):
            self._alive[key] = weakref.ref(relation, purge)

    def seed(self, relation: object, columns: Iterable[str]) -> None:
        """Record identity lineage for a materialized input relation.

        The one place where a column may be attributed to itself: these column
        names really are the input relation's column names, which is what the
        lineage graph resolves `derived_from` against.
        """
        self.record(relation, {column: (column,) for column in columns})

    def _forget(self, key: int) -> None:
        """Drop a collected relation's entry."""
        self._by_relation.pop(key, None)
        self._alive.pop(key, None)


#: The tracker for this process. One dry run runs per worker, and it is cleared
#: when a run starts.
TRACKER = PlanLineage()

session.on_reset(TRACKER.clear)


@dataclass(frozen=True, slots=True)
class SelectItem:
    """One entry of a select list, as far as lineage is concerned."""

    roots: tuple[str, ...]
    """Input column names the expression reads, in the order they appear."""

    is_star: bool = False
    """True for `*`, which stands for a number of columns nobody can count
    from the text alone."""


@dataclass(frozen=True, slots=True)
class SelectList:
    """A parsed select list, and whether it can be matched positionally."""

    items: tuple[SelectItem, ...] = ()

    resolvable: bool = False
    """False when the parse failed, the statement was not a plain select, or a
    `*` makes the item count meaningless."""

    sources: tuple[str, ...] = ()
    """Base table names the query reads, when it reads tables directly."""


def parse_expressions(expressions: Sequence[str]) -> SelectList:
    """Parse a relational-API select list, given as SQL expression fragments.

    A single fragment may itself be a list (`"status, sum(amount)"`), which is
    how DuckDB's `project` and `aggregate` are usually called, so the pieces
    are joined and parsed as one select list.
    """
    fragments = [fragment for fragment in expressions if fragment and fragment.strip()]
    if not fragments:
        return SelectList()
    return parse_query(f"SELECT {', '.join(fragments)}")


def parse_query(sql: str) -> SelectList:
    """Parse a full query and return its top-level select list."""
    parsed = database.serialize_sql(database.strip_terminator(sql))
    if parsed.get("error"):
        return SelectList()
    statements = parsed.get("statements") or []
    if len(statements) != 1:
        return SelectList()
    node = statements[0].get("node") or {}
    if node.get("type") != "SELECT_NODE":
        # A set operation or a recursive CTE. Its columns come from several
        # scopes at once, and the rules in this module handle unions through
        # `merged` rather than by reading the text.
        return SelectList()
    if (node.get("cte_map") or {}).get("map"):
        return SelectList(sources=_sources(node))
    entries = node.get("select_list") or []
    items = tuple(_item(entry) for entry in entries)
    return SelectList(
        items=items,
        resolvable=bool(items) and not any(item.is_star for item in items),
        sources=_sources(node),
    )


def positional(
    parent: Lineage | None,
    columns: Sequence[str],
    parsed: SelectList,
) -> Lineage | None:
    """Attribute output columns to select-list items by position.

    Args:
        parent: lineage of the relation the call was made on, or None to treat
            every root as an input column name in its own right, which is what
            a raw query over base tables needs.
        columns: the output column names DuckDB resolved, in order.
        parsed: the select list those columns came from.

    Returns:
        Output column name to input column names, or None when the select list
        cannot be matched to the output - a `*`, a parse failure, or a count
        that does not line up. None means "record nothing", which reports the
        columns as untraced.
    """
    if not parsed.resolvable or len(parsed.items) != len(columns):
        return None
    resolved: Lineage = {}
    for column, item in zip(columns, parsed.items, strict=True):
        roots = _through(parent, item.roots)
        if roots:
            resolved[column] = roots
    return resolved


def carry(parent: Mapping[str, tuple[str, ...]], columns: Sequence[str]) -> Lineage:
    """Lineage for an operation that keeps column names: filter, order, limit."""
    return {column: tuple(parent[column]) for column in columns if column in parent}


def extended(
    parent: Mapping[str, tuple[str, ...]],
    columns: Sequence[str],
    added: Mapping[str, SelectList],
) -> Lineage:
    """Lineage for `with_columns`, which names its expressions with keywords.

    The existing columns keep what they had; each added column is attributed to
    the roots of the expression that produced it. No positional matching is
    needed, because the caller wrote the names down.
    """
    resolved: Lineage = carry(parent, columns)
    produced = set(columns)
    for name, parsed in added.items():
        if name not in produced:
            continue
        roots = _through(parent, tuple(root for item in parsed.items for root in item.roots))
        if roots:
            resolved[name] = roots
        else:
            resolved.pop(name, None)
    return resolved


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
    suffix: str = "",
) -> Lineage:
    """Lineage for a join, matched by name from whichever side has it.

    DuckDB keeps both columns of a colliding name rather than suffixing one,
    unless a suffix was asked for, so a collision is reported as a duplicate
    column by `describe` and gets the lineage of the left-hand side here. That
    is the honest half of an ambiguous situation: the name resolves to the left
    column everywhere else too.
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
    """Lineage for a set operation: a column comes from every side that has one."""
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


def of_query(sql: str, columns: Sequence[str]) -> Lineage:
    """Lineage for a query this adapter did not watch being built.

    Resolvable only when the projection reads base tables directly; see the
    module docstring for why a subquery is left untraced rather than guessed
    through.
    """
    parsed = parse_query(sql)
    if not parsed.resolvable or not _only_base_tables(parsed):
        return {}
    return positional(None, columns, parsed) or {}


def expression_text(value: Any) -> str:
    """The SQL text of one relational-API argument, or an empty string.

    DuckDB takes either a string or an `Expression`, and `str()` of an
    expression is the SQL it stands for. An alias does not survive that
    conversion, which costs nothing here: the output name comes from the
    resolved relation, and only the roots are read off the expression.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    text = str(value)
    return text if text and not text.startswith("<") else ""


def _item(entry: Mapping[str, Any]) -> SelectItem:
    """Read one select-list entry: what it reads, and whether it is a star."""
    if str(entry.get("class", "")) in STAR_CLASSES:
        return SelectItem(roots=(), is_star=True)
    return SelectItem(roots=_column_refs(entry))


def _column_refs(node: Any) -> tuple[str, ...]:
    """Every column name an expression reads, deduplicated, in order.

    Walks the whole expression, including function arguments, cast operands and
    case branches. A qualified reference (`orders.amount`) contributes its last
    part, because that is the name the schema knows it by.
    """
    found: list[str] = []
    _collect_refs(node, found)
    return tuple(dict.fromkeys(found))


def _collect_refs(node: Any, found: list[str]) -> None:
    """Depth-first walk collecting `COLUMN_REF` names into `found`."""
    if isinstance(node, dict):
        if str(node.get("class", "")) == "COLUMN_REF":
            names = node.get("column_names") or []
            if names:
                found.append(str(names[-1]))
            return
        for value in node.values():
            _collect_refs(value, found)
        return
    if isinstance(node, list):
        for value in node:
            _collect_refs(value, found)


def _sources(node: Mapping[str, Any]) -> tuple[str, ...]:
    """Base table names the query reads, and the kinds of source it reads.

    The kinds travel with the names so `_only_base_tables` can refuse a query
    whose FROM clause is a subquery, without this function having to decide.
    """
    found: list[str] = []
    _collect_sources(node.get("from_table"), found)
    return tuple(found)


def _collect_sources(node: Any, found: list[str]) -> None:
    """Walk a FROM clause, recording `kind:name` for every source in it."""
    if isinstance(node, dict):
        kind = str(node.get("type", ""))
        if kind:
            found.append(f"{kind}:{node.get('table_name') or ''}")
        for key in ("left", "right", "child", "subquery"):
            if key in node:
                _collect_sources(node[key], found)
        return
    if isinstance(node, list):  # pragma: no cover - DuckDB nests joins pairwise
        for value in node:
            _collect_sources(value, found)


def _only_base_tables(parsed: SelectList) -> bool:
    """True when every source in the FROM clause is a table, not a subquery."""
    if not parsed.sources:
        return False
    return all(source.partition(":")[0] in RESOLVABLE_SOURCES for source in parsed.sources)


def _through(
    parent: Mapping[str, tuple[str, ...]] | None,
    roots: Sequence[str],
) -> tuple[str, ...]:
    """Map expression roots onto input columns through the parent's lineage.

    With no parent lineage the roots are taken to be input column names, which
    is true for a query over registered relations and false for a chained
    relation - which is why the chained path always has a parent.
    """
    if parent is None:
        return tuple(dict.fromkeys(roots))
    return tuple(dict.fromkeys(source for root in roots for source in parent.get(root, ())))

"""What the buffer says, without running anything.

Every editor feature needs the same two questions answered about a cursor
position: which pipeline is it in, and is it sitting on a column name. A dry
run cannot answer either - it knows schemas, not source locations - so this
module parses the buffer and indexes the places a column name can appear.

This is deliberately *not* schema inference. The index records that a string
literal is in a column-reading position; what columns exist comes from a
resolved schema, never from the source. The distinction matters: the proposal
rules out static schema inference because it breaks on real code, and an index
that claims "this literal names a column" is a far weaker claim than "this
frame has these columns".

The rule throughout is that an unrecognised shape contributes nothing. A
literal inside a call this module does not know is not indexed, so it gets no
diagnostic, no completion and no hover - silence rather than a guess.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from rowless.analysis import analyse_source
from rowless.discovery.plain import (
    FRAME_ANNOTATIONS,
    decorator_calls,
    dotted_name,
    import_aliases,
    keyword_constant,
)

#: Methods and functions whose positional string arguments name columns to
#: read. Polars-shaped, with the pandas spellings beside them: both engines
#: take bare column names positionally, and the ones that do not appear here
#: are the ones where a string means something else.
READ_SELECTORS: frozenset[str] = frozenset(
    {
        "col",
        "select",
        "drop",
        "group_by",
        "groupby",
        "sort",
        "sort_values",
        "unique",
        "drop_duplicates",
        "agg",
        "sum",
        "mean",
        "min",
        "max",
        "count",
        "first",
        "last",
        "exclude",
        "nth",
    }
)

#: Keyword arguments whose value names columns to read. `on=`, `left_on=` and
#: `right_on=` are the join keys, which is where a rename does its damage.
READ_KEYWORDS: frozenset[str] = frozenset(
    {"on", "left_on", "right_on", "by", "subset", "index", "columns_from", "key"}
)

#: Keywords whose column could belong to either side of the operation. A join
#: key is not attributed to the frame the chain started from, because
#: `orders.join(users, left_on="customer_id")` names a column of `users` as
#: often as one of `orders`, and attributing it to the wrong schema would
#: report a column that exists as missing.
AMBIGUOUS_KEYWORDS: frozenset[str] = frozenset({"on", "left_on", "right_on", "key"})

#: Calls that introduce a column name rather than read one.
CREATE_SELECTORS: frozenset[str] = frozenset({"alias", "name"})

#: Calls whose keyword arguments name new columns: `with_columns(day=...)`,
#: `assign(day=...)`, `agg(total=...)`.
CREATE_KEYWORD_CALLS: frozenset[str] = frozenset({"with_columns", "assign", "agg", "select"})

#: `rename({"old": "new"})` reads its keys and creates its values, which makes
#: it the one call where both sides of a mapping matter.
RENAME_SELECTORS: frozenset[str] = frozenset({"rename"})


class Use(StrEnum):
    """Whether a column literal reads an existing column or names a new one."""

    READ = "read"
    CREATE = "create"


@dataclass(frozen=True, slots=True)
class Span:
    """A one-line range in the buffer.

    Lines are 1-based to match `rowless.ir.Origin`; characters are 0-based to
    match LSP. Converting at the protocol edge keeps the rest of this package
    speaking rowless's coordinates rather than two systems at once.
    """

    line: int
    start: int
    end: int

    def contains(self, line: int, character: int) -> bool:
        """True when a cursor at `(line, character)` is inside this span.

        Inclusive of the end, because a cursor sitting just past the last
        character of `"amount"` is still on `amount` as far as a user is
        concerned, and that is where the cursor is while typing.
        """
        return line == self.line and self.start <= character <= self.end


@dataclass(frozen=True, slots=True)
class ColumnUse:
    """One string literal in a position where a column name belongs."""

    name: str
    span: Span
    use: Use

    function: str | None = None
    """Enclosing top-level function, when the literal is inside one."""

    frame: str | None = None
    """The parameter the call chain is rooted in, when that is resolvable.

    `orders.select("amount")` resolves to `orders`; `pl.col("amount")` does
    not, because the chain is rooted in a module. An unresolved frame is not a
    failure - a single-input pipeline has only one schema it could mean - but
    it is the difference between checking a name against one schema and
    against the union of several.
    """


@dataclass(frozen=True, slots=True)
class IndexedFunction:
    """A top-level function in the buffer, whether rowless checks it or not."""

    name: str
    line: int
    end_line: int
    name_span: Span

    params: tuple[str, ...] = ()
    param_spans: tuple[Span, ...] = ()

    returns: tuple[Span, ...] = ()
    """End of each `return` expression: where an inlay hint goes."""

    created: frozenset[str] = frozenset()
    """Column names introduced inside the function. A literal read after one
    of these is not a missing column, so these keep the live check quiet about
    columns the pipeline itself produced."""

    dynamic_lines: frozenset[int] = frozenset()
    """Lines holding a computed column name. Any of these in a function means
    its column set is partly unknown, so the live check says nothing about it
    rather than reporting a name it cannot see."""

    checked: bool = False
    """True when discovery would pick this function up: an engine-naming
    annotation, or an `@rowless.pipeline` marker. False is worth surfacing -
    "rowless is not checking this" is the question a new user actually has."""

    engine: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentIndex:
    """One buffer, indexed. Pure: text in, positions out."""

    path: str
    lines: tuple[str, ...]
    functions: tuple[IndexedFunction, ...] = ()
    columns: tuple[ColumnUse, ...] = ()
    parsed: bool = True
    """False when the buffer does not parse, which is the normal state of a
    file being typed in. Everything else is then empty, and the callers fall
    back to the last index that did parse."""

    @classmethod
    def of(cls, path: str, text: str) -> DocumentIndex:
        """Index a buffer. Never raises: a half-typed file is the common case."""
        lines = tuple(text.splitlines())
        try:
            tree = ast.parse(text, filename=path)
        except (SyntaxError, ValueError):
            return cls(path=path, lines=lines, parsed=False)
        dynamic = _dynamic_lines(text, path)
        functions: list[IndexedFunction] = []
        columns: list[ColumnUse] = []
        aliases = import_aliases(tree)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            found = list(_columns_in(node, params=_params(node)))
            functions.append(_function(node, aliases, found, dynamic))
            columns.extend(found)
        return cls(
            path=path,
            lines=lines,
            functions=tuple(functions),
            columns=tuple(columns),
        )

    def function_at(self, line: int) -> IndexedFunction | None:
        """The function containing `line`, if any. 1-based, like `Origin`."""
        for function in self.functions:
            if function.line <= line <= function.end_line:
                return function
        return None

    def function_named(self, name: str) -> IndexedFunction | None:
        """One function by name."""
        for function in self.functions:
            if function.name == name:
                return function
        return None

    def column_at(self, line: int, character: int) -> ColumnUse | None:
        """The column literal under a cursor, if it is on one."""
        for column in self.columns:
            if column.span.contains(line, character):
                return column
        return None

    def reads_in(self, function: str) -> tuple[ColumnUse, ...]:
        """Every read of an existing column inside one function."""
        return tuple(
            column
            for column in self.columns
            if column.function == function and column.use is Use.READ
        )

    def line_text(self, line: int) -> str:
        """One 1-based line, or an empty string when it is out of range."""
        if 1 <= line <= len(self.lines):
            return self.lines[line - 1]
        return ""


def _function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: dict[str, str],
    columns: list[ColumnUse],
    dynamic: frozenset[int],
) -> IndexedFunction:
    """Build the record for one top-level function."""
    params = _params(node)
    marker, engine = _marker_and_engine(node, aliases)
    body_lines = range(node.lineno, (node.end_lineno or node.lineno) + 1)
    return IndexedFunction(
        name=node.name,
        line=node.lineno,
        end_line=node.end_lineno or node.lineno,
        # `def ` is four characters, and the name follows it on the same line.
        name_span=Span(
            line=node.lineno,
            start=node.col_offset + len("def "),
            end=node.col_offset + len("def ") + len(node.name),
        ),
        params=params,
        param_spans=tuple(
            Span(
                line=argument.lineno,
                start=argument.col_offset,
                end=argument.col_offset + len(argument.arg),
            )
            for argument in node.args.args
        ),
        returns=tuple(_return_spans(node)),
        created=frozenset(column.name for column in columns if column.use is Use.CREATE),
        dynamic_lines=frozenset(line for line in dynamic if line in body_lines),
        checked=marker or engine is not None,
        engine=engine,
    )


def _marker_and_engine(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: dict[str, str],
) -> tuple[bool, str | None]:
    """Whether discovery would claim this function, and for which engine.

    The same two rules as `rowless.discovery.plain`, read off the same helpers
    rather than re-derived: an `@rowless.pipeline` marker, or an annotation
    naming a frame type. Keeping it to those helpers means an adapter that
    adds a frame type is understood here too, without a second list to update.
    """
    marker = False
    engine: str | None = None
    for name, call in decorator_calls(node, aliases):
        if name in {"rowless.pipeline", "rowless.api.pipeline"}:
            marker = True
            engine = engine or keyword_constant(call, "engine")
    if engine is None:
        annotations = [node.returns, *(argument.annotation for argument in node.args.args)]
        for annotation in annotations:
            if annotation is None:
                continue
            found = FRAME_ANNOTATIONS.get(dotted_name(annotation, aliases))
            if found is not None:
                engine = found
                break
    return marker, engine


def _params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    """Positional parameter names, which is what a dry run binds frames to."""
    return tuple(argument.arg for argument in node.args.args)


def _return_spans(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[Span]:
    """The end of every `return` expression in a function.

    Where the resolved schema goes as an inlay hint: the end of the chain is
    the place a developer is looking when they want to know what came out.
    """
    for statement in ast.walk(node):
        if not isinstance(statement, ast.Return) or statement.value is None:
            continue
        value = statement.value
        if value.end_lineno is None or value.end_col_offset is None:
            continue
        yield Span(line=value.end_lineno, start=value.end_col_offset, end=value.end_col_offset)


def _dynamic_lines(text: str, path: str) -> frozenset[int]:
    """Lines where a column name is computed, from core's own static pass.

    Reusing `rowless.analysis` rather than re-walking for this: a user who
    reads `RWL203` in the terminal and sees the same line go quiet in the
    editor is seeing one rule, not two that happen to agree.
    """
    return frozenset(
        finding.line for finding in analyse_source(text, path) if finding.kind == "dynamic_column"
    )


def _columns_in(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    params: tuple[str, ...],
) -> Iterator[ColumnUse]:
    """Every column literal inside one function, with its position and role."""
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript):
            yield from _subscript_columns(child, node.name, params)
        elif isinstance(child, ast.Call):
            yield from _call_columns(child, node.name, params)


def _subscript_columns(
    node: ast.Subscript,
    function: str,
    params: tuple[str, ...],
) -> Iterator[ColumnUse]:
    """`df["amount"]` and `df[["a", "b"]]`: the one spelling every engine shares."""
    frame = _frame_root(node.value, params)
    for literal in _literals(node.slice):
        yield _use(literal, Use.READ, function, frame)


def _call_columns(
    node: ast.Call,
    function: str,
    params: tuple[str, ...],
) -> Iterator[ColumnUse]:
    """Column literals in one call, whichever of the shapes it is."""
    selector = _called_name(node)
    frame = _frame_root(node.func, params)

    if selector in RENAME_SELECTORS:
        yield from _rename_columns(node, function, frame)
        return
    if selector in CREATE_SELECTORS:
        for literal in node.args[:1]:
            for found in _literals(literal):
                yield _use(found, Use.CREATE, function, frame)
        return

    if selector in READ_SELECTORS:
        for argument in node.args:
            for literal in _literals(argument):
                yield _use(literal, Use.READ, function, frame)
    for keyword in node.keywords:
        if keyword.arg is None:
            continue
        if keyword.arg in READ_KEYWORDS:
            attributed = None if keyword.arg in AMBIGUOUS_KEYWORDS else frame
            for literal in _literals(keyword.value):
                yield _use(literal, Use.READ, function, attributed)
        elif selector in CREATE_KEYWORD_CALLS:
            # `with_columns(day=...)` names the column in the keyword itself,
            # so there is no literal to point at. The name still has to be
            # recorded or the live check reports every later read of it.
            yield ColumnUse(
                name=keyword.arg,
                span=Span(
                    line=keyword.value.lineno,
                    start=keyword.value.col_offset,
                    end=keyword.value.col_offset,
                ),
                use=Use.CREATE,
                function=function,
                frame=frame,
            )


def _rename_columns(node: ast.Call, function: str, frame: str | None) -> Iterator[ColumnUse]:
    """`rename({"old": "new"})`: keys read, values create.

    Both halves matter and they mean opposite things, which is why this is the
    one call with its own branch. A rename is also the change the whole project
    exists to catch, so getting the key side indexed is the point.
    """
    for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
        if not isinstance(argument, ast.Dict):
            continue
        for key, value in zip(argument.keys, argument.values, strict=True):
            if key is not None:
                for literal in _literals(key):
                    yield _use(literal, Use.READ, function, frame)
            for literal in _literals(value):
                yield _use(literal, Use.CREATE, function, frame)


def _use(node: ast.Constant, use: Use, function: str, frame: str | None) -> ColumnUse:
    """A literal plus its role, as one indexed use."""
    return ColumnUse(
        name=str(node.value),
        span=_span(node),
        use=use,
        function=function,
        frame=frame,
    )


def _span(node: ast.Constant) -> Span:
    """The span of a string literal's *text*, inside its quotes.

    Inside, because the span is what a code action replaces and what a
    completion overwrites: including the quotes would make a fix delete them.
    """
    end = node.end_col_offset if node.end_col_offset is not None else node.col_offset
    return Span(line=node.lineno, start=node.col_offset + 1, end=max(end - 1, node.col_offset + 1))


def _literals(node: ast.expr) -> Iterator[ast.Constant]:
    """String constants reachable as column names from one expression.

    Follows lists, tuples and sets, because every engine accepts a list of
    names where it accepts one. Does not follow anything computed: that is
    what makes an f-string contribute nothing instead of contributing a guess.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            yield node
        return
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        for element in node.elts:
            yield from _literals(element)


def _called_name(node: ast.Call) -> str:
    """The called name, whether it was reached as a method or a function."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return getattr(func, "id", "")


def _frame_root(node: ast.expr, params: tuple[str, ...]) -> str | None:
    """The parameter a call chain is rooted in, when it is rooted in one.

    `orders.filter(...).select("amount")` roots in `orders`, so the literal is
    checked against that input's schema. `pl.col("amount")` roots in a module,
    and the answer is None: the honest reading is "some frame in this
    function", which the caller resolves by taking every input schema it has.
    """
    current = node
    while True:
        if isinstance(current, ast.Attribute):
            current = current.value
        elif isinstance(current, ast.Call):
            current = current.func
        elif isinstance(current, ast.Subscript):
            current = current.value
        else:
            break
    if isinstance(current, ast.Name) and current.id in params:
        return current.id
    return None

"""Completion, hover, inlay hints, code actions and the protocol conversion.

All of these read from a session that has already checked something, so the
fixtures build one directly rather than going through the server: what is
under test is what the editor is shown, not how the request arrived.
"""

from __future__ import annotations

import pytest
from lsprotocol import types
from rowless.diagnostics.codes import Code
from rowless.diagnostics.models import Diagnostic, Level
from rowless.ir.confidence import Confidence, Coverage
from rowless.ir.dtypes import datetime, int_, string
from rowless.ir.schema import Field, Origin, Schema
from rowless.lineage.graph import ColumnRef, Edge, Evidence, LineageGraph, Node, NodeKind
from rowless_lsp.actions import code_actions, with_ignore
from rowless_lsp.completion import complete, trigger_at
from rowless_lsp.convert import group_by_file, range_for, to_lsp
from rowless_lsp.documents import DocumentIndex
from rowless_lsp.hints import inlay_hints
from rowless_lsp.hover import hover
from rowless_lsp.session import Checked, Document, Resolved

ORDERS = Schema.from_mapping(
    {
        "order_id": int_(64),
        "status": string(),
        "amount_cents": int_(64),
        "created_at": datetime(),
    },
    name="orders",
)

OUTPUT = Schema(
    fields=(
        Field(name="day", dtype=datetime(), derived_from=("created_at",)),
        Field(name="total", dtype=int_(64), derived_from=("amount_cents",)),
    )
)

SOURCE = (
    "import polars as pl\n"
    "import rowless\n"
    "\n"
    "\n"
    '@rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="polars")\n'
    "def daily_revenue(orders: pl.LazyFrame) -> pl.LazyFrame:\n"
    '    return orders.select("amount")\n'
)

TARGET = "revenue:daily_revenue"


def _graph() -> LineageGraph:
    """A two-node graph with traced edges, as a real check produces."""
    graph = LineageGraph()
    graph.add_node(Node(id="lake://stg_orders", kind=NodeKind.SOURCE, schema=ORDERS))
    graph.add_node(Node(id=TARGET, kind=NodeKind.PYTHON, schema=OUTPUT, engine="polars"))
    graph.add_edge(
        Edge(
            source=ColumnRef("lake://stg_orders", "amount_cents"),
            target=ColumnRef(TARGET, "total"),
            evidence=Evidence.PLAN,
            transform="sum",
        )
    )
    return graph


@pytest.fixture
def document() -> Document:
    """A checked document: one pipeline, one resolved input, one output."""
    found = Document(path="revenue.py", text=SOURCE, version=1)
    found.index = DocumentIndex.of("revenue.py", SOURCE)
    found.parsed = found.index
    found.checked = Checked(
        report=_report(),  # type: ignore[arg-type]
        resolved={
            "daily_revenue": Resolved(
                inputs={"orders": ORDERS},
                uris={"orders": "lake://stg_orders"},
                target=TARGET,
                output=OUTPUT,
                coverage=Coverage.score(OUTPUT, confidence=Confidence.EXACT),
            )
        },
        graph=_graph(),
    )
    return found


def _report() -> object:
    """An empty report: these tests do not read it, only the schemas."""
    from rowless.diagnostics.models import Report

    return Report()


def at(line: int, character: int) -> types.Position:
    """An LSP position from a 1-based line, as the source is numbered."""
    return types.Position(line=line - 1, character=character)


# --- Narrowing helpers -------------------------------------------------------
#
# The protocol allows several shapes for a hover's contents, a completion's
# edit and a hint's label. This server only ever sends one of each, so these
# assert that rather than accommodating the others.


def popup(found: types.Hover | None) -> str:
    """A hover's markdown."""
    assert found is not None
    assert isinstance(found.contents, types.MarkupContent)
    return found.contents.value


def documentation(item: types.CompletionItem) -> str:
    """A completion item's markdown documentation."""
    assert isinstance(item.documentation, types.MarkupContent)
    return item.documentation.value


def replacement(item: types.CompletionItem) -> types.TextEdit:
    """A completion item's edit, which is always a plain replacement."""
    assert isinstance(item.text_edit, types.TextEdit)
    return item.text_edit


def label(hint: types.InlayHint) -> str:
    """An inlay hint's label."""
    assert isinstance(hint.label, str)
    return hint.label


def tooltip(hint: types.InlayHint) -> str:
    """An inlay hint's tooltip."""
    assert isinstance(hint.tooltip, types.MarkupContent)
    return hint.tooltip.value


def only_edit(action: types.CodeAction) -> types.TextEdit:
    """The single text edit a code action carries, for the document it names."""
    assert action.edit is not None
    assert action.edit.changes is not None
    return next(iter(action.edit.changes.values()))[0]


# --- Completion --------------------------------------------------------------


def test_completion_offers_the_resolved_input_columns(document: Document) -> None:
    """Inside `select("...")`, the schema's columns and nothing else."""
    line = SOURCE.splitlines()[6]
    items = complete(document, at(7, line.index("amount")))

    assert [item.label for item in items] == [
        "order_id",
        "status",
        "amount_cents",
        "created_at",
        "day",
        "total",
    ]


def test_completion_carries_the_dtype_and_the_source(document: Document) -> None:
    """A name on its own is half an answer: the type is why you picked it."""
    line = SOURCE.splitlines()[6]
    items = {item.label: item for item in complete(document, at(7, line.index("amount")))}

    assert items["amount_cents"].detail == "int64"
    assert "lake://stg_orders" in documentation(items["amount_cents"])


def test_completion_replaces_the_whole_literal(document: Document) -> None:
    """Not an append: completing over `amount` must not leave `amountamount`."""
    line = SOURCE.splitlines()[6]
    items = complete(document, at(7, line.index("amount") + 3))
    edit = replacement(items[0])

    assert edit.range.start.character == line.index("amount")
    assert edit.range.end.character == line.index("amount") + len("amount")


def test_completion_is_silent_outside_a_column_position(document: Document) -> None:
    """A cursor in ordinary code gets no column names."""
    assert complete(document, at(7, 4)) == []


def test_completion_is_silent_where_a_new_name_is_written() -> None:
    """`alias("...")` names a column that does not exist yet, by design."""
    trigger = trigger_at('    return orders.select(pl.col("a").alias("tot', 46)

    assert trigger is not None
    assert trigger.reads is False


def test_completion_is_silent_before_the_first_check() -> None:
    """Nothing resolved means nothing to offer. Silence, not a guess."""
    found = Document(path="revenue.py", text=SOURCE)
    found.index = DocumentIndex.of("revenue.py", SOURCE)
    line = SOURCE.splitlines()[6]

    assert complete(found, at(7, line.index("amount"))) == []


def test_completion_survives_a_buffer_that_does_not_parse(document: Document) -> None:
    """The state a file is in while the literal is being typed.

    The index falls back to the last one that parsed, which is the whole
    reason the trigger is detected from line text rather than from the tree.
    """
    half_typed = SOURCE.replace('orders.select("amount")', 'orders.select("am')
    document.replace(half_typed, version=2)

    assert document.index.parsed is False
    items = complete(document, at(7, len(half_typed.splitlines()[6])))
    assert "amount_cents" in [item.label for item in items]


# --- Hover -------------------------------------------------------------------


def test_hover_on_a_column_shows_its_type_and_origin(document: Document) -> None:
    """The question a hover is for."""
    line = SOURCE.splitlines()[6]
    assert "not in any resolved input schema" in popup(
        hover(document, at(7, line.index("amount") + 1))
    )


def test_hover_on_a_resolved_column_shows_the_dtype() -> None:
    """A column that does exist: type, nullability, and where it came from."""
    source = SOURCE.replace('"amount"', '"amount_cents"')
    found = Document(path="revenue.py", text=source)
    found.index = DocumentIndex.of("revenue.py", source)
    found.parsed = found.index
    found.checked = Checked(
        report=_report(),  # type: ignore[arg-type]
        resolved={
            "daily_revenue": Resolved(
                inputs={"orders": ORDERS},
                uris={"orders": "lake://stg_orders"},
                target=TARGET,
                output=OUTPUT,
            )
        },
        graph=_graph(),
    )
    line = source.splitlines()[6]
    contents = popup(hover(found, at(7, line.index("amount_cents") + 1)))

    assert "`amount_cents`: `int64`" in contents
    assert "lake://stg_orders" in contents


def test_hover_on_a_parameter_shows_the_whole_input_schema(document: Document) -> None:
    """What does this input actually contain: the other question."""
    line = SOURCE.splitlines()[5]
    contents = popup(hover(document, at(6, line.index("orders") + 1)))

    assert "lake://stg_orders" in contents
    assert "`amount_cents`" in contents


def test_hover_on_the_pipeline_name_reports_confidence(document: Document) -> None:
    """The honesty surface, saying the same thing `rowless explain` says."""
    line = SOURCE.splitlines()[5]
    contents = popup(hover(document, at(6, line.index("daily_revenue") + 1)))

    assert "confidence **exact**" in contents
    assert "`total`" in contents


def test_hover_on_an_unchecked_function_says_why() -> None:
    """ "Why is rowless ignoring this" is a question the server can answer."""
    source = "def helper(frame):\n    return frame\n"
    found = Document(path="revenue.py", text=source)
    found.index = DocumentIndex.of("revenue.py", source)
    found.parsed = found.index
    contents = popup(hover(found, at(1, 5)))

    assert "does not check" in contents
    assert "@rowless.pipeline" in contents


def test_hover_shows_lineage_with_its_evidence() -> None:
    """A guess and a plan walk must not read identically."""
    source = SOURCE.replace('"amount"', '"total"')
    found = Document(path="revenue.py", text=source)
    found.index = DocumentIndex.of("revenue.py", source)
    found.parsed = found.index
    found.checked = Checked(
        report=_report(),  # type: ignore[arg-type]
        resolved={
            "daily_revenue": Resolved(
                inputs={"orders": ORDERS},
                uris={"orders": "lake://stg_orders"},
                target=TARGET,
                output=OUTPUT,
            )
        },
        graph=_graph(),
    )
    line = source.splitlines()[6]
    contents = popup(hover(found, at(7, line.index("total") + 1)))

    assert "lake://stg_orders.amount_cents" in contents
    assert "(plan via sum)" in contents


# --- Inlay hints -------------------------------------------------------------


def test_a_return_gets_the_resolved_schema(document: Document) -> None:
    """The schema nobody can read off the source, put where the chain ends."""
    hints = inlay_hints(document)
    returns = [hint for hint in hints if hint.kind is types.InlayHintKind.Type]

    assert [label(hint) for hint in returns] == ["-> day: datetime[us], total: int64"]


def test_a_parameter_gets_its_source_and_width(document: Document) -> None:
    """Which table feeds which parameter, without opening the config."""
    hints = inlay_hints(document)
    params = [hint for hint in hints if hint.kind is types.InlayHintKind.Parameter]

    assert [label(hint) for hint in params] == ["lake://stg_orders (4 cols)"]


def test_an_opaque_result_is_labelled_unverified(document: Document) -> None:
    """A missing hint reads as "no information"; this project's point is that
    "rowless does not know" is information."""
    document.checked = Checked(
        report=_report(),  # type: ignore[arg-type]
        resolved={
            "daily_revenue": Resolved(
                inputs={"orders": ORDERS},
                uris={"orders": "lake://stg_orders"},
                target=TARGET,
                output=Schema.empty(),
                opaque=True,
            )
        },
    )
    returns = [hint for hint in inlay_hints(document) if hint.kind is types.InlayHintKind.Type]

    assert [label(hint) for hint in returns] == ["-> unverified"]


def test_a_wide_schema_is_truncated_with_the_rest_in_the_tooltip() -> None:
    """A hint longer than the line it annotates is not read."""
    wide = Schema.from_mapping({f"c{index}": int_(64) for index in range(12)})
    found = Document(path="revenue.py", text=SOURCE)
    found.index = DocumentIndex.of("revenue.py", SOURCE)
    found.parsed = found.index
    found.checked = Checked(
        report=_report(),  # type: ignore[arg-type]
        resolved={"daily_revenue": Resolved(inputs={"orders": ORDERS}, output=wide)},
    )
    hint = next(hint for hint in inlay_hints(found) if hint.kind is types.InlayHintKind.Type)

    assert label(hint).endswith("+8 more")
    assert "`c11`" in tooltip(hint)


def test_hints_are_limited_to_the_requested_range(document: Document) -> None:
    """The client asks for the visible region, not the file."""
    away = types.Range(
        start=types.Position(line=200, character=0),
        end=types.Position(line=220, character=0),
    )

    assert inlay_hints(document, away) == []


# --- Code actions ------------------------------------------------------------


def _finding(line: int, message: str, suggestion: str | None) -> Diagnostic:
    """One rowless finding located in the fixture document."""
    return Diagnostic(
        code=Code.COLUMN_NOT_FOUND,
        level=Level.ERROR,
        message=message,
        origin=Origin(kind="python", ref=TARGET, file="revenue.py", line=line),
        suggestion=suggestion,
    )


def test_a_near_miss_offers_the_name_that_exists(document: Document) -> None:
    """The rename case the proposal opens with, as one keystroke."""
    finding = _finding(7, "ColumnNotFound 'amount'", "did you mean 'amount_cents'?")
    whole = types.Range(
        start=types.Position(line=6, character=0),
        end=types.Position(line=6, character=40),
    )
    actions = code_actions(document, [finding], whole)
    fix = next(action for action in actions if action.title.startswith("Change"))

    assert fix.title == "Change 'amount' to 'amount_cents'"
    assert fix.is_preferred is True
    edit = only_edit(fix)
    assert edit.new_text == "amount_cents"
    line = SOURCE.splitlines()[6]
    assert edit.range.start.character == line.index("amount")
    assert edit.range.end.character == line.index("amount") + len("amount")


def test_the_fix_falls_back_to_the_resolved_schema(document: Document) -> None:
    """A finding with no suggestion still has a schema behind it."""
    finding = _finding(7, "ColumnNotFound 'amount'", None)
    whole = types.Range(
        start=types.Position(line=6, character=0),
        end=types.Position(line=6, character=40),
    )
    actions = code_actions(document, [finding], whole)

    assert any(action.title == "Change 'amount' to 'amount_cents'" for action in actions)


def test_every_finding_offers_an_ignore_comment(document: Document) -> None:
    """A checker with no way to say "not now" gets deleted rather than configured."""
    finding = _finding(7, "ColumnNotFound 'amount'", None)
    whole = types.Range(
        start=types.Position(line=6, character=0),
        end=types.Position(line=6, character=40),
    )
    actions = code_actions(document, [finding], whole)
    ignore = next(action for action in actions if action.title.startswith("Ignore"))

    assert ignore.title == "Ignore RWL101 on this line"
    assert only_edit(ignore).new_text.endswith("# rowless: ignore[RWL101]")


def test_an_existing_ignore_comment_is_extended_not_duplicated() -> None:
    """Two comments on a line are not what the parser reads."""
    line = '    return orders.select("amount")  # rowless: ignore[RWL201]'

    assert with_ignore(line, Code.COLUMN_NOT_FOUND).endswith("ignore[RWL201,RWL101]")
    assert with_ignore(line, Code.ROW_DEPENDENT_CONTROL_FLOW) == line


def test_actions_outside_the_range_are_not_offered(document: Document) -> None:
    """The client asks about the cursor, not the file."""
    finding = _finding(7, "ColumnNotFound 'amount'", None)
    elsewhere = types.Range(
        start=types.Position(line=0, character=0),
        end=types.Position(line=1, character=0),
    )

    assert code_actions(document, [finding], elsewhere) == []


# --- Conversion --------------------------------------------------------------


def test_a_finding_becomes_a_diagnostic_with_its_code(document: Document) -> None:
    """The code is public API: it goes in ignore comments and CI allowlists."""
    finding = _finding(7, "ColumnNotFound 'amount'", "did you mean 'amount_cents'?")
    converted = to_lsp(finding, document.usable)

    assert converted.code == "RWL101"
    assert converted.source == "rowless"
    assert converted.severity is types.DiagnosticSeverity.Error
    assert "did you mean 'amount_cents'?" in converted.message


def test_an_unverified_finding_is_a_warning_carrying_its_code() -> None:
    """An error squiggle for "rowless does not know" is how a server gets
    switched off; the 2xx code keeps the distinction visible."""
    finding = Diagnostic(
        code=Code.ROW_DEPENDENT_CONTROL_FLOW,
        level=Level.UNVERIFIED,
        message="row-dependent call",
        origin=Origin(kind="python", ref=TARGET, file="revenue.py", line=7),
    )
    converted = to_lsp(finding)

    assert converted.severity is types.DiagnosticSeverity.Warning
    assert converted.code == "RWL201"
    assert isinstance(converted.data, dict)
    assert converted.data["unverified"] is True


def test_the_squiggle_lands_under_the_column_not_the_line(document: Document) -> None:
    """A whole-line squiggle on a 90-character chain says nothing useful."""
    finding = _finding(7, "ColumnNotFound 'amount'", None)
    found = range_for(finding, document.usable)
    line = SOURCE.splitlines()[6]

    assert found.start.character == line.index("amount")
    assert found.end.character == line.index("amount") + len("amount")


def test_a_finding_with_no_line_still_lands_somewhere() -> None:
    """A coverage score has no line, and a diagnostic an editor cannot place
    is a diagnostic nobody reads."""
    finding = Diagnostic(
        code=Code.COVERAGE_BELOW_THRESHOLD,
        level=Level.WARNING,
        message="typed coverage 40% is below the configured floor 80%",
    )
    found = range_for(finding)

    assert found.start.line == 0


def test_related_locations_survive_the_conversion() -> None:
    """The cross-language payoff: the Python finding points at the SQL model."""
    finding = Diagnostic(
        code=Code.COLUMN_NOT_FOUND,
        level=Level.ERROR,
        message="ColumnNotFound 'amount'",
        origin=Origin(kind="python", ref=TARGET, file="revenue.py", line=7),
        related=(Origin(kind="sql", ref="stg_orders", file="models/stg_orders.sql", line=14),),
    )
    converted = to_lsp(finding)

    assert converted.related_information is not None
    assert len(converted.related_information) == 1
    assert converted.related_information[0].location.uri.endswith("stg_orders.sql")


def test_findings_are_grouped_by_the_file_they_belong_to() -> None:
    """A diagnostic published against the wrong URI is invisible."""
    python = Diagnostic(
        code=Code.COLUMN_NOT_FOUND,
        level=Level.ERROR,
        message="ColumnNotFound 'amount'",
        origin=Origin(kind="python", ref=TARGET, file="revenue.py", line=7),
    )
    sql = Diagnostic(
        code=Code.BREAKING_COLUMN_RENAMED,
        level=Level.ERROR,
        message="BREAKING stg_orders.amount -> amount_cents",
        origin=Origin(kind="sql", ref="stg_orders", file="models/stg_orders.sql", line=14),
    )
    grouped = group_by_file([python, sql])

    assert len(grouped) == 2
    assert all(len(found) == 1 for found in grouped.values())


def test_a_finding_with_no_file_goes_to_the_document_that_triggered_it() -> None:
    """Otherwise a report-level finding is dropped on the floor."""
    finding = Diagnostic(
        code=Code.COVERAGE_BELOW_THRESHOLD, level=Level.WARNING, message="below floor"
    )
    grouped = group_by_file([finding], fallback="revenue.py")

    assert sum(len(found) for found in grouped.values()) == 1


def test_completion_keeps_schema_order(document: Document) -> None:
    """A wide table's useful columns are the ones its author put first.

    Zero-padded sort keys, because a client sorts `sort_text` as a string and
    `10` would otherwise come before `2`.
    """
    line = SOURCE.splitlines()[6]
    items = complete(document, at(7, line.index("amount")))
    keys = [item.sort_text or "" for item in items]

    assert keys == sorted(keys)
    assert keys[:2] == ["0000", "0001"]

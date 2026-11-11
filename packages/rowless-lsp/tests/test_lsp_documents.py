"""The document index: which positions name columns, and which do not.

The index is the floor every other feature stands on, and its contract is
conservative: a shape it does not recognise contributes nothing. So the tests
that matter most are the negative ones - an f-string, a value that is not a
column name, a call nobody models - because a false entry here becomes a false
diagnostic, a wrong completion and a wrong hover all at once.
"""

from __future__ import annotations

from rowless_lsp.documents import DocumentIndex, Use


def index(source: str) -> DocumentIndex:
    """Index a snippet as `revenue.py`."""
    return DocumentIndex.of("revenue.py", source)


# --- Functions ---------------------------------------------------------------


def test_an_annotated_function_is_marked_checked() -> None:
    """The same rule discovery uses, so the editor agrees about what runs."""
    found = index(
        "import polars as pl\n"
        "\n"
        "def daily_revenue(orders: pl.LazyFrame) -> pl.LazyFrame:\n"
        "    return orders\n"
    )

    function = found.function_named("daily_revenue")
    assert function is not None
    assert function.checked is True
    assert function.engine == "polars"
    assert function.params == ("orders",)


def test_a_marker_without_annotations_is_also_checked() -> None:
    """`@rowless.pipeline` is the escape hatch, and it has to be honoured here."""
    found = index(
        "import rowless\n"
        "\n"
        '@rowless.pipeline(inputs={"orders": "lake://orders"}, engine="fake")\n'
        "def daily_revenue(orders):\n"
        "    return orders\n"
    )

    function = found.function_named("daily_revenue")
    assert function is not None
    assert function.checked is True
    assert function.engine == "fake"


def test_a_plain_function_is_indexed_but_not_checked() -> None:
    """Indexed so hover can say why rowless is ignoring it.

    "Why is rowless not checking my file" is the first question a new user
    has, and a server that returns nothing on hover cannot answer it.
    """
    found = index("def helper(frame):\n    return frame\n")

    function = found.function_named("helper")
    assert function is not None
    assert function.checked is False
    assert function.engine is None


def test_a_buffer_that_does_not_parse_indexes_nothing() -> None:
    """Half-typed code is the normal state, and it is not an error."""
    found = index('def daily_revenue(orders):\n    return orders.select("am\n')

    assert found.parsed is False
    assert found.functions == ()
    assert found.columns == ()


# --- Column positions --------------------------------------------------------


def test_positional_strings_in_a_selector_are_reads() -> None:
    """`select("a", "b")`: the plain case, and the one with a frame attached."""
    found = index(
        "import polars as pl\n"
        "\n"
        "def f(orders: pl.LazyFrame) -> pl.LazyFrame:\n"
        '    return orders.select("order_id", "amount")\n'
    )

    reads = found.reads_in("f")
    assert [(use.name, use.frame) for use in reads] == [
        ("order_id", "orders"),
        ("amount", "orders"),
    ]


def test_a_subscript_is_a_read_on_every_engine() -> None:
    """`df["amount"]` means the same thing in pandas and in Polars."""
    found = index(
        "import pandas as pd\n"
        "\n"
        "def f(orders: pd.DataFrame) -> pd.DataFrame:\n"
        '    return orders[["order_id", "amount"]]\n'
    )

    assert [use.name for use in found.reads_in("f")] == ["order_id", "amount"]


def test_a_join_key_is_a_read() -> None:
    """`on=` is where a rename upstream does its damage."""
    found = index(
        "import polars as pl\n"
        "\n"
        "def f(orders: pl.LazyFrame, users: pl.LazyFrame) -> pl.LazyFrame:\n"
        '    return orders.join(users, on="user_id")\n'
    )

    assert [use.name for use in found.reads_in("f")] == ["user_id"]


def test_rename_reads_its_keys_and_creates_its_values() -> None:
    """The flagship case: both halves of the mapping, meaning opposite things."""
    found = index(
        "import polars as pl\n"
        "\n"
        "def f(orders: pl.LazyFrame) -> pl.LazyFrame:\n"
        '    return orders.rename({"amount": "amount_cents"})\n'
    )

    by_use = {use.name: use.use for use in found.columns}
    assert by_use == {"amount": Use.READ, "amount_cents": Use.CREATE}


def test_an_alias_and_a_keyword_create_names() -> None:
    """A column the pipeline builds is never a missing column."""
    found = index(
        "import polars as pl\n"
        "\n"
        "def f(orders: pl.LazyFrame) -> pl.LazyFrame:\n"
        '    out = orders.with_columns(day=pl.col("created_at"))\n'
        '    return out.select(pl.col("day").alias("the_day"))\n'
    )

    function = found.function_named("f")
    assert function is not None
    assert function.created == frozenset({"day", "the_day"})
    assert {use.name for use in found.reads_in("f")} == {"created_at", "day"}


def test_a_chain_resolves_to_the_parameter_it_started_from() -> None:
    """Which schema a name is checked against depends on which frame it is."""
    found = index(
        "import polars as pl\n"
        "\n"
        "def f(orders: pl.LazyFrame, users: pl.LazyFrame) -> pl.LazyFrame:\n"
        '    return users.filter(users["active"]).select("name")\n'
    )

    assert {(use.name, use.frame) for use in found.reads_in("f")} == {
        ("active", "users"),
        ("name", "users"),
    }


def test_a_bare_column_expression_has_no_frame() -> None:
    """`pl.col("amount")` is rooted in a module, so it names no input.

    None rather than a guess at the only frame in scope: the caller decides
    what to do with an unattributed name, and for a multi-input pipeline the
    honest answer is "any of them".
    """
    found = index(
        "import polars as pl\n"
        "\n"
        "def f(orders: pl.LazyFrame) -> pl.LazyFrame:\n"
        '    return orders.filter(pl.col("status") == "paid")\n'
    )

    reads = found.reads_in("f")
    assert [(use.name, use.frame) for use in reads] == [("status", None)]


# --- What is deliberately not indexed ----------------------------------------


def test_a_computed_name_is_not_indexed_and_marks_the_function() -> None:
    """An f-string column is a gap, recorded as one rather than guessed at."""
    found = index(
        "import polars as pl\n"
        "\n"
        "def f(orders: pl.LazyFrame, currency: str) -> pl.LazyFrame:\n"
        '    return orders.select(f"amount_{currency}")\n'
    )

    assert found.reads_in("f") == ()
    function = found.function_named("f")
    assert function is not None
    assert function.dynamic_lines == frozenset({4})


def test_an_unrecognised_call_contributes_nothing() -> None:
    """Silence is the default for a shape this module does not model."""
    found = index(
        "import polars as pl\n"
        "\n"
        "def f(orders: pl.LazyFrame) -> pl.LazyFrame:\n"
        '    return orders.pipe(my_helper, "amount")\n'
    )

    assert found.columns == ()


def test_a_string_that_is_not_a_column_name_is_not_indexed() -> None:
    """A value compared against a column is not itself a column."""
    found = index(
        "import polars as pl\n"
        "\n"
        "def f(orders: pl.LazyFrame) -> pl.LazyFrame:\n"
        '    note = "paid in full"\n'
        '    return orders.select("status")\n'
    )

    assert [use.name for use in found.columns] == ["status"]
    assert "paid in full" not in {use.name for use in found.columns}


# --- Lookups -----------------------------------------------------------------


def test_a_cursor_resolves_to_the_literal_it_is_on() -> None:
    """Including one character past the end, which is where typing leaves it."""
    source = (
        "import polars as pl\n"
        "\n"
        "def f(orders: pl.LazyFrame) -> pl.LazyFrame:\n"
        '    return orders.select("amount")\n'
    )
    found = index(source)
    span = found.reads_in("f")[0].span

    assert found.column_at(span.line, span.start) is not None
    assert found.column_at(span.line, span.end) is not None
    assert found.column_at(span.line, span.start - 2) is None
    assert source.splitlines()[span.line - 1][span.start : span.end] == "amount"


def test_a_cursor_resolves_to_the_function_it_is_in() -> None:
    """Every feature starts by asking this."""
    found = index(
        "import polars as pl\n"
        "\n"
        "def first(orders: pl.LazyFrame) -> pl.LazyFrame:\n"
        "    return orders\n"
        "\n"
        "\n"
        "def second(orders: pl.LazyFrame) -> pl.LazyFrame:\n"
        "    return orders\n"
    )

    assert found.function_at(4) is not None
    assert found.function_at(4).name == "first"  # type: ignore[union-attr]
    assert found.function_at(8).name == "second"  # type: ignore[union-attr]
    assert found.function_at(1) is None


def test_the_function_name_span_covers_the_name() -> None:
    """Hovering a pipeline name is how its output schema is read."""
    source = "import polars as pl\n\ndef daily_revenue(orders: pl.LazyFrame):\n    return orders\n"
    found = index(source)
    function = found.function_named("daily_revenue")

    assert function is not None
    line = source.splitlines()[function.name_span.line - 1]
    assert line[function.name_span.start : function.name_span.end] == "daily_revenue"


def test_a_return_span_lands_at_the_end_of_the_chain() -> None:
    """Where the inlay hint goes: after the last call, not after `return`."""
    source = (
        "import polars as pl\n"
        "\n"
        "def f(orders: pl.LazyFrame) -> pl.LazyFrame:\n"
        '    return orders.select("amount")\n'
    )
    found = index(source)
    function = found.function_named("f")

    assert function is not None
    assert len(function.returns) == 1
    span = function.returns[0]
    assert source.splitlines()[span.line - 1][: span.end].endswith('select("amount")')

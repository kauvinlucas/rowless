"""The between-saves checks: what the buffer alone can prove.

Half of these assert a finding and half assert silence, which is the right
balance for this module. A live finding the next save contradicts is worse
than no live finding at all: the squiggle would appear while typing and vanish
on save, which teaches a user to stop believing the editor.
"""

from __future__ import annotations

from rowless.config import Config
from rowless.diagnostics.codes import Code
from rowless.diagnostics.models import Diagnostic, Level
from rowless.ir.dtypes import int_, string
from rowless.ir.schema import Schema
from rowless_lsp.documents import DocumentIndex
from rowless_lsp.live import live_diagnostics

ORDERS = Schema.from_mapping(
    {"order_id": int_(64), "status": string(), "amount_cents": int_(64)},
    name="orders",
)

USERS = Schema.from_mapping({"user_id": int_(64), "name": string()}, name="users")

HEADER = "import polars as pl\n\n"


def check(
    body: str,
    *,
    schemas: dict[str, dict[str, Schema]] | None = None,
    outputs: dict[str, Schema] | None = None,
    config: Config | None = None,
) -> tuple[Diagnostic, ...]:
    """Run the live checks over a one-function module."""
    source = HEADER + body
    index = DocumentIndex.of("revenue.py", source)
    return live_diagnostics(
        index,
        source,
        schemas=schemas if schemas is not None else {"f": {"orders": ORDERS}},
        outputs=outputs,
        config=config,
    )


ONE_INPUT = "def f(orders: pl.LazyFrame) -> pl.LazyFrame:\n"


# --- The finding -------------------------------------------------------------


def test_a_missing_column_is_reported_where_it_is_written() -> None:
    """The headline finding, before the file is saved.

    The proposal's own example: the source renamed `amount` to `amount_cents`
    and the pipeline still asks for the old name.
    """
    found = check(ONE_INPUT + '    return orders.select("amount")\n')

    assert len(found) == 1
    finding = found[0]
    assert finding.code is Code.COLUMN_NOT_FOUND
    assert finding.level is Level.ERROR
    assert finding.origin is not None
    assert finding.origin.line == 4
    # 1-based, like the line, and pointing at the `a` of the literal rather
    # than at the start of the statement.
    assert finding.origin.column == '    return orders.select("amount")'.index("amount") + 1
    assert finding.suggestion == "did you mean 'amount_cents'?"


def test_the_message_is_the_one_the_dry_run_will_print() -> None:
    """Word for word, because two spellings read as two different findings.

    The message is built by raising core's own `ColumnNotFound` against the
    resolved column set, which is the same object the dry run reports - so
    the near-miss cutoff is core's too, rather than a second one here.
    """
    found = check(ONE_INPUT + '    return orders.select("amount")\n')

    assert found[0].message == ("ColumnNotFound 'amount' in orders, did you mean 'amount_cents'?")


def test_no_near_miss_means_no_suggestion() -> None:
    """A wrong suggestion sends somebody to edit the wrong line.

    `amuont` is not close enough to anything in this schema by core's cutoff,
    so the finding stands on its own rather than inventing a fix.
    """
    found = check(ONE_INPUT + '    return orders.select("amuont")\n')

    assert found[0].code is Code.COLUMN_NOT_FOUND
    assert found[0].suggestion is None


def test_a_rename_key_is_checked_and_its_value_is_not() -> None:
    """`rename({"old": "new"})`: only the left side has to exist."""
    found = check(ONE_INPUT + '    return orders.rename({"amuont": "amount"})\n')

    assert [finding.message.split("'")[1] for finding in found] == ["amuont"]


def test_a_join_key_that_no_input_holds_is_reported() -> None:
    """A rename upstream of a join is the expensive version of this bug."""
    found = check(
        "def f(orders: pl.LazyFrame, users: pl.LazyFrame) -> pl.LazyFrame:\n"
        '    return orders.join(users, on="customer_id")\n',
        schemas={"f": {"orders": ORDERS, "users": USERS}},
    )

    assert [finding.code for finding in found] == [Code.COLUMN_NOT_FOUND]


def test_a_row_dependent_call_is_reported_as_unverified() -> None:
    """The same rule the runtime applies, found statically in the buffer."""
    found = check(
        ONE_INPUT
        + "    if orders.is_empty():\n"
        + "        return orders\n"
        + '    return orders.select("order_id")\n'
    )

    assert [finding.code for finding in found] == [Code.ROW_DEPENDENT_CONTROL_FLOW]
    assert found[0].level is Level.UNVERIFIED


# --- The silences ------------------------------------------------------------


def test_a_column_that_exists_is_not_reported() -> None:
    """The case that must stay quiet, or the feature is unusable."""
    assert check(ONE_INPUT + '    return orders.select("amount_cents")\n') == ()


def test_a_column_the_pipeline_creates_is_not_reported() -> None:
    """A name introduced in the chain is not a missing input column."""
    found = check(
        ONE_INPUT
        + '    out = orders.rename({"amount_cents": "amount"})\n'
        + '    return out.select("amount")\n'
    )

    assert found == ()


def test_a_column_only_in_the_last_resolved_output_is_not_reported() -> None:
    """The safety net for every chain shape this module does not model.

    If the engine itself produced the column, rowless has no business
    reporting a read of it, whatever the source looks like.
    """
    found = check(
        ONE_INPUT + '    return orders.select("total")\n',
        outputs={"f": Schema.from_mapping({"total": int_(64)})},
    )

    assert found == ()


def test_a_function_with_a_computed_name_says_nothing_about_columns() -> None:
    """One gap in a function's column set disqualifies the whole function.

    A name rowless cannot see could be the very one it is about to report
    missing, so the honest answer for the function is silence - and the
    computed name itself is still reported, which is how the user finds out.
    """
    found = check(
        "def f(orders: pl.LazyFrame, currency: str) -> pl.LazyFrame:\n"
        '    out = orders.select(f"amount_{currency}")\n'
        '    return out.select("amuont")\n'
    )

    assert [finding.code for finding in found] == [Code.DYNAMIC_COLUMN_NAMES]


def test_a_function_rowless_does_not_check_is_left_alone() -> None:
    """No annotation and no marker means no claim about its columns."""
    found = check(
        'def f(orders):\n    return orders.select("amuont")\n',
        schemas={"f": {"orders": ORDERS}},
    )

    assert found == ()


def test_nothing_is_reported_before_the_first_resolve() -> None:
    """With no schema there is nothing to compare against, so no finding."""
    assert check(ONE_INPUT + '    return orders.select("amuont")\n', schemas={}) == ()


def test_an_unattributed_name_needs_every_input_to_be_missing_it() -> None:
    """`pl.col("name")` could mean any frame in scope, so it means all of them."""
    body = (
        "def f(orders: pl.LazyFrame, users: pl.LazyFrame) -> pl.LazyFrame:\n"
        '    return orders.join(users, on="user_id").filter(pl.col("name") == "x")\n'
    )

    assert check(body, schemas={"f": {"orders": ORDERS, "users": USERS}}) == ()


def test_an_unattributed_name_missing_from_all_inputs_is_reported() -> None:
    """The other half of the same rule: absent everywhere is absent."""
    found = check(
        "def f(orders: pl.LazyFrame, users: pl.LazyFrame) -> pl.LazyFrame:\n"
        '    return orders.filter(pl.col("nope") == 1)\n',
        schemas={"f": {"orders": ORDERS, "users": USERS}},
    )

    assert [finding.code for finding in found] == [Code.COLUMN_NOT_FOUND]


# --- Configuration -----------------------------------------------------------


def test_an_ignore_comment_in_the_buffer_takes_effect_immediately() -> None:
    """Before the save. A suppression comment that needs one is not believed."""
    found = check(ONE_INPUT + '    return orders.select("amuont")  # rowless: ignore[RWL101]\n')

    assert found == ()


def test_configured_ignores_apply_in_the_editor_too(tmp_path: object) -> None:
    """One policy, not a second opinion from the editor."""
    config = Config(ignore_codes=frozenset({Code.COLUMN_NOT_FOUND}))
    found = check(ONE_INPUT + '    return orders.select("amuont")\n', config=config)

    assert found == ()


def test_fail_on_unverified_off_downgrades_the_live_level() -> None:
    """`Config.level_for` decides severity here exactly as it does in CI."""
    config = Config(fail_on_unverified=False)
    found = check(
        ONE_INPUT + "    if orders.is_empty():\n        return orders\n    return orders\n",
        config=config,
    )

    assert [finding.level for finding in found] == [Level.WARNING]

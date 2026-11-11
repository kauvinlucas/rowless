"""The operations battery for pandas: real pipelines, and the schemas they produce.

The conformance suite in core writes the universal half of the contract once;
this is the half that cannot be written there, because `df.select("a")` is not
the same call in Polars, pandas and DuckDB and inventing one abstract
vocabulary for all of them is the proxy approach the project rejects.

Every case runs through the same runtime the CLI uses, so what is asserted is
the schema a user would get. The `reason` on a case is the valuable part: it is
where an engine disagreement gets written down, and the suite prints it when
the case fails.

The pandas battery carries more disagreements than the others, and the two
worth reading first are `group_by_sum` - where the group key lands in the
index and vanishes from the schema - and `apply_with_annotation`, which is the
drift table's headline: pandas never calls the function, so without the
correction the column would keep the dtype it went in with.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rowless.errors import ColumnNotFound, DuplicateColumn
from rowless.ir.confidence import Confidence
from rowless.ir.dtypes import (
    TimeUnit,
    boolean,
    datetime,
    enum,
    float_,
    int_,
    string,
)
from rowless.ir.schema import Schema
from rowless.testing import fixtures
from rowless.testing.conformance import ConformanceCase

#: The timestamp dtype both fixture schemas carry, spelled once.
TIMESTAMP = datetime(TimeUnit.MICROSECOND, "UTC")


def annotated_cents(amount: int) -> float:
    """A UDF with a return annotation, which is what the adapter prefers.

    Deliberately a named function and not a lambda: the point of the case is
    that the annotation is read, and a lambda cannot carry one.
    """
    return amount / 100


def unknowable(amount: int) -> Any:
    """A UDF nothing can type: no annotation worth having, and it raises when probed."""
    raise RuntimeError("this UDF cannot be called with an invented value")


def conformance_cases() -> Sequence[ConformanceCase]:
    """Every operation this adapter claims to resolve, and what it resolves to."""
    import pandas as pd

    orders = {"orders": fixtures.ORDERS}
    joined = {"orders": fixtures.ORDERS, "customers": fixtures.CUSTOMERS}
    drift_prone = {"frame": fixtures.DRIFT_PRONE}

    return (
        ConformanceCase(
            name="select",
            pipeline=lambda orders: orders[["order_id", "amount"]],
            inputs=orders,
            expect=Schema.from_mapping({"order_id": int_(64), "amount": int_(64)}),
            reason="a list of names projects; a bare name would return a Series instead",
        ),
        ConformanceCase(
            name="select_is_exact",
            pipeline=lambda orders: orders[["order_id"]],
            inputs=orders,
            expect=Schema.from_mapping({"order_id": int_(64)}),
            expect_confidence=Confidence.EXACT,
            reason=(
                "pandas computed these dtypes itself while the frame was built, and a "
                "projection reads no values, so nothing here is approximated"
            ),
        ),
        ConformanceCase(
            name="series_round_trips_as_one_column",
            pipeline=lambda orders: orders["amount"],
            inputs=orders,
            expect=Schema.from_mapping({"amount": int_(64)}),
            reason="a pipeline may return a Series; it is a one-column relation named after it",
        ),
        ConformanceCase(
            name="rename",
            pipeline=lambda orders: orders.rename(columns={"amount": "amount_cents"}),
            inputs=orders,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount_cents": int_(64),
                    "created_at": TIMESTAMP,
                }
            ),
            reason="rename keeps the column position, which a projection-based rename would not",
        ),
        ConformanceCase(
            name="filter",
            pipeline=lambda orders: orders[orders["status"] == "paid"],
            inputs=orders,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": int_(64),
                    "created_at": TIMESTAMP,
                }
            ),
            reason=(
                "a boolean mask cannot change the schema, however many rows it removes - and "
                "building the mask reads no values, so the result stays verified"
            ),
        ),
        ConformanceCase(
            name="assign_and_cast",
            pipeline=lambda orders: orders.assign(amount=orders["amount"].astype("float64")),
            inputs=orders,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": float_(64),
                    "created_at": TIMESTAMP,
                }
            ),
            reason="assign replaces a column in place rather than appending it, as Polars does",
        ),
        ConformanceCase(
            name="integer_division_is_float",
            pipeline=lambda orders: orders.assign(unit_price=orders["amount"] / 2)[["unit_price"]],
            inputs=orders,
            expect=Schema.from_mapping({"unit_price": float_(64)}),
            reason=(
                "pandas divides like Python and returns float64, which is the Polars and the "
                "DuckDB answer too. // keeps the integer in all three"
            ),
        ),
        ConformanceCase(
            name="group_by_sum",
            pipeline=lambda orders: orders.groupby("status", observed=True).agg({"amount": "sum"}),
            inputs=orders,
            expect=Schema.from_mapping({"amount": int_(64)}),
            reason=(
                "the biggest disagreement in this adapter: pandas puts the group key in the "
                "INDEX, so status is not a column and not in the schema. Polars and DuckDB both "
                "produce {status, amount}. A named aggregation keeps the input dtype, which "
                "agrees with Polars and disagrees with DuckDB, where sum widens to 128 bits and "
                "the column is renamed sum(amount)"
            ),
        ),
        ConformanceCase(
            name="group_by_sum_as_index_false",
            pipeline=lambda orders: orders.groupby("status", as_index=False, observed=True).agg(
                {"amount": "sum"}
            ),
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "amount": int_(64)}),
            reason=(
                "the same aggregation with as_index=False, which is the spelling that produces "
                "what every other engine produces. Worth asserting next to the case above: the "
                "difference between them is one keyword and one column"
            ),
        ),
        ConformanceCase(
            name="group_by_named_aggregation",
            pipeline=lambda orders: orders.groupby("status", as_index=False, observed=True).agg(
                total=("amount", "sum")
            ),
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "total": int_(64)}),
            reason=(
                "named aggregation produces flat column names. The list spelling, "
                "agg(['sum', 'mean']), produces a MultiIndex that has no flat names at all, and "
                "the adapter reports that as unresolved rather than inventing amount_sum"
            ),
        ),
        ConformanceCase(
            name="merge_separate_keys",
            pipeline=lambda orders, customers: orders.merge(
                customers, left_on="order_id", right_on="customer_id", how="inner"
            ),
            inputs=joined,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": int_(64),
                    "created_at_x": TIMESTAMP,
                    "customer_id": int_(64),
                    "country": string(),
                    "created_at_y": TIMESTAMP,
                }
            ),
            reason=(
                "pandas keeps BOTH key columns and suffixes the colliding created_at _x and _y. "
                "Polars coalesces the right key away and suffixes _right; DuckDB keeps both "
                "columns under the same name. Three engines, three answers, one contract that "
                "notices all of them"
            ),
        ),
        ConformanceCase(
            name="merge_shared_key",
            pipeline=lambda orders, customers: orders.merge(customers, on="created_at", how="left"),
            inputs=joined,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": int_(64),
                    "created_at": TIMESTAMP,
                    "customer_id": int_(64),
                    "country": string(),
                }
            ),
            reason="merging on a shared name produces one column, which is the Polars answer too",
        ),
        ConformanceCase(
            name="concat_union",
            pipeline=lambda orders: pd.concat([orders, orders[orders["status"] == "paid"]]),
            inputs=orders,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": int_(64),
                    "created_at": TIMESTAMP,
                }
            ),
            reason=(
                "pd.concat is the union. Unlike pl.concat it does not require the schemas to "
                "match: a column missing from one side arrives full of NA, and the dtype that "
                "lands is the join of the two - which is why the drift table watches it"
            ),
        ),
        ConformanceCase(
            name="apply_with_annotation",
            pipeline=lambda orders: orders.assign(cents=orders["amount"].apply(annotated_cents))[
                ["cents"]
            ],
            inputs=orders,
            expect=Schema.from_mapping({"cents": float_(64)}),
            reason=(
                "the drift table's headline. pandas has no row to call the function with, so it "
                "keeps the input dtype and would report int64; the adapter reads the return "
                "annotation and casts the column to float64, which is what production produces"
            ),
        ),
        ConformanceCase(
            name="apply_with_no_inferable_type",
            pipeline=lambda orders: orders.assign(cents=orders["amount"].apply(unknowable))[
                ["cents"]
            ],
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
            reason=(
                "nothing can type this UDF: no usable annotation, and probing it raises. pandas "
                "silently reports the input dtype, which is wrong rather than unknown, so the "
                "adapter degrades the run and says why. An annotation or probe_udfs makes it go "
                "away, and both are worth having anyway"
            ),
        ),
        ConformanceCase(
            name="map_with_a_dictionary",
            pipeline=lambda orders: orders.assign(
                label=orders["status"].map({"paid": 1.5, "void": 2.5})
            )[["label"]],
            inputs=orders,
            expect=Schema.from_mapping({"label": float_(64)}),
            reason=(
                "map takes a mapping as often as a function, and drifts for the same reason - "
                "with no row to look up, pandas returns the input dtype. The values are right "
                "there, so this correction needs no inference, only their join"
            ),
        ),
        ConformanceCase(
            name="string_accessor",
            pipeline=lambda orders: orders.assign(shouted=orders["status"].str.upper())[
                ["shouted"]
            ],
            inputs=orders,
            expect=Schema.from_mapping({"shouted": string()}),
            reason="the .str accessor keeps the string dtype rather than falling back to object",
        ),
        ConformanceCase(
            name="categorical_is_an_enum",
            pipeline=lambda frame: frame[["category"]],
            inputs=drift_prone,
            expect=Schema.from_mapping({"category": enum(("a", "b", "c"))}),
            reason=(
                "a pandas Categorical carries its variants, so it is an IR enum - where a "
                "Polars Categorical has an open variant set and is only a string. The closed "
                "set is the part a contract can check"
            ),
        ),
        ConformanceCase(
            name="nullable_boolean_survives_a_mask",
            pipeline=lambda frame: frame.assign(
                flag=frame["nullable_bool"].where(frame["nullable_bool"])
            )[["flag"]],
            inputs=drift_prone,
            expect=Schema.from_mapping({"flag": boolean()}),
            reason=(
                "where() has to store NA somewhere. A masked or arrow-backed column holds it in "
                "the mask and keeps its dtype; a plain numpy bool column would become object, "
                "and a numpy integer float64. Materializing the dtype that cannot drift is "
                "cheaper than correcting the drift, which is why this adapter does"
            ),
        ),
        ConformanceCase(
            name="missing_column",
            pipeline=lambda orders: orders[["amount_cents"]],
            inputs=orders,
            raises=ColumnNotFound,
            reason=(
                "the finding from the proposal. pandas reports it as a KeyError with no list of "
                "valid columns, so the adapter supplies the frame's own columns and the "
                "near-miss suggestion - amount -> amount_cents - comes from there"
            ),
        ),
        ConformanceCase(
            name="duplicate_column",
            pipeline=lambda orders: pd.concat([orders[["amount"]], orders[["amount"]]], axis=1),
            inputs=orders,
            raises=DuplicateColumn,
            reason=(
                "pandas is content to hold two columns called amount; Polars refuses outright. "
                "A rowless schema is addressed by name, so nothing downstream could refer to "
                "either of them, and the duplicate is the finding"
            ),
        ),
        ConformanceCase(
            name="row_dependent_branch",
            pipeline=lambda orders: orders[["order_id"]] if orders.empty else orders,
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
            reason=(
                "df.empty is True on a rowless frame, so the dry run took the branch production "
                "never would. The most common row-dependent expression in pandas code, and the "
                "reason this adapter patches a property rather than only methods"
            ),
        ),
        ConformanceCase(
            name="row_dependent_scalar",
            pipeline=lambda orders: orders.assign(ceiling=orders["amount"].max()),
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
            reason=(
                "Series.max() is NaN on zero rows, so the literal column it builds would be a "
                "number in production. The Polars spelling of the same aggregate, "
                "pl.col('amount').max(), is an expression and costs no confidence - the "
                "asymmetry is pandas being eager, not rowless being inconsistent"
            ),
        ),
        ConformanceCase(
            name="unstack_builds_columns_from_values",
            pipeline=lambda orders: (
                orders.groupby(["status", "order_id"], observed=True)
                .agg({"amount": "sum"})
                .unstack()
            ),
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
            reason=(
                "unstack turns values into column names. A dry run has no values, so whatever "
                "schema it produces is not the one production gets - and nothing can correct "
                "that, which is why this is a degradation rather than a fixup"
            ),
        ),
    )

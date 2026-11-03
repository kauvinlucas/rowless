"""UDF return-type inference. A hole in the schema graph, filled honestly."""

from __future__ import annotations

import datetime as dt
import decimal

from rowless.ir.dtypes import (
    Kind,
    any_,
    boolean,
    date,
    datetime,
    int_,
    list_,
    string,
    struct,
)
from rowless.ir.schema import Schema
from rowless.udf.infer import (
    InferenceMethod,
    annotation_to_ir,
    infer_return_dtype,
    python_value_to_ir,
    synthetic_value,
)

ARGS = Schema.from_mapping({"value": string()})


def test_declared_dtype_wins() -> None:
    """The engine relies on `return_dtype=`, so rowless should too."""

    def udf(value: str) -> str:
        return value

    signature = infer_return_dtype(udf, ARGS, declared=int_(32))

    assert signature.returns == int_(32)
    assert signature.method is InferenceMethod.DECLARED


def test_annotation_is_used_before_probing() -> None:
    """A return annotation is free evidence; calling the function is not."""

    def udf(value: str) -> int:
        raise AssertionError("inference must not call an annotated function")

    signature = infer_return_dtype(udf, ARGS, allow_probe=True)

    assert signature.returns == int_(64)
    assert signature.method is InferenceMethod.ANNOTATION
    assert signature.probe_attempts == 0


def test_without_probing_the_answer_is_any_and_says_why() -> None:
    """ANY lowers coverage rather than failing a check, and explains itself."""

    def udf(value):  # type: ignore[no-untyped-def]  # the case this exists for
        return value.upper()

    signature = infer_return_dtype(udf, ARGS)

    assert signature.returns == any_()
    assert signature.method is InferenceMethod.GAVE_UP
    assert "probe_udfs" in (signature.note or "")


def test_probing_retries_with_a_more_structured_value() -> None:
    """A function that parses its input fails on a boring value."""
    calls: list[str] = []

    def udf(value):  # type: ignore[no-untyped-def]  # a function that parses
        calls.append(value)
        return value.split(",")[1]

    signature = infer_return_dtype(udf, ARGS, allow_probe=True)

    assert signature.returns == string()
    assert signature.method is InferenceMethod.PROBE
    assert signature.probe_attempts == 2
    assert calls == ["x", "a,b"]


def test_an_unprobeable_function_is_any_with_the_last_error() -> None:
    """Three attempts, then the honest answer plus the reason."""

    def udf(value):  # type: ignore[no-untyped-def]  # deliberately hostile
        raise ValueError("needs a real row")

    signature = infer_return_dtype(udf, ARGS, allow_probe=True)

    assert signature.returns == any_()
    assert signature.method is InferenceMethod.GAVE_UP
    assert "needs a real row" in (signature.note or "")


def test_annotations_cover_builtins_datetime_and_decimal() -> None:
    """The documented set, and ANY for anything outside it."""
    assert annotation_to_ir(int) == int_(64)
    assert annotation_to_ir(bool) == boolean()
    assert annotation_to_ir(dt.date) == date()
    assert annotation_to_ir(dt.datetime) == datetime()
    assert annotation_to_ir(decimal.Decimal).kind is Kind.DECIMAL
    assert annotation_to_ir(list[int]) == list_(int_(64))
    assert annotation_to_ir(int | None) == int_(64)
    assert annotation_to_ir(object) == any_()
    assert annotation_to_ir(None).kind is Kind.NULL


def test_probe_results_unify_rather_than_taking_the_first_element() -> None:
    """`[1, "a"]` is not a list of ints just because the first element was."""
    assert python_value_to_ir([1, 2]) == list_(int_(64))

    mixed = python_value_to_ir([1, "a"])
    assert mixed.kind is Kind.LIST, "it is still a list; the element type is what is unknown"
    assert not mixed.is_known
    assert "no common dtype" in str(mixed.inner[0].reason)
    assert python_value_to_ir({"a": 1}) == struct({"a": int_(64)})
    assert python_value_to_ir(None).kind is Kind.NULL


def test_synthetic_values_match_the_dtype_they_stand_for() -> None:
    """A probe argument has to be plausible, or every probe fails."""
    assert isinstance(synthetic_value(string()), str)
    assert "," in synthetic_value(string(), attempt=1)
    assert isinstance(synthetic_value(int_(64)), int)
    assert isinstance(synthetic_value(date()), dt.date)
    assert synthetic_value(datetime(tz="UTC")).tzinfo is not None
    assert synthetic_value(any_()) is None, "there is no representative value for ANY"

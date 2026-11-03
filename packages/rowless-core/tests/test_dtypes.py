"""Dtype algebra. The most load-bearing code in the project, so test it hardest."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from rowless.ir.dtypes import (
    DType,
    Kind,
    TimeUnit,
    any_,
    binary,
    boolean,
    date,
    datetime,
    decimal,
    duration,
    enum,
    float_,
    int_,
    list_,
    map_,
    null,
    string,
    struct,
    time,
    uint,
    unify,
    unknown,
)
from rowless.ir.serde import DTypeSyntaxError
from rowless.sources.literal import parse_dtype_string


def test_matches_is_structural() -> None:
    """Equal dtypes match; a differing timezone or scale does not."""
    assert int_(64).matches(int_(64))
    assert datetime(TimeUnit.MICROSECOND, "UTC").matches(datetime(TimeUnit.MICROSECOND, "UTC"))

    assert not datetime(TimeUnit.MICROSECOND, "UTC").matches(datetime(TimeUnit.MICROSECOND))
    assert not datetime(TimeUnit.NANOSECOND).matches(datetime(TimeUnit.MICROSECOND))
    assert not decimal(18, 2).matches(decimal(18, 4))
    assert not int_(32).matches(int_(64))
    assert not list_(int_(64)).matches(list_(int_(32)))


def test_any_matches_everything() -> None:
    """ANY matches any dtype, in both directions."""
    assert any_().matches(int_(64))
    assert int_(64).matches(any_())
    assert any_().matches(struct({"a": string()}))
    assert list_(any_()).matches(list_(int_(8)))


def test_unknown_never_matches() -> None:
    """UNKNOWN does not match even itself: two unknowns are not a pass."""
    assert not unknown().matches(unknown())
    assert not unknown("no mapping").matches(int_(64))
    assert not int_(64).matches(unknown())
    assert not list_(unknown()).is_known


def test_unknown_reason_does_not_affect_equality() -> None:
    """The reason is commentary: two unresolved dtypes are equally unusable."""
    assert unknown("a") == unknown("b")
    assert unknown("a").reason == "a"


def test_accepts_is_one_directional() -> None:
    """int32 flows into int64; int64 does not flow into int32."""
    assert int_(64).accepts(int_(32))
    assert not int_(32).accepts(int_(64))

    assert float_(64).accepts(int_(32))
    assert not float_(64).accepts(int_(64)), "int64 does not fit a float64 without loss"
    assert int_(64).accepts(uint(32))
    assert not int_(64).accepts(uint(64))
    assert decimal(18, 2).accepts(decimal(10, 2))
    assert not decimal(10, 2).accepts(decimal(18, 2))
    assert string().accepts(enum(("a", "b")))
    assert not enum(("a", "b")).accepts(string())


def test_accepts_refuses_a_timezone_change() -> None:
    """Naive and aware are different values, not different spellings."""
    assert not datetime(TimeUnit.MICROSECOND, "UTC").accepts(datetime(TimeUnit.MICROSECOND))
    assert datetime(TimeUnit.NANOSECOND, "UTC").accepts(datetime(TimeUnit.MICROSECOND, "UTC"))


def test_null_fits_any_slot_but_is_still_reported() -> None:
    """An all-null column carries no evidence against a declared type."""
    assert int_(64).accepts(null())
    assert not null().accepts(int_(64))
    assert null().kind is Kind.NULL


def test_unify_falls_back_to_unknown() -> None:
    """Unifying string and int64 yields UNKNOWN, not a guessed coercion."""
    assert unify(string(), int_(64)).kind is Kind.UNKNOWN
    assert unify(datetime(tz="UTC"), datetime()).kind is Kind.UNKNOWN
    assert unify(int_(64), unknown()).kind is Kind.UNKNOWN


def test_unify_widens_where_every_engine_agrees() -> None:
    """The cases that are safe are the only ones taken."""
    assert unify(int_(32), int_(64)) == int_(64)
    assert unify(int_(32), uint(8)) == int_(32)
    assert unify(uint(8), uint(32)) == uint(32)
    assert unify(null(), string()) == string()
    assert unify(int_(8), float_(32)) == float_(32)
    assert unify(list_(int_(32)), list_(int_(64))) == list_(int_(64))
    assert unify(any_(), int_(64)) == any_()


def test_unify_of_differing_enums_is_a_string() -> None:
    """The union of two variant sets is a different type; a string is honest."""
    assert unify(enum(("a",)), enum(("b",))) == string()


def test_width_validation_rejects_an_invented_type() -> None:
    """An engine with a 42-bit integer does not exist, so neither does the dtype."""
    with pytest.raises(ValueError, match="width must be one of"):
        int_(42)
    with pytest.raises(ValueError, match="precision >= scale"):
        decimal(2, 5)


def test_dtype_strings_are_quotable() -> None:
    """The spelling in a diagnostic is the spelling a contract file accepts."""
    assert str(datetime(TimeUnit.MICROSECOND, "UTC")) == "datetime[us, UTC]"
    assert str(decimal(18, 2)) == "decimal(18,2)"
    assert str(struct({"a": int_(64), "b": list_(string())})) == "struct{a:int64,b:list[string]}"
    assert str(map_(string(), int_(32))) == "map[string, int32]"
    assert str(enum(("a", "b"))) == "enum{a,b}"


def test_parse_rejects_nonsense() -> None:
    """A dtype spelling that is not in the grammar fails loudly."""
    with pytest.raises(DTypeSyntaxError):
        parse_dtype_string("int")
    with pytest.raises(DTypeSyntaxError):
        parse_dtype_string("timestamp[us]")
    with pytest.raises(DTypeSyntaxError):
        parse_dtype_string("int64 extra")


_SCALARS = st.sampled_from(
    [
        boolean(),
        *(int_(bits) for bits in (8, 16, 32, 64, 128)),
        *(uint(bits) for bits in (8, 16, 32, 64)),
        *(float_(bits) for bits in (16, 32, 64)),
        decimal(18, 2),
        decimal(38, 0),
        string(),
        binary(),
        date(),
        null(),
        unknown(),
        any_(),
        *(time(unit) for unit in TimeUnit),
        *(datetime(unit) for unit in TimeUnit),
        datetime(TimeUnit.MICROSECOND, "UTC"),
        datetime(TimeUnit.NANOSECOND, "Europe/London"),
        *(duration(unit) for unit in TimeUnit),
        enum(("a", "b")),
    ]
)


def _nested(children: st.SearchStrategy[DType]) -> st.SearchStrategy[DType]:
    """One level of nesting around whatever the child strategy produces."""
    names = st.text(alphabet="abcdefg", min_size=1, max_size=4)
    return st.one_of(
        children.map(list_),
        st.builds(map_, children, children),
        st.dictionaries(names, children, min_size=1, max_size=3).map(struct),
    )


@given(st.recursive(_SCALARS, _nested, max_leaves=6))
def test_dtype_string_round_trips(dtype: DType) -> None:
    """`parse_dtype_string(str(dtype)) == dtype` for the whole corpus.

    Property-tested with hypothesis: a dtype printed in a diagnostic has to be
    pasteable into a contract file, or the diagnostics are not actionable.
    """
    assert parse_dtype_string(str(dtype)) == dtype

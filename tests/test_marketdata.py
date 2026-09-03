"""Required market data must fail loudly, never default."""
import pytest

from exitkit import MissingMarketData, require, require_entry
from exitkit.types import Position, SignalOutput


def test_present_field_is_returned():
    assert require({"spot_price": 412.3}, "spot_price") == 412.3


def test_missing_field_raises():
    with pytest.raises(MissingMarketData):
        require({}, "spot_price")


def test_none_and_nan_are_missing():
    with pytest.raises(MissingMarketData):
        require({"spot_price": None}, "spot_price")
    with pytest.raises(MissingMarketData):
        require({"spot_price": float("nan")}, "spot_price")


def test_unparseable_value_raises():
    with pytest.raises(MissingMarketData):
        require({"spot_price": "not a price"}, "spot_price")


def test_error_names_the_field_and_what_was_present():
    with pytest.raises(MissingMarketData) as e:
        require({"implied_vol": 0.2}, "spot_price")
    assert "spot_price" in str(e.value) and "implied_vol" in str(e.value)


def test_zero_is_a_real_value_not_a_gap():
    assert require({"momentum": 0.0}, "momentum") == 0.0


def test_entry_meta_lookup(position):
    assert require_entry(position(), "implied_vol") == 0.20


def test_missing_entry_meta_raises():
    p = Position("p", 0.0, SignalOutput(direction=1, meta={}), 100.0, 1)
    with pytest.raises(MissingMarketData):
        require_entry(p, "implied_vol")

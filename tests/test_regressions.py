"""Defects found while extracting this package. Each test is named for one."""
import time

import pytest

from exitkit import (
    FixedTimeExitModel,
    MarketHoursExitModel,
    Position,
    SignalOutput,
    WindowTensor,
)


def _position(held_hours):
    return Position(
        position_id="p1",
        entry_time=time.time() - held_hours * 3600,
        entry_signal=SignalOutput(direction=1, meta={"implied_vol": 0.2}),
        entry_price=400.0,
        quantity=10,
    )


def test_holding_hours_do_not_require_a_prior_mark():
    """get_holding_hours returned holding_period/3600, and holding_period was
    only assigned inside update_pnl(). A model that did not first mark the
    position saw zero hours held."""
    p = _position(held_hours=9.0)
    assert p.get_holding_hours() == pytest.approx(9.0, abs=0.01)


def test_time_exit_fires_without_an_explicit_mark(window, market):
    """The consequence: a nine-hour position against a two-hour limit did not
    exit, because nothing had populated holding_period."""
    m = FixedTimeExitModel(max_holding_hours=2.0)
    signals = m.generate_exit_signals([_position(9.0)], market())
    assert len(signals) == 1
    assert signals[0].exit_reason == "time_limit"


def test_time_exit_still_holds_inside_the_limit(window, market):
    m = FixedTimeExitModel(max_holding_hours=8.0)
    assert m.generate_exit_signals([_position(1.0)], market()) == []


def test_marking_a_position_agrees_with_the_derived_value():
    p = _position(held_hours=3.0)
    p.update_pnl(410.0)
    assert p.get_holding_hours() == pytest.approx(3.0, abs=0.01)


def test_market_hours_model_is_callable():
    """MarketHoursExitModel raised NameError: name 'time' is not defined on
    every call - the module used time.time() without importing time, so this
    model had never run."""
    m = MarketHoursExitModel()
    out = m.generate_exit_signals([], {"timestamp": time.time()})
    assert out == []


def test_market_hours_model_runs_against_a_position(window, market):
    m = MarketHoursExitModel()
    out = m.generate_exit_signals([_position(1.0)],
                                  market(timestamp=time.time()))
    assert isinstance(out, list)


def test_exit_models_do_not_require_a_feature_window():
    """generate_exit_signals took W_t as its first required argument, and no
    model in the package ever read it - 33 mentions, zero attribute accesses.
    Callers had to build and pass a WindowTensor that was entirely ignored."""
    from exitkit import StopLossExitModel
    m = StopLossExitModel(stop_loss_pct=0.02)
    signals = m.generate_exit_signals(
        [_position(1.0)], {"spot_price": 380.0, "implied_vol": 0.2}
    )
    assert len(signals) == 1


def test_a_window_may_still_be_supplied():
    from exitkit import StopLossExitModel, WindowTensor
    m = StopLossExitModel(stop_loss_pct=0.02)
    signals = m.generate_exit_signals(
        [_position(1.0)], {"spot_price": 380.0, "implied_vol": 0.2},
        W_t=WindowTensor(),
    )
    assert len(signals) == 1


def test_market_hours_requires_an_explicit_timestamp():
    """MarketHoursExitModel read market_data.get('timestamp', time.time()).

    In a backtest that silently evaluates market hours against the operator's
    wall clock rather than simulated time - a backtest run at 02:00 would hold
    everything. It also made the test suite time-of-day dependent: this passed
    locally and failed on CI at 16:34 UTC, which is how it was found.
    """
    from exitkit import MarketHoursExitModel, MissingMarketData
    with pytest.raises(MissingMarketData):
        MarketHoursExitModel().generate_exit_signals([_position(1.0)], {})


def test_market_hours_uses_the_timestamp_it_is_given():
    """Two fixed instants, one inside the session and one outside, must not
    depend on when the suite runs."""
    from exitkit import MarketHoursExitModel
    m = MarketHoursExitModel()
    for instant in (1704207600.0, 1704250800.0):     # 2024-01-02, two times of day
        out = m.generate_exit_signals([_position(1.0)], {"timestamp": instant})
        assert isinstance(out, list)

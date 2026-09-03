"""The backtesting.py adapter. Skipped entirely when backtesting is absent."""
import pytest

pytest.importorskip("backtesting", reason="pip install exitkit[backtesting]")

import pandas as pd  # noqa: E402
from backtesting import Backtest, Strategy  # noqa: E402
from backtesting.lib import crossover  # noqa: E402
from backtesting.test import GOOG  # noqa: E402

from exitkit import FixedTimeExitModel, StopLossExitModel, TakeProfitExitModel  # noqa: E402
from exitkit.adapters.backtesting_py import (  # noqa: E402
    ExitMixin,
    market_snapshot,
    trade_to_position,
)


def sma(values, n):
    return pd.Series(values).rolling(n).mean()


class SmaCross(ExitMixin, Strategy):
    n1, n2 = 10, 30
    exit_models = ()

    def init(self):
        self.s1 = self.I(sma, self.data.Close, self.n1)
        self.s2 = self.I(sma, self.data.Close, self.n2)

    def next(self):
        self.apply_exits()
        if crossover(self.s1, self.s2) and not self.position:
            self.buy()


def run(models, **kw):
    strategy = type("S", (SmaCross,), {"exit_models": models, **kw})
    return Backtest(GOOG, strategy, cash=100_000, commission=.002,
                    finalize_trades=True).run()


# --- the mixin drives real exits --------------------------------------------

def test_no_policy_holds_a_single_position():
    assert run(())["# Trades"] == 1


def test_a_stop_produces_more_trades_than_holding():
    assert run((StopLossExitModel(0.02),))["# Trades"] > run(())["# Trades"]


def test_a_tighter_stop_exits_more_often():
    loose = run((StopLossExitModel(0.10),))["# Trades"]
    tight = run((StopLossExitModel(0.02),))["# Trades"]
    assert tight > loose


def test_a_time_limit_caps_drawdown_here():
    """Not a general claim - on this sample, capping holding time cuts the
    drawdown sharply. The point is that the policy actually took effect."""
    held = run(())["Max. Drawdown [%]"]
    timed = run((FixedTimeExitModel(max_holding_hours=24 * 30),))["Max. Drawdown [%]"]
    assert timed > held          # less negative


def test_policies_compose():
    stats = run((StopLossExitModel(0.05), TakeProfitExitModel(take_profit_pct=0.10)))
    assert stats["# Trades"] > 1


def test_each_trade_closes_at_most_once_per_bar():
    """Two policies can fire on the same trade in one bar; the second must be
    ignored rather than closing an already-closed trade."""
    stats = run((StopLossExitModel(0.01), StopLossExitModel(0.01)))
    assert stats["# Trades"] > 0


# --- the mapping ------------------------------------------------------------

def test_snapshot_supplies_every_required_field():
    captured = {}

    class Capture(SmaCross):
        exit_models = ()

        def next(self):
            if len(self.data.Close) > 30 and not captured:
                captured.update(market_snapshot(self.data))
            super().next()

    Backtest(GOOG, Capture, cash=100_000, finalize_trades=True).run()
    for field in ("spot_price", "implied_vol", "momentum", "timestamp"):
        assert field in captured, field
    assert captured["spot_price"] > 0
    assert captured["implied_vol"] >= 0


def test_snapshot_accepts_a_supplied_implied_vol():
    class Capture(SmaCross):
        exit_models = ()
        seen = {}

        def next(self):
            if len(self.data.Close) > 30 and not Capture.seen:
                Capture.seen.update(market_snapshot(self.data, implied_vol=0.42))
            super().next()

    Backtest(GOOG, Capture, cash=100_000, finalize_trades=True).run()
    assert Capture.seen["implied_vol"] == 0.42


def test_holding_time_uses_the_bar_clock_not_the_wall_clock():
    """The decisive property for a replay: a 2010 trade must not be aged to now.

    Position.get_holding_hours() derives from time.time() when no clock is
    given, so a backtest that did not pass the bar timestamp would age every
    position by fifteen years and fire every time-based exit immediately.
    """
    stats = run((FixedTimeExitModel(max_holding_hours=24 * 3650),))  # 10-year limit
    assert stats["# Trades"] == 1, "a 10-year limit should not fire on this sample"

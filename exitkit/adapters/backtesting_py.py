"""Use exitkit's exit policies inside a `backtesting.py` strategy.

    from backtesting import Backtest, Strategy
    from exitkit import FixedTimeExitModel, StopLossExitModel
    from exitkit.adapters.backtesting_py import ExitMixin

    class SmaCross(ExitMixin, Strategy):
        exit_models = [StopLossExitModel(0.02), FixedTimeExitModel(48)]

        def next(self):
            self.apply_exits()              # close whatever policy says to close
            if not self.position:
                self.buy()

`backtesting.py` is an optional dependency: ``pip install exitkit[backtesting]``.

The clock matters here. Holding time is measured against the *bar's* timestamp, not
the wall clock, so a replay ages positions by simulated time rather than by when you
happened to run it.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..base import ExitSignalModel
from ..types import Position, SignalOutput

__all__ = ["ExitMixin", "trade_to_position", "market_snapshot"]


def _epoch(value) -> float:
    """Bar timestamps arrive as pandas Timestamps, datetimes or numbers."""
    if hasattr(value, "timestamp"):
        return float(value.timestamp())
    if isinstance(value, np.datetime64):
        return float(value.astype("datetime64[s]").astype(np.int64))
    return float(value)


def trade_to_position(trade) -> Position:
    """Map one `backtesting.py` Trade onto an exitkit Position."""
    direction = 1 if trade.is_long else -1
    return Position(
        position_id=f"trade-{trade.entry_bar}-{'L' if trade.is_long else 'S'}",
        entry_time=_epoch(trade.entry_time),
        entry_signal=SignalOutput(
            direction=direction,
            signal_type="entry",
            meta={"entry_bar": trade.entry_bar, "tag": trade.tag},
        ),
        entry_price=float(trade.entry_price),
        quantity=int(abs(trade.size)),
    )


def market_snapshot(
    data,
    implied_vol: Optional[float] = None,
    momentum_lookback: int = 5,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the market-data dict exit models require, from the current bar.

    `implied_vol` has no counterpart in OHLC data, so it is estimated from
    trailing realised volatility unless you supply one. Estimating it is a
    modelling choice, not a fact - pass your own series when you have one.
    """
    close = np.asarray(data.Close, dtype=float)
    spot = float(close[-1])

    if implied_vol is None:
        window = close[-21:]
        if len(window) > 2:
            rets = np.diff(np.log(window))
            implied_vol = float(np.std(rets, ddof=1) * np.sqrt(252))
        else:
            implied_vol = 0.0

    lookback = min(momentum_lookback, len(close) - 1)
    momentum = float(close[-1] / close[-1 - lookback] - 1.0) if lookback > 0 else 0.0

    snapshot = {
        "spot_price": spot,
        "implied_vol": float(implied_vol),
        "momentum": momentum,
        "timestamp": _epoch(data.index[-1]),
    }
    if extra:
        snapshot.update(extra)
    return snapshot


class ExitMixin:
    """Mix into a `backtesting.py` Strategy to drive exits from exitkit models.

    Set `exit_models` to a list of `ExitSignalModel` instances, then call
    `self.apply_exits()` at the top of `next()`. Trades whose policy fires are
    closed on that bar.
    """

    exit_models: Sequence[ExitSignalModel] = ()
    exit_implied_vol: Optional[float] = None
    exit_extra_market_data: Optional[Dict[str, Any]] = None

    def apply_exits(self) -> List[SignalOutput]:
        """Evaluate every policy against open trades; close what fires.

        Returns the signals that fired, so a strategy can log or count them.
        """
        models = list(self.exit_models)
        if not models or not self.trades:
            return []

        by_id = {}
        positions = []
        for trade in self.trades:
            position = trade_to_position(trade)
            by_id[position.position_id] = trade
            positions.append(position)

        market_data = market_snapshot(
            self.data,
            implied_vol=self.exit_implied_vol,
            extra=self.exit_extra_market_data,
        )

        fired: List[SignalOutput] = []
        closed = set()
        for model in models:
            live = [p for p in positions if p.position_id not in closed]
            if not live:
                break
            for signal in model.generate_exit_signals(live, market_data):
                if signal.position_id in closed:
                    continue
                trade = by_id.get(signal.position_id)
                if trade is None:
                    continue
                trade.close()
                closed.add(signal.position_id)
                fired.append(signal)
        return fired

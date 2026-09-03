"""Same entry signal, different exit policies, on backtesting.py's sample data.

    pip install exitkit[backtesting]
    python examples/compare_exit_policies.py

The entry rule never changes: a 10/30 SMA crossover. Only the exit policy varies.
That is the comparison exitkit exists to make cheap.
"""
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG

from exitkit import FixedTimeExitModel, StopLossExitModel, TakeProfitExitModel
from exitkit.adapters.backtesting_py import ExitMixin


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


POLICIES = {
    "none (hold)":       (),
    "stop 2%":           (StopLossExitModel(0.02),),
    "stop 5%":           (StopLossExitModel(0.05),),
    "take profit 10%":   (TakeProfitExitModel(take_profit_pct=0.10),),
    "time limit 30d":    (FixedTimeExitModel(max_holding_hours=24 * 30),),
    "stop 5% + tp 10%":  (StopLossExitModel(0.05), TakeProfitExitModel(take_profit_pct=0.10)),
}


def main():
    print(f"{'exit policy':<20}{'return %':>10}{'trades':>8}{'max DD %':>10}{'Sharpe':>8}")
    print("-" * 56)
    for name, models in POLICIES.items():
        strategy = type("S", (SmaCross,), {"exit_models": models})
        stats = Backtest(GOOG, strategy, cash=100_000, commission=.002,
                         finalize_trades=True).run()
        print(f"{name:<20}{stats['Return [%]']:>10.1f}{stats['# Trades']:>8}"
              f"{stats['Max. Drawdown [%]']:>10.1f}{stats['Sharpe Ratio']:>8.2f}")


if __name__ == "__main__":
    main()

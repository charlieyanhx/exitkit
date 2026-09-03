"""Regenerate docs/exit_policy_comparison.png (the README figure)."""
import warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG
from exitkit import FixedTimeExitModel, StopLossExitModel, TakeProfitExitModel
from exitkit.adapters.backtesting_py import ExitMixin

def sma(v, n): return pd.Series(v).rolling(n).mean()

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
    "no exit policy":   ((), "#8B9997"),
    "stop 2%":          ((StopLossExitModel(0.02),), "#C2544A"),
    "take profit 10%":  ((TakeProfitExitModel(take_profit_pct=0.10),), "#9A6208"),
    "stop 5% + tp 10%": ((StopLossExitModel(0.05), TakeProfitExitModel(take_profit_pct=0.10)), "#3E6E9E"),
    "time limit 30d":   ((FixedTimeExitModel(max_holding_hours=24*30),), "#0B6B5B"),
}

curves, stats_rows = {}, []
for name, (models, color) in POLICIES.items():
    cls = type("S", (SmaCross,), {"exit_models": models})
    bt = Backtest(GOOG, cls, cash=100_000, commission=.002, finalize_trades=True)
    st = bt.run()
    curves[name] = (st["_equity_curve"]["Equity"], color)
    stats_rows.append((name, st["Return [%]"], st["Max. Drawdown [%]"], st["Sharpe Ratio"], st["# Trades"]))

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                              gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.12})
fig.patch.set_facecolor("#FFFFFF")
for a in (ax, ax2):
    a.set_facecolor("#FFFFFF")
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color("#DCE2E0")
    a.tick_params(colors="#5C6A68", labelsize=8.5)
    a.grid(True, axis="y", color="#E9EDEC", lw=0.8)
    a.set_axisbelow(True)

for name, (eq, color) in curves.items():
    lw = 2.2 if name == "time limit 30d" else 1.4
    ax.plot(eq.index, eq.values / 1000, color=color, lw=lw, label=name,
            alpha=1.0 if lw > 2 else 0.85)
    dd = (eq / eq.cummax() - 1) * 100
    ax2.plot(dd.index, dd.values, color=color, lw=lw * 0.7, alpha=0.85)

ax.set_ylabel("equity  ($000)", color="#5C6A68", fontsize=9)
ax2.set_ylabel("drawdown  (%)", color="#5C6A68", fontsize=9)
ax.legend(frameon=False, fontsize=8.5, labelcolor="#131A19", loc="upper left", ncol=2)
ax.set_title("Same entry signal (10/30 SMA crossover). Only the exit policy changes.",
             fontsize=10.5, color="#131A19", loc="left", pad=12, weight="bold")
ax2.set_xlabel("")
fig.text(0.125, 0.015, "exitkit · examples/compare_exit_policies.py · backtesting.py sample data (GOOG)",
         fontsize=7.5, color="#8B9997")
fig.savefig("docs/exit_policy_comparison.png",
            dpi=160, bbox_inches="tight", facecolor="#FFFFFF")
print(f"{'policy':<20}{'ret %':>9}{'maxDD %':>10}{'Sharpe':>8}{'trades':>8}")
for r in stats_rows:
    print(f"{r[0]:<20}{r[1]:>9.1f}{r[2]:>10.1f}{r[3]:>8.2f}{r[4]:>8}")

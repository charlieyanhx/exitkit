# exitkit

A catalogue of position-exit policies. Twenty-seven models in six families behind one interface,
so exit policy is something you swap and compare rather than hardcode.

Entry logic is well served by open-source backtesting libraries. Exit logic usually is not — most
ship a fixed stop and a fixed target and leave the rest to you.

```bash
pip install exitkit
```

## Use

```python
from exitkit import StopLossExitModel, Position, SignalOutput, WindowTensor
import time

position = Position(
    position_id="p1",
    entry_time=time.time() - 3600,
    entry_signal=SignalOutput(direction=1, meta={"implied_vol": 0.18}),
    entry_price=400.0,
    quantity=10,
)

model = StopLossExitModel(stop_loss_pct=0.02, trailing=True)
signals = model.generate_exit_signals(
    WindowTensor(), [position],
    {"spot_price": 391.0, "implied_vol": 0.21},
)

for s in signals:
    print(s.exit_reason, s.meta["loss_pct"])   # stop_loss -0.0225
```

Every model implements `generate_exit_signals(window, open_positions, market_data)` and returns
`SignalOutput` objects carrying the position they close and why.

## The six families

| Family | Models |
|---|---|
| **stop_loss** | fixed, adaptive, volatility-scaled, time-decayed |
| **take_profit** | fixed, partial, adaptive, scaling, momentum-aware |
| **time_based** | fixed horizon, time decay, adaptive, market hours, performance-conditioned |
| **volatility** | breakout, regime, mean-reversion, clustering |
| **signal_reversal** | reversal, strength decay, divergence, consistency |
| **convergence** | single-target (three variants) and multi-target |

`exitkit.FAMILIES` maps each family name to its classes, which is how the test suite exercises all
twenty-seven uniformly.

## Missing market data raises. It is never defaulted.

This is the design decision the package exists to enforce, and it is the defect it was extracted
with. Every model read its inputs like this:

```python
current_price = market_data.get('spot_price', 350.0)
current_vol   = market_data.get('implied_vol', 0.2)
```

Thirty-six such sites. A caller who omitted a field did not get an error — they got an exit
decision computed against a fabricated price. The volatility one is quieter still: it appears in
ratio denominators, so a missing value produces a vol ratio of exactly `1.0`, which reads as "no
change" rather than as "no data".

Inputs are now checked at the boundary and name what is missing:

```
MissingMarketData: market data is missing 'spot_price'; got: implied_vol.
Exit models require this field - supply it rather than letting a default stand
in, which silently fabricates the decision.
```

`None`, `NaN` and unparseable values count as missing; `0.0` does not.

## Two more defects found while testing

**Time-based exits could not fire.** `check_time_exit` read `position.get_holding_hours()`, which
divided `holding_period` — a field only ever assigned inside `Position.update_pnl()`. A model that
did not first mark the position saw zero hours held, so a position held nine hours against a
two-hour limit did not exit. Holding time is now derived from `entry_time` rather than depending on
another call's side effect. A derived quantity should not require someone else's bookkeeping.

**`MarketHoursExitModel` had never run.** The module called `time.time()` without importing `time`,
so every invocation raised `NameError`. Fixed, and covered.

Both have regression tests named after the defect.

## Tests

```bash
pip install -e ".[test]"
pytest -q
```

128 tests. Four of them are parametrized across all twenty-seven models, so every model must
construct, must refuse to decide on empty market data, must run on complete data, and must return
nothing when there are no positions. Adding a model to `FAMILIES` puts it under all four
automatically.

## Scope

Exit decisions only. No data feed, no order routing, no backtest loop — hand it positions and
market data from whatever you already use. `SignalOutput` is a plain dataclass, so wiring it into
an existing engine is a translation layer, not an adoption.

## Licence

MIT.

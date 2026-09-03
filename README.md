# exitkit

**Swap your exit policy the way you swap your entry signal.**

[![tests](https://github.com/charlieyanhx/exitkit/actions/workflows/tests.yml/badge.svg)](https://github.com/charlieyanhx/exitkit/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/exitkit.svg)](https://pypi.org/project/exitkit/)
[![Python](https://img.shields.io/pypi/pyversions/exitkit.svg)](https://pypi.org/project/exitkit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Twenty-seven exit models in six families behind one interface. Entry logic is well served by
open-source backtesting libraries; exit logic usually is not — most ship one stop and one target
and leave the rest to you.

```bash
pip install exitkit
```

## Thirty seconds

```python
import time
from exitkit import StopLossExitModel, Position, SignalOutput

position = Position(
    position_id="p1",
    entry_time=time.time() - 3600,
    entry_signal=SignalOutput(direction=1, meta={"implied_vol": 0.18}),
    entry_price=400.0,
    quantity=10,
)

model = StopLossExitModel(stop_loss_pct=0.02, trailing=True)

for signal in model.generate_exit_signals(
    [position], {"spot_price": 391.0, "implied_vol": 0.21}
):
    print(signal.exit_reason, signal.meta["loss_pct"], signal.confidence)
```

```
stop_loss -0.0225 1.0
```

Every model takes `(positions, market_data)` and returns `SignalOutput` objects carrying the
position they close and why. Swapping policy is swapping the constructor.

## The six families

| Family | Models |
|---|---|
| **stop_loss** | fixed, adaptive, volatility-scaled, time-decayed |
| **take_profit** | fixed, partial, adaptive, scaling, momentum-aware |
| **time_based** | fixed horizon, time decay, adaptive, market hours, performance-conditioned |
| **volatility** | breakout, regime, mean-reversion, clustering |
| **signal_reversal** | reversal, strength decay, divergence, consistency |
| **convergence** | single-target (three variants) and multi-target |

```python
from exitkit import FAMILIES

for name, models in FAMILIES.items():
    print(name, [m.__name__ for m in models])
```

`FAMILIES` is also how the test suite exercises every model uniformly — adding a model puts it
under the whole battery automatically.

## Missing market data raises

The one opinion this library holds. Required fields are checked at the boundary and name what
is absent:

```python
model.generate_exit_signals([position], {"implied_vol": 0.21})
```

```
MissingMarketData: market data is missing 'spot_price'; got: implied_vol.
Exit models require this field - supply it rather than letting a default stand
in, which silently fabricates the decision.
```

`None`, `NaN` and unparseable values count as missing. `0.0` does not.

## Where this fits

`exitkit` decides *when to close*. It does not fetch data, route orders, or run a backtest loop —
hand it positions and market data from whatever you already use.

| If you want | Use |
|---|---|
| A full backtest engine | [backtesting.py](https://github.com/kernc/backtesting.py), [vectorbt](https://github.com/polakowo/vectorbt) |
| One trailing stop, built in | `backtesting.py`'s `TrailingStrategy` |
| Intrabar stop/target fills | [wickra-backtest](https://pypi.org/project/wickra-backtest/) |
| Many exit policies to compare | **exitkit** |

`SignalOutput` is a plain dataclass, so wiring it into an existing engine is a translation layer,
not an adoption.

## Why this exists

The catalogue was extracted from a private options-research program. Writing the test suite
surfaced three defects that had survived in running code, all fixed here with regression tests
named after them.

**Thirty-six fabricated market-data fallbacks.** Every model read its inputs as
`market_data.get('spot_price', 350.0)` or `.get('implied_vol', 0.2)`. A caller who omitted a
field did not get an error — they got an exit decision computed against an invented price. The
volatility default is quieter still: it appears in ratio denominators, so a missing value
produces a vol ratio of exactly `1.0`, which reads as "no change" rather than "no data". That is
why the boundary check above exists.

**Time-based exits could not fire.** `check_time_exit` read `position.get_holding_hours()`, which
divided `holding_period` — a field only ever assigned inside `Position.update_pnl()`. A model
that did not first mark the position saw zero hours held, so a position held nine hours against a
two-hour limit did not exit. Holding time is now derived from `entry_time`; a derived quantity
should not depend on another call's side effect.

**`MarketHoursExitModel` had never run.** The module called `time.time()` without importing
`time`, so every invocation raised `NameError`.

The feature-window argument was also removed from the required position in the signature: it was
the first parameter of every model and not one of them read it.

## Tests

```bash
pip install -e ".[test]"
pytest -q
```

130 tests. Four are parametrized across all twenty-seven models, so each must construct, refuse
to decide on empty market data, run on complete data, and return nothing when there are no
positions.

## Licence

MIT. See [CHANGELOG.md](CHANGELOG.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

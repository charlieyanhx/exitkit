# Changelog

## 0.1.0

First release. Twenty-seven exit models in six families, extracted from a private
research program with three pre-existing defects fixed.

### Added
- Six families — stop-loss, take-profit, time-based, volatility, signal-reversal and
  convergence — behind one `ExitSignalModel` interface.
- `FAMILIES` registry, which the test suite uses to exercise every model uniformly.
- `MissingMarketData` — required inputs are checked at the boundary.

### Fixed
- **Thirty-six fabricated market-data fallbacks.** Models read their inputs as
  `market_data.get('spot_price', 350.0)` and `.get('implied_vol', 0.2)`, so an omitted
  field produced an exit decision computed against an invented price rather than an
  error. The volatility default sat in ratio denominators, yielding exactly `1.0` —
  "no change" rather than "no data".
- **Time-based exits could not fire.** `check_time_exit` read `get_holding_hours()`,
  which divided `holding_period` — a field only ever assigned inside `update_pnl()`. A
  model that did not first mark the position saw zero hours held, so a position held
  nine hours against a two-hour limit did not exit. Holding time is now derived from
  `entry_time`.
- **`MarketHoursExitModel` had never run**: the module called `time.time()` with no
  `import time`, raising `NameError` on every invocation.
- **`timestamp` fell back to the wall clock.** `MarketHoursExitModel` read
  `market_data.get('timestamp', time.time())`, so a backtest evaluated market hours
  against the operator's clock rather than simulated time — a run at 02:00 would hold
  everything. Found by CI, which runs in a different timezone than the author.
- **`momentum` fell back to `0.0`**, which is a meaningful value ("flat") rather than a
  gap. Both fields are now required.

### Changed
- `generate_exit_signals(positions, market_data, W_t=None)` — the feature window was
  previously the first required argument and no model read it.

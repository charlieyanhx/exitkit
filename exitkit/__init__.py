"""exitkit - a catalogue of position-exit policies.

Entry logic is well served by open-source backtesting libraries; exit logic
usually is not. This packages thirty-two exit models in six families behind one
interface, so exit policy can be swapped and compared rather than hardcoded.

    from exitkit import StopLossExitModel, Position, SignalOutput

    model = StopLossExitModel(stop_loss_pct=0.02, trailing=True)
    signals = model.generate_exit_signals(window, open_positions,
                                          {"spot_price": 412.30, "implied_vol": 0.18})

Required market-data fields raise if absent. They are never defaulted.
"""

from .base import (
    ExitSignalModel,
    PnLBasedExitModel,
    SignalBasedExitModel,
    ThresholdExitModel,
    TimeBasedExitModel,
    VolatilityBasedExitModel,
)
from .convergence import (
    Conv1ExitModel,
    Conv2ExitModel,
    Conv3ExitModel,
    ConvergenceExitModel,
    MultiConvergenceExitModel,
)
from .marketdata import MissingMarketData, require, require_entry
from .signal_reversal import (
    SignalConsistencyExitModel,
    SignalDivergenceExitModel,
    SignalReversalExitModel,
    SignalStrengthDecayExitModel,
)
from .stop_loss import (
    AdaptiveStopLossExitModel,
    StopLossExitModel,
    TimeDecayStopLossExitModel,
    VolatilityStopLossExitModel,
)
from .take_profit import (
    AdaptiveTakeProfitExitModel,
    MomentumTakeProfitExitModel,
    PartialTakeProfitExitModel,
    ScalingTakeProfitExitModel,
    TakeProfitExitModel,
)
from .time_based import (
    AdaptiveTimeExitModel,
    FixedTimeExitModel,
    MarketHoursExitModel,
    PerformanceTimeExitModel,
    TimeDecayExitModel,
)
from .types import Position, PositionTracker, SignalOutput, WindowTensor
from .volatility import (
    VolatilityBreakoutExitModel,
    VolatilityClusteringExitModel,
    VolatilityMeanReversionExitModel,
    VolatilityRegimeExitModel,
)

__version__ = "0.1.0"

FAMILIES = {
    "stop_loss": [StopLossExitModel, AdaptiveStopLossExitModel,
                  VolatilityStopLossExitModel, TimeDecayStopLossExitModel],
    "take_profit": [TakeProfitExitModel, PartialTakeProfitExitModel,
                    AdaptiveTakeProfitExitModel, ScalingTakeProfitExitModel,
                    MomentumTakeProfitExitModel],
    "time_based": [FixedTimeExitModel, TimeDecayExitModel, AdaptiveTimeExitModel,
                   MarketHoursExitModel, PerformanceTimeExitModel],
    "volatility": [VolatilityBreakoutExitModel, VolatilityRegimeExitModel,
                   VolatilityMeanReversionExitModel, VolatilityClusteringExitModel],
    "signal_reversal": [SignalReversalExitModel, SignalStrengthDecayExitModel,
                        SignalDivergenceExitModel, SignalConsistencyExitModel],
    "convergence": [ConvergenceExitModel, Conv1ExitModel, Conv2ExitModel,
                    Conv3ExitModel, MultiConvergenceExitModel],
}

__all__ = [n for fam in FAMILIES.values() for n in [c.__name__ for c in fam]] + [
    "ExitSignalModel", "ThresholdExitModel", "TimeBasedExitModel",
    "PnLBasedExitModel", "VolatilityBasedExitModel", "SignalBasedExitModel",
    "Position", "PositionTracker", "SignalOutput", "WindowTensor",
    "MissingMarketData", "require", "require_entry", "FAMILIES", "__version__",
]

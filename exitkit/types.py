"""The small value types the exit models operate on."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SignalOutput:
    """A signal and its metadata. Entries carry direction -1/+1; exits carry 0."""

    direction: int
    kappa: float = 0.0
    confidence: float = 0.0
    horizon: float = 0.0
    signal_type: str = "entry"
    position_id: Optional[str] = None
    exit_reason: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}
        if self.signal_type not in ("entry", "exit"):
            raise ValueError(f"signal_type must be 'entry' or 'exit', got {self.signal_type!r}")
        if self.signal_type == "exit" and self.direction != 0:
            raise ValueError("exit signals must have direction 0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must lie in [0, 1], got {self.confidence}")


@dataclass
class Position:
    """An open position and its running P&L."""

    position_id: str
    entry_time: float
    entry_signal: SignalOutput
    entry_price: float
    quantity: int
    current_pnl: float = 0.0
    holding_period: float = 0.0

    def update_pnl(self, current_price: float):
        self.current_pnl = (
            (current_price - self.entry_price) * self.quantity * self.entry_signal.direction
        )
        self.holding_period = time.time() - self.entry_time

    def get_pnl_pct(self) -> float:
        notional = self.entry_price * self.quantity
        return 0.0 if notional == 0 else self.current_pnl / notional

    def get_holding_hours(self, now: Optional[float] = None) -> float:
        """Hours held, measured against `now` (epoch seconds).

        Pass the simulation clock when backtesting. Omitting it measures against
        the wall clock, which is right for live trading and wrong for a replay -
        a backtest would otherwise age every position to the moment you happened
        to run it.

        This used to return ``holding_period / 3600``, where holding_period was
        only ever assigned inside update_pnl(). Any model that did not first mark
        the position saw zero hours held, so every time-based exit was
        structurally unable to fire.
        """
        reference = time.time() if now is None else now
        return max(0.0, (reference - self.entry_time) / 3600.0)


class WindowTensor:
    """A rolling window of feature arrays, most recent last."""

    def __init__(self, window_list: Optional[List[np.ndarray]] = None, K: int = 1):
        window_list = window_list or []
        self.K = K
        self.window = window_list[-K:] if window_list else []
        first = self.window[0] if self.window else None
        if first is not None and hasattr(first, "shape"):
            self.B = first.shape[0]
            self.d = first.shape[2] if len(first.shape) > 2 else first.shape[-1]
        else:
            self.B, self.d = 1, 0

    def roll(self, new_ft: np.ndarray):
        self.window.append(new_ft)
        if len(self.window) > self.K:
            self.window.pop(0)

    def __len__(self) -> int:
        return len(self.window)


class PositionTracker:
    """Track open positions by id."""

    def __init__(self, max_positions: int = 100):
        self.max_positions = max_positions
        self.positions: Dict[str, Position] = {}

    def open(self, position: Position) -> str:
        if len(self.positions) >= self.max_positions:
            raise RuntimeError(f"position limit reached ({self.max_positions})")
        if position.position_id in self.positions:
            raise ValueError(f"duplicate position id {position.position_id!r}")
        self.positions[position.position_id] = position
        return position.position_id

    def close(self, position_id: str) -> Optional[Position]:
        return self.positions.pop(position_id, None)

    def open_positions(self) -> List[Position]:
        return list(self.positions.values())

    def mark(self, current_price: float):
        for p in self.positions.values():
            p.update_pnl(current_price)

    def __len__(self) -> int:
        return len(self.positions)

import time

import numpy as np
import pytest

from exitkit import Position, SignalOutput, WindowTensor

ENTRY_PRICE = 400.0
ENTRY_VOL = 0.20


@pytest.fixture
def entry_signal():
    def _make(direction=1, **meta):
        m = {"implied_vol": ENTRY_VOL, "entry_price": ENTRY_PRICE}
        m.update(meta)
        return SignalOutput(direction=direction, kappa=1.0, confidence=0.8,
                            horizon=1.0, signal_type="entry", meta=m)
    return _make


@pytest.fixture
def position(entry_signal):
    def _make(direction=1, held_hours=1.0, quantity=10, **meta):
        return Position(
            position_id="p1",
            entry_time=time.time() - held_hours * 3600,
            entry_signal=entry_signal(direction=direction, **meta),
            entry_price=ENTRY_PRICE,
            quantity=quantity,
        )
    return _make


@pytest.fixture
def market():
    """A complete market-data dict. Tests that need a gap remove a key."""
    def _make(**over):
        d = {
            "spot_price": ENTRY_PRICE,
            "implied_vol": ENTRY_VOL,
            "momentum": 0.0,
            "timestamp": time.time(),
            "svi_fit_current": 0.20,
            "svi_fit_previous_day": 0.20,
            "svi_envelope_max": 0.25,
        }
        d.update(over)
        return d
    return _make


@pytest.fixture
def window():
    return WindowTensor([np.zeros((1, 4, 3))], K=1)

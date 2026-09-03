"""Boundary checks on the market data an exit model is handed.

Every model in this package used to read its inputs as
``market_data.get('spot_price', 350.0)`` and ``.get('implied_vol', 0.2)``.
A caller who forgot a field did not get an error - they got an exit decision
computed against a fabricated price, or a vol ratio of exactly 1.0, which is
the most innocuous-looking answer available. Thirty-six such sites existed.

Missing inputs now fail here, at the boundary, naming the field and the model.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable


class MissingMarketData(KeyError):
    """A required market-data field was absent or unusable."""

    def __init__(self, field: str, available: Iterable[str] = ()):
        have = ", ".join(sorted(available)) or "nothing"
        super().__init__(
            f"market data is missing {field!r}; got: {have}. "
            f"Exit models require this field - supply it rather than letting a "
            f"default stand in, which silently fabricates the decision."
        )
        self.field = field


def require(market_data: Dict[str, Any], field: str) -> float:
    """Read a required numeric field from a market-data dict."""
    if not isinstance(market_data, dict):
        raise MissingMarketData(field)
    value = market_data.get(field)
    if value is None:
        raise MissingMarketData(field, market_data.keys())
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise MissingMarketData(field, market_data.keys()) from None
    if value != value:  # NaN
        raise MissingMarketData(field, market_data.keys())
    return value


def require_entry(position, field: str) -> float:
    """Read a required numeric field recorded on a position's entry signal."""
    meta = getattr(getattr(position, "entry_signal", None), "meta", None) or {}
    if field not in meta or meta[field] is None:
        raise MissingMarketData(f"entry_signal.meta[{field!r}]", meta.keys())
    return require(dict(meta), field)

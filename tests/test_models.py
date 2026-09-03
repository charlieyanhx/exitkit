"""Every model in the catalogue: constructs, refuses fabricated inputs, fires."""
import pytest

from exitkit import FAMILIES, MissingMarketData

ALL_MODELS = [(fam, cls) for fam, classes in FAMILIES.items() for cls in classes]
IDS = [f"{fam}:{cls.__name__}" for fam, cls in ALL_MODELS]


@pytest.mark.parametrize("family,cls", ALL_MODELS, ids=IDS)
def test_model_constructs_with_defaults(family, cls):
    model = cls()
    assert model.name
    assert isinstance(model.get_model_info(), dict)


@pytest.mark.parametrize("family,cls", ALL_MODELS, ids=IDS)
def test_model_refuses_empty_market_data(family, cls, window, position):
    """The regression that motivated this package.

    Every model read its inputs as market_data.get('spot_price', 350.0) or
    .get('implied_vol', 0.2) - 36 sites. A caller who omitted a field got an
    exit decision computed from a fabricated price, or a vol ratio of exactly
    1.0, which looks like "no change" rather than like an error. No model may
    now return a decision it could not compute.
    """
    model = cls()
    try:
        signals = model.generate_exit_signals([position()], {})
    except MissingMarketData:
        return                      # refused - correct
    assert signals == [], (
        f"{cls.__name__} produced {len(signals)} signal(s) from empty market data"
    )


@pytest.mark.parametrize("family,cls", ALL_MODELS, ids=IDS)
def test_model_runs_on_complete_market_data(family, cls, window, position, market):
    model = cls()
    signals = model.generate_exit_signals([position()], market())
    assert isinstance(signals, list)
    for s in signals:
        assert s.signal_type == "exit" and s.direction == 0
        assert s.position_id == "p1"
        assert s.exit_reason


@pytest.mark.parametrize("family,cls", ALL_MODELS, ids=IDS)
def test_no_positions_means_no_signals(family, cls, window, market):
    assert cls().generate_exit_signals([], market()) == []


# --- behaviour, family by family -------------------------------------------

def test_stop_loss_fires_on_a_loss(window, position, market):
    from exitkit import StopLossExitModel
    m = StopLossExitModel(stop_loss_pct=0.02)
    p = position(direction=1)
    down = market(spot_price=380.0)                 # -5% against a long
    signals = m.generate_exit_signals([p], down)
    assert len(signals) == 1 and signals[0].exit_reason == "stop_loss"


def test_stop_loss_holds_on_a_small_move(window, position, market):
    from exitkit import StopLossExitModel
    m = StopLossExitModel(stop_loss_pct=0.10)
    signals = m.generate_exit_signals([position()], market(spot_price=399.0))
    assert signals == []


def test_take_profit_fires_on_a_gain(window, position, market):
    from exitkit import TakeProfitExitModel
    m = TakeProfitExitModel(take_profit_pct=0.02)
    signals = m.generate_exit_signals([position()], market(spot_price=420.0))
    assert len(signals) == 1 and signals[0].exit_reason


def test_fixed_time_exit_respects_its_horizon(window, position, market):
    from exitkit import FixedTimeExitModel
    m = FixedTimeExitModel(max_holding_hours=2.0)
    assert m.generate_exit_signals([position(held_hours=0.5)], market()) == []
    late = m.generate_exit_signals([position(held_hours=9.0)], market())
    assert len(late) == 1


def test_a_short_position_loses_when_price_rises(window, position, market):
    from exitkit import StopLossExitModel
    m = StopLossExitModel(stop_loss_pct=0.02)
    short = position(direction=-1)
    signals = m.generate_exit_signals([short], market(spot_price=420.0))
    assert len(signals) == 1, "a short must lose on a rally"


def test_signals_carry_the_position_they_close(window, position, market):
    from exitkit import StopLossExitModel
    m = StopLossExitModel(stop_loss_pct=0.01)
    signals = m.generate_exit_signals([position()], market(spot_price=350.0))
    assert signals and all(s.position_id == "p1" for s in signals)

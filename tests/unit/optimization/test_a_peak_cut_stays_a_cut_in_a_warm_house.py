"""In the dearest quarter of the day a warm house is cut, whatever its degree-minute debt.

The strategic-overshoot clause takes back a pre-heat when the house is warmer than the mode allows,
down to zero. In thermal debt (DM below -400) the price layer's debt gate turns a negative forecast
positive, and a PEAK quarter always has cheaper quarters ahead, so the clause ran on the PEAK cut:
its zero floor raised -2 to 0, at weight 1.0, and the comfort layer's coast was overridden with it.
A warm house in debt - the ordinary winter state - got neither the cut nor the coast in the quarter
where coasting is cheapest (control review F-4; the same on v0.6.2 and v0.7.0).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

MIDNIGHT = datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)
NOW = MIDNIGHT + timedelta(hours=12, minutes=5)
TARGET = 21.0


def _engine() -> DecisionEngine:
    return DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(0.7, 1.0),
        config={
            "target_indoor_temp": TARGET,
            "tolerance": 0.5,
            "optimization_mode": "balanced",
            "latitude": 59.33,
            "heating_type": "radiator",
        },
        heat_pump_model=NibeF750Profile(),
    )


def _prices(dearest: float) -> PriceData:
    """An ordinary day with its dearest two hours around noon, at `dearest` öre."""
    prices = [50.0 + (q % 7) * 3.0 for q in range(96)]
    for q in range(46, 54):
        prices[q] = dearest
    return PriceData(
        today=[
            QuarterPeriod(MIDNIGHT + timedelta(minutes=15 * q), p) for q, p in enumerate(prices)
        ],
        tomorrow=[],
        has_tomorrow=False,
    )


def _price_vote(indoor: float, degree_minutes: float, dearest: float = 400.0):
    state = NibeState(
        outdoor_temp=-10.0,
        indoor_temp=indoor,
        supply_temp=40.0,
        return_temp=35.0,
        degree_minutes=degree_minutes,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=NOW,
        compressor_hz=60,
        power_kw=2.0,
    )
    decision = _engine().calculate_decision(state, _prices(dearest), None, 0.0, 0.0)
    return next(layer for layer in decision.layers if layer.name == "Spot Price")


@pytest.fixture(autouse=True)
def _noon(freezer):
    freezer.move_to(NOW)


def test_a_warm_house_in_debt_is_cut_in_a_peak_quarter():
    vote = _price_vote(indoor=TARGET + 0.6, degree_minutes=-600.0)

    assert vote.offset < 0 and vote.weight == 1.0, (
        f"A PEAK quarter, the house 0.6 C over target and DM -600: the price layer voted "
        f"{vote.offset:+.2f} at weight {vote.weight} ({vote.reason}). The overshoot clause's zero "
        f"floor raised the PEAK cut."
    )


@pytest.mark.parametrize(
    ("indoor", "degree_minutes"), [(TARGET + 0.4, -600.0), (TARGET + 0.6, -300.0)]
)
def test_the_cut_is_the_same_cut_either_side_of_the_debt_and_overshoot_lines(
    indoor, degree_minutes
):
    """Under the overshoot limit, or out of debt, the clause never ran: the cut was already there."""
    assert _price_vote(indoor, degree_minutes).offset == pytest.approx(
        _price_vote(TARGET + 0.6, -600.0).offset
    )

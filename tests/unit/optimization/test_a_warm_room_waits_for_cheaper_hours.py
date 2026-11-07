"""In an expensive quarter a warm room waits for the cheaper hours close ahead.

Production on 6 Nov 2026, 07-09 local: EXPENSIVE quarters (381-383 ore), the room 0.6 C over its
21 C target, DM -398..-371 after a cold night, and the layers' vote, about -1, kept the pump at
56-60 Hz repaying the degree minutes. The room then climbed to 22.5 C by 13:00 on the slab's heat
while the cheapest daytime hours (260-326 ore) passed with the pump near idle.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    DM_TIER_EMERGENCY,
    OVERSHOOT_PROTECTION_DEEPEST,
    OVERSHOOT_PROTECTION_START,
    WAIT_CHEAPER_ROOM_OVER,
    WEATHER_COMP_DEFER_DM_CRITICAL,
    QuarterClassification,
)
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

MIDNIGHT = datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)
NOW = MIDNIGHT + timedelta(hours=12, minutes=5)  # inside the dearest two hours
TARGET = 21.0
WARM = TARGET + 0.6
DEEP = WEATHER_COMP_DEFER_DM_CRITICAL - 100  # the pump repaying hard
SHALLOW = WEATHER_COMP_DEFER_DM_CRITICAL + 100


def _engine() -> DecisionEngine:
    return DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(0.7, 1.0),
        config={
            "target_indoor_temp": TARGET,
            "tolerance": 1.5,
            "optimization_mode": "balanced",
            "latitude": 59.33,
            "heating_type": "radiator",
        },
        heat_pump_model=NibeF750Profile(),
    )


def _prices(cheaper_after: bool = True) -> PriceData:
    """Noon's two hours at 400 ore; after them the ordinary 50-68 ore, or 400 to the end of the day."""
    prices = [50.0 + (q % 7) * 3.0 for q in range(96)]
    for q in range(46, 54 if cheaper_after else 96):
        prices[q] = 400.0
    return PriceData(
        today=[
            QuarterPeriod(MIDNIGHT + timedelta(minutes=15 * q), p) for q, p in enumerate(prices)
        ],
        tomorrow=[],
        has_tomorrow=False,
    )


def _state(indoor: float, degree_minutes: float, at: datetime = NOW) -> NibeState:
    return NibeState(
        outdoor_temp=-2.0,
        indoor_temp=indoor,
        supply_temp=35.0,
        return_temp=30.0,
        degree_minutes=degree_minutes,
        current_offset=-1.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=at,
        compressor_hz=58,
        power_kw=2.5,
    )


def _decide(engine, indoor, degree_minutes, prices=None, at: datetime = NOW):
    return engine.calculate_decision(
        _state(indoor, degree_minutes, at), prices or _prices(), None, 0.0, 0.0
    )


@pytest.fixture(autouse=True)
def _noon(freezer):
    freezer.move_to(NOW)


def test_a_warm_room_repaying_degree_minutes_waits_in_an_expensive_quarter():
    decision = _decide(_engine(), WARM, DEEP)

    assert decision.offset == OVERSHOOT_PROTECTION_DEEPEST, decision.reasoning
    assert "Warm room waits for cheaper hours lowered" in decision.reasoning
    assert "price -" in decision.reasoning, "the note names the cheaper hours ahead"


@pytest.mark.parametrize(
    ("indoor", "degree_minutes", "cheaper_after"),
    [
        pytest.param(TARGET + WAIT_CHEAPER_ROOM_OVER - 0.1, DEEP, True, id="room-not-warm"),
        pytest.param(
            TARGET + OVERSHOOT_PROTECTION_START, DEEP, True, id="room-coasting-but-not-warm-enough"
        ),
        pytest.param(WARM, SHALLOW, True, id="pump-not-repaying"),
        pytest.param(WARM, DEEP, False, id="no-cheaper-hours-ahead"),
    ],
)
def test_without_each_condition_the_room_does_not_wait(indoor, degree_minutes, cheaper_after):
    decision = _decide(_engine(), indoor, degree_minutes, _prices(cheaper_after))

    assert decision.offset > OVERSHOOT_PROTECTION_DEEPEST, decision.reasoning
    assert "Warm room waits" not in decision.reasoning


def test_an_ordinary_quarter_does_not_wait():
    """08:05 at the day's ordinary price, with a cheaper hour from 09:00: only the class stops it."""
    prices = [100.0] * 96
    for q in range(36, 40):
        prices[q] = 50.0
    for q in range(46, 54):
        prices[q] = 400.0
    day = PriceData(
        today=[
            QuarterPeriod(MIDNIGHT + timedelta(minutes=15 * q), p) for q, p in enumerate(prices)
        ],
        tomorrow=[],
        has_tomorrow=False,
    )
    morning = MIDNIGHT + timedelta(hours=8, minutes=5)
    decision = _decide(_engine(), WARM, DEEP, day, at=morning)

    assert "Warm room waits" not in decision.reasoning


def test_once_waiting_the_room_keeps_waiting_while_the_degree_minutes_recover():
    """The hold itself lifts DM above the line; re-deciding on it switched the hold on and off."""
    engine = _engine()
    assert _decide(engine, WARM, DEEP).offset == OVERSHOOT_PROTECTION_DEEPEST

    later = NOW + timedelta(minutes=5)
    assert _decide(engine, WARM, SHALLOW, at=later).offset == OVERSHOOT_PROTECTION_DEEPEST


def test_the_wait_ends_with_the_warm_room_and_does_not_restart_on_shallow_degree_minutes():
    engine = _engine()
    _decide(engine, WARM, DEEP)
    cooled = _decide(engine, TARGET, SHALLOW, at=NOW + timedelta(minutes=5))
    warm_again = _decide(engine, WARM, SHALLOW, at=NOW + timedelta(minutes=10))

    assert "Warm room waits" not in cooled.reasoning
    assert (
        "Warm room waits" not in warm_again.reasoning
    ), "the wait starts only while the pump repays degree minutes past the line"


def test_the_thermal_debt_ladder_is_never_cut():
    engine = _engine()
    engine.price.update_prices(_prices())
    emergency = SimpleNamespace(tier=DM_TIER_EMERGENCY)

    offset, note = engine._wait_for_cheaper_hours(
        0.0, _state(WARM, DEEP), _prices(), QuarterClassification.PEAK, emergency
    )

    assert (offset, note) == (0.0, "")

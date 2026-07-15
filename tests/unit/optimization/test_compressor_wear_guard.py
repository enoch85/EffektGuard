"""When the compressor is saturated, a higher offset buys no heat - only wear and DM deficit.

The offset raises the pump's supply setpoint S1. Once the compressor is at maximum frequency a
higher S1 produces no extra heat; it only holds the machine flat out longer and deepens the
degree-minute deficit (DM = integral(BT25 - S1)), which the auxiliary heater exists to absorb
(F-124). So when CompressorHealthMonitor reports COMPRESSOR_RISK_HIGH, the decision engine
declines to ask for MORE: `_apply_compressor_wear_guard` HOLDS the offset. It never CUTS the
offset - the boost was producing no heat, so declining it costs no comfort - and it never
overrides the absolute safety floor.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import COMPRESSOR_RISK_HIGH
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

NOW = datetime(2026, 1, 15, 12, 0)


def _engine() -> DecisionEngine:
    config = {
        "target_indoor_temp": 21.0,
        "tolerance": 0.5,
        "optimization_mode": "balanced",
        "latitude": 59.33,
        "heating_type": "radiator",
        "heat_loss_coefficient": 150.0,
        "thermal_mass": 0.7,
        "insulation_quality": 1.0,
    }
    return DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(0.7, 1.0),
        config=config,
        heat_pump_model=NibeF750Profile(),
    )


def _state(degree_minutes: float, indoor: float = 20.0, hz: int = 115) -> NibeState:
    """A house in thermal debt with the compressor already flat out."""
    return NibeState(
        outdoor_temp=-12.0,
        indoor_temp=indoor,
        supply_temp=45.0,
        return_temp=40.0,
        degree_minutes=degree_minutes,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=NOW,
        compressor_hz=hz,
        power_kw=3.0,
    )


def _decide(engine: DecisionEngine, state: NibeState, risk: str | None):
    return engine.calculate_decision(
        nibe_state=state,
        price_data=None,
        weather_data=None,
        current_peak=0.0,
        current_power=3.0,
        compressor_risk=risk,
    )


def test_a_saturated_compressor_is_not_asked_for_more():
    """At HIGH risk the boost cannot produce heat, so it must not be demanded."""
    engine = _engine()
    state = _state(degree_minutes=-600.0)

    unguarded = _decide(engine, state, risk=None).offset
    guarded = _decide(engine, state, risk=COMPRESSOR_RISK_HIGH).offset

    assert unguarded > 0.5, "precondition: without the guard the engine wants to boost"
    assert guarded < unguarded, (
        f"The compressor has been above 100 Hz for over fifteen minutes - it is at maximum and "
        f"has nothing left to give. The engine still demanded {guarded:+.2f} (unguarded: "
        f"{unguarded:+.2f}). That offset buys no heat, only wear and a deeper DM deficit."
    )


def test_the_guard_holds_heat_it_never_cuts_it():
    """A wear guard that cools the house is not a wear guard, it is a fault."""
    engine = _engine()
    state = _state(degree_minutes=-600.0)

    guarded = _decide(engine, state, risk=COMPRESSOR_RISK_HIGH).offset

    assert guarded >= 0.0, (
        f"The guard reduced the offset to {guarded:+.2f}, taking heat AWAY from a house that is "
        f"already in thermal debt. It may decline to ask for MORE; it may never ask for less."
    )


def test_the_absolute_safety_floor_still_wins():
    """A house below the hard minimum gets everything, whatever the compressor is doing."""
    engine = _engine()
    freezing = _state(degree_minutes=-600.0, indoor=17.0)  # below MIN_TEMP_LIMIT

    decision = _decide(engine, freezing, risk=COMPRESSOR_RISK_HIGH)

    assert decision.is_emergency, "an indoor temperature below the floor is not negotiable"
    assert decision.offset > 5.0, (
        f"The house is at 17 C. The wear guard must not stand between it and the heat: got "
        f"{decision.offset:+.2f}."
    )


def test_a_healthy_compressor_is_left_alone():
    """The guard must be silent when the compressor has headroom."""
    engine = _engine()
    state = _state(degree_minutes=-600.0, hz=60)

    assert _decide(engine, state, risk=None).offset == _decide(engine, state, risk="OK").offset

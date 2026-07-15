"""The thermostat must not offer a setpoint the safety layer will fight.

MIN_TEMP_LIMIT (18.0 °C) is the absolute floor: below it the safety layer fires MAX_OFFSET as an
emergency. A settable minimum below the floor produces a limit cycle - above 18 °C the comfort
layer reads an overshoot and cuts to MIN_OFFSET, below it safety commands MAX_OFFSET, and every
safety boost is is_emergency=True so it bypasses the volatility blocker.

The floor is MIN_TARGET_TEMP (one default tolerance above the safety floor), and the engine clamps
any stored target below it up to it - stored options, migration or a hand-edited entry alike. To
move the floor, change MIN_TEMP_LIMIT, not the slider.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard import climate as climate_module
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    DEFAULT_TOLERANCE,
    MAX_OFFSET,
    MIN_OFFSET,
    MIN_TARGET_TEMP,
    MIN_TEMP_LIMIT,
)
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel


def _engine(target: float) -> DecisionEngine:
    return DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(0.7, 1.0),
        config={
            "target_indoor_temp": target,
            "tolerance": 0.5,
            "optimization_mode": "balanced",
            "latitude": 59.33,
            "heating_type": "radiator",
            "heat_loss_coefficient": 150.0,
            "thermal_mass": 0.7,
            "insulation_quality": 1.0,
        },
        heat_pump_model=NibeF750Profile(),
    )


def _state(indoor: float) -> NibeState:
    return NibeState(
        outdoor_temp=-5.0,
        indoor_temp=indoor,
        supply_temp=38.0,
        return_temp=33.0,
        degree_minutes=-100.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 1, 15, 12, 0),
        compressor_hz=50,
        power_kw=2.0,
    )


def test_the_thermostat_does_not_offer_a_setpoint_below_the_safety_floor():
    """The slider and the safety layer must agree on the lowest permitted temperature.

    Checked in the source: HA's CachedProperties metaclass turns `_attr_min_temp` into a descriptor,
    so reading the class attribute would compare a property to a float, not fail honestly.
    """
    # The invariant, not the constant's name: the lowest target the thermostat offers must sit far
    # enough above the safety floor that the comfort band around it clears the floor entirely.
    assert MIN_TARGET_TEMP >= MIN_TEMP_LIMIT + DEFAULT_TOLERANCE, (
        f"The lowest offered target ({MIN_TARGET_TEMP} °C) does not clear the safety floor "
        f"({MIN_TEMP_LIMIT} °C) by a tolerance ({DEFAULT_TOLERANCE} °C). A target sitting AT the "
        f"floor puts the lower half of its own comfort band inside the emergency zone: ordinary "
        f"control noise then trips a full MAX_OFFSET boost that bypasses the volatility blocker."
    )

    source = inspect.getsource(climate_module)
    assert "_attr_min_temp = MIN_TARGET_TEMP" in source, (
        "The climate entity's minimum target must be MIN_TARGET_TEMP - the lowest temperature this "
        "system can actually HOLD - rather than a number the safety layer will fight."
    )


@pytest.mark.parametrize("indoor", [17.9, 16.0, 15.0])
def test_a_setpoint_below_the_floor_is_answered_with_an_emergency(indoor):
    """Below the safety floor the engine commands MAX_OFFSET, whatever the stored target."""
    decision = _engine(target=15.0).calculate_decision(
        nibe_state=_state(indoor),
        price_data=None,
        weather_data=None,
        current_peak=0.0,
        current_power=2.0,
    )

    assert decision.is_emergency, f"precondition: {indoor} °C is below MIN_TEMP_LIMIT"
    assert decision.offset == MAX_OFFSET, (
        f"With a target of 15 °C and the house at {indoor} °C, the engine commands "
        f"{decision.offset:+.2f} - maximum heat - against the user's own setpoint."
    )


def test_the_house_is_not_driven_between_the_two_extremes():
    """The limit cycle, in one assertion, exercised against a stored target of 15 °C.

    With a 15 °C target the comfort layer read 19.0 °C as an overshoot and cut to MIN_OFFSET while
    safety read 17.9 °C as an emergency and commanded MAX_OFFSET. HA keeps the stored value across
    the upgrade, so the ENGINE must clamp the target - not just the slider - to protect existing
    owners.
    """
    engine = _engine(target=15.0)

    hot = engine.calculate_decision(
        nibe_state=_state(19.0),
        price_data=None,
        weather_data=None,
        current_peak=0.0,
        current_power=2.0,
    )
    cold = engine.calculate_decision(
        nibe_state=_state(17.9),
        price_data=None,
        weather_data=None,
        current_peak=0.0,
        current_power=2.0,
    )

    # The defect is the COMFORT end, not the span: safety legitimately commands +10 below 18 °C, so a
    # span measured from a quiet baseline is ~10 whether healthy or not. Assert the thing that broke:
    # the engine must not slam the heat off in a house its own safety layer is about to call cold.
    assert hot.offset > MIN_OFFSET / 2, (
        f"With a stored target of 15 °C and the house at 19.0 °C, the engine commands "
        f"{hot.offset:+.2f} - it reads the house as badly overheated and slams the heat off. One "
        f"degree lower, at 17.9 °C, it commands {cold.offset:+.2f}: the safety layer calls the same "
        f"house an emergency. The pump is driven between the extremes for as long as the setpoint "
        f"stands. A target the safety layer will fight is not a target."
    )

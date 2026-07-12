"""The thermostat offers 15 °C, and the safety layer calls 15 °C an emergency.

    const.py    MIN_INDOOR_TEMP = 15.0   # "minimum settable temperature"  -> climate._attr_min_temp
    const.py    MIN_TEMP_LIMIT  = 18.0   # the absolute floor: below this, the safety layer fires

So Home Assistant's thermostat card lets the owner dial their target down to 15 °C, and the
integration treats any indoor temperature below 18 °C as an absolute emergency and commands maximum
heat. Those two facts cannot both be honoured, and the result is not a compromise - it is a hard
limit cycle across the entire offset range:

    indoor 19.0 °C   ->  -10.00   comfort: "Overshoot: 3.0 °C above target, reducing heat"
    indoor 17.9 °C   ->  +10.00   SAFETY:  "Too cold (17.9 °C < 18.0 °C)"   <- EMERGENCY
    indoor 16.0 °C   ->  +10.00   SAFETY
    indoor 15.0 °C   ->  +10.00   SAFETY

The house is driven up by an emergency, driven down by a comfort overshoot, and back again. MIN_OFFSET
to MAX_OFFSET, on a real compressor, for as long as the setpoint stands. And every one of those
emergency boosts is is_emergency=True, so it bypasses the offset-volatility blocker that exists to
stop exactly this kind of thrashing.

Nothing warns the user. The slider simply offers a number the system will spend the winter fighting.

There is one honest number here: the lowest indoor temperature this system permits. It is the safety
floor. A setpoint the integration will treat as an emergency is not a setpoint, and offering it is
not a feature.

(If 18 °C is the wrong floor - for an away mode, a holiday, an unheated room - then MIN_TEMP_LIMIT is
the thing to change, deliberately, as a safety decision. Not a UI slider that quietly disagrees with
it.)
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

    Checked in the source. Home Assistant's CachedProperties metaclass rewrites `_attr_*` class
    attributes into properties on the subclass, so reading EffektGuardClimate._attr_min_temp gives
    the descriptor, not 18.0 - a class-level assertion here compares a property to a float and
    raises TypeError rather than failing honestly.
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
    """The precondition, so nobody has to take the docstring on trust.

    This is what the system does TODAY to a user who set 15 °C. It is not a hypothetical.
    """
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
    """The limit cycle, in one assertion - and the fix that actually protects existing owners.

    Above the floor the comfort layer sees a 3 °C overshoot and cuts to MIN_OFFSET. Below it, safety
    commands MAX_OFFSET. There is no equilibrium anywhere.

    The slider no longer OFFERS 15 °C - and that does nothing for the owner who set 15 °C before
    this landed, because Home Assistant keeps the stored value across the upgrade. So the ENGINE
    refuses a target below the floor, wherever it came from: stored options, a migration, a
    hand-edited entry. That is what this exercises - a config that still says 15.
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

    span = abs(hot.offset - cold.offset)

    assert span < MAX_OFFSET, (
        f"A 1.1 °C change in indoor temperature swings the commanded offset by {span:.1f} °C "
        f"({hot.offset:+.2f} at 19.0 °C, {cold.offset:+.2f} at 17.9 °C). The comfort layer sees an "
        f"overshoot against the 15 °C target and cuts; the safety layer sees a house below 18 °C "
        f"and boosts. The pump is driven between the extremes for as long as the setpoint stands, "
        f"and the emergency flag bypasses the volatility blocker that exists to prevent exactly "
        f"this."
    )

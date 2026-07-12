"""Seeing the cold coming is worthless if the response is a trickle.

The pre-heat layer's whole job is to charge the building fabric before a cold snap lands. It used
to ask for +0.83 (a constant then named WEATHER_GENTLE_OFFSET), and on the simulator's own
validated plant models that took:

    radiator house (tau 30 h, C 4.5 kWh/K)     28.4 h to fill the +/-1 C storage band
    concrete slab  (tau 80 h, C 14.4 kWh/K)    34.6 h

Against forecast horizons of 12 h and 24 h. The battery could not be charged before the cold
arrived - not once, not ever. The constant's own history records the struggle: "tuned Oct 20, was
0.5 -> 0.6 -> 0.7 -> 0.77". It was being nudged in hundredths when it needed to be tripled.

The sizing rule is not a matter of taste. The fabric must reach the edge of the storage band
within the horizon the house is given, or the pre-heat is decoration:

    energy to fill the band   = C_fabric * THERMAL_BATTERY_BAND
    surplus the offset buys   = offset * DEFAULT_CURVE_SENSITIVITY * dQ/dFlow
    time to fill              = energy / surplus        (must be <= the forecast horizon)

dQ/dFlow is the emitter's gain. Underfloor has a large one (a whole floor: EN 1264 gives about
1600 W/K for 140 m2); radiators have a much smaller one (EN 442, a few hundred W/K) - but a
radiator house also has far less mass to charge, so the two land in the same place.
"""

import pytest

from custom_components.effektguard.const import (
    DEFAULT_CURVE_SENSITIVITY,
    THERMAL_BATTERY_BAND,
    UFH_CONCRETE_PREDICTION_HORIZON,
    WEATHER_PREHEAT_OFFSET,
    WEATHER_FORECAST_HORIZON,
)

# Representative houses, taken from the simulator's validated plant configurations.
# (thermal capacitance kWh/K, emitter gain W per C of flow, the horizon this house is given)
RADIATOR_HOUSE = (4.5, 285.0, WEATHER_FORECAST_HORIZON)
CONCRETE_HOUSE = (14.4, 1614.0, UFH_CONCRETE_PREDICTION_HORIZON)


def _hours_to_fill_the_band(capacity_kwh_per_k: float, emitter_gain_w_per_k: float) -> float:
    """How long the pre-heat needs to charge the fabric to the edge of the storage band.

    An upper bound on the surplus, and therefore a LOWER bound on the time: it ignores the rising
    heat loss as the house warms, and the emitter's own lag. The real plant is slower. If the
    optimistic figure already exceeds the horizon, the pessimistic one certainly does.
    """
    energy_kwh = capacity_kwh_per_k * THERMAL_BATTERY_BAND
    surplus_kw = WEATHER_PREHEAT_OFFSET * DEFAULT_CURVE_SENSITIVITY * emitter_gain_w_per_k / 1000.0
    return energy_kwh / surplus_kw


@pytest.mark.parametrize(
    "capacity,gain,horizon,what",
    [
        (*RADIATOR_HOUSE, "a radiator house"),
        (*CONCRETE_HOUSE, "a concrete slab"),
    ],
)
def test_the_fabric_can_be_charged_before_the_cold_arrives(capacity, gain, horizon, what):
    """The battery must be full when the snap lands, or there was no point charging it."""
    hours = _hours_to_fill_the_band(capacity, gain)

    assert hours <= horizon, (
        f"{what} needs {hours:.1f} h to charge its fabric to the edge of the "
        f"{THERMAL_BATTERY_BAND:.0f} C storage band at a pre-heat of "
        f"{WEATHER_PREHEAT_OFFSET:+.2f}, and it only sees {horizon:.0f} h ahead. The cold arrives "
        f"first, every time, and the pre-heat is decoration."
    )


def test_the_preheat_is_not_a_trickle():
    """A guard on the sizing itself: a fraction of a degree cannot move a building.

    The old value was +0.83 and took 28-35 h on the simulator's validated plants. Anything of that
    order is inert, whatever it is called.
    """
    assert WEATHER_PREHEAT_OFFSET >= 1.5, (
        f"A pre-heat of {WEATHER_PREHEAT_OFFSET:+.2f} cannot charge a building's fabric inside a "
        f"forecast horizon. It was +0.83 and needed 28 hours on a radiator house."
    )


def test_the_preheat_is_bounded_and_hands_over():
    """It fills the band and stops. It is not licence to cook the house.

    The comfort layer takes charge at the edge of THERMAL_BATTERY_BAND, so a strong pre-heat is
    bounded by construction: it charges the fabric quickly, then hands over to comfort's overshoot
    protection. It must not exceed what any weather-driven layer is permitted to command.
    """
    from custom_components.effektguard.const import WEATHER_COMP_MAX_OFFSET

    assert WEATHER_PREHEAT_OFFSET <= WEATHER_COMP_MAX_OFFSET, (
        f"A pre-heat of {WEATHER_PREHEAT_OFFSET:+.2f} exceeds the bound placed on every other "
        f"weather-driven correction ({WEATHER_COMP_MAX_OFFSET:+.1f})."
    )

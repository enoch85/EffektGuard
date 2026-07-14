"""A slow house must be allowed to look further ahead than a fast one.

The pre-heat layer fires when the forecast shows a drop of at least
WEATHER_FORECAST_DROP_THRESHOLD within WEATHER_FORECAST_HORIZON - a FIXED twelve hours, for every
house, whatever it is built of.

A concrete slab does not get into thermal debt from a sudden plunge. The pump's own curve catches
that: the curve is reactive, but it is fast. The slab gets into debt from a SLOW, DEEP slide that
nothing notices, and a twelve-hour window cannot see one:

    cold snap                     drop within 12 h   fires?
    15 C over  6 h (plunge)            -15.0 C        yes
    15 C over 24 h                      -7.5 C        yes
    15 C over 48 h (two days)           -3.8 C        NO
    20 C over 72 h (three days)         -3.3 C        NO

Within any twelve hours of a two-day slide the temperature falls less than the four degrees needed
to trigger. The pre-heat NEVER fires. The slab is drained slowly, over days, and nothing sees it
coming - while the sudden plunge, which DOES trigger it, is the case that needed it least.

The code already knows the answer and cannot reach it. UFH_CONCRETE_PREDICTION_HORIZON is 24 hours,
commented "6+ hour lag, needs 24h for extreme cold (20C drops)". AdaptiveThermalModel returns it
correctly - and the engine passes the STATIC ThermalModel, whose get_prediction_horizon() returns a
hardcoded 12.0 for every thermal mass and says so in its own docstring. Every path to the concrete
horizon is severed.

Measured on the owner's slab (2-node transient, 100 mm ground slab + 60 mm screed): the room moves
+1.0 C in 2.4-4.6 h, but the slab is only ~19% charged at 14 h and ~29% at 24 h (slow time
constant ~70 h). Six hours is the lag; twenty-four is the MINIMUM horizon to plan over.
"""

import pytest

from custom_components.effektguard.const import (
    UFH_CONCRETE_PREDICTION_HORIZON,
    UFH_RADIATOR_PREDICTION_HORIZON,
    UFH_TIMBER_PREDICTION_HORIZON,
)
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

# The engine's own classification (decision_engine.py): >= 1.5 concrete, >= 1.2 timber, else
# radiator. The horizon must be derived from the SAME thresholds, or a house is one type for the
# heating curve and another for the forecast.
CONCRETE_SLAB = 1.8
TIMBER_UFH = 1.3
RADIATORS = 0.7


@pytest.mark.parametrize(
    "thermal_mass,expected,what",
    [
        (CONCRETE_SLAB, UFH_CONCRETE_PREDICTION_HORIZON, "a concrete slab"),
        (TIMBER_UFH, UFH_TIMBER_PREDICTION_HORIZON, "timber underfloor"),
        (RADIATORS, UFH_RADIATOR_PREDICTION_HORIZON, "radiators"),
    ],
)
def test_the_horizon_follows_the_thermal_mass(thermal_mass, expected, what):
    """The heavier the house, the further ahead it has to look. That is the whole point."""
    horizon = ThermalModel(thermal_mass, 1.0).get_prediction_horizon()

    assert horizon == expected, (
        f"{what} (thermal mass {thermal_mass}) needs a {expected:.0f} h horizon and got "
        f"{horizon:.0f} h. The static ThermalModel returns a hardcoded 12.0 whatever it is built "
        f"of, and it is the model the engine actually uses."
    )


def test_a_slab_looks_further_ahead_than_a_radiator():
    """Ordering, not just values: mass buys lag, and lag must buy look-ahead."""
    slab = ThermalModel(CONCRETE_SLAB, 1.0).get_prediction_horizon()
    timber = ThermalModel(TIMBER_UFH, 1.0).get_prediction_horizon()
    radiator = ThermalModel(RADIATORS, 1.0).get_prediction_horizon()

    assert slab > timber > radiator, (
        f"Horizons must be ordered by thermal lag: concrete {slab:.0f} h > timber {timber:.0f} h "
        f"> radiators {radiator:.0f} h."
    )


def test_a_two_day_slide_is_visible_to_a_slab():
    """The case that actually drains a slab: 15 C over 48 h.

    Within twelve hours it falls only 3.8 C - under the trigger. Within twenty-four it falls
    7.5 C, and the pre-heat can start while there is still time to charge the slab.
    """
    from custom_components.effektguard.const import WEATHER_FORECAST_DROP_THRESHOLD

    total_drop, over_hours = 15.0, 48.0
    slab_horizon = ThermalModel(CONCRETE_SLAB, 1.0).get_prediction_horizon()

    drop_seen = total_drop * min(slab_horizon, over_hours) / over_hours

    assert drop_seen >= abs(WEATHER_FORECAST_DROP_THRESHOLD), (
        f"A 15 C slide over two days shows only {drop_seen:.1f} C inside a {slab_horizon:.0f} h "
        f"window, under the {abs(WEATHER_FORECAST_DROP_THRESHOLD):.0f} C trigger. The pre-heat "
        f"never fires, and the slab is drained over days with nothing watching."
    )

"""A slow house must be allowed to look further ahead than a fast one.

The pre-heat layer fires on a forecast drop of at least WEATHER_FORECAST_DROP_THRESHOLD within the
prediction horizon. A concrete slab gets into thermal debt not from a sudden plunge (the pump's own
curve catches that) but from a slow, deep, multi-day slide - and a fixed 12 h horizon cannot see one:
a 15 C fall over two days shows only 3.8 C in any twelve hours, under the trigger, so the pre-heat
never fires.

Invariant: ThermalModel.get_prediction_horizon() must scale with thermal mass
(UFH_CONCRETE > UFH_TIMBER > UFH_RADIATOR), not return a single fixed value for every house.
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
        f"{horizon:.0f} h. The horizon must scale with thermal mass, not collapse to one fixed "
        f"value - this is the model the engine actually uses."
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

"""F-124: a saturated compressor is a positive feedback trap; the aux-limit tier no longer feeds it.

`DM = integral(BT25 - S1)`. Raising the curve offset raises S1 instantly; BT25 - the water the
pump actually makes - can only follow if the compressor has headroom. A SATURATED compressor has
none, so raising the offset widens the gap, DM falls FASTER, and a layer that answers falling DM
with a raise keeps raising: a positive feedback loop. The EMERGENCY tier did exactly that, +10 at
DM -1500, and on the owner's F750 it latched for hours, the immersion heater (on from about -1450)
cooking the house while DM sat at the floor.

Past the aux limit the tier is now a floor at 0: no cut deepens the shortfall, and nothing of its
own raises into a pump that has nothing left. A room above its band is let cool.
"""

from __future__ import annotations

from custom_components.effektguard.const import MAX_OFFSET
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer


class _SaturatedPump:
    outdoor_temp = -25.0
    supply_temp = 63.0  # the pump is flat out and cannot go higher
    degree_minutes = -3000.0  # the integrator floor
    current_offset = float(MAX_OFFSET)  # already asking for everything it can ask for
    is_heating = True
    is_hot_water = False

    def __init__(self, indoor_temp: float):
        self.indoor_temp = indoor_temp


def _asked(indoor_temp: float) -> float:
    layer = EmergencyLayer(climate_detector=ClimateZoneDetector(latitude=59.33))
    return layer.evaluate_layer(
        _SaturatedPump(indoor_temp),
        weather_data=None,
        price_data=None,
        target_temp=21.0,
        tolerance_range=1.0,
    ).offset


def test_the_emergency_layer_does_not_keep_raising_a_pump_that_has_nothing_left():
    """When the pump is saturated, MORE offset is not more heat - it is only more debt."""
    for indoor in (21.0, 22.9):  # inside the band, and above it on immersion heat
        asked = _asked(indoor)
        assert asked <= 0.0, (
            f"The pump is at maximum flow ({_SaturatedPump.supply_temp} C), already commanded to "
            f"{_SaturatedPump.current_offset:+.0f}, the house is at {indoor} C and degree minutes "
            f"are at the integrator floor. The emergency layer still asks for {asked:+.1f}: "
            "raising S1, which a saturated pump cannot follow, only makes DM fall faster."
        )

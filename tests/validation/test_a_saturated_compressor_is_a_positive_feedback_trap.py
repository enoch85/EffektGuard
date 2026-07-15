"""KNOWN DEFECT, RECORDED NOT FIXED. F-124 is BLOCKED-ON-OWNER in the audit.

`DM = integral(BT25 - S1)`. Raising the curve offset raises S1 instantly; BT25 - the water the
pump actually makes - can only follow if the compressor has headroom. A SATURATED compressor has
none, so raising the offset widens the gap, DM falls FASTER, the emergency layer sees them falling
and raises the offset again: a positive feedback loop. The unit test below proves the mechanism -
handed a pump at maximum flow, a house ABOVE target and DM at the integrator floor, the emergency
layer still commands +10.

On every machine that saturates, the optimiser buys MORE resistive heat than the capacity deficit
forces (1.2-1.7x on datasheet-sized systems). It does NOT cook a correctly-sized house: the pump's
own start addition arms the elpatron first (F750 -700, S-series -460, VVM 320 -760, see
test_the_plant_engages_aux_where_the_pump_does.py) and holds the house, while the controller wastes
money fighting a wall.

WHY NOT FIXED HERE: the EMERGENCY tier deliberately bypasses the anti-windup written for this
failure mode. Changing it means deciding what a pump should do when it physically cannot meet its
own curve - a heat-pump decision, not a code cleanup. The xfail is STRICT: fix the defect and the
suite goes RED, forcing whoever fixes it to come here and delete the marker.
"""

from __future__ import annotations

import pytest

from custom_components.effektguard.const import (
    DM_THRESHOLD_AUX_LIMIT,
    MAX_OFFSET,
    SAFETY_EMERGENCY_OFFSET,
)
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer


def test_the_emergency_tier_asks_for_maximum_heat_at_the_aux_limit():
    """The precondition, and it is not itself wrong - it is what a healthy pump needs."""
    assert SAFETY_EMERGENCY_OFFSET == MAX_OFFSET


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "F-124, BLOCKED-ON-OWNER. A saturated compressor cannot raise BT25, so raising S1 makes "
        "DM = integral(BT25 - S1) fall FASTER. The emergency layer answers by raising it again and "
        "latches at +10. Every machine that saturates is made worse by it: the optimiser burns "
        "1.2-1.7x the resistive heat the capacity deficit physically forces. Fixing it means "
        "deciding what a pump should do when it physically cannot meet its own curve - a heat-pump "
        "decision, not a code-cleanup one."
    ),
)
def test_the_emergency_layer_does_not_keep_raising_a_pump_that_has_nothing_left():
    """When the pump is saturated, MORE offset is not more heat - it is only more debt.

    The pump has been at maximum flow for hours and degree minutes are still collapsing. That is
    the signature of saturation: the offset is not being converted into heat. Commanding more of it
    cannot help, and it demonstrably harms.
    """
    layer = EmergencyLayer(climate_detector=ClimateZoneDetector(latitude=59.33))

    class _SaturatedPump:
        outdoor_temp = -25.0
        indoor_temp = 22.9  # already ABOVE target - the immersion heater is cooking the house
        supply_temp = 63.0  # the pump is flat out and cannot go higher
        degree_minutes = -3000.0  # the integrator floor
        current_offset = float(MAX_OFFSET)  # already asking for everything it can ask for
        is_heating = True
        is_hot_water = False

    # raises=AssertionError on the marker ensures this xfails on the ASSERTION below, not on some
    # unrelated TypeError that would silently impersonate the expected failure.
    decision = layer.evaluate_layer(
        _SaturatedPump(),
        weather_data=None,
        price_data=None,
        target_temp=21.0,
        tolerance_range=1.0,
    )

    assert decision.offset < SAFETY_EMERGENCY_OFFSET, (
        f"The pump is at maximum flow ({_SaturatedPump.supply_temp} C), already commanded to "
        f"{_SaturatedPump.current_offset:+.0f}, the house is at {_SaturatedPump.indoor_temp} C - "
        f"ABOVE target, on immersion heat - and degree minutes are at the integrator floor. The "
        f"emergency layer still asks for {decision.offset:+.1f}. Raising the offset raises S1, "
        f"which a saturated pump cannot follow, so DM falls faster still. This is the spiral, and "
        f"the aux limit ({DM_THRESHOLD_AUX_LIMIT}) is long behind us."
    )

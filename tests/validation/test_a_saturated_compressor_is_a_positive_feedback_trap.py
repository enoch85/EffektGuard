"""KNOWN DEFECT, RECORDED NOT FIXED. F-124 is marked BLOCKED-ON-OWNER in the audit.

`DM = integral(BT25 - S1)`. Raising the curve offset raises S1 - the setpoint - INSTANTLY, while
BT25 (the water the pump actually makes) can only follow if the compressor has headroom left.

When the compressor is SATURATED it has none. So raising the offset widens the gap it is measuring,
degree minutes fall FASTER, the emergency layer sees them falling and raises the offset again. A
positive feedback loop, and the owner named it before this simulation ever ran:

    "if you keep raising the DM during stress, it will never be able to get itself out of that
     spinning loop downwards, it will worsen."

THE MECHANISM IS REAL AND THE UNIT TEST BELOW PROVES IT: handed a pump at maximum flow, a house
ABOVE target and DM at the integrator floor, the emergency layer still commands +10.

HOW MUCH IT COSTS WAS ONCE OVERSTATED, BY THIS FILE. Earlier versions here reported the house
"cooked to ~30 C" and "2-5x the resistive heat" - measured on a plant whose immersion heater
waited for EffektGuard's -1500 floor. No factory-default NIBE does that: the F750 arms its
elpatron at DM -700 (menu 4.9.3), the S-series controllers near -460, the VVM 320 near -760,
and the elpatron then works DM back UP. Re-measured with the pumps' own start-addition values
(see test_the_plant_engages_aux_where_the_pump_does.py):

    SWEDISH SIZING (cold climate, -22 C). Only the F2040 saturates, BY DESIGN: NIBE declares
    Tbiv -9 C, so below -9 C its supplementary heat is SUPPOSED to run.

        airsource_f2040   38.5 kWh resistive where the deficit forced 22.4 (1.7x).
                          Indoor held (max 22.6 C); DM settles at -771 - the hardware
                          start-addition, exactly where Swedish forum reports say a real
                          pump's DM asymptotes. The damage is ~16 kWh of COP-1.0 money per
                          cold snap, not a cooked house.

    UNDERSIZED PUMPS (average-climate sizing against a Swedish winter):

        wooden_f750       76.4 kWh vs 50.8 forced (1.5x), indoor held
        concrete_f1155   164.5 kWh vs 116.9 forced (1.4x), indoor held
        villa_s1155      143.5 kWh vs 115.9 forced (1.24x - inside the tolerance bound)
        airsource_f2040  the raw trap, still: DM pegs the -3000 integrator floor, indoor
                         hits 29.2 C, 9303 violations - a machine driven below its own
                         operating envelope, where nearly all of the burn (1652 of 1743 kWh)
                         is physically forced but the controller's latching places it as
                         overshoot.

THE FINDING STANDS, at its honest size: on every machine that saturates, the optimiser buys
MORE resistive heat than the capacity deficit forces (1.4-1.7x on datasheet-sized systems),
because the emergency ladder keeps raising a setpoint the compressor cannot follow. What it no
longer claims: that a correctly-sized system gets cooked. The hardware's own elpatron catches
the house; the controller wastes money fighting a wall.

WHY THIS IS NOT FIXED HERE. The EMERGENCY tier deliberately bypasses the anti-windup written for
exactly this failure mode, and that bypass is documented twice, in the owner's own code, as
intentional. Changing it means deciding what a heat pump should do when it physically cannot meet
its own curve - a heat-pump decision, not a code cleanup. BLOCKED-ON-OWNER, and it stays that way.

The `xfail` is STRICT on purpose: if someone fixes this, the suite goes RED and they are forced to
come here and delete the marker. A known defect nobody trips over is a defect that gets forgotten.
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
        "latches at +10. EVERY machine that saturates is made worse by it, under both of NIBE's "
        "published sizing conventions: the optimiser burns 2-5x the resistive heat of a do-nothing "
        "controller - and 1.2-2.8x what the capacity deficit physically forces - and cooks the "
        "house to about 30 C while doing nothing holds 22. The only system that escapes is the one "
        "whose pump has 1.8x spare capacity. Fixing it means deciding what a pump should do when it "
        "physically cannot meet its own curve - a heat-pump decision, not a code-cleanup one."
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

    # The first version of this call passed a keyword the layer does not have, so it died on
    # TypeError before reaching the assertion - and ordinary CI counted that as the expected
    # xfail. raises=AssertionError above makes that impersonation impossible now.
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

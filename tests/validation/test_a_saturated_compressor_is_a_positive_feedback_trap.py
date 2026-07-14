"""KNOWN DEFECT, RECORDED NOT FIXED. F-124 is marked BLOCKED-ON-OWNER in the audit.

`DM = integral(BT25 - S1)`. Raising the curve offset raises S1 - the setpoint - INSTANTLY, while
BT25 (the water the pump actually makes) can only follow if the compressor has headroom left.

When the compressor is SATURATED it has none. So raising the offset widens the gap it is measuring,
and degree minutes fall FASTER. The emergency layer sees them falling and raises the offset again.
That is a positive feedback loop, and the owner named it before this simulation ever ran:

    "if you keep raising the DM during stress, it will never be able to get itself out of that
     spinning loop downwards, it will worsen."

REPRODUCED, on a plant whose first law audits to a residual of 0.00 kWh. The F2040 is the only
shipped profile whose source is OUTDOOR AIR, so it is the only one whose capacity collapses as the
weather does - and in a cold snap it saturates:

                            optimiser   do-nothing
    indoor_max                 28.4 C       22.5 C     <- the house is COOKED
    degree minutes (min)        -3000        -1516     <- pinned at the integrator floor
    immersion heat            266 kWh       16 kWh     <- sixteen times more
    minutes above the band       5360            0
    cost                     2696 SEK     2242 SEK     <- twenty per cent MORE

And the mechanism, from the trace: of the 178 samples where degree minutes are past the auxiliary
limit, the commanded offset is +10 in ALL 178. It latches at maximum and never lets go. The house
climbs from 22.9 C to 28.4 C on immersion heat while degree minutes sit at the floor, because S1 is
pinned at maximum and BT25 can never catch it.

A DO-NOTHING CONTROLLER IS BETTER THAN THIS. It spends seven samples past the aux limit; the
optimiser spends 178.

WHY THIS IS NOT FIXED HERE. The EMERGENCY tier deliberately bypasses the anti-windup that the owner
wrote for exactly this failure mode - and that bypass is documented twice, in his own code, as
intentional. Changing it means deciding what a heat pump should do when it physically cannot meet
its own curve, and that is a heat-pump decision, not a code-cleanup one. It is marked
BLOCKED-ON-OWNER and it stays that way.

The `xfail` is STRICT on purpose: if someone fixes this, the test stops failing, the suite goes RED,
and they are forced to come here and delete the marker. A known defect that nobody trips over is a
defect that gets forgotten.
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
    reason=(
        "F-124, BLOCKED-ON-OWNER. A saturated compressor cannot raise BT25, so raising S1 makes "
        "DM = integral(BT25 - S1) fall FASTER. The emergency layer answers by raising it again and "
        "latches at +10: 178 of 178 samples past the aux limit, the house cooked to 28.4 C on "
        "266 kWh of immersion heat, and degree minutes pinned at the integrator floor. A do-nothing "
        "controller does better. Fixing it means deciding what a pump should do when it physically "
        "cannot meet its own curve - a heat-pump decision, not a code-cleanup one."
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

    decision = layer.evaluate_layer(_SaturatedPump(), price_classification="normal")

    assert decision.offset < SAFETY_EMERGENCY_OFFSET, (
        f"The pump is at maximum flow ({_SaturatedPump.supply_temp} C), already commanded to "
        f"{_SaturatedPump.current_offset:+.0f}, the house is at {_SaturatedPump.indoor_temp} C - "
        f"ABOVE target, on immersion heat - and degree minutes are at the integrator floor. The "
        f"emergency layer still asks for {decision.offset:+.1f}. Raising the offset raises S1, "
        f"which a saturated pump cannot follow, so DM falls faster still. This is the spiral, and "
        f"the aux limit ({DM_THRESHOLD_AUX_LIMIT}) is long behind us."
    )

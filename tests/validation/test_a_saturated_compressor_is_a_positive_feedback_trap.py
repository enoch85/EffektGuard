"""KNOWN DEFECT, RECORDED NOT FIXED. F-124 is marked BLOCKED-ON-OWNER in the audit.

`DM = integral(BT25 - S1)`. Raising the curve offset raises S1 - the setpoint - INSTANTLY, while
BT25 (the water the pump actually makes) can only follow if the compressor has headroom left.

When the compressor is SATURATED it has none. So raising the offset widens the gap it is measuring,
and degree minutes fall FASTER. The emergency layer sees them falling and raises the offset again.
That is a positive feedback loop, and the owner named it before this simulation ever ran:

    "if you keep raising the DM during stress, it will never be able to get itself out of that
     spinning loop downwards, it will worsen."

REPRODUCED - AND THE FIRST TIME I REPRODUCED IT, MY OWN PLANT WAS INFLATING IT ABOUT THREEFOLD.

I originally published 28.4 C, 266 kWh of immersion heat, degree minutes pinned at the -3000
integrator floor and 1134 `dm_runaway` violations, and cited a compressor-side energy audit reading
0.0 % error as proof that the plant was sound. Every one of those numbers was wrong, and the audit
was an algebraic identity that could not have detected otherwise (x - y + y = x; see
test_the_simulated_plant_obeys_physics, where the honest checks now live).

The simulator had two defects of its own, both mine:

  * it clamped BT25 to the pump's maximum flow temperature but NOT S1, so degree minutes -
    the integral of (BT25 - S1) - were accumulating against a setpoint the pump was physically
    forbidden to reach. DM fell at up to 4.1 per minute no matter what any controller did.
  * its immersion heater had no thermostat, and poured 3 kW into a water node already at its
    ceiling. The clamp then deleted the heat: 183 kWh metered, paid for, and never delivered.

With a plant that obeys its own physics, `dm_runaway` disappears entirely - it was an artefact -
and the trap is smaller than I said. IT IS ALSO STILL REAL, AND STILL FAILS THE RUN:

                            optimiser   do-nothing
    indoor_max                 27.6 C       22.5 C     <- the house is still COOKED
    degree minutes (min)        -1673        -1516
    immersion heat           73.8 kWh       16 kWh     <- four and a half times more
    minutes above the band       3130            0
    cost                     2320 SEK     2242 SEK

And the mechanism is unchanged, which is the point: of the 109 samples where degree minutes are
past the auxiliary limit, the commanded offset is +10 in ALL 109. It latches at maximum and never
lets go. The house climbs to 27.6 C on immersion heat because S1 is pinned at maximum and BT25 can
never catch it.

A DO-NOTHING CONTROLLER IS STILL BETTER THAN THIS. It never cooks the house and burns a fifth of
the resistive heat.

The lesson I am keeping: a defect measured on an instrument you have not verified is a number, not
a finding. The mechanism here was right; my evidence for it was not.

AND THE RECOVERY LADDER IS OTHERWISE UNVALIDATED BY SIMULATION. The harness now reports which
layers actually voted in each run, and the picture is stark:

  * four of the five houses pass the cold snap and NEVER engage the emergency ladder at all - only
    the proactive Z-tiers fire. The thermal-debt tiers, T1/T2/T3 and the anti-windup never run.
  * give the F2040 a correctly SIZED house (160 W/K instead of 220) and it passes the same cold
    snap with the ladder still silent.
  * the ONLY scenario in which the ladder fires is the one above - the saturated pump - and that
    scenario FAILS. It still fails on the corrected plant; only its size changed.

So there is no run anywhere in which the recovery ladder engages and RECOVERS. The simulator cannot
currently tell anyone whether it works; it can only show that when it does fire, it makes things
worse. Nobody should claim a green simulation validates the degree-minute recovery tiers, and I
nearly did.

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
        "latches at +10: 109 of 109 samples past the aux limit, the house cooked to 27.6 C on "
        "73.8 kWh of immersion heat. A do-nothing controller never cooks it at all and burns a "
        "fifth of the resistive heat. Fixing it means deciding what a pump should do when it "
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

    decision = layer.evaluate_layer(_SaturatedPump(), price_classification="normal")

    assert decision.offset < SAFETY_EMERGENCY_OFFSET, (
        f"The pump is at maximum flow ({_SaturatedPump.supply_temp} C), already commanded to "
        f"{_SaturatedPump.current_offset:+.0f}, the house is at {_SaturatedPump.indoor_temp} C - "
        f"ABOVE target, on immersion heat - and degree minutes are at the integrator floor. The "
        f"emergency layer still asks for {decision.offset:+.1f}. Raising the offset raises S1, "
        f"which a saturated pump cannot follow, so DM falls faster still. This is the spiral, and "
        f"the aux limit ({DM_THRESHOLD_AUX_LIMIT}) is long behind us."
    )

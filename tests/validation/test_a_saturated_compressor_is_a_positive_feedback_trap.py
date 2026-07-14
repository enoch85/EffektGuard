"""KNOWN DEFECT, RECORDED NOT FIXED. F-124 is marked BLOCKED-ON-OWNER in the audit.

`DM = integral(BT25 - S1)`. Raising the curve offset raises S1 - the setpoint - INSTANTLY, while
BT25 (the water the pump actually makes) can only follow if the compressor has headroom left.

When the compressor is SATURATED it has none. So raising the offset widens the gap it is measuring,
and degree minutes fall FASTER. The emergency layer sees them falling and raises the offset again.
That is a positive feedback loop, and the owner named it before this simulation ever ran:

    "if you keep raising the DM during stress, it will never be able to get itself out of that
     spinning loop downwards, it will worsen."

REPRODUCED - AND EVERY NUMBER I FIRST PUBLISHED FOR IT WAS WRONG, TWICE OVER.

The first time, my own PLANT was inflating it: it integrated degree minutes against a setpoint the
pump was forbidden to reach, and its immersion heater had no thermostat. I corrected that and
reported 27.6 C and 73.8 kWh.

The second time, the PUMP MODELS themselves turned out to be invented. The owner said so plainly -
"your sim models aren't even based on real data, yet you claim it" - and he was right. The profiles
carried an 8.0 kW compressor for a machine NIBE publishes at 4.994 kW, a COP curve keyed on outdoor
temperature for machines whose heat source is 20 C house air or 0 C brine, and a capacity derating
that ran BACKWARDS to the EN 14511 rating points it cited. See
tests/validation/test_the_pump_models_match_their_datasheets.py.

CORRECTING THE MODELS DID NOT SHRINK THIS DEFECT. IT MADE IT BIGGER, AND IT FOUND A SECOND MACHINE.

The F750 could never saturate in the old simulator, because I had given it sixty per cent more
compressor than it has. On its real 4.994 kW it saturates in a Swedish cold snap and falls into
exactly the same trap - and THIS case rests on no extrapolation at all: 4.994 kW is a published
maximum-compressor-frequency figure, and an exhaust-air pump's 20 C source does not move with the
weather.

                            optimiser   do-nothing        optimiser   do-nothing
                              F750        F750              F2040       F2040
    indoor_max                26.9 C      22.6 C            27.2 C      22.5 C
    immersion heat           35.8 kWh    1.8 kWh          223.1 kWh    51.8 kWh
    minutes above band          1020          0             12325           0
    cost                     1748 SEK   1461 SEK          2952 SEK    2663 SEK

A do-nothing controller is better on BOTH machines, on cost AND on comfort.

(Those immersion figures are with each machine's OWN heater, from its datasheet - 3.5 kW at NIBE's
delivery setting for the F750. The simulator used to give all five houses the same invented 3.0 kW,
matching none of them. Correcting it moved the F750's burn from 38.1 to 35.8 kWh: the finding is
robust to the heater size, which is worth knowing rather than assuming.)

And the mechanism is identical on both, which is what makes it a mechanism rather than a mishap:

    F750:   of the  38 samples past the auxiliary limit, the commanded offset is +10 in ALL  38
    F2040:  of the 459 samples past the auxiliary limit, the commanded offset is +10 in ALL 459

It latches at maximum and never lets go. The house climbs to 27 C on immersion heat while degree
minutes sit near the floor, because S1 is pinned at maximum and BT25 can never catch it.

ONE HONEST CAVEAT, on the F2040 only. NIBE tabulates its maximum output down to -7 C and no
further - below that the manual gives a graph and no numbers - so the model HOLDS capacity at the
-7 C figure. That understates the machine, which means the F2040's saturation is an UPPER BOUND on
the trap and not a measurement of it. The F750 case carries no such caveat, and it is the one to
rely on.

AND THE RECOVERY LADDER IS STILL UNVALIDATED BY SIMULATION - THE SAME CONCLUSION, ON BETTER DATA.

I used to write here that "four of the five houses pass the cold snap and never engage the ladder
at all". That was true of the invented models. On the real ones it is THREE of five: the two
ground-source houses and the small exhaust-air flat sail through with the ladder silent, and both
saturating machines engage it and FAIL.

The point survives intact, and it is the uncomfortable one:

  * the three houses that pass never touch the emergency ladder. Only the proactive Z-tiers fire.
    The thermal-debt tiers T1/T2/T3 and the anti-windup never run at all.
  * the ONLY runs in which the ladder engages are the two above - and both of them FAIL.

So there is still no run anywhere in which the recovery ladder engages and RECOVERS. The simulator
cannot tell anyone whether it works; it can only show that when it fires, it makes things worse.
Nobody should claim a green simulation validates the degree-minute recovery tiers, and I nearly
did - twice, on models that were not real.

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
        "latches at +10 - in 38 of 38 samples past the aux limit on a real F750, and 459 of 459 on "
        "an F2040. Both houses are cooked to 27 C on immersion heat, and a do-nothing controller "
        "beats the optimiser on cost AND comfort on both. Fixing it means deciding what a pump "
        "should do when it physically cannot meet its own curve - a heat-pump decision, not a "
        "code-cleanup one."
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

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

AND THE THIRD TIME, THE HOUSES WERE INVENTED TOO.

Every house carried a heat-loss coefficient that came from nowhere, and three of the five paired a
pump with a house it was twice too big for. That decided what the simulation was ABLE to find: a
pump with double the capacity its house needs cannot saturate, cannot fall behind, and can never
exercise the recovery ladder at all. I reported "the ground-source houses never engage the emergency
ladder" as a fact about the controller. It was a fact about my sizing.

Houses are now sized from their pump's own Pdesignh - NIBE's declared design heat load - at the
EN 14825 reference design temperature. And that exposed the last trap: THE SIZING CONVENTION MOVED
THE ANSWER. NIBE publishes Pdesignh at both reference climates, and the choice between them moves
the F750 between "saturates in a cold snap" and "does not". So the finding is not allowed to rest on
one house, and it does not.

    SWEDISH SIZING (cold climate, -22 C) - the honest default for a Swedish integration.
    Only the F2040 saturates, and it does so BY DESIGN: NIBE declares Tbiv = -9 C and
    Psup = 1.1 kW, so below -9 C its supplementary heater is SUPPOSED to run.

        airsource_f2040   239 kWh of resistive heat where the capacity deficit forced 85 (2.8x),
                          house cooked to 31.5 C

    UNDERSIZED PUMPS (average-climate sizing against a Swedish winter) - the commonest
    installation fault there is, and both figures come from NIBE's own datasheet.

                          optimiser   physics forced   do-nothing      optimiser   do-nothing
                             aux                          aux           indoor       indoor
        wooden_f750        221 kWh       104 kWh         61 kWh         29.3 C       22.6 C
        concrete_f1155     685 kWh       277 kWh        135 kWh         29.8 C       22.2 C
        villa_s1155        555 kWh       283 kWh        126 kWh         29.8 C       22.2 C
        airsource_f2040   2184 kWh      2113 kWh       1066 kWh         29.1 C       23.0 C
        apartment_f730       0 kWh         0 kWh          0 kWh         22.9 C       22.8 C

EVERY MACHINE THAT SATURATES IS MADE WORSE BY THE OPTIMISER, under BOTH sizing conventions. It
burns two to five times the resistive heat of a do-nothing controller and cooks the house to about
30 C, while doing nothing holds it at 22. The only system that escapes is the apartment - the one
where the pump has 1.8x more capacity than the house needs, and where saturation cannot happen.

THAT is the finding, and it no longer depends on a house I made up. The immersion heat is now
measured against what the pump's capacity deficit PHYSICALLY FORCES, computed step by step in the
plant, so "burned 2.8x more resistive heat than it had to" is a statement about the controller and
not about the weather.

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

    decision = layer.evaluate_layer(_SaturatedPump(), price_classification="normal")

    assert decision.offset < SAFETY_EMERGENCY_OFFSET, (
        f"The pump is at maximum flow ({_SaturatedPump.supply_temp} C), already commanded to "
        f"{_SaturatedPump.current_offset:+.0f}, the house is at {_SaturatedPump.indoor_temp} C - "
        f"ABOVE target, on immersion heat - and degree minutes are at the integrator floor. The "
        f"emergency layer still asks for {decision.offset:+.1f}. Raising the offset raises S1, "
        f"which a saturated pump cannot follow, so DM falls faster still. This is the spiral, and "
        f"the aux limit ({DM_THRESHOLD_AUX_LIMIT}) is long behind us."
    )

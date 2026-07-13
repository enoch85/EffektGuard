"""Zone 5 is a rung with no step: its band is the empty set, and it has never once fired.

The proactive ladder is meant to escalate gently before the critical tiers take over:

    Z1 +1.0   Z2 +1.5   Z3 +2.0   Z4 +2.5   Z5 +3.0   then T1 +4.0, T2 +7.0, T3 +8.5, EMERGENCY +10.0

Z5's constant even says what it is for: *"Very strong prevention (bridging to WARNING)"*. It is the
last gentle rung, the one that bridges Z4 to the first critical tier.

It cannot fire. Its band is:

    if expected_dm["warning"] < degree_minutes <= zone5_threshold:

and `zone5_threshold` is `expected_dm["normal"] * PROACTIVE_ZONE5_THRESHOLD_PERCENT` with the percent
set to **1.00** - so `zone5_threshold` is exactly `normal_max`. Meanwhile every climate zone in the
table sets `dm_warning_threshold` to exactly the deep end of `dm_normal_range`:

    "dm_normal_range": (-450, -700),
    "dm_warning_threshold": -700,          # <- the same number

So `warning == normal_max == zone5_threshold`, and the condition reads `-740 < DM <= -740`. **The
empty set.** Both ends of the band are the same number, and the same temperature adjustment is added
to both, so they move together and can never separate.

The ladder therefore steps 2.5 -> 4.0 where it was designed to step 2.5 -> 3.0 -> 4.0. In effective
pull (offset x weight) that is 1.38 -> 2.60, a near doubling, at exactly the moment the house is
leaving its normal range and a gentle nudge is what is called for.

This is not a regression. It is identical on `main`: Z5 has never fired, in any release, in any
climate zone.

The tests below check the INVARIANT, not the instance. A ladder whose rungs are computed from two
thresholds that happen to be equal is one edit away from losing another rung silently, so every zone
is swept across the whole DM range and required to appear.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer, ProactiveLayer

# Stockholm, Kiruna, Malmo - three zones with different normal ranges.
LATITUDES = [59.33, 67.86, 55.60]
OUTDOOR = [-20.0, -10.0, 0.0, 5.0]


def _state(degree_minutes: float, outdoor: float) -> NibeState:
    return NibeState(
        outdoor_temp=outdoor,
        indoor_temp=20.5,  # below target, so nothing abstains on comfort grounds
        supply_temp=40.0,
        return_temp=35.0,
        degree_minutes=degree_minutes,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 1, 15, 12, 0),
        compressor_hz=50,
    )


def _zones_reachable(latitude: float, outdoor: float) -> dict[str, float]:
    """Sweep DM and collect every proactive zone that actually fires, with its offset."""
    layer = ProactiveLayer(ClimateZoneDetector(latitude=latitude), heating_type="radiator")

    seen: dict[str, float] = {}
    for dm_tenths in range(0, -16000, -10):  # 1 DM resolution, 0 to -1600
        decision = layer.evaluate_layer(
            _state(dm_tenths / 10.0, outdoor), None, 21.0, is_volatile=False
        )
        if decision.zone and decision.zone not in seen:
            seen[decision.zone] = decision.offset

    return seen


@pytest.mark.parametrize("latitude", LATITUDES)
@pytest.mark.parametrize("outdoor", OUTDOOR)
def test_zone_5_is_reachable(latitude, outdoor):
    """The bridging rung between Z4 and the first critical tier."""
    reachable = _zones_reachable(latitude, outdoor)

    assert "Z5" in reachable, (
        f"At latitude {latitude} and {outdoor} °C, no degree-minute value anywhere between 0 and "
        f"-1600 lands in Zone 5. Its band is `warning < DM <= zone5_threshold`, and "
        f"PROACTIVE_ZONE5_THRESHOLD_PERCENT = 1.00 makes zone5_threshold equal to normal_max - which "
        f"every climate zone also uses as its warning threshold. Both ends of the band are the same "
        f"number. The ladder steps 2.5 -> 4.0 where it was built to step 2.5 -> 3.0 -> 4.0. "
        f"Zones that DO fire: {sorted(reachable)}"
    )


@pytest.mark.parametrize("latitude", LATITUDES)
@pytest.mark.parametrize("outdoor", OUTDOOR)
def test_the_whole_proactive_ladder_is_reachable(latitude, outdoor):
    """Not just Z5. Every rung the code declares must have a step to stand on.

    The zone bands are computed as percentages of one threshold and bounded by another. Two of them
    coinciding deletes a rung in silence - which is exactly what happened - so this checks all five
    rather than the one we know about.
    """
    reachable = _zones_reachable(latitude, outdoor)

    missing = [zone for zone in ("Z1", "Z2", "Z3", "Z4", "Z5") if zone not in reachable]

    assert not missing, (
        f"At latitude {latitude} and {outdoor} °C the proactive ladder has rungs with no step: "
        f"{missing}. Every zone must be reachable by some degree-minute value, or it is dead code "
        f"that reads like a working safety feature. Reachable: {sorted(reachable)}"
    )


@pytest.mark.parametrize("latitude", LATITUDES)
def test_the_ladder_escalates_monotonically(latitude):
    """A ladder that goes DOWN a rung as the house gets colder is not a ladder.

    The ladder spans TWO layers. The proactive one prevents (Z1-Z5, before the warning threshold);
    the emergency one recovers (T1-T3, after it). At the handover the proactive layer correctly
    stands down to zero - so asking either layer alone to be monotonic is asking the wrong question,
    and it is the question an earlier draft of this test asked. What must never weaken as the house
    falls further into debt is the strongest thing the system ASKS FOR, across both.
    """
    proactive = ProactiveLayer(ClimateZoneDetector(latitude=latitude), heating_type="radiator")
    emergency = EmergencyLayer(ClimateZoneDetector(latitude=latitude), heating_type="radiator")

    strongest_so_far = 0.0
    previous = (0.0, "start")
    for dm in range(0, -1600, -5):
        state = _state(float(dm), -10.0)
        p = proactive.evaluate_layer(state, None, 21.0, is_volatile=False)
        e = emergency.evaluate_layer(state, None, None, 21.0, 1.0, is_volatile=False)

        asked = max(p.offset, e.offset)
        rung = e.tier if e.offset >= p.offset else p.zone

        assert asked >= strongest_so_far, (
            f"At latitude {latitude}, degree minutes fell to {dm} - the house is deeper in thermal "
            f"debt than at {previous[1]} - and the strongest boost any layer asked for DROPPED from "
            f"{strongest_so_far:+.1f} to {asked:+.1f} (now {rung}). The ladder has a rung that steps "
            f"DOWN as the house gets colder."
        )
        strongest_so_far = asked
        previous = (asked, f"DM {dm}")

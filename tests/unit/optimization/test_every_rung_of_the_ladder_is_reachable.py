"""Every rung of the proactive ladder (Z1-Z5) must be reachable by some degree-minute value.

Zone 5's band is `warning < DM <= zone5_threshold`, and `zone5_threshold` is
`normal_max * PROACTIVE_ZONE5_THRESHOLD_PERCENT`. When that percent was 1.00, zone5_threshold equalled
normal_max - and every climate zone also sets its warning threshold to normal_max - so both ends of
the band were the same number and Z5 could never fire. It is now 0.875, strictly below the warning
threshold, restoring the +3.0 rung.

Two thresholds coinciding deletes a rung silently, so every zone is swept across the whole DM range
and required to expose all five rungs, plus a monotone-escalation check across both layers.
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

    The ladder spans two layers (proactive Z1-Z5, then emergency T1-T3), and the proactive layer
    correctly stands down to zero at the handover. So the invariant is on the strongest boost EITHER
    layer asks for: it must never weaken as the house falls further into debt.
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

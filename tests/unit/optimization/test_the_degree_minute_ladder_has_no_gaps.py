"""The degree-minute ladder must be contiguous, and every rung must be reachable.

Two layers respond to thermal debt, and between them they must cover the whole range from
"slightly negative" to the auxiliary-heat limit with no DM value falling through:

    ProactiveLayer   Z1..Z5, percentages of the thermal-mass-adjusted normal_max
    EmergencyLayer   T1/T2/T3/EMERGENCY, from the adjusted warning threshold downward

Three defects lived in the seam, and all three were invisible to a unit test that only ever
asked one layer about one DM value:

  * the layers read the threshold in DIFFERENT COORDINATE SYSTEMS - proactive from the raw
    climate zone, emergency from the thermal-mass-adjusted one - so on a concrete slab
    (x1.3) proactive stopped at -740 while T1 did not start until -962. 222 degree minutes
    with no layer acting, in precisely the band the design calls its warning boundary.
  * a WARNING tier sat after T1 testing `dm < warning` where T1 had already tested
    `dm <= warning - 0`. Unreachable.
  * a CAUTION tier tested `dm < normal_max`, and every climate zone defines
    normal_max == warning. Also unreachable.

This test sweeps degree minutes one at a time for every heating type and asserts coverage
as a PROPERTY of the pair, so a future retune of any threshold or multiplier that reopens a
gap fails here rather than in somebody's house.
"""

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.effektguard.const import (
    DM_THRESHOLD_AUX_LIMIT,
)
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.thermal_layer import (
    EmergencyLayer,
    ProactiveLayer,
)

TZ = timezone(timedelta(hours=1))
HEATING_TYPES = ["radiator", "timber_ufh", "concrete_ufh"]
# Stockholm, and an outdoor temperature cold enough that the bands are wide enough to
# resolve at 1 DM steps.
LATITUDE = 59.33
OUTDOOR = -10.0


TARGET = 21.0
# Indoor sits BELOW target on purpose. EmergencyLayer has two shortcuts before the ladder:
# "above target + tolerance -> let it cool" and "AT target and the price is not cheap ->
# ignore DM entirely". With indoor == target the second one returns tier OK for every
# degree-minute value short of the aux limit, and a coverage sweep taken there measures the
# shortcut rather than the ladder. A house that is cold is the condition under which the
# ladder is supposed to act, so that is where coverage is asserted.
INDOOR = TARGET - 0.5


class _State:
    """The fields both layers read."""

    def __init__(self, dm: float):
        self.degree_minutes = dm
        self.outdoor_temp = OUTDOOR
        self.indoor_temp = INDOOR
        self.current_offset = 0.0
        self.timestamp = datetime(2026, 1, 15, 12, 0, tzinfo=TZ)
        self.flow_temp = 35.0
        self.supply_temp = 35.0
        self.is_heating = True
        self.is_hot_water = False
        self.power_kw = 2.0


def _sweep(heating_type: str, step: int = 1):
    """DM -> (emergency tier, proactive zone) from just-negative to past the aux limit."""
    detector = ClimateZoneDetector(LATITUDE)
    proactive = ProactiveLayer(climate_detector=detector, heating_type=heating_type)
    out = {}
    for dm in range(0, int(DM_THRESHOLD_AUX_LIMIT) - 100, -step):
        # A FRESH emergency layer per probe: it keeps a DM history for anti-windup, and a
        # sweep through one instance would have it comparing unrelated samples.
        emergency = EmergencyLayer(climate_detector=detector, heating_type=heating_type)
        tier = emergency.evaluate_layer(
            nibe_state=_State(float(dm)),
            weather_data=None,
            price_data=None,
            target_temp=TARGET,
            tolerance_range=0.2,
        ).tier
        zone = proactive.evaluate_layer(
            nibe_state=_State(float(dm)), weather_data=None, target_temp=TARGET
        ).zone
        out[dm] = (tier, zone)
    return out


@pytest.fixture(scope="module")
def sweeps():
    return {ht: _sweep(ht) for ht in HEATING_TYPES}


@pytest.mark.parametrize("heating_type", HEATING_TYPES)
def test_no_degree_minute_falls_through_both_layers(heating_type, sweeps):
    """Below a shallow dead band, SOME layer must be acting at every DM value."""
    detector = ClimateZoneDetector(LATITUDE)
    # Z1 starts at a small percentage of normal_max; above that both layers idling is
    # correct, so only assert coverage below Z1's own entry point. Degree minutes are
    # NEGATIVE, so Z1's entry is the greatest zoned value - `min` here would be the
    # deepest one instead, and the band inspected would start past every proactive zone
    # and miss any seam between two of them.
    zone_entry = max(
        dm for dm, (_, zone) in sweeps[heating_type].items() if zone not in ("NONE", "")
    )
    uncovered = [
        dm
        for dm, (tier, zone) in sweeps[heating_type].items()
        if dm <= zone_entry and tier == "OK" and zone in ("NONE", "")
    ]
    assert not uncovered, (
        f"{heating_type}: {len(uncovered)} degree-minute values have no layer acting, "
        f"from {max(uncovered)} to {min(uncovered)}. The proactive zones and the recovery "
        f"tiers must meet. Zone range: {detector.get_expected_dm_range(OUTDOOR)}"
    )


@pytest.mark.parametrize("heating_type", HEATING_TYPES)
def test_the_proactive_zones_hand_over_to_t1_without_a_gap(heating_type, sweeps):
    """Z5's deep edge and T1's trigger must be adjacent, not merely both present."""
    sweep = sweeps[heating_type]
    deepest_zone = min(dm for dm, (_, zone) in sweep.items() if zone not in ("NONE", ""))
    shallowest_t1 = max(dm for dm, (tier, _) in sweep.items() if tier == "T1")
    assert abs(deepest_zone - shallowest_t1) <= 1, (
        f"{heating_type}: proactive stops at DM {deepest_zone} and T1 starts at "
        f"{shallowest_t1} - a {abs(deepest_zone - shallowest_t1)} DM seam. These are the "
        "same threshold read through two code paths; they must agree."
    )


@pytest.mark.parametrize("heating_type", HEATING_TYPES)
def test_every_rung_the_hierarchy_documents_is_reachable(heating_type, sweeps):
    """Eight rungs, and no value of DM may be unable to reach any of them."""
    tiers = {tier for tier, _ in sweeps[heating_type].values()}
    zones = {zone for _, zone in sweeps[heating_type].values()}
    for rung in ("T1", "T2", "T3", "EMERGENCY"):
        assert rung in tiers, f"{heating_type}: emergency tier {rung} is unreachable"
    for rung in ("Z1", "Z2", "Z3", "Z4", "Z5"):
        assert rung in zones, f"{heating_type}: proactive zone {rung} is unreachable"


@pytest.mark.parametrize("heating_type", HEATING_TYPES)
def test_the_tiers_deepen_monotonically(heating_type, sweeps):
    """Walking DM downward, severity may only increase - never oscillate."""
    order = {"OK": 0, "Z1": 1, "Z2": 2, "Z3": 3, "Z4": 4, "Z5": 5}
    severity = {"OK": 0, "T1": 6, "T2": 7, "T3": 8, "EMERGENCY": 9}
    previous = 0
    for dm in sorted(sweeps[heating_type], reverse=True):
        tier, zone = sweeps[heating_type][dm]
        rank = severity.get(tier, 0) or order.get(zone, 0)
        assert rank >= previous or rank == 0, (
            f"{heating_type}: severity went backwards at DM {dm} " f"(tier={tier}, zone={zone})"
        )
        previous = max(previous, rank)

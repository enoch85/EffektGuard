"""DHW must not be allowed to start at a degree-minute value that aborts it on the next tick.

Two thresholds govern hot water under thermal debt: `block` (do not START below this DM) and `abort`
(STOP a running cycle below this DM). Abort must be the DEEPER of the two: heating hot water steals
the compressor from space heating, so degree minutes always sink during a cycle, and an abort
shallower than block means every cycle that starts near the block threshold trips abort immediately -
the pump starts, stops, starts, stops. The fallback constants have the relationship right
(block -340, abort -500; abort 160 DM deeper).

Invariants: in every climate zone abort < block; the reported block equals what EmergencyLayer
actually enforces (`warning - DM_CRITICAL_T2_MARGIN`); and abort never sinks past the absolute limit.
"""

from __future__ import annotations

import pytest

from custom_components.effektguard.const import (
    DM_CRITICAL_T2_MARGIN,
    DM_DHW_ABORT_FALLBACK,
    DM_DHW_BLOCK_FALLBACK,
    DM_THRESHOLD_AUX_LIMIT,
)
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer

LATITUDES = [59.33, 67.86, 55.60]  # Stockholm, Kiruna, Malmo
OUTDOOR = [-20.0, -15.0, -10.0, 0.0, 5.0]


def _thresholds(latitude: float, outdoor: float) -> tuple[float, float]:
    """(block, abort) as the running system computes them, for this zone and temperature."""
    detector = ClimateZoneDetector(latitude=latitude)
    emergency = EmergencyLayer(detector, heating_type="radiator")

    optimizer = IntelligentDHWScheduler(emergency_layer=emergency, climate_detector=detector)
    return optimizer.get_dm_block_and_abort_thresholds(outdoor)


def test_the_fallback_constants_say_which_way_round_it_goes():
    """The precondition, and the specification. Abort is DEEPER than block."""
    assert DM_DHW_ABORT_FALLBACK < DM_DHW_BLOCK_FALLBACK, (
        f"Even the fallback pair is inverted: block {DM_DHW_BLOCK_FALLBACK}, "
        f"abort {DM_DHW_ABORT_FALLBACK}."
    )


@pytest.mark.parametrize("latitude", LATITUDES)
@pytest.mark.parametrize("outdoor", OUTDOOR)
def test_dhw_never_starts_at_a_degree_minute_that_aborts_it(latitude, outdoor):
    """The whole finding, in one assertion, in every zone and at every temperature."""
    block, abort = _thresholds(latitude, outdoor)

    assert abort < block, (
        f"At latitude {latitude}, {outdoor} °C: DHW is BLOCKED from starting below {block:.0f} DM, "
        f"but a running cycle ABORTS below {abort:.0f} DM - which is {block - abort:.0f} DM "
        f"SHALLOWER. Every degree-minute value between {block:.0f} and {abort:.0f} is one where the "
        f"pump is allowed to start hot water and then told to stop it on the next tick. Heating hot "
        f"water always sinks degree minutes, so it starts, aborts, starts, aborts."
    )


@pytest.mark.parametrize("latitude", LATITUDES)
@pytest.mark.parametrize("outdoor", OUTDOOR)
def test_the_block_threshold_is_the_one_that_is_actually_enforced(latitude, outdoor):
    """What the optimizer reports as the block must be what EmergencyLayer enforces.

    `should_block_dhw` blocks at `warning - DM_CRITICAL_T2_MARGIN`. The optimizer published plain
    `warning` as `thermal_debt_threshold_block`, so the diagnostic named a threshold that blocks
    nothing - 200 DM shallower than the one that does.
    """
    detector = ClimateZoneDetector(latitude=latitude)
    emergency = EmergencyLayer(detector, heating_type="radiator")
    enforced = emergency.get_adjusted_dm_thresholds(outdoor)["warning"] - DM_CRITICAL_T2_MARGIN

    block, _ = _thresholds(latitude, outdoor)

    assert block == pytest.approx(enforced), (
        f"The optimizer reports a DHW block threshold of {block:.0f} DM, but EmergencyLayer actually "
        f"blocks at {enforced:.0f}. The published number blocks nothing."
    )


@pytest.mark.parametrize("latitude", LATITUDES)
@pytest.mark.parametrize("outdoor", OUTDOOR)
def test_abort_never_sinks_past_the_absolute_limit(latitude, outdoor):
    """The absolute limit is the floor. Below it the emergency layer owns the pump outright.

    Clamping at the limit ITSELF is deliberate: clamping at `limit + buffer` would push abort back
    ABOVE block in the coldest zone, where block already sits at -1400, re-creating the inversion
    this file exists to prevent. An abort exactly at the limit is the hardest possible stop.
    """
    _, abort = _thresholds(latitude, outdoor)

    assert abort >= DM_THRESHOLD_AUX_LIMIT, (
        f"Abort threshold {abort:.0f} is deeper than the absolute limit "
        f"{DM_THRESHOLD_AUX_LIMIT:.0f}, past which DHW cannot run at all."
    )

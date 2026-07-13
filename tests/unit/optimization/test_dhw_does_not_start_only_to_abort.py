"""DHW is allowed to start at a degree-minute value that aborts it on the next tick.

Two thresholds govern hot water under thermal debt:

  * **block** - do not START a DHW cycle if degree minutes are already this bad;
  * **abort** - STOP a running DHW cycle if degree minutes fall this far while it runs.

Abort must be the DEEPER of the two. Heating hot water steals the compressor from space heating, so
degree minutes always sink during a DHW cycle - and if abort sits shallower than block, every cycle
that starts near the block threshold trips the abort immediately. The pump starts, stops, starts,
stops.

The fallback constants state the relationship correctly:

    DM_DHW_BLOCK_FALLBACK: Final = -340.0  # Never start DHW below this DM
    DM_DHW_ABORT_FALLBACK: Final = -500.0  # Abort DHW if reached during run

Abort is 160 DM deeper than block. That is the shape of it.

The climate-aware path - the one that actually runs - inverts it:

    dm_block_threshold = dm_thresholds["warning"]
    # Abort threshold should be LESS strict (more negative) than block threshold
    # to avoid immediate abort after starting. Use 80 DM buffer beyond warning.
    dm_abort_threshold = dm_thresholds["warning"] - 80

while the block that is actually enforced comes from `EmergencyLayer.should_block_dhw`, which blocks at
`warning - DM_CRITICAL_T2_MARGIN`, i.e. **warning - 200**. So:

    Stockholm at -10 C:   warning -740   BLOCK -940   ABORT -820

**Abort is 120 DM SHALLOWER than block.** Every degree-minute value between -940 and -820 is one where
DHW is permitted to start and is aborted on the next cycle. The comment three lines above the bug says
the code exists "to prevent start-then-abort cycles when block passes but abort fails". It guarantees
them, in every climate zone.

There is a second defect in the same four lines. `dm_block_threshold` is set to `warning` (-740), but
nothing blocks at -740 - the enforced block is -940. That number is published to the owner as
`thermal_debt_threshold_block`, so the diagnostic reports a threshold the code does not use.
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

    Clamping at the limit ITSELF is deliberate, and a first draft of the fix got it wrong: clamping
    at `limit + buffer` pushed abort back ABOVE block in the coldest zone, where block already sits
    at -1400, and re-created the inversion this file exists to prevent. Deep zones have less room,
    and an abort exactly at the limit is the hardest possible stop, not a self-defeating one.
    """
    _, abort = _thresholds(latitude, outdoor)

    assert abort >= DM_THRESHOLD_AUX_LIMIT, (
        f"Abort threshold {abort:.0f} is deeper than the absolute limit "
        f"{DM_THRESHOLD_AUX_LIMIT:.0f}, past which DHW cannot run at all."
    )

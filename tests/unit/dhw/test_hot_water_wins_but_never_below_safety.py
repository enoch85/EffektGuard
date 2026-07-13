"""A scheduled shower outranks thermal debt. It does not outrank the safety floor.

Owner decision (2026-07-13): **"DHW wins, but never below safety."** A shower the owner scheduled is a
shower the owner wants, so a scheduled window beats the thermal-debt block and beats space-heating
demand. It does not beat the 18 C indoor floor or the absolute degree-minute limit. And - this is the
half that was missing - **if it may start, it may run**: whatever permits the start must be the same
thing that would stop it, or the cycle starts and aborts and starts again.

What the code did before this file existed:

`should_start_dhw()` evaluates RULE 0 (two-lane scheduling) and RULE 0 **returns early**, before RULE 1
(critical thermal debt - never start DHW) and before RULE 2 (space heating emergency - house too cold)
are ever reached. So inside a scheduled window it heated hot water at any thermal debt and any indoor
temperature. Measured:

    DM -1400 (T3 emergency tier), indoor 17.0 C  - BELOW the 18 C absolute safety floor
      should_block_dhw()  -> BLOCK: True
      should_start_dhw()  -> heat=True,  reason=DHW_SCHEDULED_PRIORITY_1.0H

The priority was real. But the same decision handed back:

    abort_conditions -> ['thermal_debt < -1100', 'indoor_temp < 20.5', ...]

Both conditions were **already true at the moment it started**. The coordinator evaluates them on the
next cycle and switches the lux boost straight back off; starts are rate-limited to an hour, so the net
behaviour in deep debt was a futile DHW start every hour, aborted five minutes later, heating no water
and cycling the compressor. RULE 0 granted the priority and the abort conditions revoked it, forever.

Note the second condition: `indoor_temp < 20.5` is target minus 0.5. That is a COMFORT threshold being
used to abort a cycle that RULE 0 had just declared more important than comfort.

So the rule now is one rule, stated once:

  * A scheduled window may start DHW at any thermal debt, and at any indoor temperature down to the
    safety floor.
  * It may not start below the safety floor, or at the absolute degree-minute limit.
  * Its abort conditions are those same two thresholds and nothing else - so a cycle that was allowed
    to begin is allowed to finish, and only genuine danger stops it.
  * A window refused for safety is not lost: it is resumed the moment the house is safe again, even
    outside the window (owner decision: "retry as soon as it is safe").
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from custom_components.effektguard.const import DM_THRESHOLD_AUX_LIMIT, MIN_TEMP_LIMIT
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.dhw_optimizer import (
    DHWDemandPeriod,
    IntelligentDHWScheduler,
)
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer

STOCKHOLM = ZoneInfo("Europe/Stockholm")
IN_THE_RUN_UP = datetime(2026, 1, 15, 6, 0, tzinfo=STOCKHOLM)  # hot water wanted at 07:00
LONG_AFTER = datetime(2026, 1, 15, 11, 0, tzinfo=STOCKHOLM)  # window long gone


def _scheduler() -> IntelligentDHWScheduler:
    detector = ClimateZoneDetector(latitude=59.33)
    return IntelligentDHWScheduler(
        demand_periods=[
            DHWDemandPeriod(
                availability_hour=7, target_temp=50.0, duration_hours=2, min_amount_minutes=5
            )
        ],
        climate_detector=detector,
        emergency_layer=EmergencyLayer(detector, heating_type="radiator"),
        user_target_temp=50.0,
    )


def _ask(scheduler, thermal_debt: float, indoor: float, when=IN_THE_RUN_UP, dhw_temp: float = 35.0):
    return scheduler.should_start_dhw(
        current_dhw_temp=dhw_temp,
        space_heating_demand_kw=5.0,
        thermal_debt_dm=thermal_debt,
        indoor_temp=indoor,
        target_indoor_temp=21.0,
        outdoor_temp=-10.0,
        price_classification="expensive",
        current_time=when,
        price_periods=None,
        hours_since_last_dhw=8.0,
    )


def test_a_scheduled_shower_beats_thermal_debt():
    """The priority itself. This is what the owner asked for and it must not regress.

    DM -1400 is deep in the T3 recovery tier and `should_block_dhw()` refuses it. The scheduled window
    overrules that, because a shower the owner scheduled is a shower the owner wants.
    """
    scheduler = _scheduler()
    emergency = scheduler.emergency_layer

    assert emergency.should_block_dhw(-1400.0, -10.0), "precondition: debt this deep blocks DHW"

    decision = _ask(scheduler, thermal_debt=-1400.0, indoor=21.0)

    assert decision.should_heat, (
        "A scheduled hot-water window was refused because of thermal debt. The owner's rule is that "
        "the shower wins: DHW beats the debt block and beats space-heating demand."
    )


def test_a_scheduled_shower_does_not_beat_the_safety_floor():
    """The house is below the temperature at which the safety layer commands maximum heat.

    Running hot water here takes the compressor away from a house that is already in trouble.
    """
    decision = _ask(_scheduler(), thermal_debt=-400.0, indoor=MIN_TEMP_LIMIT - 0.5)

    assert not decision.should_heat, (
        f"DHW was started with the house at {MIN_TEMP_LIMIT - 0.5} C - below the {MIN_TEMP_LIMIT} C "
        f"floor, where the safety layer is already commanding maximum heat. Hot water takes the "
        f"compressor away from exactly that."
    )


def test_a_scheduled_shower_does_not_beat_the_absolute_degree_minute_limit():
    """At the aux limit the immersion heater is engaging. DHW must not compete with recovery."""
    decision = _ask(_scheduler(), thermal_debt=DM_THRESHOLD_AUX_LIMIT - 50, indoor=21.0)

    assert not decision.should_heat, (
        f"DHW was started at DM {DM_THRESHOLD_AUX_LIMIT - 50}, past the absolute limit "
        f"{DM_THRESHOLD_AUX_LIMIT} where the emergency layer owns the pump."
    )


def test_if_it_may_start_it_may_run():
    """The heart of it. Nothing that permits the start may be a reason to abort.

    The scheduled path used to start at DM -1400 while handing back `thermal_debt < -1100` as an abort
    condition - true before the cycle even began. It started and aborted, once an hour, forever.
    """
    scheduler = _scheduler()
    decision = _ask(scheduler, thermal_debt=-1400.0, indoor=20.0)

    assert decision.should_heat, "precondition: this cycle is permitted to start"

    should_abort, reason = scheduler.check_abort_conditions(
        decision.abort_conditions,
        thermal_debt=-1400.0,  # the very state it was started in
        indoor_temp=20.0,
        target_indoor=21.0,
    )

    assert not should_abort, (
        f"DHW was permitted to start in this exact state and its own abort conditions "
        f"{decision.abort_conditions} fire on it immediately: {reason}. It starts, aborts, is "
        f"rate-limited for an hour, starts again, and never heats any water."
    )


def test_it_does_abort_when_the_house_actually_becomes_unsafe():
    """The other half. The priority is not a licence to freeze the house."""
    scheduler = _scheduler()
    decision = _ask(scheduler, thermal_debt=-1400.0, indoor=20.0)
    assert decision.should_heat, "precondition"

    should_abort, reason = scheduler.check_abort_conditions(
        decision.abort_conditions,
        thermal_debt=-1400.0,
        indoor_temp=MIN_TEMP_LIMIT - 0.5,  # the house has fallen below the floor while heating
        target_indoor=21.0,
    )

    assert should_abort, (
        f"The house fell below the {MIN_TEMP_LIMIT} C safety floor while hot water was being heated, "
        f"and nothing stopped it. Abort conditions were {decision.abort_conditions}."
    )


def test_a_window_refused_for_safety_is_resumed_when_the_house_recovers():
    """Owner decision: "retry as soon as it is safe". Hot water late, not hot water never.

    The 07:00 window is refused because the house is below the floor. By 11:00 the house has recovered
    and the window is long gone - but the shower was still wanted, so it is heated now.
    """
    scheduler = _scheduler()

    refused = _ask(scheduler, thermal_debt=-400.0, indoor=MIN_TEMP_LIMIT - 0.5)
    assert not refused.should_heat, "precondition: safety refused the window"

    recovered = _ask(scheduler, thermal_debt=-200.0, indoor=21.0, when=LONG_AFTER)

    assert recovered.should_heat, (
        "The scheduled window was refused for safety and then simply forgotten. The house has "
        "recovered and the hot water the owner asked for has still not been heated."
    )


def test_the_retry_does_not_fire_forever_once_the_water_is_hot():
    """It is a debt to be settled, not a standing order."""
    scheduler = _scheduler()

    refused = _ask(scheduler, thermal_debt=-400.0, indoor=MIN_TEMP_LIMIT - 0.5)
    assert not refused.should_heat, "precondition"

    # The water reached target (by the retry, or by the pump's own schedule - it does not matter).
    satisfied = _ask(scheduler, thermal_debt=-200.0, indoor=21.0, when=LONG_AFTER, dhw_temp=50.0)
    assert not satisfied.should_heat, "the water is at target; there is nothing left to settle"

    # And it stays settled.
    again = _ask(scheduler, thermal_debt=-200.0, indoor=21.0, when=LONG_AFTER, dhw_temp=49.0)
    assert (
        again.priority_reason != "DHW_SCHEDULED_RETRY_AFTER_SAFETY"
    ), "the missed-window debt was settled and must not resurrect itself"

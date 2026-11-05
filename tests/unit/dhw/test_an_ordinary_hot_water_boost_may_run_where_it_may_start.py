"""An ordinary hot-water boost may run in any state it may start in.

RULE 2 (space heating first) refused a START only with the room more than
DHW_SPACE_HEATING_DEFICIT_THRESHOLD below target AND below DHW_SPACE_HEATING_OUTDOOR_THRESHOLD
outside. Every ordinary heat decision carried a second, stricter test as an abort condition -
`indoor_temp < target - 0.5` at any outdoor temperature - which the coordinator checks on the next
cycle. At 0 C and above, with the room 0.6 C low, every boost EffektGuard started was aborted five
minutes later, the start limiter held the next one for an hour, and it started again: a five-minute
lux pulse and a compressor start/stop every 65 minutes (review DHW F1). The tank below 20 C
(DHW_SAFETY_MINIMUM) was no exception.

RULE 2 is now the one rule: it refuses the start and, decided again every cycle, stops the run.
Without the second test the safety floor the abort used to cover at mild outdoor temperatures is
RULE 2's too - the scheduled window, which outranks it, stops there as well.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from custom_components.effektguard.const import MIN_TEMP_LIMIT
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.dhw_optimizer import (
    DHWDemandPeriod,
    IntelligentDHWScheduler,
)
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer

STOCKHOLM = ZoneInfo("Europe/Stockholm")
NOW = datetime(2026, 1, 15, 10, 30, tzinfo=STOCKHOLM)  # outside the 07:00 and 18:00 windows
TARGET = 21.0
MILD = 3.0  # outdoor, at or above DHW_SPACE_HEATING_OUTDOOR_THRESHOLD
DM = -200.0


def _scheduler() -> IntelligentDHWScheduler:
    detector = ClimateZoneDetector(latitude=59.33)
    scheduler = IntelligentDHWScheduler(
        demand_periods=[
            DHWDemandPeriod(availability_hour=7, target_temp=50.0, duration_hours=2),
            DHWDemandPeriod(availability_hour=18, target_temp=50.0, duration_hours=1),
        ],
        climate_detector=detector,
        emergency_layer=EmergencyLayer(detector, heating_type="radiator"),
        user_target_temp=50.0,
    )
    scheduler.last_legionella_boost = NOW - timedelta(days=2)  # keep the hygiene rule out of it
    return scheduler


def _decide(
    scheduler,
    *,
    tank: float,
    price: str,
    indoor: float,
    outdoor: float = MILD,
    volatile: bool = False,
):
    return scheduler.should_start_dhw(
        current_dhw_temp=tank,
        space_heating_demand_kw=1.0,
        thermal_debt_dm=DM,
        indoor_temp=indoor,
        target_indoor_temp=TARGET,
        outdoor_temp=outdoor,
        price_classification=price,
        current_time=NOW,
        price_periods=None,
        is_volatile=volatile,
    )


@pytest.mark.parametrize(
    "tank,price,volatile,rule",
    [
        (15.0, "normal", False, "DHW_SAFETY_MINIMUM"),
        (40.0, "cheap", False, "DHW_COMPLETE_EMERGENCY_HEATING"),
        # A brief cheap run: RULE 2.5 waits for a stable one, the comfort top-up does not
        (40.0, "cheap", True, "DHW_COMFORT_LOW_CHEAP"),
    ],
    ids=["tank_below_20C", "completing_to_target", "comfort_low"],
)
def test_a_boost_started_in_mild_weather_is_not_stopped_by_the_room_it_started_in(
    tank, price, volatile, rule
):
    """Room 20.4 C of 21 at +3 C outside: RULE 2 lets the boost start, so nothing may stop it."""
    scheduler = _scheduler()
    started = _decide(scheduler, tank=tank, price=price, indoor=20.4, volatile=volatile)
    assert started.should_heat and started.priority_reason == rule, "precondition"

    aborts, why = scheduler.check_abort_conditions(started.abort_conditions, DM, 20.4, TARGET, tank)
    assert not aborts, (
        f"{rule} started in this exact state and its own abort conditions "
        f"{started.abort_conditions} stop it on the next cycle: {why}. It starts, is aborted, is "
        f"rate-limited for an hour and starts again."
    )
    assert _decide(
        scheduler, tank=tank, price=price, indoor=20.4, volatile=volatile
    ).should_heat, (
        "the next cycle's decision in the same state stopped the boost it had just started"
    )


@pytest.mark.parametrize("tank,price", [(15.0, "normal"), (40.0, "normal")])
def test_nothing_ordinary_starts_with_the_house_below_its_safety_floor(tank, price):
    """The abort used to be the only thing between a mild-weather boost and a house at 17.5 C."""
    decision = _decide(_scheduler(), tank=tank, price=price, indoor=MIN_TEMP_LIMIT - 0.5)

    assert not decision.should_heat, (
        f"{decision.priority_reason}: a hot-water boost started with the house at "
        f"{MIN_TEMP_LIMIT - 0.5} C, below the {MIN_TEMP_LIMIT} C floor the scheduled window "
        f"stops at"
    )
    assert decision.priority_reason == "SPACE_HEATING_EMERGENCY"

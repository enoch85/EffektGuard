"""A hot-water boost EffektGuard starts runs DHW_MIN_RUN_MINUTES, and only safety ends it sooner.

Each boost runs the compressor near full speed. Decided afresh every five minutes, a boost ended at
the first cycle that found the tank at its target or the price moved on: 19 of the 52 temporary-lux
runs on the owner's F750, 8-21 October 2026, lasted under 11 minutes (production history). A boost
now runs its minimum unless the house needs the compressor, and one the tank would finish sooner is
not started.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from homeassistant.const import STATE_OFF, STATE_ON

from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
from custom_components.effektguard.const import DHW_MIN_RUN_MINUTES, MIN_TEMP_LIMIT
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.dhw_optimizer import (
    DHWDemandPeriod,
    DHWScheduleDecision,
    IntelligentDHWScheduler,
)
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer

STOCKHOLM = ZoneInfo("Europe/Stockholm")
# A cycle reaches hot-water control a few seconds after its :X0:10 / :X5:10 slot, never the same few.
START = datetime(2026, 1, 15, 10, 0, 13, tzinfo=STOCKHOLM)
CYCLES = [START + timedelta(minutes=5 * n, seconds=s) for n, s in ((1, -2), (2, -1), (3, -2))]
LUX = "switch.temporary_lux_50004"
TARGET = 21.0
OUTDOOR = 0.0
DM = -200.0
TANK_TARGET = 45.0  # the owner's


def _scheduler(
    *, scheduled: bool = False, analyzer: PriceAnalyzer | None = None
) -> IntelligentDHWScheduler:
    detector = ClimateZoneDetector(latitude=59.33)
    scheduler = IntelligentDHWScheduler(
        demand_periods=(
            [DHWDemandPeriod(availability_hour=12, target_temp=TANK_TARGET, duration_hours=1)]
            if scheduled
            else None
        ),
        climate_detector=detector,
        emergency_layer=EmergencyLayer(detector, heating_type="radiator"),
        user_target_temp=TANK_TARGET,
        price_analyzer=analyzer,
    )
    scheduler.last_legionella_boost = START - timedelta(days=2)  # keep the hygiene rule out of it
    return scheduler


class _House:
    """The real hot-water control and switch door on a duck-typed coordinator.

    As in test_dhw_safety_stop_not_rate_limited: the lux switch is a state the service calls flip.
    """

    def __init__(self, scheduler: IntelligentDHWScheduler) -> None:
        self.lux = STATE_OFF
        self.calls: list[str] = []
        self.scheduler = scheduler

        async def switch(domain, service, data, blocking=False):
            self.calls.append(service)
            self.lux = STATE_ON if service == "turn_on" else STATE_OFF

        coordinator = MagicMock()
        coordinator.temp_lux_entity = LUX
        coordinator._last_dhw_control_time = None
        coordinator.last_update_success = True
        coordinator.hass.states.get = lambda entity_id: (
            SimpleNamespace(state=self.lux) if entity_id == LUX else None
        )
        coordinator.hass.services.async_call = AsyncMock(side_effect=switch)
        coordinator._shutdown_requested = False
        coordinator._service_boost_until = None
        coordinator._lux_boost_is_ours = False
        coordinator._lux_boost_started_at = None
        coordinator.dhw_optimizer = scheduler
        for name in ("_is_dhw_start_rate_limited", "_set_temporary_lux", "_lux_turned_off"):
            method = getattr(EffektGuardCoordinator, name)
            setattr(coordinator, name, lambda *args, _m=method: _m(coordinator, *args))
        self.coordinator = coordinator

    async def cycle(
        self,
        now: datetime,
        *,
        tank: float,
        price: str,
        dm: float = DM,
        indoor: float = 20.8,
        prices: list[QuarterPeriod] | None = None,
        decision: DHWScheduleDecision | None = None,
    ) -> DHWScheduleDecision:
        """One control cycle: the scheduler's decision on this state, applied by the real control."""
        if decision is None:
            decision = self.scheduler.should_start_dhw(
                current_dhw_temp=tank,
                space_heating_demand_kw=1.0,
                thermal_debt_dm=dm,
                indoor_temp=indoor,
                target_indoor_temp=TARGET,
                outdoor_temp=OUTDOOR,
                price_classification=price,
                current_time=now,
                price_periods=prices,
            )
        self.coordinator.data = {"dhw_planning": {"thermal_debt": dm, "indoor_temperature": indoor}}
        await EffektGuardCoordinator._apply_dhw_control(self.coordinator, decision, tank, now)
        return decision


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scheduled,dm",
    [(False, DM), (True, DM), (True, -1000.0)],
    ids=["price_moved", "scheduled_window_at_target", "scheduled_window_at_target_in_thermal_debt"],
)
async def test_a_decision_not_to_heat_waits_for_the_minimum_run(scheduled, dm):
    """The October pattern: started cheap, at its target or out of the cheap price five minutes on.

    The scheduled window outranks the thermal-debt block to start a boost, so the debt does not cut
    the boost short either - or every top-up in a cold snap is a five-minute run at near full speed.
    """
    house = _House(_scheduler(scheduled=scheduled))
    started = await house.cycle(START, tank=40.0, price="cheap", dm=dm)
    assert started.should_heat and house.calls == ["turn_on"], "precondition: a boost started"

    for now in CYCLES[:2]:
        stop = await house.cycle(now, tank=45.3, price="normal", dm=dm)
        assert not stop.should_heat and not stop.for_safety, "precondition: no heat, not safety"
        assert house.calls == ["turn_on"], (
            f"{stop.priority_reason} ended the boost {(now - START).total_seconds() / 60:.1f} min "
            f"after it started, inside its {DHW_MIN_RUN_MINUTES}-min minimum run"
        )

    await house.cycle(CYCLES[2], tank=46.0, price="normal", dm=dm)
    assert house.calls == ["turn_on", "turn_off"], (
        f"the third cycle after the start, {(CYCLES[2] - START).total_seconds():.0f} s on, did not "
        f"end the boost: {house.calls}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scheduled,dm,indoor,tank,reason",
    [
        (False, -1000.0, 20.8, 41.0, "CRITICAL_THERMAL_DEBT"),
        (False, DM, MIN_TEMP_LIMIT - 0.5, 41.0, "SPACE_HEATING_EMERGENCY"),
        (True, DM, MIN_TEMP_LIMIT - 0.5, 41.0, "DHW_SCHEDULED_BLOCKED_BY_SAFETY"),
    ],
    ids=["thermal_debt", "house_below_its_floor", "scheduled_window_house_below_its_floor"],
)
async def test_safety_ends_a_boost_inside_its_minimum_run(scheduled, dm, indoor, tank, reason):
    house = _House(_scheduler(scheduled=scheduled))
    await house.cycle(START, tank=40.0, price="cheap")
    assert house.calls == ["turn_on"], "precondition: a boost started"

    stop = await house.cycle(CYCLES[0], tank=tank, price="cheap", dm=dm, indoor=indoor)
    assert stop.priority_reason.startswith(reason), f"precondition: {stop.priority_reason}"
    assert house.calls == [
        "turn_on",
        "turn_off",
    ], f"{reason} did not stop a boost five minutes into its minimum run: {house.calls}"


@pytest.mark.asyncio
async def test_a_boost_the_tank_would_finish_inside_the_minimum_run_is_not_started():
    """One degree short of the owner's 45 C: a five-minute boost, the compressor near full speed."""
    house = _House(_scheduler())
    decision = await house.cycle(START, tank=44.0, price="cheap")

    assert decision.should_heat, f"precondition: {decision.priority_reason} decides to heat"
    assert house.calls == [], (
        f"{decision.priority_reason} started a boost the tank finishes in "
        f"{house.scheduler.estimate_heating_time(44.0, decision.target_temp) * 60:.0f} min"
    )


def _an_evening_at_its_cheapest_left() -> tuple[PriceAnalyzer, list[QuarterPeriod]]:
    """The day's cheap hours behind it: 20:45 is EXPENSIVE, and from 21:00 the cheapest hour left."""
    midnight = START.replace(hour=0, minute=0, second=0)
    today = [
        QuarterPeriod(
            start_time=midnight + timedelta(minutes=15 * k),
            price=100.0 if k < 83 else 150.0 if k < 87 else 200.0,
        )
        for k in range(96)
    ]
    analyzer = PriceAnalyzer()
    analyzer.update_prices(PriceData(today=today, tomorrow=[], has_tomorrow=False))
    return analyzer, today


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expensive,rule",
    [(False, "DHW_SAFETY_MINIMUM"), (True, "DHW_SAFETY_WINDOW_")],
    ids=["normal_price", "expensive_cheapest_left"],
)
async def test_a_tank_below_the_safety_minimum_is_heated_however_short_the_run(expensive, rule):
    if expensive:
        analyzer, prices = _an_evening_at_its_cheapest_left()
        now, price = START.replace(hour=20, minute=59, second=12), "expensive"
    else:
        analyzer, prices, now, price = None, None, START, "normal"
    house = _House(_scheduler(analyzer=analyzer))
    decision = await house.cycle(now, tank=28.0, price=price, prices=prices)

    assert decision.should_heat and decision.priority_reason.startswith(
        rule
    ), f"precondition: {decision.priority_reason}"
    assert house.scheduler.estimate_heating_time(28.0, decision.target_temp) * 60 < (
        DHW_MIN_RUN_MINUTES
    ), "precondition: shorter than the minimum run"
    assert house.calls == [
        "turn_on"
    ], f"{decision.priority_reason}: a tank below the safety minimum was left unheated"


def _preheat_decision() -> DHWScheduleDecision:
    """The shape of RULE 4.7's heat: the tank's own target among the abort conditions."""
    return DHWScheduleDecision(
        should_heat=True,
        priority_reason="PEAK_AVOIDANCE_PREHEAT_6H",
        target_temp=TANK_TARGET,
        max_runtime_minutes=45,
        abort_conditions=["thermal_debt < -900", f"dhw_temp >= {TANK_TARGET}"],
    )


@pytest.mark.asyncio
async def test_the_tank_at_its_target_does_not_end_the_minimum_run():
    house = _House(_scheduler())
    await house.cycle(START, tank=40.0, price="cheap", decision=_preheat_decision())
    assert house.calls == ["turn_on"], "precondition: a boost started"

    await house.cycle(CYCLES[0], tank=45.5, price="cheap", decision=_preheat_decision())
    assert house.calls == ["turn_on"], "the tank reaching its target ended the minimum run"

    await house.cycle(CYCLES[2], tank=47.0, price="cheap", decision=_preheat_decision())
    assert house.calls == ["turn_on", "turn_off"], "past its minimum run the target ends it"


@pytest.mark.asyncio
async def test_a_thermal_debt_abort_ends_the_minimum_run():
    house = _House(_scheduler())
    await house.cycle(START, tank=40.0, price="cheap", decision=_preheat_decision())

    await house.cycle(CYCLES[0], tank=42.0, price="cheap", dm=-950.0, decision=_preheat_decision())
    assert house.calls == ["turn_on", "turn_off"], "an abort for the house waited out the minimum"

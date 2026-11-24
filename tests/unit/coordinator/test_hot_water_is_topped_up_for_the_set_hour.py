"""Hot water for the owner's set hour: the window opens 8 h before it, and a top-up can start.

The owner's settings (production export 2026-11-22): hot water ready by 06:00 and 21:00, 45 C, at
least 10 minutes. On 4 of 10 evenings, 11-12 to 11-22, the tank held less than 10 minutes at 21:00:
- 11-17: from 20:15 the window asked for heat at every cycle with 7-9 minutes in the tank, and no
  boost started - the tank top read 42.4-43.4 C, which the heating-time estimate (replayed) finishes
  inside the 15-minute minimum run to 20:45, so the control refused to start it.
- 11-18: a boost stopped at 20:00 for a PEAK price, the tank was emptied by 20:15, and no boost
  could start before 21:00 - the 60-minute gap between boosts ran from that stop.
The owner decided (2026-11-22, 2026-11-23): the window opens 8 h before the set hour instead of 6;
a top-up the window asks for starts even if the tank would reach its target sooner than the
15-minute minimum run, and still runs those 15 minutes; the 60 minutes before EffektGuard's next
boost count from the last start - a boost_dhw call's too.

The real scheduler and the real hot-water control, on a duck-typed coordinator (as in
test_a_hot_water_boost_runs_its_minimum).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from homeassistant.const import STATE_OFF, STATE_ON

from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
from custom_components.effektguard.const import DHW_MIN_RUN_MINUTES
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
DAY = datetime(2026, 11, 18, tzinfo=STOCKHOLM)
LUX = "switch.temporary_lux_50004"
TANK_TARGET = 45.0  # the owner's
MIN_AMOUNT = 10  # the owner's
DM = -150.0


def _at(hour: int, minute: int, second: int = 10) -> datetime:
    """A control cycle: they run at :X0:10 and :X5:10."""
    return DAY.replace(hour=hour, minute=minute, second=second)


def _scheduler() -> IntelligentDHWScheduler:
    detector = ClimateZoneDetector(latitude=55.6)
    scheduler = IntelligentDHWScheduler(
        demand_periods=[
            DHWDemandPeriod(
                availability_hour=hour,
                target_temp=TANK_TARGET,
                duration_hours=hours,
                min_amount_minutes=MIN_AMOUNT,
            )
            for hour, hours in ((6, 2), (21, 3))  # as the coordinator builds them
        ],
        climate_detector=detector,
        emergency_layer=EmergencyLayer(detector, heating_type="concrete_ufh"),
        user_target_temp=TANK_TARGET,
    )
    scheduler.last_legionella_boost = DAY - timedelta(days=2)  # keep the hygiene rule out of it
    return scheduler


class _House:
    """The real hot-water control and switch door; the lux switch is a state the calls flip."""

    def __init__(self) -> None:
        self.lux = STATE_OFF
        self.calls: list[str] = []
        self.scheduler = _scheduler()

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
        coordinator.dhw_optimizer = self.scheduler
        coordinator.optimization_enabled = True
        coordinator._control_lock = asyncio.Lock()
        coordinator.nibe.get_current_state = AsyncMock(
            return_value=SimpleNamespace(degree_minutes=DM, indoor_temp=21.2)
        )
        for name in (
            "_is_dhw_start_rate_limited",
            "_set_temporary_lux",
            "_lux_turned_off",
            "async_start_dhw_boost",
            "_end_user_dhw_boost_when_due",
        ):
            method = getattr(EffektGuardCoordinator, name)
            setattr(coordinator, name, lambda *args, _m=method: _m(coordinator, *args))
        self.coordinator = coordinator

    def price(self, now: datetime, quarters: dict[int, float]) -> tuple[list, str]:
        """A day of quarters rising 50 -> 145 öre, with `quarters` set; the real price analyzer."""
        periods = [
            QuarterPeriod(
                start_time=DAY + timedelta(minutes=15 * i), price=quarters.get(i, 50.0 + i)
            )
            for i in range(96)
        ]
        data = PriceData(today=periods, tomorrow=[], has_tomorrow=False)
        self.scheduler.price_analyzer = PriceAnalyzer()
        self.scheduler.price_analyzer.update_prices(data)
        index = data.get_period_index(now)
        return periods, self.scheduler.price_analyzer.get_current_classification(index)

    def decide(
        self, now: datetime, *, tank: float, amount: float, price: str, periods: list | None = None
    ):
        return self.scheduler.should_start_dhw(
            current_dhw_temp=tank,
            space_heating_demand_kw=1.0,
            thermal_debt_dm=DM,
            indoor_temp=21.2,
            target_indoor_temp=20.5,
            outdoor_temp=2.0,
            price_classification=price,
            current_time=now,
            dhw_amount_minutes=amount,
            price_periods=periods,
        )

    async def cycle(
        self, now: datetime, *, tank: float, amount: float, price: str, periods: list | None = None
    ) -> DHWScheduleDecision:
        """One control cycle: the scheduler's decision on this state, applied by the control."""
        decision = self.decide(now, tank=tank, amount=amount, price=price, periods=periods)
        self.coordinator.data = {"dhw_planning": {"thermal_debt": DM, "indoor_temperature": 21.2}}
        await EffektGuardCoordinator._apply_dhw_control(self.coordinator, decision, tank, now)
        return decision


@pytest.mark.parametrize(
    "now,in_window",
    [(_at(13, 0), True), (_at(12, 55), False)],
    ids=["8_h_before_21", "8_h_5_min_before_21"],
)
def test_the_window_opens_eight_hours_before_the_set_hour(now, in_window):
    """13:00 is inside the 21:00 window; 12:55, before it opens, is not."""
    decision = _House().decide(now, tank=42.0, amount=9.0, price="normal")

    assert decision.priority_reason.startswith("DHW_SCHEDULED_") is in_window, (
        f"{now:%H:%M:%S}: {decision.priority_reason} - the 21:00 window should "
        f"{'' if in_window else 'not '}be open"
    )
    assert decision.should_heat is in_window, f"{now:%H:%M:%S}: {decision.priority_reason}"


@pytest.mark.asyncio
async def test_a_top_up_short_of_the_minutes_starts_and_runs_its_minimum():
    """11-17 at 20:15: 9 minutes in the tank, its top at 43.4 C, the window asking for heat.

    It starts, and it runs the 15-minute minimum although the tank reaches its target sooner.
    """
    house = _House()
    assert (
        house.scheduler.estimate_heating_time(43.4, TANK_TARGET) * 60 < DHW_MIN_RUN_MINUTES
    ), "precondition: a top-up the tank finishes inside the minimum run"
    decision = await house.cycle(_at(20, 15), tank=43.4, amount=9.0, price="normal")

    assert decision.should_heat and decision.priority_reason.startswith(
        "DHW_SCHEDULED_PRIORITY"
    ), f"precondition: {decision.priority_reason}"
    assert house.calls == ["turn_on"], (
        f"{decision.priority_reason} with {9} of {MIN_AMOUNT} minutes was not started: the tank "
        f"finishes {TANK_TARGET} C in "
        f"{house.scheduler.estimate_heating_time(43.4, TANK_TARGET) * 60:.0f} min"
    )

    full = await house.cycle(_at(20, 25), tank=45.5, amount=12.0, price="normal")
    assert not full.should_heat, f"precondition: the target is reached, {full.priority_reason}"
    assert house.calls == ["turn_on"], "the top-up stopped 10 min in, inside its minimum run"

    await house.cycle(_at(20, 30), tank=46.0, amount=12.0, price="normal")
    assert house.calls == ["turn_on", "turn_off"], "the top-up ran on past its 15 minutes"


@pytest.mark.asyncio
async def test_a_short_top_up_starts_as_the_cheapest_quarters_begin():
    """19:40, 43.4 C, 9 minutes; the cheapest 45 minutes of the window start at 19:45."""
    house = _House()
    now = _at(19, 40)
    periods, price = house.price(now, {79: 10.0, 80: 10.0, 81: 10.0})
    assert price != "peak", f"precondition: {price}"
    decision = await house.cycle(now, tank=43.4, amount=9.0, price=price, periods=periods)

    assert decision.should_heat and decision.priority_reason.startswith(
        "DHW_SCHEDULED_OPTIMAL"
    ), f"precondition: {decision.priority_reason}"
    assert house.calls == ["turn_on"], f"{decision.priority_reason} with 9 minutes was not started"


@pytest.mark.asyncio
async def test_the_hour_between_boosts_counts_from_the_start():
    """11-18: a boost from 19:30 stopped for PEAK at 20:00, the tank emptied; 20:30 may start."""
    house = _House()
    assert (
        house.scheduler.estimate_heating_time(40.9, TANK_TARGET) * 60 >= DHW_MIN_RUN_MINUTES
    ), "precondition: a boost longer than the minimum run, so only the hour between boosts decides"
    await house.cycle(_at(19, 30), tank=40.9, amount=5.0, price="expensive")
    assert house.calls == ["turn_on"], "precondition: the window started a boost"

    stop = await house.cycle(_at(20, 0), tank=40.9, amount=9.0, price="peak")
    assert stop.priority_reason == "DHW_PEAK_AVOIDED_SCHEDULED", f"precondition: {stop}"
    assert house.calls == ["turn_on", "turn_off"], "precondition: PEAK stopped it, 30 min in"

    held = await house.cycle(_at(20, 25), tank=38.9, amount=0.0, price="expensive")
    assert held.should_heat, f"precondition: {held.priority_reason}"
    assert house.calls == ["turn_on", "turn_off"], "a second start inside the hour from the first"

    again = await house.cycle(_at(20, 30), tank=38.9, amount=0.0, price="expensive")
    assert again.should_heat, f"precondition: {again.priority_reason}"
    assert house.calls == ["turn_on", "turn_off", "turn_on"], (
        "an hour after the last start, with the tank empty before 21:00, no boost started - "
        "the hour ran from the PEAK stop instead"
    )


@pytest.mark.asyncio
async def test_the_hour_after_a_boost_dhw_call_counts_from_its_start():
    """A 30-minute boost_dhw from 19:30: the window's top-up waits for 20:30, not 21:00."""
    house = _House()
    await house.coordinator.async_start_dhw_boost(30, _at(19, 30))
    await house.coordinator._end_user_dhw_boost_when_due(None, _at(20, 0))
    assert house.calls == ["turn_on", "turn_off"], "precondition: the user's boost ran its 30 min"

    held = await house.cycle(_at(20, 5), tank=40.9, amount=0.0, price="expensive")
    assert held.should_heat, f"precondition: {held.priority_reason}"
    assert house.calls == ["turn_on", "turn_off"], "a start 35 min after the user's boost started"

    await house.cycle(_at(20, 30), tank=40.4, amount=0.0, price="expensive")
    assert house.calls == [
        "turn_on",
        "turn_off",
        "turn_on",
    ], "an hour after the user's boost started, no top-up started - the hour ran from its end"

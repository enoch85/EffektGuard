"""An ordinary hot-water boost starts only at a good price: cheap or very cheap, its whole minimum run.

The comfort top-up (RULE 7) heated a tank below 45 C at a normal price too, and every rule judged a
price by the current quarter alone: a boost started in a cheap quarter's last five minutes ran on
into the next one (owner, 2026-10-23: "we don't want to heat when the price isn't good"). A boost
runs DHW_MIN_RUN_MINUTES at least, so every quarter of those is now cheap or very cheap, or it
waits. The owner's hot-water times (RULE 0) and a tank below the safety minimum (RULE 3) still heat
at any price.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from custom_components.effektguard.adapters.gespot_adapter import (
    PriceData,
    QuarterPeriod,
    index_containing,
)
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer

STOCKHOLM = ZoneInfo("Europe/Stockholm")
NOW = datetime(2026, 1, 15, 2, 5, 12, tzinfo=STOCKHOLM)
MIDNIGHT = NOW.replace(hour=0, minute=0, second=0)


def _scheduler(analyzer: PriceAnalyzer | None = None) -> IntelligentDHWScheduler:
    detector = ClimateZoneDetector(latitude=59.33)
    scheduler = IntelligentDHWScheduler(
        climate_detector=detector,
        emergency_layer=EmergencyLayer(detector, heating_type="radiator"),
        user_target_temp=50.0,
        price_analyzer=analyzer,
    )
    scheduler.last_legionella_boost = NOW - timedelta(days=2)  # keep the hygiene rule out of it
    return scheduler


def _decide(scheduler, *, tank: float, price: str, prices=None, volatile: bool = False):
    return scheduler.should_start_dhw(
        current_dhw_temp=tank,
        space_heating_demand_kw=1.0,
        thermal_debt_dm=-200.0,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=0.0,
        price_classification=price,
        current_time=NOW,
        price_periods=prices,
        is_volatile=volatile,
    )


@pytest.mark.parametrize("price", ["normal", "expensive"])
def test_a_low_tank_is_not_topped_up_at_a_price_that_is_not_good(price):
    decision = _decide(_scheduler(), tank=40.0, price=price)

    assert not decision.should_heat, (
        f"{decision.priority_reason}: a 40 C tank topped up at a {price} price, outside the "
        f"owner's hot-water times"
    )
    assert (
        decision.priority_reason == "DHW_COMFORT_WAITING_CHEAP"
    ), f"{decision.priority_reason}: a 40 C tank waiting for a cheap price is not adequate"


def _day(cheap_until: datetime) -> tuple[PriceAnalyzer, list[QuarterPeriod]]:
    """A day at 100, cheap at 10 from midnight to `cheap_until` and from 12:00 to 14:00, 200 at
    17:00-20:00, and the day's cheapest hours at 1 from 03:00 - so now is not its cheapest window.
    """

    def price(start: datetime) -> float:
        if start.hour in (3, 4):
            return 1.0
        if start < cheap_until or 12 <= start.hour < 14:
            return 10.0
        if 17 <= start.hour < 20:
            return 200.0
        return 100.0

    today = [
        QuarterPeriod(
            start_time=MIDNIGHT + timedelta(minutes=15 * k),
            price=price(MIDNIGHT + timedelta(minutes=15 * k)),
        )
        for k in range(96)
    ]
    analyzer = PriceAnalyzer()
    analyzer.update_prices(PriceData(today=today, tomorrow=[], has_tomorrow=False))
    return analyzer, today


@pytest.mark.parametrize(
    "cheap_until,heats",
    [(NOW.replace(minute=30, second=0), True), (NOW.replace(minute=15, second=0), False)],
    ids=["cheap_for_the_whole_minimum_run", "cheap_quarter_ending_in_ten_minutes"],
)
def test_a_boost_starts_only_if_every_quarter_of_its_minimum_run_is_cheap(cheap_until, heats):
    analyzer, prices = _day(cheap_until)
    now = analyzer.get_classification_for_period(prices[index_containing(prices, NOW)])
    after = analyzer.get_classification_for_period(prices[index_containing(prices, NOW) + 1])
    assert now in ("cheap", "very_cheap"), f"precondition: this quarter is {now}"
    assert (after in ("cheap", "very_cheap")) == heats, f"precondition: the next quarter is {after}"

    decision = _decide(_scheduler(analyzer), tank=40.0, price=str(now), prices=prices)

    assert decision.should_heat == heats, (
        f"{decision.priority_reason}: this quarter {now}, the next {after} - a boost started at "
        f"{NOW:%H:%M:%S} runs into it"
    )

"""A VERY_CHEAP quarter is a cheap quarter to every hot-water rule.

`is_cheap_price` counts VERY_CHEAP as cheap, but RULE 5 tested `price_classification != "cheap"`
and RULE 7 (and RULE 3.5) spelled `["cheap", "normal"]`. So the cheapest quarters of the day were
treated as dearer than CHEAP: an adequate tank waited through a very cheap night for a cheap
quarter (review DHW F4), and a low tank RULE 7 heats at cheap and at normal price was not heated
at all. Every rule now reads the one notion.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from custom_components.effektguard.adapters.gespot_adapter import QuarterPeriod
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer

STOCKHOLM = ZoneInfo("Europe/Stockholm")
NOW = datetime(2026, 1, 15, 2, 0, tzinfo=STOCKHOLM)
MIDNIGHT = NOW.replace(hour=0)


def _prices_cheapest_now() -> list[QuarterPeriod]:
    """A day at 100, with the hour from now at 10 - the cheapest window starts now."""
    return [
        QuarterPeriod(
            start_time=MIDNIGHT + timedelta(minutes=15 * k),
            price=(
                10.0
                if NOW <= MIDNIGHT + timedelta(minutes=15 * k) < NOW + timedelta(hours=1)
                else 100.0
            ),
        )
        for k in range(96)
    ]


def _decide(price: str, *, tank: float, volatile: bool, prices):
    detector = ClimateZoneDetector(latitude=59.33)
    scheduler = IntelligentDHWScheduler(
        climate_detector=detector,
        emergency_layer=EmergencyLayer(detector, heating_type="radiator"),
        user_target_temp=50.0,
        price_analyzer=PriceAnalyzer(),
    )
    scheduler.last_legionella_boost = NOW - timedelta(days=2)  # keep the hygiene rule out of it
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


@pytest.mark.parametrize(
    "tank,volatile,with_prices",
    [(47.0, False, True), (40.0, True, False)],
    ids=["adequate_tank_in_the_cheapest_window", "low_tank_in_a_brief_cheap_run"],
)
def test_a_very_cheap_quarter_heats_where_a_cheap_one_does(tank, volatile, with_prices):
    prices = _prices_cheapest_now() if with_prices else None
    cheap = _decide("cheap", tank=tank, volatile=volatile, prices=prices)
    very_cheap = _decide("very_cheap", tank=tank, volatile=volatile, prices=prices)

    assert cheap.should_heat, f"precondition: a cheap quarter heats here ({cheap.priority_reason})"
    assert very_cheap.should_heat, (
        f"a CHEAP quarter heats this tank ({cheap.priority_reason}) and a VERY_CHEAP one does not "
        f"({very_cheap.priority_reason})"
    )

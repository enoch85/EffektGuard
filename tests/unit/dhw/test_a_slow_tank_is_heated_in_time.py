"""A tank too slow to heat inside the scheduled window is heated in time.

The scheduled window opens DHW_SCHEDULED_WINDOW_HOURS before a demand period and heats a tank that
has not reached its target. A tank that needs longer than that used to fall to a separate
pre-schedule rule outside the window, reachable for no other tank (F-143) - and that rule waited for
the cheapest hour before the deadline even with no hour to spare, so the tank was late. The window
now opens as early as the tank needs.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from custom_components.effektguard.adapters.gespot_adapter import QuarterPeriod
from custom_components.effektguard.const import DHW_HEATING_RATE_MIN
from custom_components.effektguard.optimization.dhw_optimizer import (
    DHWDemandPeriod,
    IntelligentDHWScheduler,
)
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer

NOW = datetime(2026, 1, 15, 0, 48, tzinfo=ZoneInfo("Europe/Stockholm"))  # 6.2 h before 07:00


def test_a_slow_tank_starts_when_the_time_left_is_all_it_needs():
    """31 -> 60 C at the slowest heating rate takes 5.8 h, 6.3 h with the buffer; 6.2 h are left.

    The current quarter is expensive and a cheap stretch starts two hours later - too late."""
    scheduler = IntelligentDHWScheduler(
        demand_periods=[DHWDemandPeriod(availability_hour=7, target_temp=60.0, duration_hours=2)],
        price_analyzer=PriceAnalyzer(),
        user_target_temp=60.0,
    )
    scheduler.learned_heating_rate = DHW_HEATING_RATE_MIN
    scheduler.last_legionella_boost = NOW - timedelta(days=2)
    start = NOW.replace(minute=45)
    prices = [200.0] * 8 + [20.0] * 12 + [200.0] * 20
    quarters = [
        QuarterPeriod(start_time=start + timedelta(minutes=15 * k), price=price)
        for k, price in enumerate(prices)
    ]

    decision = scheduler.should_start_dhw(
        current_dhw_temp=31.0,
        space_heating_demand_kw=1.0,
        thermal_debt_dm=-100.0,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=-5.0,
        price_classification="expensive",
        current_time=NOW,
        price_periods=quarters,
    )

    assert decision.should_heat, decision.priority_reason

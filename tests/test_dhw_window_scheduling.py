"""Test DHW window-based scheduling (Phase 5.2).

Tests the sliding window algorithm that finds the cheapest 45-minute
continuous window for DHW heating across today and tomorrow's prices.
"""

import pytest
from datetime import datetime, timedelta
from custom_components.effektguard.optimization.dhw_optimizer import (
    IntelligentDHWScheduler,
    DHWDemandPeriod,
)
from custom_components.effektguard.adapters.gespot_adapter import QuarterPeriod


def create_mock_quarters(base_time: datetime, prices: list[float]) -> list[QuarterPeriod]:
    """Create mock QuarterPeriod objects for testing.

    Args:
        base_time: Starting datetime (should be on quarter boundary)
        prices: List of prices for consecutive 15-min periods

    Returns:
        List of QuarterPeriod objects
    """
    quarters = []
    for i, price in enumerate(prices):
        period_time = base_time + timedelta(minutes=i * 15)
        quarters.append(
            QuarterPeriod(
                quarter_of_day=(period_time.hour * 4 + period_time.minute // 15) % 96,
                hour=period_time.hour,
                minute=period_time.minute,
                price=price,
                is_daytime=(6 <= period_time.hour < 22),
            )
        )
    return quarters


def test_finds_cheapest_window_tomorrow():
    """Test Phase 5.2: Window scheduling finds tomorrow's cheaper prices.

    Real-world scenario from Oct 17-18, 2025:
    - Today Q94 (23:45): 63.94 öre/kWh
    - Tomorrow Q30-Q32 (07:30-08:15): 38.00 öre/kWh average (40% cheaper!)
    """
    scheduler = IntelligentDHWScheduler()

    # Create mock price data: expensive today evening, cheap tomorrow morning
    current_time = datetime(2025, 10, 17, 23, 45)

    # Today's last quarters (expensive)
    today_prices = [63.94] * 4  # Q92-Q95 (23:00-00:00)
    today_quarters = create_mock_quarters(datetime(2025, 10, 17, 23, 0), today_prices)

    # Tomorrow morning (cheap)
    tomorrow_prices = [
        45.0,  # 07:00 Q28
        42.0,  # 07:15 Q29
        38.0,  # 07:30 Q30
        38.0,  # 07:45 Q31
        38.0,  # 08:00 Q32
        40.0,  # 08:15 Q33
        50.0,  # 08:30 Q34
    ]
    tomorrow_quarters = create_mock_quarters(datetime(2025, 10, 18, 7, 0), tomorrow_prices)

    # Combine price data
    all_quarters = today_quarters + tomorrow_quarters

    # Find cheapest 45-minute window
    window = scheduler.find_cheapest_dhw_window(
        current_time=current_time,
        lookahead_hours=12,
        dhw_duration_minutes=45,
        price_periods=all_quarters,
    )

    assert window is not None, "Should find optimal window"
    assert window["avg_price"] < 40.0, f"Should find cheap window, got {window['avg_price']}"
    assert window["hours_until"] > 7.0, "Should be ~8 hours away"
    assert window["hours_until"] < 9.0, "Should be ~8 hours away"
    assert len(window["quarters"]) == 3, "45 min = 3 quarters"

    # Verify it found the 07:30-08:15 window (Q30-Q32)
    assert window["start_time"].hour == 7
    assert window["start_time"].minute == 30


def test_lookahead_until_demand_period():
    """Test that lookahead stops at next demand period."""
    demand_period = DHWDemandPeriod(
        start_hour=7,
        target_temp=55.0,
        duration_hours=2,
    )
    scheduler = IntelligentDHWScheduler(demand_periods=[demand_period])

    # Current time: 23:45 (7.25 hours until 07:00 demand period)
    current_time = datetime(2025, 10, 17, 23, 45)

    lookahead = scheduler.get_lookahead_hours(current_time)

    # Should look until demand period (~7 hours), not full 24h
    assert lookahead == 7, f"Expected 7 hours lookahead, got {lookahead}"


def test_lookahead_defaults_to_24h():
    """Test that lookahead defaults to 24h when no demand period configured."""
    scheduler = IntelligentDHWScheduler(demand_periods=[])

    current_time = datetime(2025, 10, 17, 23, 45)

    lookahead = scheduler.get_lookahead_hours(current_time)

    assert lookahead == 24, "Should default to 24h when no demand period"


def test_window_handles_insufficient_data():
    """Test graceful handling when not enough price data available."""
    scheduler = IntelligentDHWScheduler()

    current_time = datetime(2025, 10, 17, 23, 45)

    # Only 2 quarters available, but need 3 for 45-minute window
    limited_quarters = create_mock_quarters(datetime(2025, 10, 18, 7, 0), [38.0, 38.0])

    window = scheduler.find_cheapest_dhw_window(
        current_time=current_time,
        lookahead_hours=12,
        dhw_duration_minutes=45,
        price_periods=limited_quarters,
    )

    assert window is None, "Should return None when insufficient data"


def test_window_finds_continuous_periods():
    """Test that window only considers continuous 15-minute periods."""
    scheduler = IntelligentDHWScheduler()

    current_time = datetime(2025, 10, 17, 23, 0)

    # Create quarters with a gap
    quarters = []
    quarters.extend(create_mock_quarters(datetime(2025, 10, 18, 7, 0), [40.0, 35.0]))
    # Gap here (30 minutes missing)
    quarters.extend(create_mock_quarters(datetime(2025, 10, 18, 8, 0), [30.0, 30.0, 30.0]))

    window = scheduler.find_cheapest_dhw_window(
        current_time=current_time,
        lookahead_hours=12,
        dhw_duration_minutes=45,
        price_periods=quarters,
    )

    # Should find the 08:00-08:45 window (continuous), not mix with 07:00 gap
    assert window is not None
    assert window["start_time"].hour == 8
    assert window["avg_price"] == 30.0


def test_window_multiple_cheap_periods():
    """Test that window finds absolute cheapest among multiple cheap periods."""
    scheduler = IntelligentDHWScheduler()

    current_time = datetime(2025, 10, 17, 23, 0)

    # Two cheap periods: morning (40 öre avg) and afternoon (35 öre avg)
    morning_quarters = create_mock_quarters(
        datetime(2025, 10, 18, 7, 0), [42.0, 40.0, 38.0, 40.0]  # Avg ~40
    )
    afternoon_quarters = create_mock_quarters(
        datetime(2025, 10, 18, 14, 0), [37.0, 35.0, 33.0, 35.0]  # Avg ~35
    )

    all_quarters = morning_quarters + afternoon_quarters

    window = scheduler.find_cheapest_dhw_window(
        current_time=current_time,
        lookahead_hours=20,
        dhw_duration_minutes=45,
        price_periods=all_quarters,
    )

    assert window is not None
    assert window["start_time"].hour == 14, "Should find afternoon (cheaper)"
    assert window["avg_price"] < 36.0, "Should find absolute cheapest window"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

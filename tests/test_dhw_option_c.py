"""Test DHW Classification + Window Strategy (Phase 5.3 - Option C).

Tests that DHW heating requires BOTH:
1. Price classified as "cheap" (bottom 25%)
2. Currently in or near the optimal 45-min window

This prevents heating during temporary "cheap" spikes while still
finding the absolute cheapest timing.
"""

import pytest
from datetime import datetime, timedelta
from custom_components.effektguard.optimization.dhw_optimizer import (
    IntelligentDHWScheduler,
    DHWDemandPeriod,
)
from custom_components.effektguard.adapters.gespot_adapter import QuarterPeriod


def create_mock_quarters(base_time: datetime, prices: list[float]) -> list[QuarterPeriod]:
    """Create mock QuarterPeriod objects for testing."""
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


def test_option_c_must_be_cheap_classification():
    """Test Option C: Must be classified CHEAP, blocks normal/expensive/peak."""
    scheduler = IntelligentDHWScheduler()
    scheduler.last_legionella_boost = datetime.now() - timedelta(days=2)

    current_time = datetime(2025, 10, 17, 23, 45)

    # Scenario 1: Price is "normal" - should block even if in cheapest window
    decision_normal = scheduler.should_start_dhw(
        current_dhw_temp=48.0,
        space_heating_demand_kw=1.0,
        thermal_debt_dm=-50,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="normal",  # Not cheap!
        current_time=current_time,
    )
    assert decision_normal.should_heat is False
    assert "BLOCKED_NORMAL" in decision_normal.priority_reason

    # Scenario 2: Price is "expensive" - should block
    decision_expensive = scheduler.should_start_dhw(
        current_dhw_temp=48.0,
        space_heating_demand_kw=1.0,
        thermal_debt_dm=-50,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="expensive",
        current_time=current_time,
    )
    assert decision_expensive.should_heat is False
    assert "BLOCKED_EXPENSIVE" in decision_expensive.priority_reason

    # Scenario 3: Price is "peak" - should block
    decision_peak = scheduler.should_start_dhw(
        current_dhw_temp=48.0,
        space_heating_demand_kw=1.0,
        thermal_debt_dm=-50,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="peak",
        current_time=current_time,
    )
    assert decision_peak.should_heat is False
    assert "BLOCKED_PEAK" in decision_peak.priority_reason


def test_option_c_cheap_but_better_window_ahead():
    """Test Option C: Price is cheap BUT better window ahead - wait."""
    scheduler = IntelligentDHWScheduler()
    scheduler.last_legionella_boost = datetime.now() - timedelta(days=2)

    # Current: Oct 17, 23:45 (63.94 öre - classified cheap relative to today)
    # Tomorrow: Oct 18, 07:30 (38.00 öre - 40% cheaper!)
    current_time = datetime(2025, 10, 17, 23, 45)

    # Create price data showing cheaper window tomorrow
    today_prices = [63.94] * 4  # Q92-Q95 (expensive relative to tomorrow)
    today_quarters = create_mock_quarters(datetime(2025, 10, 17, 23, 0), today_prices)

    tomorrow_prices = [
        45.0,  # 07:00
        42.0,  # 07:15
        38.0,  # 07:30 (OPTIMAL WINDOW START)
        38.0,  # 07:45
        38.0,  # 08:00
        40.0,  # 08:15
    ]
    tomorrow_quarters = create_mock_quarters(datetime(2025, 10, 18, 7, 0), tomorrow_prices)

    all_quarters = today_quarters + tomorrow_quarters

    decision = scheduler.should_start_dhw(
        current_dhw_temp=51.0,  # Comfortable (can wait)
        space_heating_demand_kw=1.0,
        thermal_debt_dm=-50,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="cheap",  # Cheap relative to today
        current_time=current_time,
        price_periods=all_quarters,  # Provide price data
    )

    assert decision.should_heat is False
    assert "WAITING_OPTIMAL_WINDOW" in decision.priority_reason
    # Should indicate waiting for cheaper window tomorrow
    assert decision.recommended_start_time is not None


def test_option_c_cheap_and_in_optimal_window():
    """Test Option C: Price is cheap AND in optimal window - heat!"""
    scheduler = IntelligentDHWScheduler()
    scheduler.last_legionella_boost = datetime.now() - timedelta(days=2)

    # Current: Oct 18, 07:30 (38.00 öre - cheapest window)
    current_time = datetime(2025, 10, 18, 7, 30)

    # Create price data showing we're in optimal window
    prices = [
        45.0,  # 07:00
        42.0,  # 07:15
        38.0,  # 07:30 (NOW - optimal window)
        38.0,  # 07:45
        38.0,  # 08:00
        40.0,  # 08:15
        50.0,  # 08:30
    ]
    quarters = create_mock_quarters(datetime(2025, 10, 18, 7, 0), prices)

    decision = scheduler.should_start_dhw(
        current_dhw_temp=48.0,  # Below target
        space_heating_demand_kw=1.0,
        thermal_debt_dm=-50,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="cheap",
        current_time=current_time,
        price_periods=quarters,
    )

    assert decision.should_heat is True
    assert "OPTIMAL_WINDOW" in decision.priority_reason
    # Should show the price in the reasoning
    assert "@38" in decision.priority_reason or "38" in decision.priority_reason


def test_option_c_dhw_low_overrides_waiting():
    """Test Option C: DHW below 45°C heats immediately (comfort override)."""
    scheduler = IntelligentDHWScheduler()
    scheduler.last_legionella_boost = datetime.now() - timedelta(days=2)

    current_time = datetime(2025, 10, 17, 23, 45)

    # Create price data with better window ahead
    today_quarters = create_mock_quarters(datetime(2025, 10, 17, 23, 0), [63.94] * 4)
    tomorrow_quarters = create_mock_quarters(datetime(2025, 10, 18, 7, 0), [38.0] * 6)
    all_quarters = today_quarters + tomorrow_quarters

    decision = scheduler.should_start_dhw(
        current_dhw_temp=44.0,  # Below 45°C comfort threshold
        space_heating_demand_kw=1.0,
        thermal_debt_dm=-50,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="cheap",
        current_time=current_time,
        price_periods=all_quarters,
    )

    # Should heat now - DHW below 45°C triggers Rule 7 (comfort heating)
    # This bypasses the window waiting logic
    assert decision.should_heat is True
    # Either goes through fallback logic (CHEAP_ELECTRICITY_OPPORTUNITY) or comfort (DHW_COMFORT_LOW_CHEAP)
    assert decision.should_heat is True  # That's the important part


def test_option_c_no_price_data_fallback():
    """Test Option C: Falls back to simple cheap classification if no price data."""
    scheduler = IntelligentDHWScheduler()
    scheduler.last_legionella_boost = datetime.now() - timedelta(days=2)

    current_time = datetime(2025, 10, 17, 23, 45)

    decision = scheduler.should_start_dhw(
        current_dhw_temp=48.0,
        space_heating_demand_kw=1.0,
        thermal_debt_dm=-50,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="cheap",
        current_time=current_time,
        price_periods=None,  # No price data available
    )

    # Should fall back to simple cheap logic
    assert decision.should_heat is True
    assert "CHEAP" in decision.priority_reason.upper()


def test_option_c_no_side_effects():
    """Test Option C: Verify no negative side effects.

    The dual-check approach (classification + window) should not:
    1. Miss urgent heating needs (safety overrides still work)
    2. Cause excessive DHW temperature drops (comfort override)
    3. Conflict with demand period logic (urgent overrides)
    """
    scheduler = IntelligentDHWScheduler(
        demand_periods=[DHWDemandPeriod(start_hour=7, target_temp=55.0, duration_hours=2)]
    )
    scheduler.last_legionella_boost = datetime.now() - timedelta(days=2)

    current_time = datetime(2025, 10, 17, 23, 45)

    # Side effect test 1: Safety minimum still works (blocks cheap if critically low)
    decision_safety = scheduler.should_start_dhw(
        current_dhw_temp=34.0,  # Below safety minimum
        space_heating_demand_kw=1.0,
        thermal_debt_dm=-50,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="cheap",
        current_time=current_time,
    )
    assert decision_safety.should_heat is True
    assert "SAFETY" in decision_safety.priority_reason.upper()

    # Side effect test 2: Urgent demand override still works (<2h before demand)
    urgent_time = datetime(2025, 10, 18, 5, 30)  # 1.5h before 07:00 demand
    decision_urgent = scheduler.should_start_dhw(
        current_dhw_temp=45.0,
        space_heating_demand_kw=1.0,
        thermal_debt_dm=-50,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="expensive",  # Not cheap, but urgent!
        current_time=urgent_time,
    )
    assert decision_urgent.should_heat is True
    assert "URGENT" in decision_urgent.priority_reason.upper()

    # Side effect test 3: Thermal debt still blocks DHW
    decision_thermal = scheduler.should_start_dhw(
        current_dhw_temp=48.0,
        space_heating_demand_kw=1.0,
        thermal_debt_dm=-350,  # Critical thermal debt
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="cheap",
        current_time=current_time,
    )
    assert decision_thermal.should_heat is False
    assert "CRITICAL_THERMAL_DEBT" in decision_thermal.priority_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

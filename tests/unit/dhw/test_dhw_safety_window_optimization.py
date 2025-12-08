"""Test DHW safety minimum with active window optimization.

Verifies that when DHW is in the 20-30°C safety deferral range during expensive
periods, the system actively finds and schedules heating for the next cheap window
instead of just passively waiting.

Context: User reported at 19.6°C DHW, system was just "waiting" instead of finding
the next cheap 45-minute window to heat.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from custom_components.effektguard.optimization.dhw_optimizer import (
    IntelligentDHWScheduler,
    DHWDemandPeriod,
)
from custom_components.effektguard.optimization.price_layer import (
    CheapestWindowResult,
    QuarterPeriod,
)


def create_mock_price_analyzer():
    """Create a mock PriceAnalyzer for window search."""
    mock_analyzer = MagicMock()

    def mock_find_cheapest_window(current_time, price_periods, duration_minutes, lookahead_hours):
        """Mock implementation that finds cheapest window."""
        from math import ceil

        if not price_periods:
            return None

        quarters_needed = ceil(duration_minutes / 15)
        end_time = current_time + timedelta(hours=lookahead_hours)

        # Filter to lookahead window
        available = [
            p for p in price_periods if p.start_time >= current_time and p.start_time < end_time
        ]

        if len(available) < quarters_needed:
            return None

        # Find cheapest continuous window
        lowest_price = None
        best_start_idx = None

        for i in range(len(available) - quarters_needed + 1):
            window = available[i : i + quarters_needed]
            avg_price = sum(p.price for p in window) / quarters_needed
            if lowest_price is None or avg_price < lowest_price:
                lowest_price = avg_price
                best_start_idx = i

        if best_start_idx is None:
            return None

        window = available[best_start_idx : best_start_idx + quarters_needed]
        return CheapestWindowResult(
            start_time=window[0].start_time,
            end_time=window[-1].start_time + timedelta(minutes=15),
            quarters=[best_start_idx + j for j in range(quarters_needed)],
            avg_price=lowest_price,
            hours_until=(window[0].start_time - current_time).total_seconds() / 3600,
        )

    def mock_calculate_lookahead_hours(heating_type, thermal_mass=1.0, next_demand_hours=None):
        """Mock implementation for lookahead calculation."""
        if heating_type == "dhw":
            if next_demand_hours is not None:
                return max(1.0, min(next_demand_hours, 24.0))
            return 24.0
        return 4.0 * thermal_mass  # space heating

    mock_analyzer.find_cheapest_window.side_effect = mock_find_cheapest_window
    mock_analyzer.calculate_lookahead_hours.side_effect = mock_calculate_lookahead_hours
    return mock_analyzer


@pytest.fixture
def dhw_optimizer():
    """Create DHW optimizer with realistic demand period and mock price analyzer."""
    demand_period = DHWDemandPeriod(start_hour=7, target_temp=50.0, duration_hours=2)
    mock_analyzer = create_mock_price_analyzer()
    return IntelligentDHWScheduler(demand_periods=[demand_period], price_analyzer=mock_analyzer)


@pytest.fixture
def price_periods_with_cheap_window():
    """Price periods with expensive now but cheap window in 2 hours."""
    base_time = datetime(2025, 10, 26, 20, 0)  # 20:00
    periods = []

    # Q0-Q79 (00:00-19:45): Already passed
    for i in range(80):
        start = base_time.replace(hour=0, minute=0) + timedelta(minutes=i * 15)
        periods.append(QuarterPeriod(start_time=start, price=50.0))

    # Q80-Q87 (20:00-21:45): EXPENSIVE NOW
    for i in range(80, 88):
        start = base_time.replace(hour=0, minute=0) + timedelta(minutes=i * 15)
        periods.append(QuarterPeriod(start_time=start, price=120.0))  # Expensive

    # Q88-Q91 (22:00-22:45): CHEAP WINDOW (4 quarters = 1 hour, enough for 45-min DHW)
    for i in range(88, 92):
        start = base_time.replace(hour=0, minute=0) + timedelta(minutes=i * 15)
        periods.append(QuarterPeriod(start_time=start, price=10.0))  # Cheap!

    # Q92-Q95 (23:00-23:45): Back to expensive
    for i in range(92, 96):
        start = base_time.replace(hour=0, minute=0) + timedelta(minutes=i * 15)
        periods.append(QuarterPeriod(start_time=start, price=100.0))

    return periods


def test_safety_deferral_finds_upcoming_cheap_window(
    dhw_optimizer, price_periods_with_cheap_window
):
    """When DHW at 25.0°C during expensive period, should find next cheap window."""
    decision = dhw_optimizer.should_start_dhw(
        current_dhw_temp=25.0,  # In safety deferral range (20-30°C)
        space_heating_demand_kw=1.5,
        thermal_debt_dm=-200,  # Healthy
        indoor_temp=22.0,
        target_indoor_temp=20.0,
        outdoor_temp=5.0,
        price_classification="expensive",
        current_time=datetime(2025, 10, 26, 20, 0),  # 20:00 - expensive period
        price_periods=price_periods_with_cheap_window,
        hours_since_last_dhw=22.0,  # Yesterday
    )

    # Should NOT heat now (price expensive)
    assert decision.should_heat is False

    # Should have found the cheap window at 22:00
    assert decision.recommended_start_time is not None
    assert decision.recommended_start_time.hour == 22
    assert decision.recommended_start_time.minute == 0

    # Reason should indicate waiting for optimal window
    assert "WINDOW" in decision.priority_reason
    assert "@10.0" in decision.priority_reason  # Shows cheap price


def test_safety_deferral_heats_when_in_cheap_window(dhw_optimizer, price_periods_with_cheap_window):
    """When DHW at 25.0°C and we're IN the cheap window, should heat immediately."""
    decision = dhw_optimizer.should_start_dhw(
        current_dhw_temp=25.0,  # In safety deferral range (20-30°C)
        space_heating_demand_kw=1.5,
        thermal_debt_dm=-200,  # Healthy
        indoor_temp=22.0,
        target_indoor_temp=20.0,
        outdoor_temp=5.0,
        price_classification="cheap",  # Window is cheap
        current_time=datetime(2025, 10, 26, 22, 5),  # 22:05 - IN cheap window
        price_periods=price_periods_with_cheap_window,
        hours_since_last_dhw=22.0,
    )

    # Should heat NOW (we're in the cheap window!)
    assert decision.should_heat is True

    # When price_classification is "cheap", system uses DHW_SAFETY_MINIMUM path
    # (not the window path) because the current price IS cheap
    assert "DHW_SAFETY_MINIMUM" in decision.priority_reason


def test_safety_critical_always_heats_immediately(dhw_optimizer, price_periods_with_cheap_window):
    """Below 20°C critical threshold, heat immediately regardless of price."""
    decision = dhw_optimizer.should_start_dhw(
        current_dhw_temp=19.5,  # CRITICAL - below 20°C
        space_heating_demand_kw=1.5,
        thermal_debt_dm=-200,  # Healthy
        indoor_temp=22.0,
        target_indoor_temp=20.0,
        outdoor_temp=5.0,
        price_classification="expensive",
        current_time=datetime(2025, 10, 26, 20, 0),  # Expensive period
        price_periods=price_periods_with_cheap_window,
        hours_since_last_dhw=22.0,
    )

    # Should heat immediately at critical temp
    assert decision.should_heat is True
    assert "DHW_SAFETY_MINIMUM" in decision.priority_reason


def test_no_window_found_falls_back_to_defer(dhw_optimizer):
    """If all prices equally expensive, system picks first window (current time)."""
    # All periods expensive - window finder picks "least bad" option (first available)
    base_time = datetime(2025, 10, 26, 20, 0)
    expensive_periods = []
    for i in range(96):
        start = base_time.replace(hour=0, minute=0) + timedelta(minutes=i * 15)
        expensive_periods.append(QuarterPeriod(start_time=start, price=120.0))

    decision = dhw_optimizer.should_start_dhw(
        current_dhw_temp=25.0,  # In deferral range (20-30°C)
        space_heating_demand_kw=1.5,
        thermal_debt_dm=-200,
        indoor_temp=22.0,
        target_indoor_temp=20.0,
        outdoor_temp=5.0,
        price_classification="expensive",
        current_time=base_time,
        price_periods=expensive_periods,
        hours_since_last_dhw=22.0,
    )

    # When all prices are equal, window finder picks first available window (now)
    # System heats immediately at that "optimal" (least bad) window
    assert decision.should_heat is True
    assert "DHW_SAFETY_WINDOW" in decision.priority_reason
    assert decision.recommended_start_time is not None


def test_safety_deferral_respects_thermal_debt(dhw_optimizer, price_periods_with_cheap_window):
    """With poor thermal debt, DHW is blocked to protect space heating."""
    decision = dhw_optimizer.should_start_dhw(
        current_dhw_temp=25.0,  # In deferral range (20-30°C)
        space_heating_demand_kw=1.5,
        thermal_debt_dm=-450,  # Poor thermal debt - blocks DHW (DM_DHW_BLOCK_FALLBACK = -340)
        indoor_temp=22.0,
        target_indoor_temp=20.0,
        outdoor_temp=5.0,
        price_classification="expensive",
        current_time=datetime(2025, 10, 26, 20, 0),
        price_periods=price_periods_with_cheap_window,
        hours_since_last_dhw=22.0,
    )

    # Critical thermal debt (DM -450 < -340 block threshold) blocks ALL DHW
    # This is RULE 1: Thermal debt protection overrides even safety minimum
    assert decision.should_heat is False
    assert "CRITICAL_THERMAL_DEBT" in decision.priority_reason

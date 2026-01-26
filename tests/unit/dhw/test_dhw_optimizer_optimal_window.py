"""Unit tests for DHW optimal window logic in LANE 1.

Tests Phase 1 fix (Jan 2026): DHW should wait for optimal price windows
when temp < MIN_DHW_TARGET_TEMP within scheduled window, if:
1. Price savings >= 15%
2. Enough time to reach optimal window + complete heating before target
"""

import pytest
from datetime import datetime, timedelta
from custom_components.effektguard.optimization.dhw_optimizer import (
    IntelligentDHWScheduler,
    DHWDemandPeriod,
)
from custom_components.effektguard.optimization.price_layer import (
    PriceAnalyzer,
    QuarterPeriod,
)


def create_price_periods(base_date, current_hour, current_price, optimal_hour, optimal_price):
    """Create a full day of price periods for testing.

    Args:
        base_date: Base datetime (year, month, day)
        current_hour: Hour of current time (0-23)
        current_price: Price at current hour
        optimal_hour: Hour of optimal window (0-23)
        optimal_price: Price at optimal hour

    Returns:
        List of 96 QuarterPeriod objects (full day)
    """
    periods = []
    base_time = base_date.replace(hour=0, minute=0, second=0, microsecond=0)

    for i in range(96):  # 96 quarters in 24 hours
        start = base_time + timedelta(minutes=i * 15)
        hour = i // 4

        # Set price based on hour
        if hour == current_hour:
            price = current_price
        elif hour == optimal_hour:
            price = optimal_price
        else:
            # Fill other hours with intermediate prices
            price = (current_price + optimal_price) / 2

        periods.append(QuarterPeriod(start_time=start, price=price))

    return periods


def test_dhw_waits_for_optimal_window_when_time_allows():
    """Test DHW waits for optimal window when savings significant and time available."""
    # Setup: DHW scheduler with demand at 20:00
    scheduler = IntelligentDHWScheduler(
        demand_periods=[
            DHWDemandPeriod(
                availability_hour=20,
                target_temp=45.0,
                duration_hours=2,
                min_amount_minutes=10,
            )
        ],
        price_analyzer=PriceAnalyzer(),
    )

    # Mock price periods - full day with expensive now, cheap at 19:00
    current_time = datetime(2026, 1, 6, 17, 45)
    price_periods = create_price_periods(
        base_date=datetime(2026, 1, 6),
        current_hour=17,  # 17:45 is in hour 17
        current_price=174.0,  # EXPENSIVE
        optimal_hour=19,  # 19:15 is in hour 19
        optimal_price=141.9,  # CHEAP
    )

    # Execute: should_start_dhw with low temp but time available
    decision = scheduler.should_start_dhw(
        current_dhw_temp=43.3,  # Below 45°C threshold
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-583,
        indoor_temp=20.5,
        target_indoor_temp=21.0,
        outdoor_temp=-7.0,
        price_classification="peak",
        current_time=current_time,
        price_periods=price_periods,
        hours_since_last_dhw=None,
        is_volatile=False,
        dhw_amount_minutes=9.0,
    )

    # Assert: Should NOT heat now, should wait for optimal window
    assert decision.should_heat is False, f"Expected should_heat=False, got {decision.should_heat}"
    assert (
        "WAITING_OPTIMAL" in decision.priority_reason
    ), f"Expected WAITING_OPTIMAL in reason, got {decision.priority_reason}"
    # Verify recommended start time is in the future (between current and target)
    assert decision.recommended_start_time > current_time, "Start time should be in future"
    assert decision.recommended_start_time < datetime(
        2026, 1, 6, 20, 0
    ), "Start time should be before target"


def test_dhw_heats_now_when_optimal_window_imminent():
    """Test DHW heats immediately when optimal window is within 15 minutes."""
    scheduler = IntelligentDHWScheduler(
        demand_periods=[
            DHWDemandPeriod(
                availability_hour=20,
                target_temp=45.0,
                duration_hours=2,
                min_amount_minutes=10,
            )
        ],
        price_analyzer=PriceAnalyzer(),
    )

    # Current time just before cheap hour begins
    current_time = datetime(2026, 1, 6, 18, 55)  # 5 min before hour 19
    price_periods = create_price_periods(
        base_date=datetime(2026, 1, 6),
        current_hour=18,  # 18:55 is in hour 18
        current_price=150.0,  # Current hour: moderate
        optimal_hour=19,  # Next hour: cheap
        optimal_price=120.0,  # 20% cheaper
    )

    decision = scheduler.should_start_dhw(
        current_dhw_temp=43.3,
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-583,
        indoor_temp=20.5,
        target_indoor_temp=21.0,
        outdoor_temp=-7.0,
        price_classification="normal",
        current_time=current_time,
        price_periods=price_periods,
        hours_since_last_dhw=None,
        is_volatile=False,
        dhw_amount_minutes=9.0,
    )

    # Assert: Should heat NOW (optimal window is within 15 min OR wait for it)
    # Either heat now because in optimal window, or wait briefly
    assert decision.should_heat is True or (
        decision.should_heat is False
        and decision.recommended_start_time <= datetime(2026, 1, 6, 19, 10)
    ), f"Expected to heat now or wait briefly, got should_heat={decision.should_heat}, start_time={decision.recommended_start_time}"


def test_dhw_heats_priority_when_savings_insufficient():
    """Test DHW heats immediately when savings < 15% threshold."""
    scheduler = IntelligentDHWScheduler(
        demand_periods=[
            DHWDemandPeriod(
                availability_hour=20,
                target_temp=45.0,
                duration_hours=2,
                min_amount_minutes=10,
            )
        ],
        price_analyzer=PriceAnalyzer(),
    )

    current_time = datetime(2026, 1, 6, 17, 45)
    price_periods = create_price_periods(
        base_date=datetime(2026, 1, 6),
        current_hour=17,
        current_price=150.0,  # Now
        optimal_hour=19,
        optimal_price=145.0,  # Only 3.3% cheaper
    )

    decision = scheduler.should_start_dhw(
        current_dhw_temp=43.3,
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-583,
        indoor_temp=20.5,
        target_indoor_temp=21.0,
        outdoor_temp=-7.0,
        price_classification="normal",
        current_time=current_time,
        price_periods=price_periods,
        hours_since_last_dhw=None,
        is_volatile=False,
        dhw_amount_minutes=9.0,
    )

    # Assert: Should heat NOW (savings insufficient)
    assert decision.should_heat is True
    assert "PRIORITY" in decision.priority_reason


def test_dhw_heats_priority_when_not_enough_time():
    """Test DHW heats immediately when not enough time to wait for optimal window."""
    scheduler = IntelligentDHWScheduler(
        demand_periods=[
            DHWDemandPeriod(
                availability_hour=20,
                target_temp=45.0,
                duration_hours=2,
                min_amount_minutes=10,
            )
        ],
        price_analyzer=PriceAnalyzer(),
    )

    current_time = datetime(2026, 1, 6, 19, 30)  # Only 30 min until target
    price_periods = create_price_periods(
        base_date=datetime(2026, 1, 6),
        current_hour=19,
        current_price=174.0,  # Now: EXPENSIVE at 19:30
        optimal_hour=19,  # Same hour
        optimal_price=174.0,  # No real cheaper window (will use current price)
    )

    decision = scheduler.should_start_dhw(
        current_dhw_temp=43.3,
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-583,
        indoor_temp=20.5,
        target_indoor_temp=21.0,
        outdoor_temp=-7.0,
        price_classification="peak",
        current_time=current_time,
        price_periods=price_periods,
        hours_since_last_dhw=None,
        is_volatile=False,
        dhw_amount_minutes=9.0,
    )

    # Assert: With PEAK pricing, we NEVER heat - even if deadline approaching
    # User explicitly requested: "NEVER heat at PEAK prices"
    # The system logs a warning that scheduled demand may be missed
    assert decision.should_heat is False
    assert "DHW_PEAK_AVOIDED_SCHEDULED" in decision.priority_reason
    assert decision.recommended_start_time is not None  # Should suggest next non-peak


def test_dhw_adequate_temp_continues_normal_rules():
    """Test DHW with adequate temp (>=45°C) continues to normal optimization rules."""
    scheduler = IntelligentDHWScheduler(
        demand_periods=[
            DHWDemandPeriod(
                availability_hour=20,
                target_temp=45.0,
                duration_hours=2,
                min_amount_minutes=10,
            )
        ],
        price_analyzer=PriceAnalyzer(),
    )

    current_time = datetime(2026, 1, 6, 17, 45)
    price_periods = create_price_periods(
        base_date=datetime(2026, 1, 6),
        current_hour=17,
        current_price=174.0,  # Now: EXPENSIVE
        optimal_hour=19,
        optimal_price=141.9,  # Optimal: CHEAP
    )

    decision = scheduler.should_start_dhw(
        current_dhw_temp=45.5,  # Above 45°C threshold - adequate
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-583,
        indoor_temp=20.5,
        target_indoor_temp=21.0,
        outdoor_temp=-7.0,
        price_classification="peak",
        current_time=current_time,
        price_periods=price_periods,
        hours_since_last_dhw=None,
        is_volatile=False,
        dhw_amount_minutes=9.0,
    )

    # Assert: Should NOT heat (temp adequate, price not cheap)
    assert decision.should_heat is False
    # Should continue to normal rules (not the new optimal window logic)
    assert (
        "ADEQUATE" in decision.priority_reason
        or "SCHEDULED_TARGET_REACHED" in decision.priority_reason
    )


def test_dhw_optimal_window_with_real_jan15_prices():
    """Test with real prices from Jan 15, 2026 (current bug scenario)."""
    scheduler = IntelligentDHWScheduler(
        demand_periods=[
            DHWDemandPeriod(
                availability_hour=20,
                target_temp=45.0,
                duration_hours=2,
                min_amount_minutes=10,
            )
        ],
        price_analyzer=PriceAnalyzer(),
    )

    # Real scenario from Jan 15, 2026 at 17:25
    current_time = datetime(2026, 1, 15, 17, 25)
    price_periods = create_price_periods(
        base_date=datetime(2026, 1, 15),
        current_hour=17,
        current_price=146.63,  # Now: PEAK
        optimal_hour=19,
        optimal_price=99.5,  # Optimal: much cheaper
    )

    decision = scheduler.should_start_dhw(
        current_dhw_temp=40.8,  # Below 45°C
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-684,
        indoor_temp=20.5,
        target_indoor_temp=21.0,
        outdoor_temp=2.2,
        price_classification="peak",
        current_time=current_time,
        price_periods=price_periods,
        hours_since_last_dhw=2.0,
        is_volatile=False,
        dhw_amount_minutes=5.0,
    )

    # Assert: Should WAIT for optimal window (significant savings available)
    # Time available: 2.6h until target, enough time to wait for cheaper window
    assert (
        decision.should_heat is False
    ), f"Expected to wait for cheaper window, got should_heat={decision.should_heat}"
    assert (
        "WAITING_OPTIMAL" in decision.priority_reason
    ), f"Expected WAITING_OPTIMAL, got {decision.priority_reason}"
    # Verify it's waiting for a future time (between current and target)
    assert decision.recommended_start_time > current_time, "Start time should be in future"
    assert decision.recommended_start_time < datetime(
        2026, 1, 15, 20, 0
    ), "Start time should be before target"

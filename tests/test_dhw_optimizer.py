"""Test DHW optimizer scheduling logic.

Verifies that DHW scheduler:
1. Heats during cheapest hours before demand periods
2. Respects thermal debt and space heating priority
3. Handles urgent vs optimal pre-heating correctly
"""

import pytest
from datetime import datetime, timedelta
from custom_components.effektguard.optimization.dhw_optimizer import (
    IntelligentDHWScheduler,
    DHWDemandPeriod,
)


def test_dhw_smart_scheduling_cheapest_hours():
    """Test that DHW heats during cheap hours before demand period."""
    # Setup: Morning shower at 7:00 AM
    demand_period = DHWDemandPeriod(
        start_hour=7,
        target_temp=55.0,
        duration_hours=2,
    )
    scheduler = IntelligentDHWScheduler(demand_periods=[demand_period])

    # IMPORTANT: Set recent Legionella boost to prevent it from triggering
    scheduler.last_legionella_boost = datetime.now() - timedelta(days=2)

    # Current time: 5:00 AM (2 hours before demand)
    current_time = datetime(2025, 10, 15, 5, 0, 0)

    # Scenario 1: CHEAP price - should heat
    decision = scheduler.should_start_dhw(
        current_dhw_temp=45.0,  # Below target
        space_heating_demand_kw=2.0,  # Low demand
        thermal_debt_dm=-50,  # Safe level
        indoor_temp=21.0,  # Comfortable
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="cheap",
        current_time=current_time,
    )

    assert decision.should_heat is True
    assert "OPTIMAL_PREHEAT_DEMAND_2H_CHEAP" in decision.priority_reason
    assert decision.target_temp == 55.0

    # Scenario 2: EXPENSIVE price at 5:00 AM - should NOT heat yet
    # DHW at 50°C (5°C below 55°C target) - can wait for cheaper electricity
    decision_expensive = scheduler.should_start_dhw(
        current_dhw_temp=50.0,  # Close to target (55°C), can wait for cheaper prices
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-50,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="expensive",
        current_time=current_time,
    )

    assert decision_expensive.should_heat is False  # Wait for cheaper hour

    # Scenario 3: URGENT - 1 hour before demand - heat regardless of price
    urgent_time = datetime(2025, 10, 15, 6, 0, 0)  # 6:00 AM (1h before 7:00)
    decision_urgent = scheduler.should_start_dhw(
        current_dhw_temp=45.0,
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-50,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="expensive",  # Even with expensive price!
        current_time=urgent_time,
    )

    assert decision_urgent.should_heat is True
    assert "URGENT_DEMAND" in decision_urgent.priority_reason


def test_dhw_thermal_debt_blocks_heating():
    """Test that critical thermal debt blocks DHW heating."""
    scheduler = IntelligentDHWScheduler()

    decision = scheduler.should_start_dhw(
        current_dhw_temp=35.0,  # Low temp
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-250,  # CRITICAL - below -240 threshold
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="cheap",
        current_time=datetime.now(),
    )

    assert decision.should_heat is False
    assert decision.priority_reason == "CRITICAL_THERMAL_DEBT"


def test_dhw_space_heating_priority():
    """Test that space heating gets priority over DHW."""
    scheduler = IntelligentDHWScheduler()

    # House too cold + freezing outside = block DHW
    decision = scheduler.should_start_dhw(
        current_dhw_temp=40.0,
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-50,
        indoor_temp=20.0,  # 1°C below target
        target_indoor_temp=21.0,
        outdoor_temp=-5.0,  # Freezing
        price_classification="cheap",
        current_time=datetime.now(),
    )

    assert decision.should_heat is False
    assert decision.priority_reason == "SPACE_HEATING_EMERGENCY"


def test_dhw_safety_minimum():
    """Test that safety minimum temperature forces heating."""
    scheduler = IntelligentDHWScheduler()

    # DHW below safety threshold (Legionella risk)
    decision = scheduler.should_start_dhw(
        current_dhw_temp=30.0,  # BELOW 35°C safety minimum
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-50,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="expensive",  # Even with expensive price!
        current_time=datetime.now(),
    )

    assert decision.should_heat is True
    assert decision.priority_reason == "DHW_SAFETY_MINIMUM"
    assert decision.target_temp == 50.0  # Heat to normal target
    assert decision.max_runtime_minutes == 30  # Limited to prevent thermal debt


def test_dhw_legionella_detection():
    """Test BT7 monitoring detects NIBE's Legionella boost.

    NOTE: We no longer TRIGGER Legionella ourselves - we MONITOR NIBE's schedule.
    This test verifies automatic detection when BT7 sensor reaches 63°C+ and cools down.
    """
    scheduler = IntelligentDHWScheduler()

    # Initially, no Legionella boost detected
    assert scheduler.last_legionella_boost is None

    # Simulate BT7 temperature rising during NIBE's Legionella boost
    now = datetime.now()

    # Normal temperature readings (build up history)
    for i in range(10):
        scheduler.update_bt7_temperature(50.0 + i * 0.1, now - timedelta(minutes=150 - i * 15))

    # NIBE starts Legionella boost - temperature rises to 65°C
    scheduler.update_bt7_temperature(60.0, now - timedelta(minutes=60))
    scheduler.update_bt7_temperature(63.0, now - timedelta(minutes=45))
    scheduler.update_bt7_temperature(65.0, now - timedelta(minutes=30))  # Peak

    # Temperature starts cooling down (boost complete)
    scheduler.update_bt7_temperature(63.5, now - timedelta(minutes=15))
    scheduler.update_bt7_temperature(61.0, now)  # Cooled 4°C from peak

    # Detection should have triggered automatically
    assert scheduler.last_legionella_boost is not None

    # Recent boost should prevent unnecessary DHW heating
    decision = scheduler.should_start_dhw(
        current_dhw_temp=50.0,  # Normal temp after boost
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-50,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="normal",
        current_time=now,
    )

    # Should NOT heat - DHW is at adequate temperature after Legionella boost
    assert decision.should_heat is False
    assert decision.priority_reason == "DHW_ADEQUATE"


def test_dhw_bt7_history_tracking():
    """Test BT7 temperature history deque management."""
    scheduler = IntelligentDHWScheduler()

    now = datetime.now()

    # Add 50 temperature readings (exceeds maxlen=48)
    for i in range(50):
        scheduler.update_bt7_temperature(
            temp=45.0 + i * 0.1, timestamp=now + timedelta(minutes=i * 15)
        )

    # Deque should limit to 48 entries (12 hours of 15-min readings)
    assert len(scheduler.bt7_history) == 48

    # Should only contain latest 48 readings (tuples: timestamp, temp)
    temps = [temp for _, temp in scheduler.bt7_history]
    assert min(temps) >= 45.2  # First 2 readings dropped (0.0, 0.1)


def test_dhw_cheap_electricity_opportunity():
    """Test opportunistic heating during cheap electricity."""
    scheduler = IntelligentDHWScheduler()

    # IMPORTANT: Set recent Legionella boost to prevent it from triggering
    scheduler.last_legionella_boost = datetime.now() - timedelta(days=2)

    decision = scheduler.should_start_dhw(
        current_dhw_temp=48.0,  # Slightly below normal
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-50,  # Healthy
        indoor_temp=21.2,  # Comfortable (within 0.3°C)
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="cheap",
        current_time=datetime.now(),
    )

    assert decision.should_heat is True
    assert decision.priority_reason == "CHEAP_ELECTRICITY_OPPORTUNITY"
    assert decision.target_temp == 55.0  # Extra buffer during cheap period


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

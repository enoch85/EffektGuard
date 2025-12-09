"""Tests for DHW Next Boost Time functionality.

Verifies that all blocking decisions provide recommended_start_time
to prevent "Unknown" sensor state.

Phase 1 implementation tests for DHW_NEXT_BOOST_IMPLEMENTATION_PLAN.md
"""

import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from custom_components.effektguard.const import (
    COMPRESSOR_MIN_CYCLE_MINUTES,
    DHW_COOLING_RATE,
    DHW_SCHEDULING_WINDOW_MAX,
    DHW_SCHEDULING_WINDOW_MIN,
    DM_RECOVERY_MAX_HOURS,
    INDOOR_TEMP_RECOVERY_MAX_HOURS,
    SPACE_HEATING_DEMAND_DROP_HOURS,
)
from custom_components.effektguard.optimization.dhw_optimizer import (
    IntelligentDHWScheduler,
)
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from tests.conftest import create_mock_price_analyzer


# ==============================================================================
# TEST FIXTURES
# ==============================================================================


@pytest.fixture
def scheduler_with_climate():
    """Scheduler with climate zone detector (Stockholm)."""
    climate_detector = ClimateZoneDetector(latitude=59.33)  # Stockholm
    mock_analyzer = create_mock_price_analyzer()
    scheduler = IntelligentDHWScheduler(
        climate_detector=climate_detector, price_analyzer=mock_analyzer
    )
    scheduler.last_legionella_boost = datetime.now() - timedelta(days=2)
    return scheduler


@pytest.fixture
def scheduler_no_climate():
    """Scheduler without climate zone detector (fallback thresholds)."""
    mock_analyzer = create_mock_price_analyzer()
    scheduler = IntelligentDHWScheduler(price_analyzer=mock_analyzer)
    scheduler.last_legionella_boost = datetime.now() - timedelta(days=2)
    return scheduler


@pytest.fixture
def current_time():
    """Fixed current time for testing."""
    return datetime(2025, 11, 30, 14, 0, 0, tzinfo=ZoneInfo("Europe/Stockholm"))


# ==============================================================================
# BLOCKING CASES - MUST HAVE recommended_start_time
# ==============================================================================


def test_critical_thermal_debt_has_next_time(scheduler_with_climate, current_time):
    """Verify CRITICAL_THERMAL_DEBT provides recommended_start_time."""
    # Test with thermal debt below warning threshold
    decision = scheduler_with_climate.should_start_dhw(
        current_dhw_temp=45.0,
        space_heating_demand_kw=3.0,
        thermal_debt_dm=-500,  # Below warning threshold
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="normal",
        current_time=current_time,
        price_periods=[],
        hours_since_last_dhw=10.0,
    )

    assert decision.should_heat is False
    assert decision.priority_reason == "CRITICAL_THERMAL_DEBT"
    assert decision.recommended_start_time is not None
    assert decision.recommended_start_time > current_time
    # Should estimate recovery time (constrained by DHW_COOLING_RATE min, DM_RECOVERY_MAX_HOURS max)
    hours_until = (decision.recommended_start_time - current_time).total_seconds() / 3600
    assert DHW_COOLING_RATE <= hours_until <= DM_RECOVERY_MAX_HOURS


def test_space_heating_emergency_has_next_time(scheduler_with_climate, current_time):
    """Verify SPACE_HEATING_EMERGENCY provides recommended_start_time."""
    decision = scheduler_with_climate.should_start_dhw(
        current_dhw_temp=45.0,
        space_heating_demand_kw=3.0,
        thermal_debt_dm=-100,  # OK
        indoor_temp=20.0,  # Too cold (deficit > 0.5)
        target_indoor_temp=21.0,
        outdoor_temp=-5.0,  # Cold outside
        price_classification="normal",
        current_time=current_time,
        price_periods=[],
        hours_since_last_dhw=10.0,
    )

    assert decision.should_heat is False
    assert decision.priority_reason == "SPACE_HEATING_EMERGENCY"
    assert decision.recommended_start_time is not None
    assert decision.recommended_start_time > current_time
    # Should estimate indoor temp recovery (min from COMPRESSOR_MIN_CYCLE_MINUTES, max from const)
    hours_until = (decision.recommended_start_time - current_time).total_seconds() / 3600
    min_hours = COMPRESSOR_MIN_CYCLE_MINUTES / 60.0
    assert min_hours <= hours_until <= INDOOR_TEMP_RECOVERY_MAX_HOURS


def test_high_space_heating_demand_has_next_time(scheduler_with_climate, current_time):
    """Verify HIGH_SPACE_HEATING_DEMAND provides recommended_start_time."""
    decision = scheduler_with_climate.should_start_dhw(
        current_dhw_temp=45.0,
        space_heating_demand_kw=7.0,  # High demand
        thermal_debt_dm=-100,  # OK but negative
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=0.0,
        price_classification="normal",
        current_time=current_time,
        price_periods=[],
        hours_since_last_dhw=10.0,
    )

    assert decision.should_heat is False
    assert decision.priority_reason == "HIGH_SPACE_HEATING_DEMAND"
    assert decision.recommended_start_time is not None
    assert decision.recommended_start_time > current_time
    # Should estimate time until demand drops (from const.py)
    hours_until = (decision.recommended_start_time - current_time).total_seconds() / 3600
    assert hours_until == SPACE_HEATING_DEMAND_DROP_HOURS


def test_dhw_adequate_has_next_time(scheduler_with_climate, current_time):
    """Verify DHW_ADEQUATE provides recommended_start_time."""
    decision = scheduler_with_climate.should_start_dhw(
        current_dhw_temp=55.0,  # Very warm
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-50,  # Healthy
        indoor_temp=21.5,
        target_indoor_temp=21.0,
        outdoor_temp=10.0,
        price_classification="expensive",
        current_time=current_time,
        price_periods=[],
        hours_since_last_dhw=5.0,
    )

    assert decision.should_heat is False
    assert "DHW_ADEQUATE" in decision.priority_reason  # Can be DHW_ADEQUATE or DHW_ADEQUATE_WAITING_CHEAP_*
    assert decision.recommended_start_time is not None
    assert decision.recommended_start_time > current_time
    # Should estimate when DHW will cool (constrained by DHW_SCHEDULING_WINDOW_MIN/MAX)
    hours_until = (decision.recommended_start_time - current_time).total_seconds() / 3600
    assert DHW_SCHEDULING_WINDOW_MIN <= hours_until <= DHW_SCHEDULING_WINDOW_MAX


# ==============================================================================
# HEATING CASES - SHOULD HAVE recommended_start_time = current_time
# ==============================================================================


def test_dhw_safety_minimum_has_current_time(scheduler_with_climate, current_time):
    """Verify DHW_SAFETY_MINIMUM sets recommended_start_time=current_time."""
    decision = scheduler_with_climate.should_start_dhw(
        current_dhw_temp=25.0,  # Below safety minimum
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-50,  # Healthy
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="normal",
        current_time=current_time,
        price_periods=[],
        hours_since_last_dhw=10.0,
    )

    assert decision.should_heat is True
    assert decision.priority_reason == "DHW_SAFETY_MINIMUM"
    assert decision.recommended_start_time is not None
    assert decision.recommended_start_time == current_time


def test_dhw_hygiene_boost_has_current_time(scheduler_with_climate, current_time):
    """Verify DHW_HYGIENE_BOOST sets recommended_start_time=current_time."""
    # Force Legionella boost needed
    scheduler_with_climate.last_legionella_boost = None

    decision = scheduler_with_climate.should_start_dhw(
        current_dhw_temp=45.0,
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-50,  # Healthy
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="cheap",
        current_time=current_time,
        price_periods=[],
        hours_since_last_dhw=10.0,
    )

    assert decision.should_heat is True
    assert decision.priority_reason == "DHW_HYGIENE_BOOST"
    assert decision.recommended_start_time is not None
    assert decision.recommended_start_time == current_time


def test_dhw_complete_emergency_heating_has_current_time(scheduler_with_climate, current_time):
    """Verify DHW_COMPLETE_EMERGENCY_HEATING sets recommended_start_time=current_time."""
    decision = scheduler_with_climate.should_start_dhw(
        current_dhw_temp=35.0,  # In 30-45°C range
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-50,  # Healthy
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="cheap",
        current_time=current_time,
        price_periods=[],
        hours_since_last_dhw=10.0,
    )

    assert decision.should_heat is True
    assert decision.priority_reason == "DHW_COMPLETE_EMERGENCY_HEATING"
    assert decision.recommended_start_time is not None
    assert decision.recommended_start_time == current_time


def test_dhw_max_wait_exceeded_has_current_time(scheduler_with_climate, current_time):
    """Verify DHW_MAX_WAIT_EXCEEDED sets recommended_start_time=current_time."""
    decision = scheduler_with_climate.should_start_dhw(
        current_dhw_temp=45.0,
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-50,  # Healthy
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="cheap",
        current_time=current_time,
        price_periods=[],
        hours_since_last_dhw=40.0,  # Exceeded max wait
    )

    assert decision.should_heat is True
    assert "DHW_MAX_WAIT_EXCEEDED" in decision.priority_reason
    assert decision.recommended_start_time is not None
    assert decision.recommended_start_time == current_time


def test_dhw_comfort_low_cheap_has_current_time(scheduler_with_climate, current_time):
    """Verify CHEAP_ELECTRICITY_OPPORTUNITY sets recommended_start_time=current_time (fallback when no price data)."""
    decision = scheduler_with_climate.should_start_dhw(
        current_dhw_temp=47.0,  # Below minimum target (45°C) but above emergency range
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-50,  # Healthy
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="cheap",
        current_time=current_time,
        price_periods=None,  # No price data triggers fallback
        hours_since_last_dhw=10.0,
    )

    assert decision.should_heat is True
    assert decision.priority_reason == "CHEAP_ELECTRICITY_OPPORTUNITY"
    assert decision.recommended_start_time is not None
    assert decision.recommended_start_time == current_time


def test_cheap_no_window_data_has_current_time(scheduler_with_climate, current_time):
    """Verify CHEAP_ELECTRICITY_OPPORTUNITY sets recommended_start_time=current_time."""
    decision = scheduler_with_climate.should_start_dhw(
        current_dhw_temp=48.0,  # Needs heating
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-50,  # Healthy
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="cheap",
        current_time=current_time,
        price_periods=None,  # No price data
        hours_since_last_dhw=10.0,
    )

    assert decision.should_heat is True
    assert decision.priority_reason == "CHEAP_ELECTRICITY_OPPORTUNITY"
    assert decision.recommended_start_time is not None
    assert decision.recommended_start_time == current_time


def test_cheap_electricity_opportunity_has_current_time(scheduler_with_climate, current_time):
    """Verify CHEAP_ELECTRICITY_OPPORTUNITY sets recommended_start_time=current_time."""
    decision = scheduler_with_climate.should_start_dhw(
        current_dhw_temp=48.0,  # Needs heating
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-50,  # Healthy
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="cheap",
        current_time=current_time,
        price_periods=None,  # No price data
        hours_since_last_dhw=10.0,
    )

    assert decision.should_heat is True
    # Could be CHEAP_NO_WINDOW_DATA or CHEAP_ELECTRICITY_OPPORTUNITY
    assert "CHEAP" in decision.priority_reason
    assert decision.recommended_start_time is not None
    assert decision.recommended_start_time == current_time


# ==============================================================================
# DM RECOVERY TIME ESTIMATION TESTS
# ==============================================================================


def test_dm_recovery_estimation_mild_weather(scheduler_with_climate, current_time):
    """Test DM recovery estimation in mild weather (fast recovery)."""
    hours = scheduler_with_climate._estimate_dm_recovery_time(
        current_dm=-435.0, target_dm=-350.0, outdoor_temp=10.0  # Mild weather
    )

    # Deficit = 85 DM, rate = 40 DM/h (mild weather)
    # Expected: 85/40 = 2.125 hours
    assert 2.0 <= hours <= 2.5


def test_dm_recovery_estimation_cold_weather(scheduler_with_climate, current_time):
    """Test DM recovery estimation in cold weather (slow recovery)."""
    hours = scheduler_with_climate._estimate_dm_recovery_time(
        current_dm=-435.0, target_dm=-350.0, outdoor_temp=-5.0  # Very cold
    )

    # Deficit = 85 DM, rate = 20 DM/h (very cold)
    # Expected: 85/20 = 4.25 hours
    assert 4.0 <= hours <= 5.0


def test_dm_recovery_minimum_constraint(scheduler_with_climate, current_time):
    """Test DM recovery time minimum constraint (reuses DHW_COOLING_RATE)."""
    hours = scheduler_with_climate._estimate_dm_recovery_time(
        current_dm=-355.0, target_dm=-350.0, outdoor_temp=10.0  # Small deficit
    )

    # Deficit = 5 DM, would calculate to 0.125h, but constrained to DHW_COOLING_RATE minimum
    assert hours == DHW_COOLING_RATE


def test_dm_recovery_minimum_constraint(scheduler_with_climate, current_time):
    """Test DM recovery time minimum constraint (reuses DHW_COOLING_RATE: 0.5h)."""
    hours = scheduler_with_climate._estimate_dm_recovery_time(
        current_dm=-355.0, target_dm=-350.0, outdoor_temp=10.0  # Small deficit
    )

    # Deficit = 5 DM, would calculate to 0.125h, constrained to DHW_COOLING_RATE
    assert hours == DHW_COOLING_RATE


# ==============================================================================
# FALLBACK THRESHOLD TESTS (No Climate Detector)
# ==============================================================================


def test_fallback_thresholds_without_climate(scheduler_no_climate, current_time):
    """Verify fallback thresholds work when climate detector unavailable."""
    decision = scheduler_no_climate.should_start_dhw(
        current_dhw_temp=45.0,
        space_heating_demand_kw=3.0,
        thermal_debt_dm=-700,  # Below fallback threshold
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=5.0,
        price_classification="normal",
        current_time=current_time,
        price_periods=[],
        hours_since_last_dhw=10.0,
    )

    assert decision.should_heat is False
    assert decision.priority_reason == "CRITICAL_THERMAL_DEBT"
    assert decision.recommended_start_time is not None
    assert decision.recommended_start_time > current_time


# ==============================================================================
# EDGE CASES
# ==============================================================================


def test_all_blocking_reasons_have_recommended_time(scheduler_with_climate, current_time):
    """Verify all blocking reasons provide recommended_start_time.

    This is the critical regression test - ensures we never return
    blocking decision without next opportunity timestamp.
    """
    blocking_scenarios = [
        # (current_dhw_temp, space_heating_demand_kw, thermal_debt_dm,
        #  indoor_temp, outdoor_temp, price_classification, hours_since_last_dhw)
        # CRITICAL_THERMAL_DEBT
        (45.0, 3.0, -500, 21.0, 5.0, "normal", 10.0),
        # SPACE_HEATING_EMERGENCY
        (45.0, 3.0, -100, 20.0, -5.0, "normal", 10.0),
        # HIGH_SPACE_HEATING_DEMAND
        (45.0, 7.0, -100, 21.0, 0.0, "normal", 10.0),
        # DHW_ADEQUATE
        (55.0, 2.0, -50, 21.5, 10.0, "expensive", 5.0),
    ]

    for scenario in blocking_scenarios:
        (
            dhw_temp,
            heating_demand,
            thermal_debt,
            indoor_temp,
            outdoor_temp,
            price_class,
            hours_since,
        ) = scenario

        decision = scheduler_with_climate.should_start_dhw(
            current_dhw_temp=dhw_temp,
            space_heating_demand_kw=heating_demand,
            thermal_debt_dm=thermal_debt,
            indoor_temp=indoor_temp,
            target_indoor_temp=21.0,
            outdoor_temp=outdoor_temp,
            price_classification=price_class,
            current_time=current_time,
            price_periods=[],
            hours_since_last_dhw=hours_since,
        )

        # ALL blocking decisions MUST have recommended_start_time
        if not decision.should_heat:
            assert (
                decision.recommended_start_time is not None
            ), f"Blocking reason '{decision.priority_reason}' missing recommended_start_time!"
            assert decision.recommended_start_time >= current_time


# ==============================================================================
# PHASE 3: DATACLASS VALIDATION TESTS
# ==============================================================================


def test_dataclass_validation_blocks_missing_recommended_time():
    """Test that DHWScheduleDecision validation catches missing recommended_start_time.

    Phase 3: Prevents regression - ensures new blocking logic can't be added
    without providing next opportunity timestamp.
    """
    from custom_components.effektguard.optimization.dhw_optimizer import DHWScheduleDecision

    # Valid: Blocking decision WITH recommended_start_time
    valid_decision = DHWScheduleDecision(
        should_heat=False,
        priority_reason="TEST_BLOCKING",
        target_temp=0.0,
        max_runtime_minutes=0,
        abort_conditions=[],
        recommended_start_time=datetime.now() + timedelta(hours=1),
    )
    assert valid_decision.recommended_start_time is not None

    # Invalid: Blocking decision WITHOUT recommended_start_time should raise ValueError
    with pytest.raises(
        ValueError, match="blocks heating but doesn't provide recommended_start_time"
    ):
        DHWScheduleDecision(
            should_heat=False,
            priority_reason="TEST_BLOCKING_INVALID",
            target_temp=0.0,
            max_runtime_minutes=0,
            abort_conditions=[],
            recommended_start_time=None,  # Invalid for blocking decision!
        )


def test_dataclass_validation_allows_heating_without_timestamp():
    """Test that heating decisions can optionally omit recommended_start_time.

    Heating decisions (should_heat=True) can have None for recommended_start_time
    since the sensor will show "pending" or the actual heating start time.
    """
    from custom_components.effektguard.optimization.dhw_optimizer import DHWScheduleDecision

    # Valid: Heating decision without recommended_start_time
    heating_decision = DHWScheduleDecision(
        should_heat=True,
        priority_reason="TEST_HEATING",
        target_temp=50.0,
        max_runtime_minutes=45,
        abort_conditions=["thermal_debt < -500"],
        recommended_start_time=None,  # OK for heating decisions
    )
    assert heating_decision.should_heat is True

    # Also valid: Heating decision with recommended_start_time
    heating_with_time = DHWScheduleDecision(
        should_heat=True,
        priority_reason="TEST_HEATING_SCHEDULED",
        target_temp=50.0,
        max_runtime_minutes=45,
        abort_conditions=[],
        recommended_start_time=datetime.now(),
    )
    assert heating_with_time.recommended_start_time is not None

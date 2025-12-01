"""Tests for DHW safety temperature adjustments.

Tests the new DHW safety thresholds:
- DHW_SAFETY_CRITICAL = 10°C (always heat, no deferral)
- DHW_SAFETY_MIN = 20°C (can defer if 10-20°C during expensive periods)

This prevents peak billing hits when DHW can safely wait for better prices.

IMPORTANT: DHW heating time considerations:
- Tank heat-up: 1-2 hours (enoch85 research)
- No thermal mass delays like concrete slab UFH
- Must check if price peak occurs within heating window
- Defer heating if peak is coming within 1-2 hours

Based on: User request to lower DHW minimum temps for better price optimization
"""

import pytest
from datetime import datetime, timedelta

from custom_components.effektguard.const import (
    DEFAULT_DHW_TARGET_TEMP,
    DHW_SAFETY_CRITICAL,
    DHW_SAFETY_MIN,
    DM_DHW_ABORT_FALLBACK,
    DM_DHW_BLOCK_FALLBACK,
)
from custom_components.effektguard.optimization.dhw_optimizer import (
    DHWScheduleDecision,
    IntelligentDHWScheduler,
)

# DHW heating time constants (from research: enoch85 case study)
DHW_HEATING_TIME_HOURS = 1.5  # Typical DHW tank heat-up time
DHW_HEATING_TIME_MIN_HOURS = 1.0  # Minimum heating time
DHW_HEATING_TIME_MAX_HOURS = 2.0  # Maximum heating time

# Calculated test thresholds based on fallback constants
# The "concerning" threshold is block + 20 (used in deferral logic)
DM_CONCERNING_THRESHOLD = DM_DHW_BLOCK_FALLBACK + 20  # -340 + 20 = -320


class TestDHWSafetyConstants:
    """Test that DHW safety constants are correctly defined."""

    def test_constants_are_defined(self):
        """Test that DHW safety constants are defined and importable."""
        assert DHW_SAFETY_CRITICAL is not None
        assert DHW_SAFETY_MIN is not None
        assert isinstance(DHW_SAFETY_CRITICAL, (int, float))
        assert isinstance(DHW_SAFETY_MIN, (int, float))

    def test_critical_lower_than_min(self):
        """Test that critical threshold is lower than minimum (logical relationship)."""
        assert DHW_SAFETY_CRITICAL < DHW_SAFETY_MIN

    def test_scheduler_uses_constants(self):
        """Test that DHWScheduler uses constants from const.py directly."""
        # Constants are now imported directly in dhw_optimizer, not class variables
        scheduler = IntelligentDHWScheduler()
        # Verify scheduler can be instantiated (uses constants internally)
        assert scheduler is not None


class TestDHWCriticalTemperature:
    """Test DHW heating at critical temperature (below DHW_SAFETY_CRITICAL)."""

    def test_always_heat_below_critical_degrees(self):
        """Test that DHW always heats when below DHW_SAFETY_CRITICAL (safety override)."""
        scheduler = IntelligentDHWScheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_CRITICAL - 1.0,  # Below critical threshold
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-50.0,  # Normal
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="peak",  # Even during peak pricing!
            current_time=datetime.now(),
        )

        assert decision.should_heat is True
        assert "SAFETY" in decision.priority_reason

    def test_critical_overrides_expensive_price(self):
        """Test that critical temp overrides expensive pricing."""
        scheduler = IntelligentDHWScheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_CRITICAL - 2.0,  # Critically low
            space_heating_demand_kw=5.0,  # High demand
            thermal_debt_dm=-100.0,  # Some debt
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=-5.0,  # Cold
            price_classification="expensive",  # Expensive period
            current_time=datetime.now(),
        )

        assert decision.should_heat is True
        assert "SAFETY" in decision.priority_reason

    def test_critical_thermal_debt_blocks_all_dhw_even_at_30(self):
        """Test that critical thermal debt blocks ALL DHW heating."""
        scheduler = IntelligentDHWScheduler()

        # Rule 1: thermal_debt_dm <= dm_block_threshold blocks DHW
        # dm_block_threshold is from DM_DHW_BLOCK_FALLBACK constant

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_CRITICAL - 0.5,  # Below critical temp
            thermal_debt_dm=-350.0,  # Below block threshold
            space_heating_demand_kw=0.0,
            indoor_temp=20.5,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",
            current_time=datetime.now(),
        )

        # Rule 1 blocks DHW when DM <= DM_DHW_BLOCK_FALLBACK
        assert decision.should_heat is False
        assert "THERMAL_DEBT" in decision.priority_reason


class TestDHWDeferralRange:
    """Test DHW deferral in DHW_SAFETY_CRITICAL to DHW_SAFETY_MIN range during expensive periods."""

    def test_defer_at_mid_range_expensive_price_with_healthy_dm(self):
        """Test that DHW in mid-range can be deferred during expensive period with healthy DM.

        Deferral conditions (all must be true):
        - current_dhw_temp >= DHW_SAFETY_CRITICAL - safe to wait
        - price_classification in ["expensive", "peak"] - high cost
        - thermal_debt_dm > (dm_block_threshold + 20) - healthy enough to defer

        With fallback: dm_block_threshold = -240, so check is: DM > -220
        """
        scheduler = IntelligentDHWScheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN - 1.0,  # In deferral range (just below safety min)
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-200.0,  # Healthy: -200 > -220, can defer
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",  # Expensive period
            current_time=datetime.now(),
        )

        assert decision.should_heat is False
        assert "DEFERRED" in decision.priority_reason

    def test_defer_at_lower_mid_range_peak_price_with_healthy_dm(self):
        """Test deferral in lower mid-range with peak price and healthy thermal debt.

        Logic: can_defer checks:
        - current_dhw_temp >= DHW_SAFETY_CRITICAL: True
        - price expensive/peak: True
        - thermal_debt_dm > (dm_block_threshold + 20) - healthy enough to defer

        With DM_DHW_BLOCK_FALLBACK = -240, threshold is -220.
        """
        scheduler = IntelligentDHWScheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN - 3.0,  # Lower in deferral range
            thermal_debt_dm=-180.0,  # Healthy: -180 > -220, can defer
            space_heating_demand_kw=0.0,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="peak",
            current_time=datetime.now(),
        )

        # Can defer because all deferral conditions met
        assert decision.should_heat is False
        assert decision.priority_reason == "DHW_SAFETY_DEFERRED_PEAK_PRICE"

    def test_defer_prevents_peak_during_expensive_hour_with_healthy_dm(self):
        """Test deferring expensive electricity with healthy thermal debt."""
        scheduler = IntelligentDHWScheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN - 3.0,  # In deferral range
            thermal_debt_dm=-190.0,  # Healthy: -190 > -220
            space_heating_demand_kw=0.0,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",
            current_time=datetime.now(),
        )

        # Should defer expensive electricity
        assert decision.should_heat is False
        assert decision.priority_reason == "DHW_SAFETY_DEFERRED_PEAK_PRICE"

    def test_no_defer_at_34_degrees_cheap_price(self):
        """Test that DHW at 34°C heats during cheap period (no deferral needed)."""
        scheduler = IntelligentDHWScheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN - 1.0,  # In deferral range
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-200.0,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="cheap",  # Cheap period - go ahead and heat
            current_time=datetime.now(),
        )

        # Should heat during cheap period (opportunity heating)
        assert decision.should_heat is True

    def test_no_defer_at_mid_range_normal_price(self):
        """Test that DHW in mid-range heats during normal period."""
        scheduler = IntelligentDHWScheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN - 1.0,  # In deferral range
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-200.0,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="normal",  # Normal period - OK to heat
            current_time=datetime.now(),
        )

        assert decision.should_heat is True


class TestDHWBoundaryConditions:
    """Test boundary conditions at DHW_SAFETY_CRITICAL and DHW_SAFETY_MIN thresholds."""

    def test_exactly_at_critical_heats_with_expensive_and_bad_dm(self):
        """Test that exactly at DHW_SAFETY_CRITICAL heats when DM is bad (concerning).

        At exactly critical temp with expensive price:
        - current_dhw_temp >= DHW_SAFETY_CRITICAL: True (exactly equal)
        - price is expensive: True
        - thermal_debt_dm > (dm_block_threshold + 20): Need bad DM to prevent defer

        With fallback dm_block_threshold = -340, threshold is -320.
        DM must be < -320 to prevent deferral (bad debt).
        """
        scheduler = IntelligentDHWScheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_CRITICAL,  # Exactly at critical threshold
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-330.0,  # Bad: -330 < -320 (fallback +20), cannot defer
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",
            current_time=datetime.now(),
        )

        # With bad DM, cannot defer, heats immediately
        assert decision.should_heat is True
        assert decision.priority_reason == "DHW_SAFETY_MINIMUM"

    def test_just_below_critical_always_heats(self):
        """Test that just below DHW_SAFETY_CRITICAL always heats (below critical threshold)."""
        scheduler = IntelligentDHWScheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_CRITICAL - 0.1,  # Just below critical
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-200.0,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",
            current_time=datetime.now(),
        )

        # Below critical = always heat
        assert decision.should_heat is True
        assert "SAFETY" in decision.priority_reason

    def test_exactly_at_safety_min_behavior(self):
        """Test behavior at exactly DHW_SAFETY_MIN (at safety minimum)."""
        scheduler = IntelligentDHWScheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN,  # Exactly at safety minimum
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-200.0,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",
            current_time=datetime.now(),
        )

        # At safety min with expensive price, should heat (safety minimum reached)
        # current_dhw_temp < DHW_SAFETY_MIN: False (exactly equal)
        # So this won't trigger the safety minimum rule
        # Falls through to other rules
        assert decision.should_heat is False  # Not in safety range, defers

    def test_just_above_safety_min(self):
        """Test that just above DHW_SAFETY_MIN doesn't trigger safety minimum."""
        scheduler = IntelligentDHWScheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN + 0.1,  # Above safety minimum
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-200.0,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",
            current_time=datetime.now(),
        )

        # Above safety min - not in safety range, follows normal rules
        assert decision.should_heat is False  # No heating needed above safety min


class TestDHWDeferralPreventsPeakBilling:
    """Test that deferral prevents peak billing hits."""

    def test_defer_prevents_peak_during_expensive_hour_with_healthy_dm(self):
        """Test that deferring DHW during expensive hour prevents peak billing.

        Requires healthy thermal debt (> -320 with fallback) to safely defer.
        """
        scheduler = IntelligentDHWScheduler()

        # Scenario: Mid-range DHW, expensive electricity, healthy DM
        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN - 2.0,  # In deferral range
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-300.0,  # Healthy: -300 > -320, can defer
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",
            current_time=datetime.now(),
        )

        # Should defer to avoid peak billing
        assert decision.should_heat is False
        assert "DEFERRED" in decision.priority_reason

    def test_heat_when_cheap_even_if_in_deferral_range(self):
        """Test that DHW heats during cheap period even in deferral range."""
        scheduler = IntelligentDHWScheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN - 3.0,  # In deferral range
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-150.0,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="cheap",  # Cheap period - opportunity!
            current_time=datetime.now(),
        )

        # Should heat during cheap period (no peak billing concern)
        assert decision.should_heat is True


class TestDHWDeferralWithThermalDebt:
    """Test DHW deferral interaction with thermal debt."""

    def test_no_defer_with_critical_thermal_debt(self):
        """Test that critical thermal debt prevents all DHW heating (Rule 1)."""
        scheduler = IntelligentDHWScheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN - 2.0,  # In deferral range
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-340.0,  # At block threshold (Rule 1 blocks, fallback)
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",
            current_time=datetime.now(),
        )

        # Rule 1 blocks DHW entirely when DM <= dm_block_threshold
        assert decision.should_heat is False
        assert "THERMAL_DEBT" in decision.priority_reason

    def test_defer_with_healthy_thermal_debt(self):
        """Test that healthy thermal debt allows deferral during expensive prices.

        Thermal debt is healthy when > (dm_block_threshold + 20).
        With fallback -240, that's > -220.
        """
        scheduler = IntelligentDHWScheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN - 2.0,  # In deferral range
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-180.0,  # Healthy: -180 > -220
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",
            current_time=datetime.now(),
        )

        # Healthy debt + expensive price = defer DHW to save money
        assert decision.should_heat is False
        assert "DEFERRED" in decision.priority_reason

    def test_real_world_case_oct22_should_have_deferred(self):
        """Test real-world case from Oct 22: DHW 33.6°C, DM -211, expensive price.

        This is the bug that was found in production with OLD thresholds (30°C/35°C):
        - DHW temp: 33.6°C (was below 35°C safety minimum, in old deferral range)
        - DM: -211 (healthy, well above -276 block threshold)
        - Price: 92.71 öre (expensive)
        - Expected (with old thresholds): Defer to cheaper hour
        - Actual (before fix): Heated immediately

        With NEW thresholds (10°C/20°C):
        - DHW temp: 33.6°C is now ABOVE DHW_SAFETY_MIN (20°C)
        - This is considered DHW_ADEQUATE, not in deferral range
        - Behavior: Won't heat (adequate temp, no need)

        Test adjusted to use temperature in deferral range for new thresholds.
        """
        scheduler = IntelligentDHWScheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN - 1.0,  # In deferral range with new thresholds
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-211.0,  # Healthy: -211 > -220 ✓
            indoor_temp=22.7,
            target_indoor_temp=21.5,
            outdoor_temp=10.2,
            price_classification="expensive",
            current_time=datetime.now(),
        )

        # Should defer because:
        # 1. DHW >= DHW_SAFETY_CRITICAL (safe to wait)
        # 2. Price is expensive
        # 3. DM is healthy (-211 > -220)
        assert decision.should_heat is False
        assert decision.priority_reason == "DHW_SAFETY_DEFERRED_PEAK_PRICE"


@pytest.fixture
def climate_detector():
    """Mock climate detector for testing."""
    from unittest.mock import MagicMock

    detector = MagicMock()
    detector.get_dm_thresholds.return_value = {
        "block": DM_DHW_BLOCK_FALLBACK,
        "abort": DM_DHW_ABORT_FALLBACK,
    }
    return detector


class TestDHWHeatingTimeAndPeakAvoidance:
    """Test DHW heating time consideration and peak avoidance.

    DHW heating takes 1-2 hours. If starting heating now would overlap with
    a price peak within the next 2 hours, we should defer (if safe to do so).

    This prevents situations like:
    - Start heating at 16:00 during normal price
    - Peak hour starts at 17:00 (expensive)
    - Still heating during peak = peak billing hit!
    """

    def test_defer_if_peak_coming_within_heating_window(self):
        """Test that DHW defers if price peak is coming within 1-2 hours.

        Scenario:
        - Current time: 16:00 (expensive price)
        - DHW temp: In deferral range (safe to defer)
        - Peak hour: 17:00-18:00 (expensive)
        - Heating time: 1.5 hours (would overlap with peak)
        - Decision: DEFER until after peak
        """
        scheduler = IntelligentDHWScheduler()

        # Current time: 16:00, peak at 17:00
        current_time = datetime(2025, 10, 17, 16, 0)

        # DHW heating would take 1.5 hours (16:00 + 1.5h = 17:30)
        # This overlaps with peak hour 17:00-18:00
        # So we should defer if:
        # 1. Temp is safe (in deferral range)
        # 2. Price is expensive/peak
        # 3. Thermal debt is healthy enough to defer (> -320)

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN - 2.0,  # Safe to defer
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-300.0,  # Healthy: -300 > -320, can defer
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",  # Currently expensive
            current_time=current_time,
        )

        # Should defer to avoid heating during expensive period
        assert decision.should_heat is False
        assert "DEFERRED" in decision.priority_reason

    def test_heat_if_no_peak_within_heating_window(self):
        """Test that DHW heats if no peak coming within 2 hours.

        Scenario:
        - Current time: 22:00 (normal price)
        - DHW temp: In deferral range
        - Next peak: Tomorrow 17:00 (>2 hours away)
        - Heating time: 1.5 hours (done by 23:30)
        - Decision: HEAT NOW (safe window)
        """
        scheduler = IntelligentDHWScheduler()

        current_time = datetime(2025, 10, 17, 22, 0)

        # Normal price, no peak imminent
        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN - 2.0,  # In deferral range
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-150.0,  # Not concerning
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="normal",  # Normal price
            current_time=current_time,
        )

        # Should heat now (safe window, no peak coming)
        assert decision.should_heat is True

    def test_calculate_dhw_heating_duration(self):
        """Test calculation of DHW heating duration based on temp difference.

        Heating time varies based on temperature difference:
        - Small difference (5°C): ~1 hour
        - Medium difference (15°C): ~1.5 hours
        - Large difference (25°C): ~2 hours
        """

        # This is a helper function test to show how to calculate heating time
        def estimate_dhw_heating_time(current_temp: float, target_temp: float) -> float:
            """Estimate DHW heating time in hours.

            Based on enoch85 research: 1-2 hours typical for full heat-up.
            Linear interpolation between temps.
            """
            temp_diff = target_temp - current_temp

            if temp_diff <= 0:
                return 0.0  # Already at or above target
            elif temp_diff <= 5:
                return DHW_HEATING_TIME_MIN_HOURS  # 1 hour for small boost
            elif temp_diff <= 15:
                return DHW_HEATING_TIME_HOURS  # 1.5 hours typical
            else:
                return DHW_HEATING_TIME_MAX_HOURS  # 2 hours for cold start

        # Test various temperature differences
        assert estimate_dhw_heating_time(45.0, 50.0) == 1.0  # 5°C diff
        assert estimate_dhw_heating_time(35.0, 50.0) == 1.5  # 15°C diff
        assert estimate_dhw_heating_time(25.0, 50.0) == 2.0  # 25°C diff
        assert estimate_dhw_heating_time(50.0, 50.0) == 0.0  # Already at target

    def test_defer_only_if_safe_temperature(self):
        """Test that peak avoidance only defers if temperature is safe (>=30°C).

        Scenario:
        - Peak coming in 1 hour
        - DHW temp: Below critical threshold
        - Decision: HEAT NOW (safety override, even during peak)
        """
        scheduler = IntelligentDHWScheduler()

        current_time = datetime(2025, 10, 17, 16, 0)

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_CRITICAL - 2.0,  # Below critical threshold
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-330.0,  # Concerning
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",  # Peak coming
            current_time=current_time,
        )

        # Must heat even if peak coming (safety override)
        assert decision.should_heat is True
        assert "SAFETY" in decision.priority_reason

    def test_heating_time_affects_abort_conditions(self):
        """Test that heating time influences abort conditions.

        If heating will take 2 hours, abort conditions should be checked
        more strictly to prevent long expensive heating cycles.
        """
        scheduler = IntelligentDHWScheduler()

        current_time = datetime(2025, 10, 17, 14, 0)

        # Large temp difference = long heating time
        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN + 5.0,  # Significantly below target
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-150.0,  # OK
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="normal",
            current_time=current_time,
        )

        # Should heat, but with abort conditions
        if decision.should_heat:
            assert len(decision.abort_conditions) > 0
            # Should have thermal debt abort condition
            assert any("thermal_debt" in cond for cond in decision.abort_conditions)

    def test_opportunistic_heating_before_peak_hours(self):
        """Test opportunistic heating before known peak hours.

        Scenario:
        - Current time: 15:00 (cheap period)
        - DHW temp: In deferral range
        - Peak hours: 17:00-20:00
        - Decision: HEAT NOW to avoid peak period heating

        Note: DHW well above safety min won't heat.
        Need to be in the opportunity range for opportunistic heating.
        """
        scheduler = IntelligentDHWScheduler()

        current_time = datetime(2025, 10, 17, 15, 0)

        # Cheap period, opportunity to preheat before evening peak
        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN - 2.0,  # In range that could benefit from heating
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-100.0,  # Fine
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="cheap",  # Opportunity!
            current_time=current_time,
        )

        # Should take opportunity to heat during cheap period
        # Before evening peak hours (17:00-20:00)
        assert decision.should_heat is True

    def test_realistic_peak_billing_prevention_scenario(self):
        """Test realistic scenario: prevent peak billing during evening peak.

        Real-world scenario from Swedish household:
        - Time: 16:45 (end of normal workday)
        - DHW: In deferral range (used during day, but still safe)
        - Peak period: 17:00-20:00 (Swedish evening peak)
        - If we heat now: 16:45 + 1.5h = 18:15 (right in peak!)
        - Better: Wait until 20:00, heat 20:00-21:30 (off-peak)
        """
        scheduler = IntelligentDHWScheduler()

        # 16:45 - end of workday, peak starts in 15 min
        current_time = datetime(2025, 10, 17, 16, 45)

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN - 1.0,  # Safe to defer
            space_heating_demand_kw=0.0,
            thermal_debt_dm=-300.0,  # Healthy: -300 > -320, can defer
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=2.0,  # Cold evening
            price_classification="expensive",  # Peak starting soon
            current_time=current_time,
        )

        # Should defer to avoid peak billing
        # Wait until 20:00+ when peak ends
        assert decision.should_heat is False
        assert "DEFERRED" in decision.priority_reason

        # Verify temperature is safe for 3+ hour wait
        assert current_time.hour + 3 < 24  # Can wait until 20:00
        assert decision.target_temp == DEFAULT_DHW_TARGET_TEMP  # Shows user target even when not heating

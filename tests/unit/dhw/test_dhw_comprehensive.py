"""Comprehensive DHW optimization tests.

Tests all aspects of DHW (Domestic Hot Water) optimization:
1. Safety rules (thermal debt, space heating priority, safety minimum)
2. Window-based scheduling (finds absolute cheapest periods)
3. Demand period targeting (ready by configured time)
4. History tracking and maximum wait enforcement
5. Legionella detection and prevention
6. Automatic control via temporary lux switch

This replaces multiple separate DHW test files with one comprehensive suite.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from custom_components.effektguard.const import (
    DHW_SAFETY_RUNTIME_MINUTES,
    DHW_URGENT_RUNTIME_MINUTES,
    DHW_NORMAL_RUNTIME_MINUTES,
    DHW_EXTENDED_RUNTIME_MINUTES,
    DHW_SAFETY_MIN,
    DHW_SAFETY_CRITICAL,
    DHW_MAX_WAIT_HOURS,
    MIN_DHW_TARGET_TEMP,
    DEFAULT_DHW_TARGET_TEMP,
    DHW_PREHEAT_TARGET_OFFSET,
    DM_DHW_BLOCK_FALLBACK,
    DEFAULT_INDOOR_TEMP,
    DHW_LEGIONELLA_PREVENT_TEMP,
    DHW_LEGIONELLA_MAX_DAYS,
)
from custom_components.effektguard.optimization.dhw_optimizer import (
    IntelligentDHWScheduler,
    DHWDemandPeriod,
    DHWScheduleDecision,
)
from custom_components.effektguard.adapters.gespot_adapter import QuarterPeriod


# ==============================================================================
# TEST FIXTURES AND HELPERS
# ==============================================================================


def create_mock_quarters(base_time: datetime, prices: list[float]) -> list[QuarterPeriod]:
    """Create mock QuarterPeriod objects for testing.

    Args:
        base_time: Starting datetime (should be on quarter boundary)
        prices: List of prices for consecutive 15-min periods

    Returns:
        List of QuarterPeriod objects
    """
    from zoneinfo import ZoneInfo

    quarters = []
    for i, price in enumerate(prices):
        period_time = base_time + timedelta(minutes=i * 15)
        # Make timezone-aware if not already
        if period_time.tzinfo is None:
            period_time = period_time.replace(tzinfo=ZoneInfo("Europe/Stockholm"))

        quarters.append(
            QuarterPeriod(
                start_time=period_time,
                price=price,
            )
        )
    return quarters


def create_mock_price_analyzer():
    """Create a mock PriceAnalyzer for testing DHW optimizer.

    Returns a MagicMock that provides the find_cheapest_window and
    calculate_lookahead_hours methods required by DHW optimizer.
    """
    from custom_components.effektguard.optimization.price_layer import CheapestWindowResult

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


def create_dhw_scheduler(**kwargs):
    """Create DHW scheduler with mock price_analyzer.

    All DHW schedulers need price_analyzer - layers are required, not optional.
    """
    if "price_analyzer" not in kwargs:
        kwargs["price_analyzer"] = create_mock_price_analyzer()
    return IntelligentDHWScheduler(**kwargs)


@pytest.fixture
def scheduler_with_morning_demand():
    """Scheduler with morning shower demand at 7 AM."""
    from tests.conftest import create_mock_price_analyzer

    demand_period = DHWDemandPeriod(
        start_hour=7,
        target_temp=55.0,
        duration_hours=2,
    )
    mock_analyzer = create_mock_price_analyzer()
    scheduler = IntelligentDHWScheduler(
        demand_periods=[demand_period], price_analyzer=mock_analyzer
    )
    scheduler.last_legionella_boost = datetime.now() - timedelta(days=2)
    return scheduler


@pytest.fixture
def mock_hass():
    """Create mock Home Assistant instance."""
    hass = MagicMock()
    hass.config.latitude = 59.33  # Stockholm
    hass.config.longitude = 18.07
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.states = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda func, *args: func(*args))
    return hass


# ==============================================================================
# SAFETY RULES - RULE 1-4 (HIGHEST PRIORITY)
# ==============================================================================


class TestSafetyRules:
    """Test safety rules that override all other logic."""

    def test_critical_thermal_debt_blocks_dhw(self):
        """RULE 1: Critical thermal debt blocks DHW to protect space heating."""
        scheduler = create_dhw_scheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_CRITICAL,  # Even critically low!
            space_heating_demand_kw=1.0,
            thermal_debt_dm=DM_DHW_BLOCK_FALLBACK,  # CRITICAL thermal debt
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="cheap",
            current_time=datetime(2025, 10, 18, 12, 0),
        )

        assert decision.should_heat is False
        assert decision.priority_reason == "CRITICAL_THERMAL_DEBT"

    def test_space_heating_emergency_blocks_dhw(self):
        """RULE 2: Space heating emergency blocks DHW."""
        scheduler = create_dhw_scheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN + 2.0,  # Just above minimum
            space_heating_demand_kw=3.5,  # HIGH space demand
            thermal_debt_dm=-50,
            indoor_temp=19.5,  # Too cold!
            target_indoor_temp=21.0,
            outdoor_temp=-5.0,
            price_classification="cheap",
            current_time=datetime(2025, 10, 18, 12, 0),
        )

        assert decision.should_heat is False
        assert decision.priority_reason == "SPACE_HEATING_EMERGENCY"

    def test_dhw_safety_minimum_forces_heating(self):
        """RULE 3: DHW below safety minimum forces heating (Legionella risk)."""
        scheduler = create_dhw_scheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN - 2.0,  # 33°C - Below safety minimum!
            space_heating_demand_kw=2.0,
            thermal_debt_dm=-325,  # Bad DM: -325 < -320, cannot defer (but not critical)
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",  # Even during expensive!
            current_time=datetime(2025, 10, 18, 12, 0),
        )

        assert decision.should_heat is True
        assert decision.priority_reason == "DHW_SAFETY_MINIMUM"
        assert decision.max_runtime_minutes == DHW_SAFETY_RUNTIME_MINUTES
        # Two-tier strategy: Emergency heating only heats to DHW_SAFETY_MIN (20°C), not full target
        assert decision.target_temp == DHW_SAFETY_MIN

    def test_two_tier_emergency_heating_completes_during_cheap(self):
        """RULE 2.5: Two-tier emergency heating - complete to comfort level during cheap prices.

        Scenario:
        1. Family showers, DHW drops to 9°C (emergency)
        2. System heats to DHW_SAFETY_MIN (20°C) immediately
        3. Waits for cheap price to complete heating to user target (50°C)

        This prevents wasting money heating to full temp during expensive periods.
        """
        from datetime import timedelta

        scheduler = create_dhw_scheduler()
        current_time = datetime(2025, 10, 18, 12, 0)

        # Set recent Legionella boost so hygiene boost doesn't trigger
        scheduler.last_legionella_boost = current_time - timedelta(days=7)

        # After emergency heating reached 20°C, now at cheap price
        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN + 2.0,  # Just above safety min (22°C)
            space_heating_demand_kw=2.0,
            thermal_debt_dm=-50,  # Healthy (above -80 spare capacity threshold)
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="cheap",  # Cheap price - complete heating now!
            current_time=current_time,
        )

        assert decision.should_heat is True
        assert decision.priority_reason == "DHW_COMPLETE_EMERGENCY_HEATING"
        assert decision.target_temp == 50.0  # Full user target

    def test_two_tier_emergency_heating_waits_during_expensive(self):
        """RULE 2.5: Two-tier emergency heating - waits for cheap prices if at safe temp.

        After emergency heating to 20°C, should NOT continue heating during expensive prices.
        Waits for cheaper period to complete to full target.
        """
        from datetime import timedelta

        scheduler = create_dhw_scheduler()
        current_time = datetime(2025, 10, 18, 17, 0)  # Evening peak

        # Set recent Legionella boost so hygiene boost doesn't trigger
        scheduler.last_legionella_boost = current_time - timedelta(days=7)

        # At 22°C (safe level), expensive price
        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN + 2.0,  # Just above safety min (22°C)
            space_heating_demand_kw=2.0,
            thermal_debt_dm=-50,  # Healthy
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",  # Expensive - wait for cheaper!
            current_time=current_time,
        )

        # Should NOT heat during expensive period (temp is safe at 22°C)
        assert decision.should_heat is False
        assert decision.priority_reason == "DHW_ADEQUATE"

    def test_dhw_safety_deferred_for_peak_pricing(self):
        """RULE 3: Safety minimum CAN be deferred if temp safe, DM healthy, and price peak."""
        scheduler = create_dhw_scheduler()

        # For defer to work: temp >= DHW_SAFETY_CRITICAL AND price expensive/peak AND DM > block+20
        # With fallback thresholds: block=-340, so need DM > -320 (healthy)
        decision = scheduler.should_start_dhw(
            current_dhw_temp=DHW_SAFETY_MIN
            - 3.0,  # Above critical but below safety min (deferral range)
            space_heating_demand_kw=2.0,
            thermal_debt_dm=-300,  # Healthy: -300 > -320, can defer
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="peak",  # Peak pricing
            current_time=datetime(2025, 10, 18, 12, 0),
        )

        assert decision.should_heat is False
        assert decision.priority_reason == "DHW_SAFETY_DEFERRED_PEAK_PRICE"

    def test_high_space_demand_delays_dhw(self):
        """RULE 4: High space heating demand delays DHW."""
        from datetime import timedelta

        scheduler = create_dhw_scheduler()
        current_time = datetime(2025, 10, 18, 12, 0)

        # Set recent Legionella boost so hygiene boost doesn't trigger
        scheduler.last_legionella_boost = current_time - timedelta(days=7)

        decision = scheduler.should_start_dhw(
            current_dhw_temp=MIN_DHW_TARGET_TEMP,
            space_heating_demand_kw=7.0,  # High demand!
            thermal_debt_dm=-80,  # Some thermal debt
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="cheap",
            current_time=current_time,
        )

        assert decision.should_heat is False
        assert decision.priority_reason == "HIGH_SPACE_HEATING_DEMAND"

    def test_hygiene_boost_after_14_days(self):
        """RULE 2.7: Hygiene boost when DHW hasn't been above 60°C in 14 days.

        With new low thresholds (10°C/20°C), bacteria can grow in 20-45°C range.
        Ensure high-temp cycle (60°C) every 14 days for Legionella prevention.
        """
        from datetime import timedelta

        scheduler = create_dhw_scheduler()
        current_time = datetime(2025, 10, 18, 12, 0)

        # Simulate last Legionella boost was 15 days ago
        scheduler.last_legionella_boost = current_time - timedelta(days=15)

        decision = scheduler.should_start_dhw(
            current_dhw_temp=45.0,  # Normal temp, but needs hygiene boost
            space_heating_demand_kw=2.0,
            thermal_debt_dm=-50,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="cheap",  # Only during cheap prices
            current_time=current_time,
        )

        assert decision.should_heat is True
        assert decision.priority_reason == "DHW_HYGIENE_BOOST"
        assert decision.target_temp == DHW_LEGIONELLA_PREVENT_TEMP

    def test_hygiene_boost_never_had_high_temp(self):
        """RULE 2.7: Hygiene boost when never had high-temp cycle.

        If last_legionella_boost is None (never tracked), trigger hygiene boost.
        """
        scheduler = create_dhw_scheduler()

        # last_legionella_boost is None (never set)
        assert scheduler.last_legionella_boost is None

        decision = scheduler.should_start_dhw(
            current_dhw_temp=40.0,
            space_heating_demand_kw=2.0,
            thermal_debt_dm=-50,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="cheap",
            current_time=datetime(2025, 10, 18, 12, 0),
        )

        assert decision.should_heat is True
        assert decision.priority_reason == "DHW_HYGIENE_BOOST"
        assert decision.target_temp == DHW_LEGIONELLA_PREVENT_TEMP

    def test_hygiene_boost_waits_for_cheap_prices(self):
        """RULE 2.7: Hygiene boost only during cheap prices.

        Even if 14+ days since high-temp cycle, wait for cheap electricity.
        """
        from datetime import timedelta

        scheduler = create_dhw_scheduler()
        current_time = datetime(2025, 10, 18, 17, 0)  # Evening peak

        # 15 days since last boost, but expensive price
        scheduler.last_legionella_boost = current_time - timedelta(days=15)

        decision = scheduler.should_start_dhw(
            current_dhw_temp=45.0,
            space_heating_demand_kw=2.0,
            thermal_debt_dm=-50,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",  # Wait for cheaper!
            current_time=current_time,
        )

        # Should NOT heat during expensive period (wait for cheap)
        assert decision.should_heat is False
        assert "DHW_ADEQUATE_WAITING_CHEAP" in decision.priority_reason

    def test_no_hygiene_boost_within_14_days(self):
        """RULE 2.7: No hygiene boost if recent high-temp cycle.

        If had high-temp cycle within 14 days, no need for hygiene boost.
        """
        from datetime import timedelta

        scheduler = create_dhw_scheduler()
        current_time = datetime(2025, 10, 18, 12, 0)

        # Last boost was only 10 days ago (within 14-day window)
        scheduler.last_legionella_boost = current_time - timedelta(days=10)

        decision = scheduler.should_start_dhw(
            current_dhw_temp=45.0,
            space_heating_demand_kw=2.0,
            thermal_debt_dm=-50,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="cheap",
            current_time=current_time,
        )

        # Should heat for comfort, NOT hygiene boost
        assert decision.should_heat is True
        assert decision.priority_reason != "DHW_HYGIENE_BOOST"


# ==============================================================================
# MAXIMUM WAIT ENFORCEMENT - RULE 3.5
# ==============================================================================


class TestMaximumWaitEnforcement:
    """Test 36-hour maximum wait enforcement."""

    def test_max_wait_forces_dhw_after_36_hours(self):
        """RULE 4: Max 36h wait enforces DHW heating during cheap/normal prices only."""
        from datetime import timedelta

        scheduler = create_dhw_scheduler()
        current_time = datetime(2025, 10, 20, 12, 0)

        # Set recent Legionella boost so hygiene boost doesn't trigger
        scheduler.last_legionella_boost = current_time - timedelta(days=7)

        # Test with cheap pricing - should heat
        decision = scheduler.should_start_dhw(
            current_dhw_temp=MIN_DHW_TARGET_TEMP,
            space_heating_demand_kw=1.0,
            thermal_debt_dm=-50,
            indoor_temp=DEFAULT_INDOOR_TEMP,
            target_indoor_temp=DEFAULT_INDOOR_TEMP,
            outdoor_temp=0.0,
            price_classification="cheap",
            current_time=current_time,
            hours_since_last_dhw=36.1,
        )

        assert decision.should_heat is True
        assert "DHW_MAX_WAIT" in decision.priority_reason
        assert "36.1" in decision.priority_reason

    def test_max_wait_deferred_during_expensive(self):
        """RULE 4: Max 36h wait does NOT override expensive pricing."""
        scheduler = create_dhw_scheduler()

        # Even with max wait exceeded, expensive pricing blocks
        decision = scheduler.should_start_dhw(
            current_dhw_temp=MIN_DHW_TARGET_TEMP,
            space_heating_demand_kw=1.0,
            thermal_debt_dm=-50,
            indoor_temp=DEFAULT_INDOOR_TEMP,
            target_indoor_temp=DEFAULT_INDOOR_TEMP,
            outdoor_temp=0.0,
            price_classification="expensive",  # Expensive pricing
            current_time=datetime(2025, 10, 20, 12, 0),
            hours_since_last_dhw=36.1,
        )

        # Should wait for better prices, not heat during expensive
        assert decision.should_heat is False
        assert "DHW_ADEQUATE_WAITING_CHEAP" in decision.priority_reason

    def test_max_wait_deferred_during_peak(self):
        """RULE 4: Max 36h wait does NOT override peak pricing."""
        scheduler = create_dhw_scheduler()

        # Even with max wait exceeded, peak pricing blocks
        decision = scheduler.should_start_dhw(
            current_dhw_temp=MIN_DHW_TARGET_TEMP,
            space_heating_demand_kw=1.0,
            thermal_debt_dm=-50,
            indoor_temp=DEFAULT_INDOOR_TEMP,
            target_indoor_temp=DEFAULT_INDOOR_TEMP,
            outdoor_temp=0.0,
            price_classification="peak",  # Peak pricing (even worse!)
            current_time=datetime(2025, 10, 20, 12, 0),
            hours_since_last_dhw=36.1,
        )

        # Should definitely NOT heat during peak
        assert decision.should_heat is False
        assert "DHW_ADEQUATE_WAITING_CHEAP" in decision.priority_reason

    def test_max_wait_under_36_hours_no_heating(self):
        """DHW adequate at 48°C, under 36h wait, no heating needed."""
        scheduler = create_dhw_scheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=MIN_DHW_TARGET_TEMP + 3.0,  # Adequate
            space_heating_demand_kw=1.0,
            thermal_debt_dm=-50,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="expensive",
            current_time=datetime(2025, 10, 18, 12, 0),
            hours_since_last_dhw=24.0,  # Under limit
        )

        assert decision.should_heat is False
        assert "DHW_ADEQUATE_WAITING_CHEAP" in decision.priority_reason


# ==============================================================================
# DEMAND PERIOD TARGETING - RULE 5
# ==============================================================================


class TestDemandPeriodTargeting:
    """Test targeting DHW ready by demand period (e.g., morning shower)."""

    def test_urgent_demand_within_30_min(self, scheduler_with_morning_demand):
        """RULE 5: Urgent demand (<30 min) triggers heating during cheap/normal prices."""
        from zoneinfo import ZoneInfo

        # 6:45 AM, shower at 7:00 AM (15 minutes away = 0.25 hours)
        current_time = datetime(2025, 10, 18, 6, 45, tzinfo=ZoneInfo("Europe/Stockholm"))

        decision = scheduler_with_morning_demand.should_start_dhw(
            current_dhw_temp=45.0,  # Below target 55°C
            space_heating_demand_kw=2.0,
            thermal_debt_dm=-50,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="normal",
            current_time=current_time,
        )

        # Should trigger urgent heating (0.25 hours < 0.5 hours threshold)
        assert decision.should_heat is True
        assert "URGENT_DEMAND" in decision.priority_reason
        assert decision.max_runtime_minutes == DHW_URGENT_RUNTIME_MINUTES

    def test_urgent_demand_blocks_during_peak(self, scheduler_with_morning_demand):
        """RULE 5: Urgent demand still blocks during peak pricing."""
        from zoneinfo import ZoneInfo

        current_time = datetime(2025, 10, 18, 6, 45, tzinfo=ZoneInfo("Europe/Stockholm"))

        decision = scheduler_with_morning_demand.should_start_dhw(
            current_dhw_temp=45.0,
            space_heating_demand_kw=2.0,
            thermal_debt_dm=-50,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="peak",  # Peak!
            current_time=current_time,
        )

        # Should not heat during peak even with urgent demand
        assert decision.should_heat is False


# ==============================================================================
# WINDOW-BASED SCHEDULING - RULE 6
# ==============================================================================


class TestWindowBasedScheduling:
    """Test sliding window algorithm for finding cheapest periods."""

    def test_finds_cheapest_window_tomorrow(self):
        """Finds tomorrow's cheaper prices across day boundary."""
        from zoneinfo import ZoneInfo

        from custom_components.effektguard.optimization.price_layer import PriceAnalyzer

        # Use PriceAnalyzer directly (the layer that provides window search)
        analyzer = PriceAnalyzer()
        current_time = datetime(2025, 10, 17, 23, 45, tzinfo=ZoneInfo("Europe/Stockholm"))

        # Today evening: expensive
        today_prices = [63.94] * 4  # Q92-Q95
        today_quarters = create_mock_quarters(datetime(2025, 10, 17, 23, 0), today_prices)

        # Tomorrow morning: CHEAP (40% cheaper!)
        tomorrow_prices = [38.00, 37.50, 38.50, 39.00, 40.00]  # Q0-Q4
        tomorrow_quarters = create_mock_quarters(datetime(2025, 10, 18, 0, 0), tomorrow_prices)

        all_quarters = today_quarters + tomorrow_quarters

        # Find cheapest window using the layer directly
        optimal_window = analyzer.find_cheapest_window(
            current_time=current_time,
            price_periods=all_quarters,
            duration_minutes=DHW_NORMAL_RUNTIME_MINUTES,
            lookahead_hours=8.0,
        )

        assert optimal_window is not None
        assert optimal_window.avg_price < 40.0  # Should find tomorrow's cheap prices
        assert optimal_window.hours_until > 0  # In the future

    def test_waits_for_optimal_window(self):
        """Waits for better window if DHW comfortable."""
        from datetime import timedelta
        from zoneinfo import ZoneInfo

        # Create scheduler with mock price analyzer (required for window search)
        mock_analyzer = create_mock_price_analyzer()
        scheduler = IntelligentDHWScheduler(price_analyzer=mock_analyzer)
        current_time = datetime(2025, 10, 17, 23, 45, tzinfo=ZoneInfo("Europe/Stockholm"))

        # Set recent Legionella boost so hygiene boost doesn't trigger
        scheduler.last_legionella_boost = current_time - timedelta(days=7)

        # Create price data with cheaper window ahead (but not too close)
        # Need to be outside the 0.25h (15 min) optimal window trigger
        prices = [60.0] * 8 + [35.0] * 8  # Cheaper after 2 hours
        quarters = create_mock_quarters(datetime(2025, 10, 17, 23, 0), prices)

        decision = scheduler.should_start_dhw(
            current_dhw_temp=MIN_DHW_TARGET_TEMP + 3.0,  # Adequate
            space_heating_demand_kw=2.0,
            thermal_debt_dm=-50,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="cheap",  # Current hour is cheap
            current_time=current_time,
            price_periods=quarters,
        )

        # Should wait for better window (or DHW adequate)
        assert decision.should_heat is False
        assert (
            "WAITING_OPTIMAL_WINDOW" in decision.priority_reason
            or decision.priority_reason == "DHW_ADEQUATE"
        )

    def test_heats_in_optimal_window(self):
        """Heats when in the optimal window."""
        from datetime import timedelta
        from zoneinfo import ZoneInfo

        # Create scheduler with mock price analyzer (required for window search)
        mock_analyzer = create_mock_price_analyzer()
        scheduler = IntelligentDHWScheduler(price_analyzer=mock_analyzer)
        current_time = datetime(2025, 10, 18, 4, 0, tzinfo=ZoneInfo("Europe/Stockholm"))

        # Set recent Legionella boost so hygiene boost doesn't trigger
        scheduler.last_legionella_boost = current_time - timedelta(days=7)

        # Create price data - current time is cheapest
        prices = [30.0, 30.5, 31.0] + [50.0] * 8  # First 3 quarters cheapest
        quarters = create_mock_quarters(datetime(2025, 10, 18, 4, 0), prices)

        decision = scheduler.should_start_dhw(
            current_dhw_temp=MIN_DHW_TARGET_TEMP + 3.0,
            space_heating_demand_kw=2.0,
            thermal_debt_dm=-50,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="cheap",
            current_time=current_time,
            price_periods=quarters,
        )

        # Should heat now (in optimal window)
        assert decision.should_heat is True
        assert "OPTIMAL_WINDOW" in decision.priority_reason


# ==============================================================================
# LEGIONELLA DETECTION
# ==============================================================================


class TestLegionellaDetection:
    """Test automatic Legionella boost detection."""

    def test_detects_legionella_boost_completion(self):
        """Detects when NIBE's automatic Legionella boost completes."""
        scheduler = create_dhw_scheduler()
        now = datetime.now()

        # Build up temperature history showing Legionella boost
        for i in range(10):
            scheduler.update_bt7_temperature(50.0 + i * 0.1, now - timedelta(minutes=150 - i * 15))

        # NIBE starts Legionella boost
        scheduler.update_bt7_temperature(60.0, now - timedelta(minutes=60))
        scheduler.update_bt7_temperature(63.0, now - timedelta(minutes=45))
        scheduler.update_bt7_temperature(65.0, now - timedelta(minutes=30))  # Peak

        # Cooling down
        scheduler.update_bt7_temperature(63.5, now - timedelta(minutes=15))
        scheduler.update_bt7_temperature(61.0, now)  # Cooled 4°C

        # Should have detected boost
        assert scheduler.last_legionella_boost is not None

    def test_dhw_adequate_after_legionella(self):
        """DHW adequate after Legionella boost, no heating needed."""
        scheduler = create_dhw_scheduler()
        now = datetime.now()

        # Simulate completed Legionella boost
        scheduler.last_legionella_boost = now - timedelta(hours=2)

        decision = scheduler.should_start_dhw(
            current_dhw_temp=DEFAULT_DHW_TARGET_TEMP,  # Comfortable after boost
            space_heating_demand_kw=2.0,
            thermal_debt_dm=-50,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="normal",
            current_time=now,
        )

        assert decision.should_heat is False
        assert "DHW_ADEQUATE" in decision.priority_reason


# ==============================================================================
# CHEAP ELECTRICITY OPPORTUNITIES
# ==============================================================================


class TestCheapElectricityOpportunities:
    """Test opportunistic heating during cheap periods."""

    def test_heats_during_cheap_when_low(self):
        """RULE 7 / RULE 2.5: Heats below minimum target during cheap prices.

        Can be either:
        - RULE 2.5: DHW_COMPLETE_EMERGENCY_HEATING (20-45°C range, completing emergency heating)
        - RULE 7: DHW_COMFORT_LOW_CHEAP (below MIN_DHW_TARGET_TEMP)
        Both are valid for this temperature range during cheap prices.
        """
        from datetime import timedelta

        scheduler = create_dhw_scheduler()
        current_time = datetime(2025, 10, 18, 12, 0)

        # Set recent Legionella boost so hygiene boost doesn't trigger
        scheduler.last_legionella_boost = current_time - timedelta(days=7)

        decision = scheduler.should_start_dhw(
            current_dhw_temp=MIN_DHW_TARGET_TEMP - 3.0,  # Below MIN_DHW_TARGET_TEMP (42°C)
            space_heating_demand_kw=2.0,
            thermal_debt_dm=-50,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="cheap",
            current_time=current_time,
        )

        assert decision.should_heat is True
        # Can be either completing emergency heating or comfort heating
        assert (
            "CHEAP" in decision.priority_reason.upper()
            or "EMERGENCY" in decision.priority_reason.upper()
        )

    def test_no_heat_when_dhw_adequate(self):
        """RULE 8: No heating when DHW adequate."""
        scheduler = create_dhw_scheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=MIN_DHW_TARGET_TEMP + 3.0,  # Adequate
            space_heating_demand_kw=2.0,
            thermal_debt_dm=-50,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="normal",
            current_time=datetime(2025, 10, 18, 12, 0),
        )

        assert decision.should_heat is False
        assert "DHW_ADEQUATE" in decision.priority_reason


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestIntegrationScenarios:
    """Test realistic multi-factor scenarios."""

    def test_typical_morning_scenario(self, scheduler_with_morning_demand):
        """Typical scenario: finds cheap night price before morning shower."""
        from zoneinfo import ZoneInfo

        current_time = datetime(2025, 10, 18, 2, 0, tzinfo=ZoneInfo("Europe/Stockholm"))  # 2 AM

        # Create realistic price curve (cheap at night, expensive in morning)
        prices = [30.0] * 12 + [45.0] * 8 + [60.0] * 12  # Night cheap
        quarters = create_mock_quarters(datetime(2025, 10, 18, 2, 0), prices)

        decision = scheduler_with_morning_demand.should_start_dhw(
            current_dhw_temp=MIN_DHW_TARGET_TEMP + 1.0,  # Getting low
            space_heating_demand_kw=1.5,
            thermal_debt_dm=-50,
            indoor_temp=DEFAULT_INDOOR_TEMP,
            target_indoor_temp=DEFAULT_INDOOR_TEMP,
            outdoor_temp=5.0,
            price_classification="cheap",
            current_time=current_time,
            price_periods=quarters,
        )

        # Should find optimal window and heat
        assert decision.should_heat is True or "WAITING" in decision.priority_reason

    def test_cold_weather_priority(self):
        """Cold weather: space heating takes priority over DHW."""
        scheduler = create_dhw_scheduler()

        decision = scheduler.should_start_dhw(
            current_dhw_temp=MIN_DHW_TARGET_TEMP - 1.0,  # Low but not critical
            space_heating_demand_kw=8.0,  # HIGH demand (cold weather)
            thermal_debt_dm=-150,  # Significant thermal debt
            indoor_temp=20.0,  # Slightly cold
            target_indoor_temp=21.0,
            outdoor_temp=-5.0,  # Cold outside
            price_classification="cheap",
            current_time=datetime(2025, 10, 18, 12, 0),
        )

        # Space heating should take priority
        assert decision.should_heat is False
        assert (
            "SPACE_HEATING" in decision.priority_reason
            or "THERMAL_DEBT" in decision.priority_reason
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

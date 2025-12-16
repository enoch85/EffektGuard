"""Regression test for October 28, 2025 DHW failure.

System identified optimal window at 04:00 (8.6öre/kWh) but failed to heat
due to spare capacity check blocking despite being 62 DM above warning threshold.

Root causes fixed:
1. Spare capacity check DELETED (redundant with RULE 1)
2. Window timing extended from 10 min to 15 min (matches coordinator cycle)
3. Comparison operator fixed from > to >= for MIN_DHW_TARGET_TEMP
4. RULE 1 enhanced logging added

This test ensures the October 28 scenario would now work correctly.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from custom_components.effektguard.const import (
    MIN_DHW_TARGET_TEMP,
)
from custom_components.effektguard.optimization.dhw_optimizer import (
    IntelligentDHWScheduler,
    DHWDemandPeriod,
)
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.adapters.gespot_adapter import QuarterPeriod
from tests.conftest import create_mock_price_analyzer


def create_dhw_scheduler(**kwargs):
    """Create DHW scheduler with mock price_analyzer."""
    if "price_analyzer" not in kwargs:
        kwargs["price_analyzer"] = create_mock_price_analyzer()
    return IntelligentDHWScheduler(**kwargs)


def create_oct28_price_periods():
    """Create realistic price data from October 28, 2025.

    Window at 04:00 (Q16) was 8.6 öre/kWh (cheapest).
    """
    # Start at midnight
    base_time = datetime(2025, 10, 28, 0, 0, 0, tzinfo=ZoneInfo("Europe/Stockholm"))

    # Realistic prices (öre/kWh) for each 15-min quarter
    # Q16 = 04:00 is the cheapest at 8.6
    prices = [
        # 00:00-01:00 (Q0-Q3)
        12.5,
        12.3,
        11.8,
        11.5,
        # 01:00-02:00 (Q4-Q7)
        10.2,
        9.8,
        9.5,
        9.3,
        # 02:00-03:00 (Q8-Q11)
        9.1,
        9.0,
        8.9,
        8.8,
        # 03:00-04:00 (Q12-Q15)
        8.7,
        8.7,
        8.6,
        8.6,
        # 04:00-05:00 (Q16-Q19) ← OPTIMAL WINDOW HERE
        8.6,
        8.6,
        8.7,
        8.9,  # Q16 is 8.6 (cheapest)
        # 05:00-06:00 (Q20-Q23)
        9.2,
        9.5,
        9.8,
        10.1,
        # 06:00-07:00 (Q24-Q27)
        10.5,
        11.2,
        12.5,
        13.8,
        # 07:00-08:00 (Q28-Q31)
        15.2,
        16.5,
        17.8,
        18.9,
    ]

    quarters = []
    for i, price in enumerate(prices):
        period_time = base_time + timedelta(minutes=i * 15)
        quarters.append(
            QuarterPeriod(
                start_time=period_time,
                price=price,
            )
        )

    return quarters


def test_oct28_failure_scenario_at_03_45():
    """Reproduce October 28, 2025 failure: DHW at 36.1°C, DM -254, 15 min before optimal window.

    At 03:45:
    - DHW temp: 36.1°C (below MIN_DHW_TARGET_TEMP of 45°C)
    - DM: -254 (62 DM better than warning -316, SAFE!)
    - Optimal window: 04:00 (15 minutes away, Q16 at 8.6 öre/kWh)
    - Price classification: cheap

    Expected: Should heat (within 15-min window buffer)
    Old behavior: Blocked by spare capacity check
    New behavior: Heats (spare capacity deleted, window timing fixed)
    """
    # Setup: Moderate Cold zone (Malmö/Southern Sweden)
    climate_detector = ClimateZoneDetector(latitude=55.60)  # Malmö
    demand_period = DHWDemandPeriod(availability_hour=7, target_temp=50.0, duration_hours=2)
    scheduler = create_dhw_scheduler(
        demand_periods=[demand_period],
        climate_detector=climate_detector,
    )

    # At 03:45 - 15 minutes before optimal window at 04:00
    current_time = datetime(2025, 10, 28, 3, 45, 0, tzinfo=ZoneInfo("Europe/Stockholm"))

    # Set last legionella boost to prevent hygiene boost from firing
    scheduler.last_legionella_boost = current_time - timedelta(days=2)

    price_periods = create_oct28_price_periods()

    # October 28 conditions from logs
    decision = scheduler.should_start_dhw(
        current_dhw_temp=36.1,  # Actual temp from logs (below 45°C minimum)
        space_heating_demand_kw=2.5,
        thermal_debt_dm=-254,  # Actual DM from logs (62 DM above warning -316)
        indoor_temp=23.4,
        target_indoor_temp=21.0,
        outdoor_temp=8.2,  # Actual outdoor temp
        price_classification="cheap",
        current_time=current_time,
        price_periods=price_periods,
        hours_since_last_dhw=8.4,  # Last heating at 19:24 Oct 27
    )

    # With fixes applied:
    # 1. Spare capacity check DELETED (no longer blocks)
    # 2. Window timing extended to 15 min (03:45 is within buffer)
    # 3. RULE 1 check: -254 > -316? YES ✅ (passes)
    # Result: Should heat
    assert decision.should_heat is True, (
        f"Should heat when within 15min of optimal window at 03:45. "
        f"DM -254 is 62 DM above warning -316 (safe!). "
        f"Reason: {decision.priority_reason}"
    )

    # Should be optimal window activation or DHW low cheap heating
    assert "OPTIMAL_WINDOW" in decision.priority_reason or "DHW" in decision.priority_reason

    # Verify DM is actually safe (better than warning)
    dm_range = climate_detector.get_expected_dm_range(8.2)
    warning_threshold = dm_range["warning"]  # Should be around -316 for T2-level
    assert (
        -254 > warning_threshold
    ), f"Test setup error: DM -254 should be better than warning {warning_threshold}"


def test_oct28_at_04_00_exactly():
    """At exactly 04:00 (the scheduled window start), should definitely heat.

    This is the absolute optimal window - must heat regardless of DM
    (as long as RULE 1 doesn't block, which it won't at -249).
    """
    climate_detector = ClimateZoneDetector(latitude=55.60)
    demand_period = DHWDemandPeriod(availability_hour=7, target_temp=50.0, duration_hours=2)
    scheduler = create_dhw_scheduler(
        demand_periods=[demand_period],
        climate_detector=climate_detector,
    )

    # Exactly at the scheduled window (04:00)
    current_time = datetime(2025, 10, 28, 4, 0, 0, tzinfo=ZoneInfo("Europe/Stockholm"))

    # Set last legionella boost to prevent hygiene boost from firing
    scheduler.last_legionella_boost = current_time - timedelta(days=2)

    price_periods = create_oct28_price_periods()

    decision = scheduler.should_start_dhw(
        current_dhw_temp=36.1,
        space_heating_demand_kw=2.5,
        thermal_debt_dm=-249,  # Slightly better than at 03:45
        indoor_temp=23.4,
        target_indoor_temp=21.0,
        outdoor_temp=8.2,
        price_classification="cheap",
        current_time=current_time,
        price_periods=price_periods,
        hours_since_last_dhw=8.7,
    )

    # MUST heat when exactly in the optimal window
    assert decision.should_heat is True, (
        f"Must heat at 04:00 (optimal window start). " f"Reason: {decision.priority_reason}"
    )

    # Should be either optimal window activation OR emergency heating completion
    # (both are valid - emergency heating gets priority when DHW is in 30-45°C range)
    valid_reasons = ["OPTIMAL_WINDOW", "DHW_COMPLETE_EMERGENCY_HEATING", "DHW_COMFORT_LOW_CHEAP"]
    assert any(reason in decision.priority_reason for reason in valid_reasons), (
        f"Should use optimal window or emergency heating logic. " f"Got: {decision.priority_reason}"
    )

    # Should target reasonable temp (either pre-heat or user target)
    assert (
        50.0 <= decision.target_temp <= 60.0
    ), f"Target temp should be 50-60°C, got {decision.target_temp}°C"


def test_oct28_at_03_50_also_within_window():
    """At 03:50 (10 min before window), should also heat with new 15-min buffer."""
    climate_detector = ClimateZoneDetector(latitude=55.60)
    demand_period = DHWDemandPeriod(availability_hour=7, target_temp=50.0, duration_hours=2)
    scheduler = create_dhw_scheduler(
        demand_periods=[demand_period],
        climate_detector=climate_detector,
    )

    # At 03:50 - 10 minutes before window
    current_time = datetime(2025, 10, 28, 3, 50, 0, tzinfo=ZoneInfo("Europe/Stockholm"))

    # Set last legionella boost to prevent hygiene boost from firing
    scheduler.last_legionella_boost = current_time - timedelta(days=2)

    price_periods = create_oct28_price_periods()

    decision = scheduler.should_start_dhw(
        current_dhw_temp=36.1,
        space_heating_demand_kw=2.5,
        thermal_debt_dm=-252,
        indoor_temp=23.4,
        target_indoor_temp=21.0,
        outdoor_temp=8.2,
        price_classification="cheap",
        current_time=current_time,
        price_periods=price_periods,
        hours_since_last_dhw=8.5,
    )

    # Should heat with 15-min buffer (0.25 hours = 15 minutes)
    assert decision.should_heat is True, (
        f"Should heat at 03:50 (10 min before window, within 15-min buffer). "
        f"Reason: {decision.priority_reason}"
    )


def test_rule1_blocks_at_warning_threshold():
    """RULE 1 should block DHW when thermal debt at/below warning threshold.

    This test verifies that RULE 1 still provides proper thermal debt protection
    even though spare capacity check has been deleted.
    """
    climate_detector = ClimateZoneDetector(latitude=55.60)
    scheduler = create_dhw_scheduler(climate_detector=climate_detector)

    # Get warning threshold for conditions
    dm_range = climate_detector.get_expected_dm_range(-1.0)
    warning_threshold = dm_range["warning"]  # Around -316 for Moderate Cold at -1°C

    current_time = datetime(2025, 10, 28, 4, 0, 0, tzinfo=ZoneInfo("Europe/Stockholm"))
    price_periods = create_oct28_price_periods()

    # Test 1: DM exactly at warning = should block
    decision = scheduler.should_start_dhw(
        current_dhw_temp=36.0,
        space_heating_demand_kw=2.5,
        thermal_debt_dm=warning_threshold,  # Exactly at warning
        indoor_temp=21.5,
        target_indoor_temp=21.0,
        outdoor_temp=-1.0,
        price_classification="cheap",
        current_time=current_time,
        price_periods=price_periods,
        hours_since_last_dhw=8.0,
    )

    assert (
        decision.should_heat is False
    ), f"RULE 1 should block when DM at warning threshold {warning_threshold}"
    assert decision.priority_reason == "CRITICAL_THERMAL_DEBT"

    # Test 2: DM below warning = should block
    decision2 = scheduler.should_start_dhw(
        current_dhw_temp=36.0,
        space_heating_demand_kw=2.5,
        thermal_debt_dm=warning_threshold - 50,  # Past warning
        indoor_temp=21.5,
        target_indoor_temp=21.0,
        outdoor_temp=-1.0,
        price_classification="cheap",
        current_time=current_time,
        price_periods=price_periods,
        hours_since_last_dhw=8.0,
    )

    assert decision2.should_heat is False
    assert decision2.priority_reason == "CRITICAL_THERMAL_DEBT"

    # Test 3: DM above warning = should allow (if other conditions met)
    decision3 = scheduler.should_start_dhw(
        current_dhw_temp=36.0,
        space_heating_demand_kw=2.5,
        thermal_debt_dm=warning_threshold + 60,  # 60 DM better than warning
        indoor_temp=21.5,
        target_indoor_temp=21.0,
        outdoor_temp=-1.0,
        price_classification="cheap",
        current_time=current_time,
        price_periods=price_periods,
        hours_since_last_dhw=8.0,
    )

    # Should heat (not blocked by RULE 1, DHW below minimum, cheap price)
    assert decision3.should_heat is True, (
        f"Should heat when DM above warning. "
        f"DM={warning_threshold + 60}, warning={warning_threshold}. "
        f"Reason: {decision3.priority_reason}"
    )


def test_comparison_operator_fix():
    """Verify >= operator fix for MIN_DHW_TARGET_TEMP waiting logic.

    October 28 logs showed DHW at 36.1°C trying to use > comparison with 45°C,
    which incorrectly failed. Should use >= so exactly 45°C also waits for window.
    
    Updated: With the new "adequate + not cheap = wait" rule, DHW at 45°C with
    normal prices will wait for cheap prices, which is the correct behavior.
    """
    climate_detector = ClimateZoneDetector(latitude=55.60)
    demand_period = DHWDemandPeriod(availability_hour=7, target_temp=50.0, duration_hours=2)
    scheduler = create_dhw_scheduler(
        demand_periods=[demand_period],
        climate_detector=climate_detector,
    )

    # Test: DHW exactly at MIN_DHW_TARGET_TEMP (45°C)
    # Should wait for cheap prices (not heat during normal prices)
    current_time = datetime(2025, 10, 28, 2, 0, 0, tzinfo=ZoneInfo("Europe/Stockholm"))
    price_periods = create_oct28_price_periods()

    decision = scheduler.should_start_dhw(
        current_dhw_temp=MIN_DHW_TARGET_TEMP,  # Exactly 45°C
        space_heating_demand_kw=2.0,
        thermal_debt_dm=-200,  # Good thermal debt
        indoor_temp=21.5,
        target_indoor_temp=21.0,
        outdoor_temp=8.0,
        price_classification="normal",  # Not cheap - should wait for cheap
        current_time=current_time,
        price_periods=price_periods,
        hours_since_last_dhw=4.0,
    )

    # Should wait for cheap prices (DHW adequate at 45°C, price is normal)
    assert decision.should_heat is False, (
        f"Should wait for cheap prices when DHW adequate at MIN_DHW_TARGET_TEMP (45°C) "
        f"and price is normal. Reason: {decision.priority_reason}"
    )
    # Accept either waiting reason - the key is that we're not heating
    assert "WAITING" in decision.priority_reason or "DHW_ADEQUATE" in decision.priority_reason


def test_october_28_would_not_need_manual_boost():
    """Verify complete October 28 scenario - user would NOT need to manually boost.

    This is the end-to-end test: System should heat during optimal window,
    bringing DHW from 36.1°C to 50°C+ without manual intervention.
    """
    climate_detector = ClimateZoneDetector(latitude=55.60)
    demand_period = DHWDemandPeriod(availability_hour=7, target_temp=50.0, duration_hours=2)
    scheduler = create_dhw_scheduler(
        demand_periods=[demand_period],
        climate_detector=climate_detector,
    )

    price_periods = create_oct28_price_periods()

    # Simulate coordinator updates every 5 minutes from 03:45 to 04:00
    test_times = [
        datetime(2025, 10, 28, 3, 45, 0, tzinfo=ZoneInfo("Europe/Stockholm")),
        datetime(2025, 10, 28, 3, 50, 0, tzinfo=ZoneInfo("Europe/Stockholm")),
        datetime(2025, 10, 28, 3, 55, 0, tzinfo=ZoneInfo("Europe/Stockholm")),
        datetime(2025, 10, 28, 4, 0, 0, tzinfo=ZoneInfo("Europe/Stockholm")),
    ]

    # Set last legionella boost to prevent hygiene boost from firing
    scheduler.last_legionella_boost = test_times[0] - timedelta(days=2)

    heated = False
    for current_time in test_times:
        decision = scheduler.should_start_dhw(
            current_dhw_temp=36.1,
            space_heating_demand_kw=2.5,
            thermal_debt_dm=-254,
            indoor_temp=23.4,
            target_indoor_temp=21.0,
            outdoor_temp=8.2,
            price_classification="cheap",
            current_time=current_time,
            price_periods=price_periods,
            hours_since_last_dhw=8.4,
        )

        if decision.should_heat:
            heated = True
            break

    # Should have heated at one of these times
    assert heated, (
        "System should have heated DHW during optimal window period (03:45-04:00). "
        "User should NOT need manual boost!"
    )

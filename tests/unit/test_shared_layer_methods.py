"""Tests for shared layer methods.

Tests the reusable methods in optimization layer files:
- dhw_optimizer.format_planning_summary()
- dhw_optimizer.check_abort_conditions()
- effect_layer.estimate_power_consumption()
- effect_layer.estimate_power_from_compressor()
- thermal_layer.get_thermal_debt_status()
"""

import pytest

from custom_components.effektguard.const import (
    COMPRESSOR_HZ_MIN,
    COMPRESSOR_POWER_MAX_KW,
    COMPRESSOR_POWER_MIN_KW,
    COMPRESSOR_TEMP_EXTREME_COLD_THRESHOLD,
    COMPRESSOR_TEMP_FACTOR_EXTREME_COLD,
    COMPRESSOR_TEMP_FACTOR_MILD,
    DEFAULT_HEAT_PUMP_POWER_KW,
    POWER_MULTIPLIER_COLD,
    POWER_MULTIPLIER_MILD,
    POWER_MULTIPLIER_VERY_COLD,
    POWER_STANDBY_KW,
    POWER_TEMP_COLD_THRESHOLD,
    POWER_TEMP_VERY_COLD_THRESHOLD,
    SPACE_HEATING_DEMAND_HIGH_THRESHOLD,
    SPACE_HEATING_DEMAND_LOW_THRESHOLD,
    SPACE_HEATING_DEMAND_MODERATE_THRESHOLD,
)
from custom_components.effektguard.optimization.thermal_layer import get_thermal_debt_status
from tests.conftest import create_mock_price_analyzer


def create_dhw_scheduler(**kwargs):
    """Create DHW scheduler with mock price_analyzer."""
    from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler

    if "price_analyzer" not in kwargs:
        kwargs["price_analyzer"] = create_mock_price_analyzer()
    return IntelligentDHWScheduler(**kwargs)


class TestGetThermalDebtStatus:
    """Tests for get_thermal_debt_status helper function."""

    def test_ok_status_with_margin(self):
        """Test OK status when thermal debt is healthy."""
        dm_thresholds = {"block": -700, "abort": -1500}
        result = get_thermal_debt_status(-200, dm_thresholds)

        assert "OK" in result
        assert "500" in result  # -200 - (-700) = 500 margin

    def test_warning_status(self):
        """Test WARNING status when between block and abort thresholds."""
        dm_thresholds = {"block": -700, "abort": -1500}
        result = get_thermal_debt_status(-900, dm_thresholds)

        assert "WARNING" in result
        # -900 is past -700 block threshold

    def test_critical_status(self):
        """Test CRITICAL status when past abort threshold."""
        dm_thresholds = {"block": -700, "abort": -1500}
        result = get_thermal_debt_status(-1600, dm_thresholds)

        assert "CRITICAL" in result
        assert "past abort" in result

    def test_supports_warning_key_alias(self):
        """Test that 'warning' key works as alias for 'block'."""
        dm_thresholds = {"warning": -700, "critical": -1500}
        result = get_thermal_debt_status(-200, dm_thresholds)

        assert "OK" in result

    def test_uses_fallback_thresholds(self):
        """Test fallback thresholds when keys missing."""
        dm_thresholds = {}
        result = get_thermal_debt_status(-200, dm_thresholds)

        assert "OK" in result  # -200 > default -700


class TestDHWOptimizerFormatPlanningSummary:
    """Tests for IntelligentDHWScheduler.format_planning_summary()."""

    @pytest.fixture
    def scheduler(self):
        """Create a basic DHW scheduler instance."""
        return create_dhw_scheduler()

    def test_formats_basic_summary(self, scheduler):
        """Test basic summary formatting."""
        result = scheduler.format_planning_summary(
            recommendation="Heat now",
            current_temp=42.0,
            target_temp=50.0,
            thermal_debt=-200,
            dm_thresholds={"block": -700, "abort": -1500},
            space_heating_demand=2.5,
            price_classification="CHEAP",
            weather_opportunity=None,
        )

        assert "DHW Planning Summary" in result
        assert "42.0°C" in result
        assert "50°C" in result
        assert "CHEAP" in result
        assert "Heat now" in result

    def test_includes_thermal_debt_status(self, scheduler):
        """Test thermal debt status in summary."""
        result = scheduler.format_planning_summary(
            recommendation="Block DHW",
            current_temp=45.0,
            target_temp=50.0,
            thermal_debt=-800,
            dm_thresholds={"block": -700, "abort": -1500},
            space_heating_demand=4.0,
            price_classification="normal",
            weather_opportunity=None,
        )

        assert "WARNING" in result

    def test_includes_weather_opportunity(self, scheduler):
        """Test weather opportunity in summary."""
        result = scheduler.format_planning_summary(
            recommendation="Heat now",
            current_temp=42.0,
            target_temp=50.0,
            thermal_debt=-200,
            dm_thresholds={"block": -700, "abort": -1500},
            space_heating_demand=1.0,
            price_classification="CHEAP",
            weather_opportunity="Unusually warm (+5.0°C)",
        )

        assert "Unusually warm" in result

    def test_heating_demand_categories(self, scheduler):
        """Test heating demand category display."""
        # High demand
        result_high = scheduler.format_planning_summary(
            recommendation="Wait",
            current_temp=45.0,
            target_temp=50.0,
            thermal_debt=-200,
            dm_thresholds={"block": -700, "abort": -1500},
            space_heating_demand=SPACE_HEATING_DEMAND_HIGH_THRESHOLD + 1.0,
            price_classification="normal",
            weather_opportunity=None,
        )
        assert "HIGH" in result_high

        # Moderate demand
        result_moderate = scheduler.format_planning_summary(
            recommendation="Wait",
            current_temp=45.0,
            target_temp=50.0,
            thermal_debt=-200,
            dm_thresholds={"block": -700, "abort": -1500},
            space_heating_demand=SPACE_HEATING_DEMAND_MODERATE_THRESHOLD + 0.5,
            price_classification="normal",
            weather_opportunity=None,
        )
        assert "MODERATE" in result_moderate

        # Low demand
        result_low = scheduler.format_planning_summary(
            recommendation="Wait",
            current_temp=45.0,
            target_temp=50.0,
            thermal_debt=-200,
            dm_thresholds={"block": -700, "abort": -1500},
            space_heating_demand=SPACE_HEATING_DEMAND_LOW_THRESHOLD + 0.1,
            price_classification="normal",
            weather_opportunity=None,
        )
        assert "LOW" in result_low


class TestDHWOptimizerCheckAbortConditions:
    """Tests for IntelligentDHWScheduler.check_abort_conditions()."""

    @pytest.fixture
    def scheduler(self):
        """Create a basic DHW scheduler instance."""
        return create_dhw_scheduler()

    def test_no_abort_when_no_conditions(self, scheduler):
        """Test no abort when abort_conditions is empty."""
        should_abort, reason = scheduler.check_abort_conditions(
            abort_conditions=[],
            thermal_debt=-200,
            indoor_temp=21.0,
            target_indoor=21.0,
        )

        assert should_abort is False
        assert reason is None

    def test_abort_on_thermal_debt_threshold(self, scheduler):
        """Test abort when thermal debt crosses threshold."""
        should_abort, reason = scheduler.check_abort_conditions(
            abort_conditions=["thermal_debt < -500"],
            thermal_debt=-600,
            indoor_temp=21.0,
            target_indoor=21.0,
        )

        assert should_abort is True
        assert "Thermal debt" in reason
        assert "-600" in reason

    def test_no_abort_when_thermal_debt_above_threshold(self, scheduler):
        """Test no abort when thermal debt is above threshold."""
        should_abort, reason = scheduler.check_abort_conditions(
            abort_conditions=["thermal_debt < -500"],
            thermal_debt=-400,
            indoor_temp=21.0,
            target_indoor=21.0,
        )

        assert should_abort is False
        assert reason is None

    def test_abort_on_indoor_temp_threshold(self, scheduler):
        """Test abort when indoor temp drops below threshold."""
        should_abort, reason = scheduler.check_abort_conditions(
            abort_conditions=["indoor_temp < 20.5"],
            thermal_debt=-200,
            indoor_temp=20.0,
            target_indoor=21.0,
        )

        assert should_abort is True
        assert "Indoor" in reason
        assert "20.0" in reason

    def test_multiple_conditions_first_triggered(self, scheduler):
        """Test first triggered condition aborts."""
        should_abort, reason = scheduler.check_abort_conditions(
            abort_conditions=["thermal_debt < -500", "indoor_temp < 20.5"],
            thermal_debt=-600,  # This triggers first
            indoor_temp=21.0,  # This is fine
            target_indoor=21.0,
        )

        assert should_abort is True
        assert "Thermal debt" in reason

    def test_handles_malformed_condition(self, scheduler):
        """Test graceful handling of malformed conditions."""
        should_abort, reason = scheduler.check_abort_conditions(
            abort_conditions=["malformed condition", "thermal_debt < invalid"],
            thermal_debt=-200,
            indoor_temp=21.0,
            target_indoor=21.0,
        )

        # Should not crash, should return False
        assert should_abort is False
        assert reason is None


class TestEffectLayerEstimatePowerConsumption:
    """Tests for EffectManager.estimate_power_consumption()."""

    @pytest.fixture
    def effect_manager(self, hass):
        """Create an EffectManager instance."""
        from custom_components.effektguard.optimization.effect_layer import EffectManager

        return EffectManager(hass)

    def test_standby_power_when_not_heating(self, effect_manager):
        """Test standby power returned when not heating."""
        result = effect_manager.estimate_power_consumption(is_heating=False, outdoor_temp=5.0)

        assert result == POWER_STANDBY_KW

    def test_base_power_at_mild_temp(self, effect_manager):
        """Test base power at mild temperatures."""
        result = effect_manager.estimate_power_consumption(
            is_heating=True, outdoor_temp=POWER_TEMP_COLD_THRESHOLD + 5.0
        )

        assert result == DEFAULT_HEAT_PUMP_POWER_KW * POWER_MULTIPLIER_MILD

    def test_increased_power_at_cold_temp(self, effect_manager):
        """Test increased power at cold temperatures."""
        result = effect_manager.estimate_power_consumption(
            is_heating=True,
            outdoor_temp=(POWER_TEMP_VERY_COLD_THRESHOLD + POWER_TEMP_COLD_THRESHOLD) / 2,
        )

        assert result == DEFAULT_HEAT_PUMP_POWER_KW * POWER_MULTIPLIER_COLD

    def test_maximum_power_at_very_cold_temp(self, effect_manager):
        """Test maximum power at very cold temperatures."""
        result = effect_manager.estimate_power_consumption(
            is_heating=True, outdoor_temp=POWER_TEMP_VERY_COLD_THRESHOLD - 5.0
        )

        assert result == DEFAULT_HEAT_PUMP_POWER_KW * POWER_MULTIPLIER_VERY_COLD


class TestEffectLayerEstimatePowerFromCompressor:
    """Tests for EffectManager.estimate_power_from_compressor()."""

    @pytest.fixture
    def effect_manager(self, hass):
        """Create an EffectManager instance."""
        from custom_components.effektguard.optimization.effect_layer import EffectManager

        return EffectManager(hass)

    def test_standby_power_when_compressor_off(self, effect_manager):
        """Test standby power when compressor is off (0 Hz)."""
        result = effect_manager.estimate_power_from_compressor(compressor_hz=0, outdoor_temp=5.0)

        assert result == POWER_STANDBY_KW

    def test_minimum_power_at_minimum_hz(self, effect_manager):
        """Test minimum power at minimum compressor Hz."""
        result = effect_manager.estimate_power_from_compressor(
            compressor_hz=COMPRESSOR_HZ_MIN, outdoor_temp=10.0
        )

        # Should be close to minimum power with mild temp factor
        assert abs(result - COMPRESSOR_POWER_MIN_KW * COMPRESSOR_TEMP_FACTOR_MILD) < 0.5

    def test_power_increases_with_hz(self, effect_manager):
        """Test power increases with compressor Hz."""
        result_low = effect_manager.estimate_power_from_compressor(
            compressor_hz=30, outdoor_temp=5.0
        )
        result_high = effect_manager.estimate_power_from_compressor(
            compressor_hz=70, outdoor_temp=5.0
        )

        assert result_high > result_low

    def test_temperature_factor_applied(self, effect_manager):
        """Test temperature factor increases power at cold temps."""
        result_mild = effect_manager.estimate_power_from_compressor(
            compressor_hz=50, outdoor_temp=10.0
        )
        result_cold = effect_manager.estimate_power_from_compressor(
            compressor_hz=50, outdoor_temp=COMPRESSOR_TEMP_EXTREME_COLD_THRESHOLD - 5.0
        )

        assert result_cold > result_mild

    def test_power_clamped_to_max(self, effect_manager):
        """Test power is clamped to maximum."""
        result = effect_manager.estimate_power_from_compressor(
            compressor_hz=100,  # Above normal range
            outdoor_temp=COMPRESSOR_TEMP_EXTREME_COLD_THRESHOLD - 10.0,  # Very cold
        )

        # Should be clamped before temp factor
        assert result <= COMPRESSOR_POWER_MAX_KW * COMPRESSOR_TEMP_FACTOR_EXTREME_COLD


class TestThermalDebtStatusEdgeCases:
    """Edge case tests for thermal debt status."""

    def test_exactly_at_block_threshold(self):
        """Test exactly at block threshold."""
        dm_thresholds = {"block": -700, "abort": -1500}
        result = get_thermal_debt_status(-700, dm_thresholds)

        # Exactly at threshold - should be WARNING (below block)
        assert "WARNING" in result or "OK" in result

    def test_exactly_at_abort_threshold(self):
        """Test exactly at abort threshold."""
        dm_thresholds = {"block": -700, "abort": -1500}
        result = get_thermal_debt_status(-1500, dm_thresholds)

        # Should be WARNING (at abort threshold, not past it)
        # The condition is < abort, so -1500 is NOT past -1500
        assert "WARNING" in result or "CRITICAL" in result

    def test_positive_thermal_debt(self):
        """Test positive thermal debt (rare but possible)."""
        dm_thresholds = {"block": -700, "abort": -1500}
        result = get_thermal_debt_status(100, dm_thresholds)

        assert "OK" in result
        assert "800" in result  # 100 - (-700) = 800 margin


class TestDHWOptimizerCalculateRecommendation:
    """Tests for IntelligentDHWScheduler.calculate_recommendation()."""

    @pytest.fixture
    def scheduler(self):
        """Create a basic DHW scheduler instance."""
        return create_dhw_scheduler()

    def test_returns_dhw_recommendation_dataclass(self, scheduler):
        """Test that calculate_recommendation returns DHWRecommendation."""
        from datetime import datetime

        from custom_components.effektguard.optimization.dhw_optimizer import DHWRecommendation

        result = scheduler.calculate_recommendation(
            current_dhw_temp=45.0,
            thermal_debt=-200,
            space_heating_demand=1.5,
            outdoor_temp=5.0,
            indoor_temp=21.5,
            target_indoor=21.0,
            price_classification="normal",
            current_time=datetime.now(),
            price_periods=None,
            hours_since_last_dhw=12.0,
        )

        assert isinstance(result, DHWRecommendation)
        assert result.recommendation
        assert result.summary
        assert isinstance(result.details, dict)

    def test_blocks_dhw_when_indoor_cooling_rapidly(self, scheduler):
        """Test DHW is blocked when indoor temp is cooling rapidly."""
        from datetime import datetime

        result = scheduler.calculate_recommendation(
            current_dhw_temp=45.0,
            thermal_debt=-200,
            space_heating_demand=2.0,
            outdoor_temp=0.0,
            indoor_temp=20.5,  # Below target
            target_indoor=21.0,
            price_classification="cheap",
            current_time=datetime.now(),
            price_periods=None,
            hours_since_last_dhw=8.0,
            thermal_trend_rate=-0.5,  # Cooling rapidly
        )

        assert "Block" in result.recommendation or "INDOOR_COOLING" in result.details.get(
            "priority_reason", ""
        )
        assert result.details["should_heat"] is False

    def test_includes_thermal_debt_status_in_details(self, scheduler):
        """Test thermal debt status is included in details."""
        from datetime import datetime

        result = scheduler.calculate_recommendation(
            current_dhw_temp=45.0,
            thermal_debt=-200,
            space_heating_demand=1.0,
            outdoor_temp=5.0,
            indoor_temp=21.5,
            target_indoor=21.0,
            price_classification="normal",
            current_time=datetime.now(),
            price_periods=None,
            hours_since_last_dhw=12.0,
        )

        assert "thermal_debt_status" in result.details
        assert "OK" in result.details["thermal_debt_status"]

    def test_includes_climate_zone_in_details(self, scheduler):
        """Test climate zone is included in details."""
        from datetime import datetime

        result = scheduler.calculate_recommendation(
            current_dhw_temp=45.0,
            thermal_debt=-200,
            space_heating_demand=1.0,
            outdoor_temp=5.0,
            indoor_temp=21.5,
            target_indoor=21.0,
            price_classification="normal",
            current_time=datetime.now(),
            price_periods=None,
            hours_since_last_dhw=12.0,
            climate_zone_name="Nordic",
        )

        assert result.details.get("climate_zone") == "Nordic"

    def test_formats_planning_summary(self, scheduler):
        """Test planning summary is properly formatted."""
        from datetime import datetime

        result = scheduler.calculate_recommendation(
            current_dhw_temp=42.0,
            thermal_debt=-200,
            space_heating_demand=1.5,
            outdoor_temp=5.0,
            indoor_temp=21.5,
            target_indoor=21.0,
            price_classification="CHEAP",
            current_time=datetime.now(),
            price_periods=None,
            hours_since_last_dhw=12.0,
        )

        assert "DHW Planning Summary" in result.summary
        assert "42.0°C" in result.summary
        assert "CHEAP" in result.summary


class TestEstimateDmRecoveryTime:
    """Tests for shared estimate_dm_recovery_time function.

    Phase 10: Moved from dhw_optimizer._estimate_dm_recovery_time to thermal_layer
    for shared reuse between DHW and space heating.
    """

    def test_recovery_time_basic(self):
        """Test basic DM recovery estimation."""
        from custom_components.effektguard.optimization.thermal_layer import (
            estimate_dm_recovery_time,
        )

        # 100 DM deficit at mild temp (>5°C)
        # Rate is 40 DM/h at mild, so 100/40 = 2.5h
        result = estimate_dm_recovery_time(
            current_dm=-400,
            target_dm=-300,
            outdoor_temp=10.0,
        )

        # Should be around 2.5 hours at mild conditions
        assert 1.0 < result < 5.0

    def test_recovery_time_cold_weather(self):
        """Test slower recovery in cold weather."""
        from custom_components.effektguard.optimization.thermal_layer import (
            estimate_dm_recovery_time,
        )

        # Same deficit at different temperatures
        mild_result = estimate_dm_recovery_time(-400, -300, outdoor_temp=10.0)
        cold_result = estimate_dm_recovery_time(-400, -300, outdoor_temp=2.0)
        very_cold_result = estimate_dm_recovery_time(-400, -300, outdoor_temp=-5.0)

        # Recovery should be slower in colder weather
        assert very_cold_result > cold_result
        assert cold_result > mild_result

    def test_recovery_time_minimum_clamping(self):
        """Test minimum recovery time clamping at 0.5h."""
        from custom_components.effektguard.optimization.thermal_layer import (
            estimate_dm_recovery_time,
        )

        # Very small deficit that would calculate to <0.5h
        result = estimate_dm_recovery_time(
            current_dm=-100,
            target_dm=-99,  # 1 DM deficit
            outdoor_temp=10.0,
        )

        # Should be clamped to minimum 0.5h
        assert result >= 0.5

    def test_recovery_time_maximum_clamping(self):
        """Test maximum recovery time clamping at 12h."""
        from custom_components.effektguard.optimization.thermal_layer import (
            estimate_dm_recovery_time,
        )
        from custom_components.effektguard.const import DM_RECOVERY_MAX_HOURS

        # Very large deficit at very cold temp
        result = estimate_dm_recovery_time(
            current_dm=-1500,
            target_dm=-200,  # 1300 DM deficit
            outdoor_temp=-20.0,
        )

        # Should be clamped to maximum
        assert result <= DM_RECOVERY_MAX_HOURS


class TestDhwOptimizerEmergencyLayerIntegration:
    """Tests for DHW optimizer using shared EmergencyLayer.

    Phase 10: DHW optimizer now accepts optional emergency_layer parameter
    to use shared thermal debt blocking logic.
    """

    @pytest.fixture
    def emergency_layer(self):
        """Create an EmergencyLayer for testing."""
        from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer
        from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector

        climate_detector = ClimateZoneDetector(latitude=59.33)  # Stockholm
        return EmergencyLayer(
            climate_detector=climate_detector,
            price_analyzer=None,
            heating_type="radiator",
        )

    def test_scheduler_accepts_emergency_layer(self, emergency_layer):
        """Test that scheduler accepts emergency_layer parameter."""
        from custom_components.effektguard.optimization.dhw_optimizer import (
            IntelligentDHWScheduler,
        )

        scheduler = IntelligentDHWScheduler(
            emergency_layer=emergency_layer,
        )

        assert scheduler.emergency_layer is emergency_layer

    def test_scheduler_uses_emergency_layer_for_blocking(self, emergency_layer):
        """Test that scheduler uses emergency_layer.should_block_dhw()."""
        from datetime import datetime

        from custom_components.effektguard.optimization.dhw_optimizer import (
            IntelligentDHWScheduler,
        )

        scheduler = IntelligentDHWScheduler(
            emergency_layer=emergency_layer,
        )

        # Test with thermal debt that should be blocked
        # EmergencyLayer blocks at T2 threshold (warning - 200 DM margin)
        result = scheduler.should_start_dhw(
            current_dhw_temp=45.0,
            space_heating_demand_kw=1.0,
            thermal_debt_dm=-1400,  # Deep thermal debt, should be blocked
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=0.0,
            price_classification="normal",
            current_time=datetime.now(),
        )

        # Should be blocked due to thermal debt
        assert result.should_heat is False
        assert result.priority_reason == "CRITICAL_THERMAL_DEBT"

    def test_scheduler_delegates_recovery_time_estimation(self):
        """Test that _estimate_dm_recovery_time delegates to shared function."""
        from custom_components.effektguard.optimization.thermal_layer import (
            estimate_dm_recovery_time,
        )

        scheduler = create_dhw_scheduler()

        # Both should return the same result
        scheduler_result = scheduler._estimate_dm_recovery_time(
            current_dm=-400,
            target_dm=-300,
            outdoor_temp=5.0,
        )
        shared_result = estimate_dm_recovery_time(
            current_dm=-400,
            target_dm=-300,
            outdoor_temp=5.0,
        )

        assert scheduler_result == shared_result
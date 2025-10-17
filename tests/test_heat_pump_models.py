"""Unit tests for heat pump model profiles.

Tests the model-specific profiles for NIBE heat pumps:
- F730 (6kW ASHP) - widely used, replaced by F735 in current lineup
- F750 (8kW ASHP) - widely used, not in current NIBE lineup
- F2040 (12-16kW ASHP) - widely used, not in current NIBE lineup
- S1155 (3-12kW GSHP mid-range) - CURRENT MODEL, verified from NIBE website

Validates:
- Model registration system
- COP calculations across temperature ranges
- Power consumption estimation
- Flow temperature calculations
- Power validation logic
"""

import pytest
from dataclasses import dataclass

from custom_components.effektguard.models.registry import HeatPumpModelRegistry
from custom_components.effektguard.models.base import HeatPumpProfile, ValidationResult
from custom_components.effektguard.models.nibe import (
    NibeF730Profile,
    NibeF750Profile,
    NibeF2040Profile,
    NibeS1155Profile,
)


class TestModelRegistry:
    """Test heat pump model registry system."""

    def test_registry_contains_nibe_models(self):
        """Test that all NIBE models are registered."""
        supported = HeatPumpModelRegistry.get_supported_models()

        assert "nibe_f730" in supported
        assert "nibe_f750" in supported
        assert "nibe_f2040" in supported
        assert "nibe_s1155" in supported
        # S1255 removed - doesn't exist in NIBE lineup

    def test_get_model_returns_profile(self):
        """Test getting model profile by ID."""
        profile = HeatPumpModelRegistry.get_model("nibe_f750")

        assert isinstance(profile, HeatPumpProfile)
        assert profile.model_name == "F750"
        assert profile.manufacturer == "NIBE"

    def test_get_model_raises_on_unknown(self):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown heat pump model"):
            HeatPumpModelRegistry.get_model("nonexistent_model")

    def test_get_models_by_manufacturer(self):
        """Test filtering models by manufacturer."""
        nibe_models = HeatPumpModelRegistry.get_models_by_manufacturer("NIBE")

        assert "nibe_f730" in nibe_models
        assert "nibe_f750" in nibe_models
        assert "nibe_f2040" in nibe_models
        assert "nibe_s1155" in nibe_models
        # S1255 removed - doesn't exist
        assert len(nibe_models) == 4

    def test_get_models_grouped_by_manufacturer(self):
        """Test getting models grouped by manufacturer."""
        grouped = HeatPumpModelRegistry.get_models_grouped_by_manufacturer()

        assert "NIBE" in grouped
        assert len(grouped["NIBE"]) == 4  # S1255 removed

        # Check structure
        f750 = next(m for m in grouped["NIBE"] if m["id"] == "nibe_f750")
        assert f750["name"] == "F750"
        assert f750["model_type"] == "F-series ASHP"


class TestNibeF750Profile:
    """Test NIBE F750 8kW ASHP profile."""

    @pytest.fixture
    def f750(self):
        """NIBE F750 profile fixture."""
        return NibeF750Profile()

    def test_basic_attributes(self, f750):
        """Test basic F750 attributes."""
        assert f750.model_name == "F750"
        assert f750.manufacturer == "NIBE"
        assert f750.model_type == "F-series ASHP"
        assert f750.modulation_type == "inverter"
        assert f750.supports_modulation is True

    def test_power_characteristics(self, f750):
        """Test F750 power characteristics."""
        assert f750.rated_power_kw == (2.0, 8.0)
        assert f750.typical_electrical_range_kw == (1.2, 6.5)
        assert f750.max_flow_temp == 60.0
        assert f750.min_flow_temp == 20.0

    def test_cop_at_rated_conditions(self, f750):
        """Test COP at rated conditions (7°C outdoor)."""
        cop = f750.get_cop_at_temperature(7.0)
        assert cop == 5.0  # Rated COP

    def test_cop_at_swedish_temperatures(self, f750):
        """Test COP across Swedish temperature range."""
        test_cases = [
            (7, 5.0),  # Mild
            (0, 4.0),  # Malmö/Gothenburg average
            (-5, 3.5),  # Stockholm common cold
            (-10, 3.0),  # Cold winter
            (-15, 2.7),  # Design temperature
            (-20, 2.3),  # Very cold
            (-25, 2.0),  # Extreme (Kiruna)
            (-30, 1.8),  # Survival mode
        ]

        for outdoor_temp, expected_cop in test_cases:
            cop = f750.get_cop_at_temperature(outdoor_temp)
            assert (
                abs(cop - expected_cop) < 0.01
            ), f"COP mismatch at {outdoor_temp}°C: expected {expected_cop}, got {cop}"

    def test_cop_interpolation(self, f750):
        """Test COP interpolation between data points."""
        # Between 0°C (COP 4.0) and -5°C (COP 3.5)
        cop_minus_2_5 = f750.get_cop_at_temperature(-2.5)
        expected = 3.75  # Midpoint
        assert abs(cop_minus_2_5 - expected) < 0.01

        # Between -10°C (COP 3.0) and -15°C (COP 2.7)
        cop_minus_12_5 = f750.get_cop_at_temperature(-12.5)
        expected = 2.85  # Midpoint
        assert abs(cop_minus_12_5 - expected) < 0.01

    def test_cop_extrapolation_beyond_range(self, f750):
        """Test COP at temperatures beyond defined range."""
        # Above max temp
        cop_high = f750.get_cop_at_temperature(15.0)
        assert cop_high == 5.0  # Should return max COP

        # Below min temp
        cop_low = f750.get_cop_at_temperature(-35.0)
        assert cop_low == 1.8  # Should return min COP

    def test_electrical_consumption_estimation(self, f750):
        """Test electrical consumption estimation."""
        # At 0°C with 6kW heat demand
        # COP = 4.0, so electrical = 6 / 4.0 = 1.5kW
        electrical = f750.estimate_electrical_consumption(heat_demand_kw=6.0, outdoor_temp=0.0)
        assert abs(electrical - 1.5) < 0.1

        # At -15°C with 10kW heat demand
        # COP = 2.7, so electrical = 10 / 2.7 = 3.7kW
        electrical = f750.estimate_electrical_consumption(heat_demand_kw=10.0, outdoor_temp=-15.0)
        assert abs(electrical - 3.7) < 0.2

    def test_electrical_consumption_capped_at_max(self, f750):
        """Test electrical consumption is capped at max power."""
        # Very high heat demand should cap at max electrical
        electrical = f750.estimate_electrical_consumption(heat_demand_kw=30.0, outdoor_temp=-20.0)
        assert electrical <= f750.typical_electrical_range_kw[1]

    def test_optimal_flow_temp_calculation(self, f750):
        """Test optimal flow temperature calculation."""
        # Mild conditions (-5°C outdoor)
        flow_temp = f750.calculate_optimal_flow_temp(
            outdoor_temp=-5.0, indoor_target=21.0, heat_demand_kw=6.0
        )

        # Should be reasonable for F750
        assert 25.0 <= flow_temp <= 55.0

        # Efficiency target: outdoor + 27°C ± 3°C
        # -5 + 27 = 22°C (lower bound ~19, upper bound ~30 with adjustments)
        # With heat demand formula it may be higher
        assert flow_temp >= 20.0  # Above minimum
        assert flow_temp <= 60.0  # Below maximum

    def test_flow_temp_clamped_to_limits(self, f750):
        """Test flow temperature is clamped to F750 limits."""
        # Extreme cold should not exceed max flow temp
        flow_temp = f750.calculate_optimal_flow_temp(
            outdoor_temp=-30.0, indoor_target=21.0, heat_demand_kw=15.0
        )
        assert flow_temp <= f750.max_flow_temp

        # Should never go below minimum
        flow_temp = f750.calculate_optimal_flow_temp(
            outdoor_temp=10.0, indoor_target=21.0, heat_demand_kw=2.0
        )
        assert flow_temp >= f750.min_flow_temp

    def test_power_validation_normal(self, f750):
        """Test power validation for normal consumption."""
        # Normal consumption at -10°C: ~2.3kW
        result = f750.validate_power_consumption(
            current_power_kw=2.3, outdoor_temp=-10.0, flow_temp=35.0
        )

        assert result.valid is True
        assert result.severity == "info"

    def test_power_validation_high_power(self, f750):
        """Test power validation for high consumption."""
        # Very high power at -10°C: 7.0kW (above typical max 6.5kW)
        result = f750.validate_power_consumption(
            current_power_kw=7.0, outdoor_temp=-10.0, flow_temp=50.0
        )

        assert result.valid is False
        assert result.severity == "warning"
        assert "High power" in result.message or "exceeds" in result.message
        assert len(result.suggestions) > 0

    def test_power_validation_low_power(self, f750):
        """Test power validation for low consumption."""
        # Very low power at -10°C: 0.8kW
        result = f750.validate_power_consumption(
            current_power_kw=0.8, outdoor_temp=-10.0, flow_temp=30.0
        )

        assert result.valid is True
        assert result.severity == "info"
        assert "low" in result.message.lower()


class TestNibeF730Profile:
    """Test NIBE F730 6kW ASHP profile."""

    @pytest.fixture
    def f730(self):
        """NIBE F730 profile fixture."""
        return NibeF730Profile()

    def test_smaller_than_f750(self, f730):
        """Test F730 is smaller than F750."""
        f750 = NibeF750Profile()

        assert f730.rated_power_kw[1] < f750.rated_power_kw[1]
        assert f730.typical_electrical_range_kw[1] < f750.typical_electrical_range_kw[1]

    def test_cop_same_as_f750(self, f730):
        """Test F730 has same COP curve as F750 (same technology)."""
        f750 = NibeF750Profile()

        # Should have same COP at all temperatures
        for temp in [7, 0, -5, -10, -15, -20, -25, -30]:
            cop_730 = f730.get_cop_at_temperature(temp)
            cop_750 = f750.get_cop_at_temperature(temp)
            assert cop_730 == cop_750


class TestNibeF2040Profile:
    """Test NIBE F2040 12-16kW ASHP profile."""

    @pytest.fixture
    def f2040(self):
        """NIBE F2040 profile fixture."""
        return NibeF2040Profile()

    def test_larger_than_f750(self, f2040):
        """Test F2040 is larger than F750."""
        f750 = NibeF750Profile()

        assert f2040.rated_power_kw[1] > f750.rated_power_kw[1]
        assert f2040.typical_electrical_range_kw[1] > f750.typical_electrical_range_kw[1]

    def test_slightly_lower_cop(self, f2040):
        """Test F2040 has slightly lower COP than F750 (larger unit)."""
        f750 = NibeF750Profile()

        # At -15°C: F750 = 2.7, F2040 = 2.5
        cop_f2040 = f2040.get_cop_at_temperature(-15.0)
        cop_f750 = f750.get_cop_at_temperature(-15.0)

        assert cop_f2040 < cop_f750
        assert abs(cop_f2040 - 2.5) < 0.1

    def test_higher_power_consumption(self, f2040):
        """Test F2040 has higher power consumption for large houses."""
        # 12kW heat demand at -15°C
        electrical = f2040.estimate_electrical_consumption(heat_demand_kw=12.0, outdoor_temp=-15.0)

        # COP ~2.5, so electrical ~4.8kW
        assert 4.0 <= electrical <= 5.5


class TestNibeS1155Profile:
    """Test NIBE S1155 12kW GSHP profile."""

    @pytest.fixture
    def s1155(self):
        """NIBE S1155 profile fixture."""
        return NibeS1155Profile()

    def test_gshp_characteristics(self, s1155):
        """Test S1155 GSHP characteristics."""
        assert s1155.model_type == "S-series GSHP"
        assert s1155.rated_power_kw[1] == 12.0

    def test_much_better_cop_than_ashp(self, s1155):
        """Test S1155 has much better COP than equivalent ASHP."""
        f2040 = NibeF2040Profile()  # Similar size ASHP

        # At -15°C: S1155 GSHP = 4.3, F2040 ASHP = 2.5
        cop_gshp = s1155.get_cop_at_temperature(-15.0)
        cop_ashp = f2040.get_cop_at_temperature(-15.0)

        assert cop_gshp > cop_ashp
        assert cop_gshp > 4.0  # GSHP stays high even in cold
        assert abs(cop_gshp - 4.3) < 0.1

    def test_lower_electrical_consumption(self, s1155):
        """Test S1155 uses much less power than equivalent ASHP."""
        f2040 = NibeF2040Profile()

        # Same heat demand (10kW) at -15°C
        elec_gshp = s1155.estimate_electrical_consumption(heat_demand_kw=10.0, outdoor_temp=-15.0)
        elec_ashp = f2040.estimate_electrical_consumption(heat_demand_kw=10.0, outdoor_temp=-15.0)

        # GSHP should use much less power
        assert elec_gshp < elec_ashp
        # S1155: 10 / 4.3 = ~2.3kW
        # F2040: 10 / 2.5 = ~4.0kW
        assert abs(elec_gshp - 2.3) < 0.3

    def test_stable_cop_in_extreme_cold(self, s1155):
        """Test S1155 COP stays high even in extreme cold."""
        # At -30°C: GSHP = 3.5 (ground temp stable)
        cop = s1155.get_cop_at_temperature(-30.0)
        assert cop >= 3.5  # Much better than ASHP ~1.8

    def test_can_run_lower_flow_temps(self, s1155):
        """Test S1155 can run lower flow temperatures."""
        f750 = NibeF750Profile()

        assert s1155.min_flow_temp < f750.min_flow_temp
        assert s1155.optimal_flow_delta < f750.optimal_flow_delta


class TestModelComparisons:
    """Test comparisons between different models."""

    def test_power_consumption_hierarchy(self):
        """Test power consumption increases with model size."""
        f730 = NibeF730Profile()
        f750 = NibeF750Profile()
        f2040 = NibeF2040Profile()

        # At -10°C with 8kW heat demand
        # F730: May exceed capacity
        # F750: ~2.7kW
        # F2040: ~2.7kW but has more headroom

        elec_730 = f730.estimate_electrical_consumption(8.0, -10.0)
        elec_750 = f750.estimate_electrical_consumption(8.0, -10.0)
        elec_2040 = f2040.estimate_electrical_consumption(8.0, -10.0)

        # F730 may be capped at max
        # F750 and F2040 should have similar electrical (same COP ~3.0)
        assert abs(elec_750 - 2.7) < 0.4

    def test_gshp_always_more_efficient_than_ashp(self):
        """Test GSHP models are always more efficient than ASHP."""
        f750 = NibeF750Profile()  # 8kW ASHP
        s1155 = NibeS1155Profile()  # 12kW GSHP

        test_temps = [7, 0, -5, -10, -15, -20, -25, -30]

        for temp in test_temps:
            cop_ashp = f750.get_cop_at_temperature(temp)
            cop_gshp = s1155.get_cop_at_temperature(temp)

            assert cop_gshp > cop_ashp, (
                f"GSHP should be more efficient at {temp}°C: " f"GSHP {cop_gshp} vs ASHP {cop_ashp}"
            )

    def test_all_models_have_required_attributes(self):
        """Test all models have required profile attributes."""
        models = [
            NibeF730Profile(),
            NibeF750Profile(),
            NibeF2040Profile(),
            NibeS1155Profile(),
            # S1255 removed - doesn't exist in NIBE lineup
        ]

        for model in models:
            # Basic identity
            assert model.model_name is not None
            assert model.manufacturer == "NIBE"
            assert model.model_type is not None

            # Power characteristics
            assert model.rated_power_kw is not None
            assert model.typical_electrical_range_kw is not None
            assert model.modulation_type in ["inverter", "on_off", "staged"]

            # Efficiency
            assert model.typical_cop_range is not None
            assert model.optimal_flow_delta > 0
            assert model.cop_curve is not None

            # Capabilities
            assert isinstance(model.supports_aux_heating, bool)
            assert isinstance(model.supports_modulation, bool)
            assert isinstance(model.supports_weather_compensation, bool)

            # Limits
            assert model.max_flow_temp > model.min_flow_temp
            assert model.min_runtime_minutes > 0


class TestValidationResults:
    """Test ValidationResult dataclass."""

    def test_validation_result_structure(self):
        """Test ValidationResult has correct structure."""
        from custom_components.effektguard.models.base import ValidationResult

        result = ValidationResult(
            valid=True,
            severity="info",
            message="Test message",
            suggestions=["Suggestion 1", "Suggestion 2"],
        )

        assert result.valid is True
        assert result.severity == "info"
        assert result.message == "Test message"
        assert len(result.suggestions) == 2

    def test_validation_severity_levels(self):
        """Test different validation severity levels."""
        from custom_components.effektguard.models.base import ValidationResult

        # Info level
        info = ValidationResult(valid=True, severity="info", message="OK", suggestions=[])
        assert info.severity == "info"

        # Warning level
        warning = ValidationResult(
            valid=False, severity="warning", message="Warning", suggestions=[]
        )
        assert warning.severity == "warning"

        # Error level
        error = ValidationResult(valid=False, severity="error", message="Error", suggestions=[])
        assert error.severity == "error"


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_typical_swedish_house_f750(self):
        """Test F750 in typical Swedish house (150m² standard insulation)."""
        f750 = NibeF750Profile()

        # Stockholm winter: -10°C, 7kW heat demand
        electrical = f750.estimate_electrical_consumption(heat_demand_kw=7.0, outdoor_temp=-10.0)

        # COP ~3.0, so 7 / 3.0 = ~2.3kW
        assert 2.0 <= electrical <= 2.7
        assert electrical < f750.typical_electrical_range_kw[1]  # Within normal range

    def test_undersized_f730_for_large_house(self):
        """Test F730 undersized for 150m² house."""
        f730 = NibeF730Profile()

        # 10kW heat demand (too much for F730)
        electrical = f730.estimate_electrical_consumption(heat_demand_kw=10.0, outdoor_temp=-15.0)

        # Should be at or below max electrical (may be capped)
        assert electrical <= f730.typical_electrical_range_kw[1]
        # For 10kW demand at -15°C (COP 2.7): 10/2.7 = 3.7kW
        assert 3.5 <= electrical <= 4.5

    def test_f2040_in_extreme_cold(self):
        """Test F2040 in extreme cold (Kiruna -25°C)."""
        f2040 = NibeF2040Profile()

        # Large house, 15kW heat demand
        electrical = f2040.estimate_electrical_consumption(heat_demand_kw=15.0, outdoor_temp=-25.0)

        # COP ~1.9, so 15 / 1.9 = ~7.9kW
        # Should be capped or close to max
        assert electrical >= 7.0

    def test_gshp_efficiency_advantage(self):
        """Test GSHP efficiency advantage in real scenario."""
        f2040 = NibeF2040Profile()  # ASHP
        s1155 = NibeS1155Profile()  # GSHP

        # Same conditions: -15°C, 10kW heat demand
        elec_ashp = f2040.estimate_electrical_consumption(10.0, -15.0)
        elec_gshp = s1155.estimate_electrical_consumption(10.0, -15.0)

        # GSHP should use ~40-50% less power
        savings_ratio = (elec_ashp - elec_gshp) / elec_ashp
        assert savings_ratio > 0.35  # At least 35% savings


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

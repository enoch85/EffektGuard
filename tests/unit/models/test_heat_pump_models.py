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
        assert "nibe_f1155" in nibe_models
        assert "nibe_f2040" in nibe_models
        assert "nibe_s1155" in nibe_models
        # S1255 removed - doesn't exist
        assert len(nibe_models) == 5

    def test_get_models_grouped_by_manufacturer(self):
        """Test getting models grouped by manufacturer."""
        grouped = HeatPumpModelRegistry.get_models_grouped_by_manufacturer()

        assert "NIBE" in grouped
        assert len(grouped["NIBE"]) == 5  # F1155 added for issue #18

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
        """The F750's published maximum output is 4.994 kW (EN 14511, part no. 066 063), not 8.0.

        It is an exhaust-air pump: output is bounded by the house's ventilation air, so the old
        (2.0, 8.0) kW rating was invented.
        """
        assert f750.rated_power_kw == (1.144, 4.994)
        assert f750.max_heat_output_kw == 4.994
        assert f750.max_flow_temp == 60.0

    def test_cop_matches_the_datasheet(self, f750):
        """It used to assert COP 5.0 "at rated conditions (7 C outdoor)". Both halves were wrong.

        The number 5.0 appears nowhere in the F750's datasheet, and "7 C outdoor" is not a condition
        this machine is rated at. Its three published points are all A20(12) - twenty-degree extract
        air - and the outdoor air never touches its evaporator.
        """
        published = {p.condition: (p.heat_output_kw, p.cop) for p in f750.datasheet_points}

        assert published == {
            "A20(12)W35, exhaust air flow 108 m3/h (30 l/s) min compressor frequency": (
                1.144,
                4.20,
            ),
            "A20(12)W35, exhaust air flow 252 m3/h (70 l/s) min compressor frequency": (
                1.498,
                4.72,
            ),
            "A20(12)W45, exhaust air flow 252 m3/h (70 l/s) max compressor frequency": (
                4.994,
                2.43,
            ),
        }
        assert f750.typical_cop_range == (2.43, 4.72)

    def test_the_display_curve_is_labelled_as_a_proxy_not_a_measurement(self, f750):
        """The outdoor-keyed COP curve is a dashboard proxy derived from the published endpoints.

        It used to assert a fabricated 8-point table (COP 5.0 at 7 C down to 1.8 at -30 C) as fact,
        keyed on a variable this exhaust-air machine does not respond to. Nothing computes from it -
        the simulator's physics comes from `datasheet_points`.
        """
        curve = f750.cop_curve

        assert max(curve.values()) == pytest.approx(4.72), "the best published COP, min freq at W35"
        assert min(curve.values()) == pytest.approx(2.43), "the worst, max freq at W45"
        assert 5.0 not in curve.values(), "COP 5.0 is not a figure NIBE publishes for this machine"

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

    def test_the_f730_is_not_a_smaller_f750(self, f730):
        """At A20(12)W45 max frequency NIBE publishes 5.35 kW (F730) vs 4.994 kW (F750).

        The F730 is the stronger machine at full tilt; the old "F730 < F750" ordering was invented.
        """
        f750 = NibeF750Profile()

        assert f730.max_heat_output_kw == 5.35
        assert f750.max_heat_output_kw == 4.994
        assert f730.max_heat_output_kw > f750.max_heat_output_kw

    def test_the_f730_does_not_share_the_f750s_cop_curve(self, f730):
        """Two different machines must not carry byte-identical COP curves.

        The old `test_cop_same_as_f750` demanded they stay equal ("same technology"). NIBE publishes
        COP 5.32 for the F730 at A20(12)W35 min frequency, and 4.72 for the F750.
        """
        f750 = NibeF750Profile()

        assert f730.typical_cop_range == (2.43, 5.32)
        assert f750.typical_cop_range == (2.43, 4.72)
        assert f730.cop_curve != f750.cop_curve


class TestNibeF2040Profile:
    """Test NIBE F2040 12-16kW ASHP profile."""

    @pytest.fixture
    def f2040(self):
        """NIBE F2040 profile fixture."""
        return NibeF2040Profile()

    def test_the_f2040_is_an_air_source_machine_and_the_only_one(self, f2040):
        """It is the ONLY profile here whose heat source really is the outdoor air.

        Which makes it the only one for which an outdoor-keyed COP curve was ever meaningful - and
        its curve is now the datasheet's own W35 rows, not a template.
        """
        assert f2040.cop_curve == {7: 4.65, 2: 3.76, -7: 2.68}
        assert f2040.max_flow_temp == 58.0, "the datasheet says 58, the profile used to say 63"
        assert not f2040.supports_aux_heating, (
            "The F2040 is an outdoor monobloc and has NO immersion heater - its technical "
            "specifications table has no such row. The profile claimed 'True # Larger immersion "
            "heaters'. The electric addition lives in the paired indoor module."
        )

    def test_its_capacity_rises_as_the_weather_cools(self, f2040):
        """It does not derate. It ramps up. The simulator had this backwards and cited EN 14511.

        NIBE publishes 3.86 kW at A7/W35 and 6.60 kW at A-7/W35. An inverter is throttled back at
        its mild rating point and opens up as the load arrives. What collapses in the cold is the
        COP - 4.65 to 2.68 - and not the capacity.
        """
        by_source = {p.source_temp_c: p for p in f2040.datasheet_points if p.flow_temp_c == 35.0}

        assert by_source[-7.0].heat_output_kw > by_source[7.0].heat_output_kw
        assert by_source[-7.0].cop < by_source[7.0].cop


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

    def test_can_run_lower_flow_temps(self, s1155):
        """Test S1155 can run lower flow temperatures."""
        f750 = NibeF750Profile()

        assert s1155.min_flow_temp < f750.min_flow_temp
        assert s1155.optimal_flow_delta < f750.optimal_flow_delta


class TestModelComparisons:
    """Test comparisons between different models."""

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
            # The pump's own factory aux-start, per installer manual - the simulator's plant
            # fires the elpatron here. Must sit above EffektGuard's -1500 emergency floor.
            assert model.aux_start_dm > -1500


class TestValidationResults:
    """Test ValidationResult dataclass."""

    def test_validation_result_structure(self):
        """Test ValidationResult has correct structure."""

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

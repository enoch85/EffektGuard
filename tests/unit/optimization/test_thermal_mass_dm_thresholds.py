"""Tests for thermal mass-aware DM threshold adjustments.

Validates that high thermal mass systems (concrete slab UFH) get tighter DM thresholds
to prevent v0.1.0 solar gain overshoot problem.

Test Categories:
1. Multiplier application (concrete 1.3×, timber 1.15×, radiator 1.0×)
2. Critical threshold preservation (always -1500)
3. Real-world scenario prevention (v0.1.0 failure mode)
4. Climate zone integration (thermal mass × climate awareness)
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from custom_components.effektguard.const import (
    DM_THERMAL_MASS_BUFFER_CONCRETE,
    DM_THERMAL_MASS_BUFFER_TIMBER,
    DM_THERMAL_MASS_BUFFER_RADIATOR,
    DM_THRESHOLD_AUX_LIMIT,
)
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.adapters.nibe_adapter import NibeState


@pytest.fixture
def base_config():
    """Base configuration for decision engine."""
    return {
        "target_indoor_temp": 21.5,
        "tolerance": 0.5,
        "latitude": 59.33,  # Stockholm
        "enable_weather_compensation": True,
        "weather_compensation_weight": 0.49,
    }


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for decision engine."""
    price_analyzer = MagicMock()
    effect_manager = MagicMock()
    thermal_model = MagicMock()
    return price_analyzer, effect_manager, thermal_model


class TestThermalMassMultipliers:
    """Test thermal mass buffer multipliers are applied correctly."""

    def test_concrete_slab_30_percent_tighter(self, base_config, mock_dependencies):
        """Concrete slab should get 1.3× tighter thresholds (30% more conservative)."""
        base_config["heating_type"] = "concrete_ufh"
        engine = DecisionEngine(*mock_dependencies, base_config)

        # Stockholm at 10°C: base warning ~-276
        base_thresholds = engine.climate_detector.get_expected_dm_range(outdoor_temp=10.0)
        base_warning = base_thresholds["warning"]

        # Apply thermal mass adjustment
        adjusted = engine._get_thermal_mass_adjusted_thresholds(base_thresholds, "concrete_ufh")

        # Should be 30% tighter (more negative)
        expected_warning = base_warning * DM_THERMAL_MASS_BUFFER_CONCRETE
        assert adjusted["warning"] == pytest.approx(expected_warning, abs=1)
        assert adjusted["warning"] < base_warning  # More negative = tighter

        # Verify ~30% tighter
        tightening_factor = adjusted["warning"] / base_warning
        assert tightening_factor == pytest.approx(1.3, abs=0.01)

    def test_timber_15_percent_tighter(self, base_config, mock_dependencies):
        """Timber UFH should get 1.15× tighter thresholds (15% more conservative)."""
        base_config["heating_type"] = "timber"
        engine = DecisionEngine(*mock_dependencies, base_config)

        base_thresholds = engine.climate_detector.get_expected_dm_range(outdoor_temp=10.0)
        base_warning = base_thresholds["warning"]

        adjusted = engine._get_thermal_mass_adjusted_thresholds(base_thresholds, "timber")

        # Should be 15% tighter
        expected_warning = base_warning * DM_THERMAL_MASS_BUFFER_TIMBER
        assert adjusted["warning"] == pytest.approx(expected_warning, abs=1)

        # Verify ~15% tighter
        tightening_factor = adjusted["warning"] / base_warning
        assert tightening_factor == pytest.approx(1.15, abs=0.01)

    def test_radiator_standard_thresholds(self, base_config, mock_dependencies):
        """Radiators should keep standard thresholds (1.0× = no adjustment)."""
        base_config["heating_type"] = "radiator"
        engine = DecisionEngine(*mock_dependencies, base_config)

        base_thresholds = engine.climate_detector.get_expected_dm_range(outdoor_temp=10.0)
        base_warning = base_thresholds["warning"]

        adjusted = engine._get_thermal_mass_adjusted_thresholds(base_thresholds, "radiator")

        # Should be identical (1.0× multiplier)
        assert adjusted["warning"] == pytest.approx(base_warning, abs=1)

        tightening_factor = adjusted["warning"] / base_warning
        assert tightening_factor == pytest.approx(1.0, abs=0.01)

    def test_unknown_type_defaults_to_radiator(self, base_config, mock_dependencies):
        """Unknown heating type should default to radiator (1.0× = safe default)."""
        base_config["heating_type"] = "unknown_type"
        engine = DecisionEngine(*mock_dependencies, base_config)

        base_thresholds = engine.climate_detector.get_expected_dm_range(outdoor_temp=10.0)
        adjusted = engine._get_thermal_mass_adjusted_thresholds(base_thresholds, "unknown_type")

        # Should use radiator default (1.0×)
        assert adjusted["warning"] == pytest.approx(base_thresholds["warning"], abs=1)


class TestCriticalThresholdPreservation:
    """Test that critical threshold (-1500) is never adjusted."""

    def test_concrete_preserves_critical_1500(self, base_config, mock_dependencies):
        """Concrete slab should keep -1500 critical threshold (absolute maximum)."""
        base_config["heating_type"] = "concrete_ufh"
        engine = DecisionEngine(*mock_dependencies, base_config)

        base_thresholds = engine.climate_detector.get_expected_dm_range(outdoor_temp=-30.0)
        adjusted = engine._get_thermal_mass_adjusted_thresholds(base_thresholds, "concrete_ufh")

        # Critical should always be -1500
        assert adjusted["critical"] == DM_THRESHOLD_AUX_LIMIT
        assert adjusted["critical"] == -1500

    def test_all_types_preserve_critical(self, base_config, mock_dependencies):
        """All heating types should keep -1500 critical threshold."""
        engine = DecisionEngine(*mock_dependencies, base_config)

        for heating_type in ["concrete_ufh", "timber", "radiator", "unknown"]:
            base_thresholds = engine.climate_detector.get_expected_dm_range(outdoor_temp=0.0)
            adjusted = engine._get_thermal_mass_adjusted_thresholds(base_thresholds, heating_type)

            assert adjusted["critical"] == DM_THRESHOLD_AUX_LIMIT
            assert adjusted["critical"] == -1500


class TestRealWorldScenarioPrevention:
    """Test that thermal mass buffer prevents v0.1.0 failure mode."""

    def test_prevents_v010_dm_700_overshoot(self, base_config, mock_dependencies):
        """Verify thermal mass buffer prevents DM -700 during solar gain (v0.1.0 failure)."""
        # v0.1.0 scenario: Stockholm at 10°C, indoor rising from solar gain
        # Base warning: -276, v0.1.0 allowed DM to deepen to -700
        # Result: 1.5°C temperature drop at sunset
        base_config["heating_type"] = "concrete_ufh"
        base_config["latitude"] = 59.33  # Stockholm
        engine = DecisionEngine(*mock_dependencies, base_config)

        # Get base and adjusted thresholds for Stockholm at 10°C
        base_thresholds = engine.climate_detector.get_expected_dm_range(outdoor_temp=10.0)
        base_warning = base_thresholds["warning"]  # ~-276

        adjusted = engine._get_thermal_mass_adjusted_thresholds(base_thresholds, "concrete_ufh")
        adjusted_warning = adjusted["warning"]  # ~-359 (30% tighter)

        # v0.1.0 allowed DM -700 (beyond base warning -276)
        v010_dm = -700

        # With thermal mass buffer, system should intervene much earlier
        # Adjusted warning should be significantly tighter than base
        assert adjusted_warning < base_warning  # More negative = tighter
        assert adjusted_warning > v010_dm  # Still not as extreme as -700

        # Verify concrete gets meaningful tightening
        # Should activate T1 recovery ~-359 instead of allowing -700
        tightening = abs(adjusted_warning - base_warning)
        assert tightening > 50  # At least 50 DM tighter

    def test_concrete_activates_t1_earlier_than_radiator(self, base_config, mock_dependencies):
        """Concrete slab should activate T1 recovery earlier than radiators."""
        engine = DecisionEngine(*mock_dependencies, base_config)

        outdoor_temp = 10.0
        base_thresholds = engine.climate_detector.get_expected_dm_range(outdoor_temp)

        concrete_thresholds = engine._get_thermal_mass_adjusted_thresholds(
            base_thresholds, "concrete_ufh"
        )
        radiator_thresholds = engine._get_thermal_mass_adjusted_thresholds(
            base_thresholds, "radiator"
        )

        # Concrete should have tighter (more negative) warning threshold
        assert concrete_thresholds["warning"] < radiator_thresholds["warning"]

        # Concrete activates T1 at DM -359, radiator at DM -276 (Stockholm 10°C example)
        gap = concrete_thresholds["warning"] - radiator_thresholds["warning"]
        assert gap < -50  # At least 50 DM earlier intervention for concrete


class TestClimateZoneIntegration:
    """Test thermal mass buffer works with climate-aware thresholds."""

    def test_arctic_concrete_vs_mild_concrete(self, base_config, mock_dependencies):
        """Concrete in Arctic should have different thresholds than mild climate."""
        base_config["heating_type"] = "concrete_ufh"

        # Test Arctic (Kiruna) at -30°C
        base_config["latitude"] = 67.85  # Kiruna
        arctic_engine = DecisionEngine(*mock_dependencies, base_config.copy())
        arctic_base = arctic_engine.climate_detector.get_expected_dm_range(outdoor_temp=-30.0)
        arctic_adjusted = arctic_engine._get_thermal_mass_adjusted_thresholds(
            arctic_base, "concrete_ufh"
        )

        # Test Mild (Paris) at 5°C
        base_config["latitude"] = 48.85  # Paris
        mild_engine = DecisionEngine(*mock_dependencies, base_config.copy())
        mild_base = mild_engine.climate_detector.get_expected_dm_range(outdoor_temp=5.0)
        mild_adjusted = mild_engine._get_thermal_mass_adjusted_thresholds(mild_base, "concrete_ufh")

        # Arctic should have much deeper thresholds than mild climate
        assert arctic_adjusted["warning"] < mild_adjusted["warning"]  # More negative = deeper

        # Both should apply 1.3× multiplier
        arctic_factor = arctic_adjusted["warning"] / arctic_base["warning"]
        mild_factor = mild_adjusted["warning"] / mild_base["warning"]
        assert arctic_factor == pytest.approx(1.3, abs=0.01)
        assert mild_factor == pytest.approx(1.3, abs=0.01)

    def test_all_climates_preserve_multiplier_ratio(self, base_config, mock_dependencies):
        """All climate zones should apply same thermal mass multiplier."""
        base_config["heating_type"] = "concrete_ufh"

        test_climates = [
            (67.85, -30.0, "Kiruna (Extreme Cold)"),
            (59.33, -10.0, "Stockholm (Cold)"),
            (55.68, 0.0, "Copenhagen (Moderate Cold)"),
            (48.85, 5.0, "Paris (Moderate)"),
        ]

        for latitude, outdoor_temp, name in test_climates:
            base_config["latitude"] = latitude
            engine = DecisionEngine(*mock_dependencies, base_config.copy())

            base_thresholds = engine.climate_detector.get_expected_dm_range(outdoor_temp)
            adjusted = engine._get_thermal_mass_adjusted_thresholds(base_thresholds, "concrete_ufh")

            # All should apply 1.3× multiplier
            factor = adjusted["warning"] / base_thresholds["warning"]
            assert factor == pytest.approx(1.3, abs=0.01), f"{name} failed to apply 1.3× multiplier"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_alternative_concrete_naming(self, base_config, mock_dependencies):
        """Test alternative concrete UFH naming conventions."""
        engine = DecisionEngine(*mock_dependencies, base_config)
        base_thresholds = engine.climate_detector.get_expected_dm_range(outdoor_temp=10.0)

        # Both should apply concrete multiplier
        for name in ["concrete_ufh", "concrete_slab"]:
            adjusted = engine._get_thermal_mass_adjusted_thresholds(base_thresholds, name)
            factor = adjusted["warning"] / base_thresholds["warning"]
            assert factor == pytest.approx(1.3, abs=0.01)

    def test_alternative_timber_naming(self, base_config, mock_dependencies):
        """Test alternative timber UFH naming conventions."""
        engine = DecisionEngine(*mock_dependencies, base_config)
        base_thresholds = engine.climate_detector.get_expected_dm_range(outdoor_temp=10.0)

        # Both should apply timber multiplier
        for name in ["timber", "timber_ufh"]:
            adjusted = engine._get_thermal_mass_adjusted_thresholds(base_thresholds, name)
            factor = adjusted["warning"] / base_thresholds["warning"]
            assert factor == pytest.approx(1.15, abs=0.01)

    def test_empty_string_defaults_to_radiator(self, base_config, mock_dependencies):
        """Empty heating type should default to radiator (safe default)."""
        engine = DecisionEngine(*mock_dependencies, base_config)
        base_thresholds = engine.climate_detector.get_expected_dm_range(outdoor_temp=10.0)

        adjusted = engine._get_thermal_mass_adjusted_thresholds(base_thresholds, "")

        # Should apply radiator multiplier (1.0×)
        assert adjusted["warning"] == pytest.approx(base_thresholds["warning"], abs=1)

    def test_normal_min_and_max_both_adjusted(self, base_config, mock_dependencies):
        """Both normal_min and normal_max should be adjusted, not just warning."""
        base_config["heating_type"] = "concrete_ufh"
        engine = DecisionEngine(*mock_dependencies, base_config)

        base_thresholds = engine.climate_detector.get_expected_dm_range(outdoor_temp=10.0)
        adjusted = engine._get_thermal_mass_adjusted_thresholds(base_thresholds, "concrete_ufh")

        # All thresholds except critical should be adjusted
        min_factor = adjusted["normal_min"] / base_thresholds["normal_min"]
        max_factor = adjusted["normal_max"] / base_thresholds["normal_max"]
        warning_factor = adjusted["warning"] / base_thresholds["warning"]

        assert min_factor == pytest.approx(1.3, abs=0.01)
        assert max_factor == pytest.approx(1.3, abs=0.01)
        assert warning_factor == pytest.approx(1.3, abs=0.01)

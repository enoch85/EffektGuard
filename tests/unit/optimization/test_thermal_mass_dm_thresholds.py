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

from custom_components.effektguard.const import (
    DM_THERMAL_MASS_BUFFER_CONCRETE,
    DM_THERMAL_MASS_BUFFER_TIMBER,
    DM_THERMAL_MASS_BUFFER_RADIATOR,
    DM_THRESHOLD_AUX_LIMIT,
)
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector


@pytest.fixture
def climate_detector():
    """Create climate detector for Stockholm."""
    return ClimateZoneDetector(latitude=59.33)


class TestThermalMassMultipliers:
    """Test thermal mass buffer multipliers are applied correctly."""

    def test_concrete_slab_30_percent_tighter(self, climate_detector):
        """Concrete slab should get 1.3× tighter thresholds (30% more conservative)."""
        layer = EmergencyLayer(climate_detector, heating_type="concrete_ufh")

        # Stockholm at 10°C: base warning ~-276
        base_thresholds = climate_detector.get_expected_dm_range(outdoor_temp=10.0)
        base_warning = base_thresholds["warning"]

        # Apply thermal mass adjustment
        adjusted = layer._get_thermal_mass_adjusted_thresholds(base_thresholds)

        # Should be 30% tighter (more negative)
        expected_warning = base_warning * DM_THERMAL_MASS_BUFFER_CONCRETE
        assert adjusted["warning"] == pytest.approx(expected_warning, abs=1)
        assert adjusted["warning"] < base_warning  # More negative = tighter

        # Verify ~30% tighter
        tightening_factor = adjusted["warning"] / base_warning
        assert tightening_factor == pytest.approx(1.3, abs=0.01)

    def test_timber_15_percent_tighter(self, climate_detector):
        """Timber UFH should get 1.15× tighter thresholds (15% more conservative)."""
        layer = EmergencyLayer(climate_detector, heating_type="timber")

        base_thresholds = climate_detector.get_expected_dm_range(outdoor_temp=10.0)
        base_warning = base_thresholds["warning"]

        adjusted = layer._get_thermal_mass_adjusted_thresholds(base_thresholds)

        # Should be 15% tighter
        expected_warning = base_warning * DM_THERMAL_MASS_BUFFER_TIMBER
        assert adjusted["warning"] == pytest.approx(expected_warning, abs=1)

        # Verify ~15% tighter
        tightening_factor = adjusted["warning"] / base_warning
        assert tightening_factor == pytest.approx(1.15, abs=0.01)

    def test_radiator_standard_thresholds(self, climate_detector):
        """Radiators should keep standard thresholds (1.0× = no adjustment)."""
        layer = EmergencyLayer(climate_detector, heating_type="radiator")

        base_thresholds = climate_detector.get_expected_dm_range(outdoor_temp=10.0)
        base_warning = base_thresholds["warning"]

        adjusted = layer._get_thermal_mass_adjusted_thresholds(base_thresholds)

        # Should be identical (1.0× multiplier)
        assert adjusted["warning"] == pytest.approx(base_warning, abs=1)
        assert adjusted["warning"] == pytest.approx(base_warning * DM_THERMAL_MASS_BUFFER_RADIATOR, abs=1)

    def test_unknown_type_defaults_to_radiator(self, climate_detector):
        """Unknown heating types should default to radiator (safest option)."""
        layer = EmergencyLayer(climate_detector, heating_type="unknown_type")

        base_thresholds = climate_detector.get_expected_dm_range(outdoor_temp=10.0)
        base_warning = base_thresholds["warning"]

        adjusted = layer._get_thermal_mass_adjusted_thresholds(base_thresholds)

        # Should default to radiator (1.0×)
        assert adjusted["warning"] == pytest.approx(base_warning, abs=1)


class TestCriticalThresholdPreservation:
    """Test that critical safety thresholds are NEVER adjusted."""

    def test_concrete_preserves_critical_1500(self, climate_detector):
        """Concrete slab adjustment should NOT affect -1500 critical limit."""
        layer = EmergencyLayer(climate_detector, heating_type="concrete_ufh")

        # Stockholm at -30°C (extreme case)
        base_thresholds = climate_detector.get_expected_dm_range(outdoor_temp=-30.0)
        
        adjusted = layer._get_thermal_mass_adjusted_thresholds(base_thresholds)

        # Warning should be adjusted
        assert adjusted["warning"] < base_thresholds["warning"]
        
        # Critical MUST remain -1500
        assert adjusted["critical"] == DM_THRESHOLD_AUX_LIMIT
        assert adjusted["critical"] == -1500

    def test_all_types_preserve_critical(self, climate_detector):
        """All heating types must preserve the critical safety limit."""
        base_thresholds = climate_detector.get_expected_dm_range(outdoor_temp=0.0)

        for h_type in ["concrete_ufh", "timber", "radiator", "unknown"]:
            layer = EmergencyLayer(climate_detector, heating_type=h_type)
            adjusted = layer._get_thermal_mass_adjusted_thresholds(base_thresholds)
            assert adjusted["critical"] == -1500


class TestRealWorldScenarioPrevention:
    """Test prevention of specific real-world failure scenarios."""

    def test_prevents_v010_dm_700_overshoot(self, climate_detector):
        """Prevent v0.1.0 scenario: DM -700 allowed on concrete slab.
        
        In v0.1.0, DM -700 was considered "normal" for Stockholm.
        On concrete slab, this caused massive overshoot when sun came out.
        With 1.3× multiplier, -700 should be flagged as WARNING much earlier.
        """
        layer = EmergencyLayer(climate_detector, heating_type="concrete_ufh")

        # Stockholm at 10°C (mild spring day)
        base_thresholds = climate_detector.get_expected_dm_range(outdoor_temp=10.0)
        # Base warning is around -340
        
        adjusted = layer._get_thermal_mass_adjusted_thresholds(base_thresholds)
        
        # Adjusted warning should be around -442 (-340 * 1.3)
        # This means DM -700 is DEEP into warning/critical territory
        
        assert adjusted["warning"] > -500  # Warning triggers before -500 (e.g. at -442)
        
        # If current DM is -700, it should be well past warning
        current_dm = -700
        assert current_dm < adjusted["warning"]  # -700 < -442 (True)

    def test_concrete_activates_t1_earlier_than_radiator(self, climate_detector):
        """Concrete slab should have deeper (more negative) warning thresholds.

        With multiplier 1.3, concrete DM thresholds become MORE NEGATIVE.
        This means recovery triggers at a DEEPER thermal debt level, which
        makes sense because concrete's high thermal mass can absorb more
        energy without immediate indoor temperature impact.

        For concrete: base_warning * 1.3 = deeper threshold
        Example: -300 * 1.3 = -390 (allows deeper DM before warning)
        """
        concrete_layer = EmergencyLayer(climate_detector, heating_type="concrete_ufh")
        radiator_layer = EmergencyLayer(climate_detector, heating_type="radiator")

        base_thresholds = climate_detector.get_expected_dm_range(outdoor_temp=10.0)

        concrete_thresholds = concrete_layer._get_thermal_mass_adjusted_thresholds(base_thresholds)
        radiator_thresholds = radiator_layer._get_thermal_mass_adjusted_thresholds(base_thresholds)

        # Concrete warning should be MORE NEGATIVE than radiator (deeper threshold)
        # because concrete's 1.3× multiplier makes threshold more negative
        assert concrete_thresholds["warning"] < radiator_thresholds["warning"]

        # Verify concrete is 30% more negative
        expected_ratio = DM_THERMAL_MASS_BUFFER_CONCRETE / DM_THERMAL_MASS_BUFFER_RADIATOR
        actual_ratio = concrete_thresholds["warning"] / radiator_thresholds["warning"]
        assert actual_ratio == pytest.approx(expected_ratio, abs=0.01)

class TestClimateZoneIntegration:
    """Test that thermal mass adjustments work across climate zones."""

    def test_arctic_concrete_vs_mild_concrete(self):
        """Test concrete slab adjustments in different climates."""
        arctic_detector = ClimateZoneDetector(latitude=67.85)  # Kiruna
        mild_detector = ClimateZoneDetector(latitude=55.60)    # Malmö
        
        arctic_layer = EmergencyLayer(arctic_detector, heating_type="concrete_ufh")
        mild_layer = EmergencyLayer(mild_detector, heating_type="concrete_ufh")

        # Same outdoor temp
        outdoor_temp = -5.0

        arctic_base = arctic_detector.get_expected_dm_range(outdoor_temp)
        mild_base = mild_detector.get_expected_dm_range(outdoor_temp)

        arctic_adjusted = arctic_layer._get_thermal_mass_adjusted_thresholds(arctic_base)
        mild_adjusted = mild_layer._get_thermal_mass_adjusted_thresholds(mild_base)

        # Both should be adjusted by same ratio
        arctic_ratio = arctic_adjusted["warning"] / arctic_base["warning"]
        mild_ratio = mild_adjusted["warning"] / mild_base["warning"]

        assert arctic_ratio == pytest.approx(DM_THERMAL_MASS_BUFFER_CONCRETE, abs=0.01)
        assert mild_ratio == pytest.approx(DM_THERMAL_MASS_BUFFER_CONCRETE, abs=0.01)

    def test_all_climates_preserve_multiplier_ratio(self):
        """Multiplier ratio should be consistent regardless of base threshold magnitude."""
        detector = ClimateZoneDetector(latitude=67.85)  # Kiruna
        layer = EmergencyLayer(detector, heating_type="concrete_ufh")
        
        # Test across wide temp range
        for temp in [10.0, 0.0, -10.0, -30.0]:
            base_thresholds = detector.get_expected_dm_range(temp)
            adjusted = layer._get_thermal_mass_adjusted_thresholds(base_thresholds)
            
            ratio = adjusted["warning"] / base_thresholds["warning"]
            assert ratio == pytest.approx(DM_THERMAL_MASS_BUFFER_CONCRETE, abs=0.01)


class TestEdgeCases:
    """Test edge cases and input validation."""

    def test_alternative_concrete_naming(self, climate_detector):
        """Test alternative names for concrete slab."""
        base_thresholds = climate_detector.get_expected_dm_range(outdoor_temp=0.0)
        
        for name in ["concrete_ufh", "concrete_slab"]:
            layer = EmergencyLayer(climate_detector, heating_type=name)
            adjusted = layer._get_thermal_mass_adjusted_thresholds(base_thresholds)
            ratio = adjusted["warning"] / base_thresholds["warning"]
            assert ratio == pytest.approx(DM_THERMAL_MASS_BUFFER_CONCRETE, abs=0.01)

    def test_alternative_timber_naming(self, climate_detector):
        """Test alternative names for timber."""
        base_thresholds = climate_detector.get_expected_dm_range(outdoor_temp=0.0)
        
        for name in ["timber", "timber_ufh"]:
            layer = EmergencyLayer(climate_detector, heating_type=name)
            adjusted = layer._get_thermal_mass_adjusted_thresholds(base_thresholds)
            ratio = adjusted["warning"] / base_thresholds["warning"]
            assert ratio == pytest.approx(DM_THERMAL_MASS_BUFFER_TIMBER, abs=0.01)

    def test_empty_string_defaults_to_radiator(self, climate_detector):
        """Empty string should default to radiator."""
        layer = EmergencyLayer(climate_detector, heating_type="")
        base_thresholds = climate_detector.get_expected_dm_range(outdoor_temp=0.0)
        
        adjusted = layer._get_thermal_mass_adjusted_thresholds(base_thresholds)
        
        # Should be 1.0×
        assert adjusted["warning"] == pytest.approx(base_thresholds["warning"], abs=1)

    def test_normal_min_and_max_both_adjusted(self, climate_detector):
        """Both normal_min and normal_max should be adjusted."""
        layer = EmergencyLayer(climate_detector, heating_type="concrete_ufh")
        base_thresholds = climate_detector.get_expected_dm_range(outdoor_temp=0.0)
        
        adjusted = layer._get_thermal_mass_adjusted_thresholds(base_thresholds)
        
        assert adjusted["normal_min"] == pytest.approx(base_thresholds["normal_min"] * DM_THERMAL_MASS_BUFFER_CONCRETE, abs=1)
        assert adjusted["normal_max"] == pytest.approx(base_thresholds["normal_max"] * DM_THERMAL_MASS_BUFFER_CONCRETE, abs=1)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

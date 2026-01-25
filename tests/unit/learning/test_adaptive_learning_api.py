"""Tests for AdaptiveThermalModel API compatibility with ThermalModel.

Tests the insulation_quality property that bridges the old ThermalModel API
with the new adaptive learning approach.
"""

from custom_components.effektguard.optimization.adaptive_learning import (
    AdaptiveThermalModel,
)


class TestInsulationQualityProperty:
    """Test insulation_quality property getter and setter."""

    def test_default_insulation_quality(self):
        """Test insulation_quality returns default 1.0 before learning."""
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Before learning, should return default 1.0
        assert model.insulation_quality == 1.0

    def test_insulation_quality_setter(self):
        """Test insulation_quality setter stores heat loss coefficient."""
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Set insulation quality to better than normal
        model.insulation_quality = 1.5

        # Should convert to heat loss coefficient
        # 1.5 → 180/1.5 = 120 W/°C
        assert "heat_loss_coefficient" in model.learned_parameters
        assert abs(model.learned_parameters["heat_loss_coefficient"] - 120.0) < 1.0

    def test_insulation_quality_setter_poor_insulation(self):
        """Test setting poor insulation quality."""
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Set to poor insulation
        model.insulation_quality = 0.5

        # Should convert to high heat loss
        # 0.5 → 180/0.5 = 360 W/°C
        assert abs(model.learned_parameters["heat_loss_coefficient"] - 360.0) < 1.0

    def test_insulation_quality_setter_excellent_insulation(self):
        """Test setting excellent insulation quality."""
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Set to excellent insulation
        model.insulation_quality = 2.0

        # Should convert to low heat loss
        # 2.0 → 180/2.0 = 90 W/°C
        assert abs(model.learned_parameters["heat_loss_coefficient"] - 90.0) < 1.0

    def test_insulation_quality_roundtrip(self):
        """Test setting and getting insulation quality returns same value."""
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Set various values
        test_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

        for value in test_values:
            model.insulation_quality = value

            # Getter should return same value (with small tolerance)
            retrieved = model.insulation_quality
            assert abs(retrieved - value) < 0.01, f"Expected {value}, got {retrieved}"

    def test_insulation_quality_clamping(self):
        """Test insulation quality is clamped to valid range."""
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Set value outside range (too low)
        model.insulation_quality = 0.3
        assert 0.5 <= model.insulation_quality <= 2.0

        # Set value outside range (too high)
        model.insulation_quality = 3.0
        assert 0.5 <= model.insulation_quality <= 2.0


class TestLearnedParameterOverride:
    """Test that learned parameters override manual settings."""

    def test_learned_parameters_override_manual(self):
        """Test learned parameters override initial insulation quality."""
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Set initial value
        model.insulation_quality = 1.0
        assert model.insulation_quality == 1.0

        # Simulate learning better insulation with high confidence
        model.learned_parameters = {
            "heat_loss_coefficient": 150.0,  # Better than 180
            "confidence": 0.85,  # High confidence
            "thermal_mass": 1.2,
            "heating_efficiency": 0.42,
            "thermal_decay_rate": -0.08,
        }

        # Should use learned value (180/150 = 1.2)
        learned_quality = model.insulation_quality
        assert learned_quality > 1.0
        assert learned_quality < 1.3
        assert abs(learned_quality - 1.2) < 0.01

    def test_insufficient_confidence_uses_default(self):
        """Test low confidence learned parameters still return their value.

        Note: Unlike other learned operations, insulation_quality property
        returns the value from learned_parameters regardless of confidence.
        This allows manual configuration to work via the setter.
        """
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Set learned parameters but with low confidence
        model.learned_parameters = {
            "heat_loss_coefficient": 150.0,  # Would give 180/150 = 1.2
            "confidence": 0.40,  # Below threshold (0.7)
            "thermal_mass": 1.2,
            "heating_efficiency": 0.42,
            "thermal_decay_rate": -0.08,
        }

        # Should return the value from heat_loss_coefficient (1.2)
        # Even with low confidence, because manual config needs to work
        assert abs(model.insulation_quality - 1.2) < 0.01

    def test_learned_parameters_within_valid_range(self):
        """Test learned values are clamped to valid range."""
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Simulate learning very low heat loss (excellent insulation)
        # Should be clamped to max 2.0 quality
        model.learned_parameters = {
            "heat_loss_coefficient": 50.0,  # Would give 180/50 = 3.6
            "confidence": 0.85,
            "thermal_mass": 1.2,
            "heating_efficiency": 0.42,
            "thermal_decay_rate": -0.08,
        }

        quality = model.insulation_quality
        assert quality <= 2.0

        # Simulate learning very high heat loss (poor insulation)
        # Should be clamped to min 0.5 quality
        model.learned_parameters = {
            "heat_loss_coefficient": 400.0,  # Would give 180/400 = 0.45
            "confidence": 0.85,
            "thermal_mass": 1.2,
            "heating_efficiency": 0.42,
            "thermal_decay_rate": -0.08,
        }

        quality = model.insulation_quality
        assert quality >= 0.5


class TestAPICompatibility:
    """Test API compatibility with ThermalModel."""

    def test_same_api_as_thermal_model(self):
        """Test AdaptiveThermalModel has same API as ThermalModel."""
        from custom_components.effektguard.optimization.thermal_layer import ThermalModel

        adaptive_model = AdaptiveThermalModel(initial_thermal_mass=1.2)
        thermal_model = ThermalModel(thermal_mass=1.2, insulation_quality=1.5)

        # Both should have thermal_mass attribute
        assert hasattr(adaptive_model, "thermal_mass")
        assert hasattr(thermal_model, "thermal_mass")

        # Both should have insulation_quality attribute/property
        assert hasattr(adaptive_model, "insulation_quality")
        assert hasattr(thermal_model, "insulation_quality")

        # Both should be readable
        _ = adaptive_model.thermal_mass
        _ = adaptive_model.insulation_quality
        _ = thermal_model.thermal_mass
        _ = thermal_model.insulation_quality

        # Both should be writable
        adaptive_model.thermal_mass = 1.5
        adaptive_model.insulation_quality = 1.2
        thermal_model.thermal_mass = 1.5
        thermal_model.insulation_quality = 1.2

    def test_config_reload_pattern(self):
        """Test config reload pattern works with AdaptiveThermalModel.

        Simulates what coordinator.py:1913 does during config reload.
        """
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Simulate user changing config
        new_options = {
            "thermal_mass": 1.5,
            "insulation_quality": 1.8,
        }

        # Config reload sets these attributes
        model.thermal_mass = new_options["thermal_mass"]
        model.insulation_quality = new_options["insulation_quality"]

        # Verify values are set correctly
        assert model.thermal_mass == 1.5
        assert abs(model.insulation_quality - 1.8) < 0.01

    def test_decision_engine_pattern(self):
        """Test decision engine pattern works with AdaptiveThermalModel.

        Simulates what decision_engine.py:968 does in prediction layer.
        """
        model = AdaptiveThermalModel(initial_thermal_mass=1.2)
        model.insulation_quality = 1.5

        # Decision engine reads these attributes
        thermal_mass = model.thermal_mass
        insulation_quality = model.insulation_quality

        # Should not raise AttributeError
        assert thermal_mass == 1.2
        assert abs(insulation_quality - 1.5) < 0.01


class TestConversionFormulas:
    """Test heat loss coefficient to insulation quality conversion."""

    def test_baseline_conversion(self):
        """Test baseline 180 W/°C = 1.0 quality."""
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        model.learned_parameters = {
            "heat_loss_coefficient": 180.0,
            "confidence": 0.85,
            "thermal_mass": 1.0,
            "heating_efficiency": 0.42,
            "thermal_decay_rate": -0.08,
        }

        assert abs(model.insulation_quality - 1.0) < 0.01

    def test_poor_insulation_conversion(self):
        """Test high heat loss = poor insulation quality."""
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # 360 W/°C → 0.5 quality
        model.learned_parameters = {
            "heat_loss_coefficient": 360.0,
            "confidence": 0.85,
            "thermal_mass": 1.0,
            "heating_efficiency": 0.42,
            "thermal_decay_rate": -0.08,
        }

        assert abs(model.insulation_quality - 0.5) < 0.01

    def test_excellent_insulation_conversion(self):
        """Test low heat loss = excellent insulation quality."""
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # 90 W/°C → 2.0 quality
        model.learned_parameters = {
            "heat_loss_coefficient": 90.0,
            "confidence": 0.85,
            "thermal_mass": 1.0,
            "heating_efficiency": 0.42,
            "thermal_decay_rate": -0.08,
        }

        assert abs(model.insulation_quality - 2.0) < 0.01


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_confidence(self):
        """Test zero confidence still returns value from learned_parameters.

        Manual configuration sets values in learned_parameters,
        so we return those even with zero confidence.
        """
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        model.learned_parameters = {
            "heat_loss_coefficient": 150.0,  # Would give 180/150 = 1.2
            "confidence": 0.0,
            "thermal_mass": 1.0,
            "heating_efficiency": 0.42,
            "thermal_decay_rate": -0.08,
        }

        # Should return value from heat_loss_coefficient
        assert abs(model.insulation_quality - 1.2) < 0.01

    def test_missing_learned_parameters(self):
        """Test missing learned parameters uses default."""
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Empty learned parameters
        model.learned_parameters = {}

        assert model.insulation_quality == 1.0

    def test_none_learned_parameters(self):
        """Test None learned parameters uses default."""
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Set to None (shouldn't happen but handle gracefully)
        model.learned_parameters = None

        assert model.insulation_quality == 1.0

    def test_very_low_heat_loss_clamped(self):
        """Test very low heat loss is clamped to max quality."""
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Extreme case: 10 W/°C (unrealistic but test clamping)
        model.learned_parameters = {
            "heat_loss_coefficient": 10.0,
            "confidence": 0.85,
            "thermal_mass": 1.0,
            "heating_efficiency": 0.42,
            "thermal_decay_rate": -0.08,
        }

        quality = model.insulation_quality
        assert quality == 2.0  # Should be clamped to max

    def test_very_high_heat_loss_clamped(self):
        """Test very high heat loss is clamped to min quality."""
        model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Extreme case: 1000 W/°C (unrealistic but test clamping)
        model.learned_parameters = {
            "heat_loss_coefficient": 1000.0,
            "confidence": 0.85,
            "thermal_mass": 1.0,
            "heating_efficiency": 0.42,
            "thermal_decay_rate": -0.08,
        }

        quality = model.insulation_quality
        assert quality == 0.5  # Should be clamped to min

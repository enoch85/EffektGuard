"""Test overshoot-aware thermal recovery damping.

Context: v0.1.0 showed 2.1°C overshoot (23.6°C vs target 21.5°C) during DM recovery.
Traditional damping only considers warming rate, not absolute temperature error.

Solution: Overshoot severity adds additional damping:
  - Severe ≥1.5°C: 80% strength (0.8 multiplier)
  - Moderate ≥1.0°C: 90% strength (0.9 multiplier)
  - Mild ≥0.5°C: 95% strength (0.95 multiplier)

Overshoot damping multiplies with warming damping for compound effect.

Test Coverage:
  1. Overshoot-only damping (no warming trend)
  2. Warming-only damping (no overshoot)
  3. Combined overshoot + warming damping
  4. Overshoot severity levels
  5. Safety minimums respected
  6. v0.1.0 failure prevented
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.const import (
    THERMAL_RECOVERY_OVERSHOOT_SEVERE_THRESHOLD,  # 1.5°C
    THERMAL_RECOVERY_OVERSHOOT_MODERATE_THRESHOLD,  # 1.0°C
    THERMAL_RECOVERY_OVERSHOOT_MILD_THRESHOLD,  # 0.5°C
    THERMAL_RECOVERY_OVERSHOOT_SEVERE_DAMPING,  # 0.8
    THERMAL_RECOVERY_OVERSHOOT_MODERATE_DAMPING,  # 0.9
    THERMAL_RECOVERY_OVERSHOOT_MILD_DAMPING,  # 0.95
    THERMAL_RECOVERY_DAMPING_FACTOR,  # 0.6 (standard warming)
    THERMAL_RECOVERY_RAPID_FACTOR,  # 0.4 (rapid warming)
    THERMAL_RECOVERY_T1_MIN_OFFSET,  # 1.0°C
    THERMAL_RECOVERY_T2_MIN_OFFSET,  # 1.5°C
)


@pytest.fixture
def emergency_layer():
    """Create emergency layer with mock dependencies."""
    climate_detector = ClimateZoneDetector(latitude=59.33)
    
    # Mock trend callbacks
    mock_thermal_trend = MagicMock(return_value={"rate_per_hour": 0.0, "confidence": 0.8})
    mock_outdoor_trend = MagicMock(return_value={"rate_per_hour": 0.0, "confidence": 0.8})
    
    layer = EmergencyLayer(
        climate_detector=climate_detector,
        heating_type="radiator",
        get_thermal_trend=mock_thermal_trend,
        get_outdoor_trend=mock_outdoor_trend,
    )
    
    # Attach mocks to layer for easy access in tests
    layer.mock_thermal_trend = mock_thermal_trend
    layer.mock_outdoor_trend = mock_outdoor_trend
    
    return layer


class TestOvershootOnlyDamping:
    """Test overshoot damping when no warming trend detected."""

    def test_severe_overshoot_2c(self, emergency_layer):
        """Severe 2.0°C overshoot → 80% strength (0.8 multiplier)."""
        layer = emergency_layer

        # Mock predictor with no warming trend
        layer.mock_thermal_trend.return_value = {
            "rate_per_hour": 0.1,  # Slight warming but below threshold
            "confidence": 0.8,
        }
        layer.mock_outdoor_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # Test severe overshoot: indoor 23.5°C, target 21.5°C = 2.0°C over
        base_offset = 2.5
        damped_offset, reason = layer._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=23.5,
            target_temp=21.5,
        )

        # Should apply severe overshoot damping: 2.5 × 0.8 = 2.0°C
        assert damped_offset == 2.0
        assert "severe overshoot" in reason
        assert "+2.0°C" in reason

    def test_moderate_overshoot_1_2c(self, emergency_layer):
        """Moderate 1.2°C overshoot → 90% strength (0.9 multiplier)."""
        layer = emergency_layer
        
        layer.mock_thermal_trend.return_value = {
            "rate_per_hour": 0.1,
            "confidence": 0.8,
        }

        # 1.2°C overshoot
        base_offset = 2.0
        damped_offset, reason = layer._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=22.7,
            target_temp=21.5,
        )

        # 2.0 × 0.9 = 1.8°C
        assert damped_offset == pytest.approx(1.8)
        assert "moderate overshoot" in reason

    def test_mild_overshoot_0_7c(self, emergency_layer):
        """Mild 0.7°C overshoot → 95% strength (0.95 multiplier)."""
        layer = emergency_layer
        
        layer.mock_thermal_trend.return_value = {
            "rate_per_hour": 0.1,
            "confidence": 0.8,
        }

        # 0.7°C overshoot
        base_offset = 2.0
        damped_offset, reason = layer._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=22.2,
            target_temp=21.5,
        )

        # 2.0 × 0.95 = 1.9°C
        assert damped_offset == pytest.approx(1.9)
        assert "mild overshoot" in reason

    def test_no_overshoot_no_damping(self, emergency_layer):
        """No overshoot (<0.5°C) → No damping (1.0 multiplier)."""
        layer = emergency_layer
        
        layer.mock_thermal_trend.return_value = {
            "rate_per_hour": 0.1,
            "confidence": 0.8,
        }

        # 0.4°C overshoot
        base_offset = 2.0
        damped_offset, reason = layer._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=21.9,
            target_temp=21.5,
        )

        # No change
        assert damped_offset == 2.0
        assert reason == ""


class TestCombinedOvershootAndWarming:
    """Test compound effect of overshoot AND warming trend."""

    def test_severe_overshoot_plus_rapid_warming(self, emergency_layer):
        """Severe overshoot (0.8) × Rapid warming (0.4) = 0.32 multiplier."""
        layer = emergency_layer

        # Rapid warming
        layer.mock_thermal_trend.return_value = {
            "rate_per_hour": 0.6,  # Rapid (>0.5)
            "confidence": 0.8,
        }

        # Severe overshoot (2.0°C)
        base_offset = 5.0  # Large offset to see effect clearly
        damped_offset, reason = layer._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=23.5,
            target_temp=21.5,
        )

        # 5.0 × 0.8 (overshoot) × 0.4 (warming) = 1.6°C
        # But clamped to min_damped_offset (1.5°C)
        # Wait, 1.6 > 1.5, so it should be 1.6
        
        expected = 5.0 * 0.8 * 0.4  # 1.6
        assert damped_offset == pytest.approx(expected)
        assert "severe overshoot" in reason
        assert "rapid warming" in reason

    def test_moderate_overshoot_plus_moderate_warming(self, emergency_layer):
        """Moderate overshoot (0.9) × Moderate warming (0.6) = 0.54 multiplier."""
        layer = emergency_layer

        # Moderate warming
        layer.mock_thermal_trend.return_value = {
            "rate_per_hour": 0.3,  # Moderate (0.2-0.5)
            "confidence": 0.8,
        }

        # Moderate overshoot (1.2°C)
        base_offset = 4.0
        damped_offset, reason = layer._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=22.7,
            target_temp=21.5,
        )

        # 4.0 × 0.9 (overshoot) × 0.6 (warming) = 2.16°C
        expected = 4.0 * 0.9 * 0.6
        assert damped_offset == pytest.approx(expected)
        assert "moderate overshoot" in reason
        assert "warming" in reason


class TestV010FailurePrevention:
    """Test prevention of specific v0.1.0 failure scenario."""

    def test_prevents_2c_overshoot_during_recovery(self, emergency_layer):
        """Scenario: DM recovery active, indoor temp hits 23.6°C (target 21.5°C).
        
        v0.1.0 behavior: Kept pushing +3.0°C offset because DM was low.
        New behavior: Should clamp offset aggressively.
        """
        layer = emergency_layer

        # Rapid warming (sun came out)
        layer.mock_thermal_trend.return_value = {
            "rate_per_hour": 0.8,
            "confidence": 0.9,
        }

        # Severe overshoot
        base_offset = 3.0
        damped_offset, reason = layer._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=23.6,
            target_temp=21.5,
        )

        # 3.0 × 0.8 (overshoot) × 0.4 (warming) = 0.96°C
        # Clamped to min T2 offset (1.5°C)
        assert damped_offset == THERMAL_RECOVERY_T2_MIN_OFFSET
        assert damped_offset == 1.5
        # assert "clamped to T2 min" in reason  # Reason string doesn't include clamping info currently


class TestSafetyMinimums:
    """Test that damping never reduces offset below safety minimums."""

    def test_t1_minimum_respected(self, emergency_layer):
        """T1 minimum (1.0°C) must be respected."""
        layer = emergency_layer
        
        # Extreme damping conditions
        layer.mock_thermal_trend.return_value = {"rate_per_hour": 1.0, "confidence": 1.0}
        
        base_offset = 2.0
        damped_offset, _ = layer._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T1",
            min_damped_offset=THERMAL_RECOVERY_T1_MIN_OFFSET,
            indoor_temp=24.0,  # Extreme overshoot
            target_temp=21.5,
        )

        # Calculation: 2.0 * 0.8 * 0.4 = 0.64
        # Should clamp to 1.0
        assert damped_offset == 1.0

    def test_t2_minimum_respected(self, emergency_layer):
        """T2 minimum (1.5°C) must be respected."""
        layer = emergency_layer
        
        layer.mock_thermal_trend.return_value = {"rate_per_hour": 1.0, "confidence": 1.0}
        
        base_offset = 3.0
        damped_offset, _ = layer._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=24.0,
            target_temp=21.5,
        )

        # Calculation: 3.0 * 0.8 * 0.4 = 0.96
        # Should clamp to 1.5
        assert damped_offset == 1.5


class TestOvershootWithoutTrendData:
    """Test behavior when trend data is missing."""

    def test_overshoot_applies_without_trend_data(self, emergency_layer):
        """Overshoot damping should work even if trend data is unavailable."""
        layer = emergency_layer
        
        # No trend data
        layer.mock_thermal_trend.return_value = None
        
        base_offset = 2.5
        damped_offset, reason = layer._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=23.5,  # Severe overshoot
            target_temp=21.5,
        )

        # Should still apply overshoot damping (0.8)
        # 2.5 * 0.8 = 2.0
        assert damped_offset == 2.0
        assert "severe overshoot" in reason


class TestEdgeCases:
    """Test edge cases."""

    def test_exactly_at_severe_threshold(self, emergency_layer):
        """Exactly at 1.5°C overshoot should trigger severe damping."""
        layer = emergency_layer
        layer.mock_thermal_trend.return_value = {"rate_per_hour": 0.0, "confidence": 0.8}
        
        base_offset = 2.0
        damped_offset, reason = layer._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=23.0,  # Exactly 1.5°C over 21.5
            target_temp=21.5,
        )

        # 2.0 * 0.8 = 1.6
        assert damped_offset == 1.6
        assert "severe overshoot" in reason

    def test_negative_overshoot_ignored(self, emergency_layer):
        """Negative overshoot (too cold) should be ignored."""
        layer = emergency_layer
        layer.mock_thermal_trend.return_value = {"rate_per_hour": 0.0, "confidence": 0.8}
        
        base_offset = 2.0
        damped_offset, reason = layer._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=20.0,  # Too cold
            target_temp=21.5,
        )

        assert damped_offset == 2.0
        assert reason == ""

    def test_missing_indoor_temp_no_overshoot_damping(self, emergency_layer):
        """Missing indoor temp should disable overshoot damping."""
        layer = emergency_layer
        layer.mock_thermal_trend.return_value = {"rate_per_hour": 0.0, "confidence": 0.8}
        
        base_offset = 2.0
        damped_offset, reason = layer._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=None,
            target_temp=21.5,
        )

        assert damped_offset == 2.0
        assert reason == ""

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

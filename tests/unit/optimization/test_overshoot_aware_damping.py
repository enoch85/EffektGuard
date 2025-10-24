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

from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.price_analyzer import PriceAnalyzer
from custom_components.effektguard.optimization.effect_manager import EffectManager
from custom_components.effektguard.optimization.thermal_model import ThermalModel
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
def decision_engine():
    """Create decision engine with mock dependencies."""
    hass = MagicMock()
    hass.config.latitude = 59.33  # Stockholm

    engine = DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(hass),
        thermal_model=ThermalModel(thermal_mass=1.0, insulation_quality=1.0),
        config={"latitude": 59.33, "target_indoor_temp": 21.5, "tolerance": 1.0},
    )

    # Mock predictor with sufficient history
    engine.predictor = MagicMock()
    engine.predictor.state_history = [None] * 16  # 4 hours of 15-min samples
    engine.predictor.__len__ = MagicMock(return_value=16)

    return engine


class TestOvershootOnlyDamping:
    """Test overshoot damping when no warming trend detected."""

    def test_severe_overshoot_2c(self, decision_engine):
        """Severe 2.0°C overshoot → 80% strength (0.8 multiplier)."""
        engine = decision_engine

        # Mock predictor with no warming trend
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.1,  # Slight warming but below threshold
            "confidence": 0.8,
        }
        engine.predictor.get_outdoor_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # Test severe overshoot: indoor 23.5°C, target 21.5°C = 2.0°C over
        base_offset = 2.5
        damped_offset, reason = engine._apply_thermal_recovery_damping(
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

    def test_moderate_overshoot_1_2c(self, decision_engine):
        """Moderate 1.2°C overshoot → 90% strength (0.9 multiplier)."""
        engine = decision_engine
        # Mock predictor
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }
        # Mock outdoor predictor
        engine.predictor.get_outdoor_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # Moderate overshoot: indoor 22.7°C, target 21.5°C = 1.2°C over
        base_offset = 2.0
        damped_offset, reason = engine._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T1",
            min_damped_offset=THERMAL_RECOVERY_T1_MIN_OFFSET,
            indoor_temp=22.7,
            target_temp=21.5,
        )

        # Should apply moderate overshoot damping: 2.0 × 0.9 = 1.8°C
        assert damped_offset == 1.8
        assert "moderate overshoot" in reason

    def test_mild_overshoot_0_7c(self, decision_engine):
        """Mild 0.7°C overshoot → 95% strength (0.95 multiplier)."""
        engine = decision_engine
        # Mock predictor
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }
        # Mock outdoor predictor
        engine.predictor.get_outdoor_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # Mild overshoot: indoor 22.2°C, target 21.5°C = 0.7°C over
        base_offset = 2.0
        damped_offset, reason = engine._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T1",
            min_damped_offset=THERMAL_RECOVERY_T1_MIN_OFFSET,
            indoor_temp=22.2,
            target_temp=21.5,
        )

        # Should apply mild overshoot damping: 2.0 × 0.95 = 1.9°C
        assert damped_offset == 1.9
        assert "mild overshoot" in reason

    def test_no_overshoot_no_damping(self, decision_engine):
        """No overshoot (<0.5°C) → no overshoot damping applied."""
        engine = decision_engine
        # Mock predictor
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }
        # Mock outdoor predictor
        engine.predictor.get_outdoor_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # No overshoot: indoor 21.8°C, target 21.5°C = 0.3°C (below threshold)
        base_offset = 2.0
        damped_offset, reason = engine._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T1",
            min_damped_offset=THERMAL_RECOVERY_T1_MIN_OFFSET,
            indoor_temp=21.8,
            target_temp=21.5,
        )

        # Should NOT apply damping (no warming, no overshoot)
        assert damped_offset == 2.0
        assert reason == ""


class TestCombinedOvershootAndWarming:
    """Test compound damping when both overshoot and warming occur."""

    def test_severe_overshoot_plus_rapid_warming(self, decision_engine):
        """Severe overshoot (0.8) + rapid warming (0.4) → compound 0.32."""
        engine = decision_engine
        # Mock predictor
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.6,  # Rapid warming (≥0.5°C/h)
            "confidence": 0.8,
        }
        # Mock outdoor predictor
        engine.predictor.get_outdoor_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # Severe overshoot + rapid warming (worst case scenario)
        base_offset = 3.0
        damped_offset, reason = engine._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=23.5,  # +2.0°C overshoot
            target_temp=21.5,
        )

        # Should apply compound damping: 3.0 × 0.4 × 0.8 = 0.96°C
        # But respects minimum: max(0.96, 1.5) = 1.5°C
        assert damped_offset == THERMAL_RECOVERY_T2_MIN_OFFSET  # 1.5°C minimum
        assert "warming" in reason
        assert "overshoot" in reason

    def test_moderate_overshoot_plus_moderate_warming(self, decision_engine):
        """Moderate overshoot (0.9) + moderate warming (0.6) → compound 0.54."""
        engine = decision_engine
        # Mock predictor
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.4,  # Moderate warming (0.3-0.5°C/h)
            "confidence": 0.8,
        }
        # Mock outdoor predictor
        engine.predictor.get_outdoor_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # Moderate overshoot + moderate warming
        base_offset = 2.5
        damped_offset, reason = engine._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T1",
            min_damped_offset=THERMAL_RECOVERY_T1_MIN_OFFSET,
            indoor_temp=22.6,  # +1.1°C overshoot
            target_temp=21.5,
        )

        # Should apply compound damping: 2.5 × 0.6 × 0.9 = 1.35°C
        # Respects minimum: max(1.35, 1.0) = 1.35°C
        assert damped_offset == pytest.approx(1.35, rel=0.01)
        assert "warming" in reason
        assert "overshoot" in reason


class TestV010FailurePrevention:
    """Test that overshoot awareness prevents v0.1.0 failure mode."""

    def test_prevents_2c_overshoot_during_recovery(self, decision_engine):
        """Verify 2.1°C overshoot like v0.1.0 would be damped to prevent escalation."""
        engine = decision_engine
        # Mock predictor
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.3,  # Moderate warming from solar gain
            "confidence": 0.8,
        }
        # Mock outdoor predictor
        engine.predictor.get_outdoor_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # v0.1.0 scenario: DM recovery active, indoor 23.6°C vs target 21.5°C
        base_offset = 2.5  # T2 strong recovery offset
        damped_offset, reason = engine._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=23.6,
            target_temp=21.5,
        )

        # Compound damping: 2.5 × 0.6 (warming) × 0.8 (severe overshoot) = 1.2°C
        # Respects minimum: max(1.2, 1.5) = 1.5°C
        assert damped_offset == THERMAL_RECOVERY_T2_MIN_OFFSET
        assert "severe overshoot" in reason
        assert "warming" in reason

        # This prevents further heat buildup that caused v0.1.0's 2.1°C overshoot


class TestSafetyMinimums:
    """Test that safety minimums are always respected."""

    def test_t1_minimum_respected(self, decision_engine):
        """T1 minimum (1.0°C) enforced even with extreme damping."""
        engine = decision_engine
        # Mock predictor
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.7,  # Rapid warming
            "confidence": 0.8,
        }
        # Mock outdoor predictor
        engine.predictor.get_outdoor_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # Extreme damping scenario
        base_offset = 1.5
        damped_offset, _ = engine._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T1",
            min_damped_offset=THERMAL_RECOVERY_T1_MIN_OFFSET,
            indoor_temp=24.0,  # Severe overshoot
            target_temp=21.5,
        )

        # Compound: 1.5 × 0.4 × 0.8 = 0.48°C → respects min 1.0°C
        assert damped_offset >= THERMAL_RECOVERY_T1_MIN_OFFSET

    def test_t2_minimum_respected(self, decision_engine):
        """T2 minimum (1.5°C) enforced even with extreme damping."""
        engine = decision_engine
        # Mock predictor
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.7,  # Rapid warming
            "confidence": 0.8,
        }
        # Mock outdoor predictor
        engine.predictor.get_outdoor_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # Extreme damping scenario
        base_offset = 2.0
        damped_offset, _ = engine._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=24.0,  # Severe overshoot
            target_temp=21.5,
        )

        # Compound: 2.0 × 0.4 × 0.8 = 0.64°C → respects min 1.5°C
        assert damped_offset >= THERMAL_RECOVERY_T2_MIN_OFFSET


class TestOvershootWithoutTrendData:
    """Test overshoot damping when trend confidence insufficient."""

    def test_overshoot_applies_without_trend_data(self, decision_engine):
        """Overshoot damping works even when trend confidence too low."""
        engine = decision_engine
        # Mock predictor
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.2,  # Insufficient confidence (<0.4)
        }
        # Mock outdoor predictor
        engine.predictor.get_outdoor_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.2,
        }

        # Severe overshoot but low trend confidence
        base_offset = 2.5
        damped_offset, reason = engine._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T2",
            min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
            indoor_temp=23.5,  # +2.0°C overshoot
            target_temp=21.5,
        )

        # Should still apply overshoot damping: 2.5 × 0.8 = 2.0°C
        assert damped_offset == 2.0
        assert "severe overshoot" in reason


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exactly_at_severe_threshold(self, decision_engine):
        """Overshoot exactly at 1.5°C → triggers severe damping."""
        engine = decision_engine
        # Mock predictor
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }
        # Mock outdoor predictor
        engine.predictor.get_outdoor_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # Exactly at threshold: indoor 23.0°C, target 21.5°C = 1.5°C
        base_offset = 2.0
        damped_offset, reason = engine._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T1",
            min_damped_offset=THERMAL_RECOVERY_T1_MIN_OFFSET,
            indoor_temp=23.0,
            target_temp=21.5,
        )

        # Should apply severe damping: 2.0 × 0.8 = 1.6°C
        assert damped_offset == 1.6
        assert "severe overshoot" in reason

    def test_negative_overshoot_ignored(self, decision_engine):
        """Negative overshoot (indoor < target) → no overshoot damping."""
        engine = decision_engine
        # Mock predictor
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }
        # Mock outdoor predictor
        engine.predictor.get_outdoor_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # Negative overshoot: indoor 20.5°C, target 21.5°C = -1.0°C (cold!)
        base_offset = 2.0
        damped_offset, reason = engine._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T1",
            min_damped_offset=THERMAL_RECOVERY_T1_MIN_OFFSET,
            indoor_temp=20.5,
            target_temp=21.5,
        )

        # Should NOT apply overshoot damping (no warming either)
        assert damped_offset == 2.0
        assert reason == ""

    def test_missing_indoor_temp_no_overshoot_damping(self, decision_engine):
        """Missing indoor temp → overshoot damping skipped safely."""
        engine = decision_engine
        # Mock predictor
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }
        # Mock outdoor predictor
        engine.predictor.get_outdoor_trend.return_value = {
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # Missing indoor temp (None)
        base_offset = 2.0
        damped_offset, reason = engine._apply_thermal_recovery_damping(
            base_offset=base_offset,
            tier_name="T1",
            min_damped_offset=THERMAL_RECOVERY_T1_MIN_OFFSET,
            indoor_temp=None,  # Missing data
            target_temp=21.5,
        )

        # Should safely skip overshoot damping, return base offset
        assert damped_offset == 2.0
        assert reason == ""

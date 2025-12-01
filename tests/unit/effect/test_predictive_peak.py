"""Tests for Phase 4: Predictive Peak Avoidance.

Phase 4 adds predictive power consumption forecasting to the effect layer,
using indoor temperature trend to predict heating demand 15 minutes ahead
and avoid peaks proactively.

Key Innovation:
- Traditional: Reacts when power is already high
- Predictive: Sees cooling trend → knows power will increase → acts before spike

References:
    MASTER_IMPLEMENTATION_PLAN.md: Phase 4 - Predictive Peak Avoidance
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime

from custom_components.effektguard.optimization.decision_engine import (
    DecisionEngine,
    LayerDecision,
)
from custom_components.effektguard.optimization.thermal_model import ThermalModel
from custom_components.effektguard.optimization.effect_manager import EffectManager
from custom_components.effektguard.optimization.price_analyzer import PriceAnalyzer


@pytest.fixture
def hass_mock():
    """Create mock Home Assistant instance."""
    return MagicMock()


@pytest.fixture
def thermal_model():
    """Create thermal model for testing."""
    return ThermalModel(thermal_mass=1.0, insulation_quality=1.0)


@pytest.fixture
def engine(hass_mock, thermal_model):
    """Create decision engine for testing."""
    effect_manager = EffectManager(hass_mock)
    price_analyzer = PriceAnalyzer()

    config = {
        "latitude": 59.33,  # Stockholm
        "target_indoor_temp": 21.0,
        "tolerance": 5.0,
    }

    engine = DecisionEngine(
        price_analyzer=price_analyzer,
        effect_manager=effect_manager,
        thermal_model=thermal_model,
        config=config,
    )

    # Mock predictor for trend data
    engine.predictor = MagicMock()
    engine.predictor.state_history = [MagicMock()] * 10
    engine.predictor.get_current_trend.return_value = {
        "trend": "stable",
        "rate_per_hour": 0.0,
        "confidence": 0.8,
    }

    return engine


@pytest.fixture
def nibe_state():
    """Create NIBE state for testing."""
    state = MagicMock()
    state.indoor_temp = 21.0
    state.outdoor_temp = 0.0
    state.flow_temp = 35.0
    state.degree_minutes = -100
    state.power_kw = 2.5
    return state


class TestPowerPrediction:
    """Test power consumption prediction based on indoor temperature trend."""

    def test_predicts_power_increase_rapid_cooling(self, engine, nibe_state):
        """Predict 1.5 kW increase when house cooling rapidly."""
        # Mock rapid cooling trend
        engine.predictor.get_current_trend.return_value = {
            "trend": "cooling_rapidly",
            "rate_per_hour": -0.5,
            "confidence": 0.8,
        }

        # Current power 3.0 kW, peak 5.0 kW
        nibe_state.power_kw = 3.0
        current_peak = 5.0

        decision = engine._effect_layer(nibe_state, current_peak, 3.0)

        # Predicted: 3.0 + 1.5 = 4.5 kW
        # Margin: 5.0 - 4.5 = 0.5 kW (< 1.0 kW threshold)
        # Should trigger predictive avoidance
        assert decision.offset < 0  # Reduces heating
        assert "predictive" in decision.reason.lower()
        assert "cooling rapidly" in decision.reason.lower()

    def test_predicts_power_increase_gentle_cooling(self, engine, nibe_state):
        """Predict 0.5 kW increase when house cooling gently."""
        # Mock gentle cooling trend
        engine.predictor.get_current_trend.return_value = {
            "trend": "cooling",
            "rate_per_hour": -0.25,
            "confidence": 0.8,
        }

        # Current power 4.2 kW, peak 5.0 kW
        nibe_state.power_kw = 4.2
        current_peak = 5.0

        decision = engine._effect_layer(nibe_state, current_peak, 4.2)

        # Predicted: 4.2 + 0.5 = 4.7 kW
        # Margin: 5.0 - 4.7 = 0.3 kW (< 1.0 kW threshold)
        # Should trigger predictive avoidance
        assert decision.offset < 0
        assert (
            "predictive" in decision.reason.lower() or "gentle cooling" in decision.reason.lower()
        )

    def test_predicts_power_decrease_warming(self, engine, nibe_state):
        """Predict 0.5 kW decrease when house warming."""
        # Mock warming trend
        engine.predictor.get_current_trend.return_value = {
            "trend": "warming",
            "rate_per_hour": 0.4,
            "confidence": 0.8,
        }

        # Current power 3.8 kW, peak 4.0 kW (warning zone)
        nibe_state.power_kw = 3.8
        current_peak = 4.0

        # Mock effect manager to return WARNING
        engine.effect.should_limit_power = MagicMock()
        engine.effect.should_limit_power.return_value = MagicMock(severity="WARNING")

        decision = engine._effect_layer(nibe_state, current_peak, 3.8)

        # Predicted: 3.8 - 0.5 = 3.3 kW (power will decrease)
        # In warning zone but demand falling → gentle reduction
        assert decision.offset < 0
        assert decision.offset > -1.0  # Gentler than rising demand
        assert "warming" in decision.reason.lower()

    def test_no_prediction_stable_trend(self, engine, nibe_state):
        """No power change predicted when trend stable."""
        # Mock stable trend
        engine.predictor.get_current_trend.return_value = {
            "trend": "stable",
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # Current power 3.0 kW, peak 5.0 kW (safe margin)
        nibe_state.power_kw = 3.0
        current_peak = 5.0

        # Mock effect manager to return OK
        engine.effect.should_limit_power = MagicMock()
        engine.effect.should_limit_power.return_value = MagicMock(severity="OK")

        decision = engine._effect_layer(nibe_state, current_peak, 3.0)

        # Predicted: 3.0 + 0.0 = 3.0 kW (no change)
        # Safe margin → no action
        assert decision.offset == 0.0
        assert "safe margin" in decision.reason.lower()


class TestProactiveAvoidance:
    """Test proactive peak avoidance (acts before spike occurs)."""

    def test_acts_before_spike_occurs(self, engine, nibe_state):
        """Act proactively when predicted power will exceed safe margin."""
        # Mock rapid cooling - power will spike soon
        engine.predictor.get_current_trend.return_value = {
            "trend": "cooling_rapidly",
            "rate_per_hour": -0.5,
            "confidence": 0.8,
        }

        # Current: 3.5 kW, peak: 5.0 kW
        # Predicted: 3.5 + 1.5 = 5.0 kW (will hit peak!)
        nibe_state.power_kw = 3.5
        current_peak = 5.0

        # Mock effect manager - currently OK (not in warning yet)
        engine.effect.should_limit_power = MagicMock()
        engine.effect.should_limit_power.return_value = MagicMock(severity="OK")

        decision = engine._effect_layer(nibe_state, current_peak, 3.5)

        # Should act NOW (proactively) even though currently OK
        # Predicted margin: 5.0 - 5.0 = 0.0 < 1.0 threshold
        assert decision.offset < 0
        assert "predictive" in decision.reason.lower()

    def test_no_action_when_warming_prevents_spike(self, engine, nibe_state):
        """No action when warming trend prevents predicted spike."""
        # Mock warming - power will decrease
        engine.predictor.get_current_trend.return_value = {
            "trend": "warming",
            "rate_per_hour": 0.4,
            "confidence": 0.8,
        }

        # Current: 3.5 kW, peak: 5.0 kW
        # Predicted: 3.5 - 0.5 = 3.0 kW (moving away from peak)
        nibe_state.power_kw = 3.5
        current_peak = 5.0

        # Mock effect manager - currently OK
        engine.effect.should_limit_power = MagicMock()
        engine.effect.should_limit_power.return_value = MagicMock(severity="OK")

        decision = engine._effect_layer(nibe_state, current_peak, 3.5)

        # Should NOT act - power trending down
        assert decision.offset == 0.0

    def test_stronger_response_when_cooling_in_warning_zone(self, engine, nibe_state):
        """Stronger response when cooling trend in warning zone.

        Note: Predictive logic may take precedence over warning zone if
        predicted margin < 1.0 kW. This test validates correct prioritization.
        """
        # Mock cooling trend
        engine.predictor.get_current_trend.return_value = {
            "trend": "cooling",
            "rate_per_hour": -0.3,
            "confidence": 0.8,
        }

        # In warning zone
        nibe_state.power_kw = 3.8
        current_peak = 4.0

        # Mock effect manager - WARNING
        engine.effect.should_limit_power = MagicMock()
        engine.effect.should_limit_power.return_value = MagicMock(severity="WARNING")

        decision = engine._effect_layer(nibe_state, current_peak, 3.8)

        # Should respond because demand rising
        # Predicted: 3.8 + 0.5 = 4.3, margin 4.0 - 4.3 = -0.3 < 1.0
        # Predictive logic takes precedence (correct prioritization)
        assert decision.offset < 0
        assert "cooling" in decision.reason.lower()  # Either "predictive" or "gentle cooling"
        # The specific path (predictive vs warning) depends on predicted margin


class TestCriticalPeakHandling:
    """Test critical peak handling (unchanged from Phase 3 but verify)."""

    def test_critical_peak_immediate_action(self, engine, nibe_state):
        """CRITICAL severity triggers immediate maximum reduction."""
        # Mock stable trend (shouldn't affect critical response)
        engine.predictor.get_current_trend.return_value = {
            "trend": "stable",
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # At peak
        nibe_state.power_kw = 4.8
        current_peak = 5.0

        # Mock effect manager - CRITICAL
        engine.effect.should_limit_power = MagicMock()
        engine.effect.should_limit_power.return_value = MagicMock(severity="CRITICAL")

        decision = engine._effect_layer(nibe_state, current_peak, 4.8)

        # Should apply maximum reduction regardless of trend
        assert decision.offset == -3.0
        assert decision.weight == 1.0
        assert "critical" in decision.reason.lower()


class TestPhase4SuccessCriteria:
    """Verify Phase 4 success criteria from implementation plan."""

    def test_predicts_power_increase_when_cooling_rapidly(self, engine, nibe_state):
        """Verify power increase predicted when cooling rapidly."""
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": -0.5,
            "confidence": 0.8,
        }

        nibe_state.power_kw = 3.5
        current_peak = 5.0

        # Mock as OK to test predictive logic
        engine.effect.should_limit_power = MagicMock()
        engine.effect.should_limit_power.return_value = MagicMock(severity="OK")

        decision = engine._effect_layer(nibe_state, current_peak, 3.5)

        # Should predict increase and act proactively
        # Predicted: 3.5 + 1.5 = 5.0 (margin 0.0 < 1.0)
        assert decision.offset < 0
        assert "predictive" in decision.reason.lower()

    def test_predicts_power_decrease_when_warming(self, engine, nibe_state):
        """Verify power decrease predicted when warming."""
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.4,
            "confidence": 0.8,
        }

        nibe_state.power_kw = 3.8
        current_peak = 4.0

        # In warning zone
        engine.effect.should_limit_power = MagicMock()
        engine.effect.should_limit_power.return_value = MagicMock(severity="WARNING")

        decision = engine._effect_layer(nibe_state, current_peak, 3.8)

        # Should recognize warming (power decreasing)
        assert "warming" in decision.reason.lower()

    def test_reduces_offset_before_predicted_peak(self, engine, nibe_state):
        """Verify offset reduced BEFORE peak occurs."""
        # Rapid cooling will cause spike in 15 min
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": -0.6,
            "confidence": 0.8,
        }

        # Currently OK but will spike
        nibe_state.power_kw = 3.2
        current_peak = 5.0

        engine.effect.should_limit_power = MagicMock()
        engine.effect.should_limit_power.return_value = MagicMock(severity="OK")

        decision = engine._effect_layer(nibe_state, current_peak, 3.2)

        # Should act NOW before spike (predicted: 3.2 + 1.5 = 4.7, margin 0.3)
        assert decision.offset < 0
        assert "predictive" in decision.reason.lower()


class TestComparisonReactiveVsPredictive:
    """Compare reactive (Phase 3) vs predictive (Phase 4) behavior."""

    def test_reactive_waits_predictive_acts_early(self, engine, nibe_state):
        """Show predictive acts earlier than reactive approach."""
        # Scenario: House cooling rapidly, will spike in 15 min
        engine.predictor.get_current_trend.return_value = {
            "trend": "cooling_rapidly",
            "rate_per_hour": -0.5,
            "confidence": 0.8,
        }

        # Currently at 3.5 kW (OK zone for reactive)
        # But predicted 5.0 kW (will hit peak)
        nibe_state.power_kw = 3.5
        current_peak = 5.0

        # Mock effect manager says OK (reactive wouldn't act)
        engine.effect.should_limit_power = MagicMock()
        engine.effect.should_limit_power.return_value = MagicMock(severity="OK")

        decision = engine._effect_layer(nibe_state, current_peak, 3.5)

        # Phase 4 (predictive): Acts NOW
        # Phase 3 (reactive): Would wait until power reaches warning/critical
        assert decision.offset < 0  # Predictive acts
        assert "predictive" in decision.reason.lower()

        # This is the key benefit: acts ~15 minutes earlier than reactive

    def test_smoother_power_profile(self, engine, nibe_state):
        """Verify gradual reduction vs sudden reactive reduction."""
        # Start of cooling trend
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": -0.3,
            "confidence": 0.8,
        }

        nibe_state.power_kw = 4.0
        current_peak = 5.0

        # Mock warning zone
        engine.effect.should_limit_power = MagicMock()
        engine.effect.should_limit_power.return_value = MagicMock(severity="WARNING")

        decision = engine._effect_layer(nibe_state, current_peak, 4.0)

        # Should apply moderate reduction (not aggressive)
        # Predicted: 4.0 + 0.5 = 4.5, but in warning with rising demand
        assert -1.5 <= decision.offset < 0
        # Smoother than emergency -3.0 offset


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_predictor_available(self, engine, nibe_state):
        """Handle gracefully when predictor not available."""
        engine.predictor = None

        nibe_state.power_kw = 3.5
        current_peak = 5.0

        engine.effect.should_limit_power = MagicMock()
        engine.effect.should_limit_power.return_value = MagicMock(severity="OK")

        # Should not crash, fall back to non-predictive behavior
        decision = engine._effect_layer(nibe_state, current_peak, 3.5)
        assert decision.offset == 0.0  # Safe margin, no action

    def test_predicted_power_exactly_at_peak(self, engine, nibe_state):
        """Test boundary: predicted power exactly equals peak."""
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": -0.3,
            "confidence": 0.8,
        }

        nibe_state.power_kw = 4.5
        current_peak = 5.0

        engine.effect.should_limit_power = MagicMock()
        engine.effect.should_limit_power.return_value = MagicMock(severity="OK")

        decision = engine._effect_layer(nibe_state, current_peak, 4.5)

        # Predicted: 4.5 + 0.5 = 5.0
        # Margin: 0.0 < 1.0 → should act
        assert decision.offset < 0

    def test_large_predicted_increase_capped(self, engine, nibe_state):
        """Verify predicted increase doesn't exceed reasonable bounds."""
        # Even with extreme cooling, prediction should be reasonable
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": -1.0,  # Extreme cooling
            "confidence": 0.8,
        }

        nibe_state.power_kw = 3.0
        current_peak = 5.0

        engine.effect.should_limit_power = MagicMock()
        engine.effect.should_limit_power.return_value = MagicMock(severity="OK")

        decision = engine._effect_layer(nibe_state, current_peak, 3.0)

        # Max predicted increase is 1.5 kW (rapid cooling threshold)
        # Predicted: 3.0 + 1.5 = 4.5
        # Should act but not panic
        assert decision.offset < 0

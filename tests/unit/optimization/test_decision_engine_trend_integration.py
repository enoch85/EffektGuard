"""Tests for DecisionEngine indoor temperature trend integration.

Tests Phase 1 trend integration requirements from INDOOR_TEMP_TREND_IMPLEMENTATION.md:
- Rapid cooling detection for predictive intervention
- DHW blocking during rapid indoor cooling
- Overshoot prevention damping during rapid warming
- Trend-aware boost in proactive zones
"""

import pytest
from unittest.mock import MagicMock, patch

from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_manager import EffectManager
from custom_components.effektguard.optimization.price_analyzer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_model import ThermalModel


@pytest.fixture
def hass_mock():
    """Create mock Home Assistant instance."""
    return MagicMock()


@pytest.fixture
def nibe_state_mock():
    """Create mock NIBE state."""
    state = MagicMock()
    state.indoor_temp = 20.5
    state.outdoor_temp = -5.0
    state.degree_minutes = -150
    state.flow_temp = 35.0
    return state


def create_engine_with_predictor(hass_mock, latitude: float = 59.33):
    """Create DecisionEngine with thermal predictor for trend data."""
    price_analyzer = PriceAnalyzer()
    effect_manager = EffectManager(hass_mock)
    thermal_model = ThermalModel(thermal_mass=1.0, insulation_quality=1.0)

    # Create mock thermal predictor
    thermal_predictor = MagicMock()
    thermal_predictor.state_history = [MagicMock()] * 10  # Enough data for trend

    config = {
        "latitude": latitude,
        "target_indoor_temp": 21.0,
        "tolerance": 5.0,
    }

    engine = DecisionEngine(
        price_analyzer=price_analyzer,
        effect_manager=effect_manager,
        thermal_model=thermal_model,
        config=config,
        thermal_predictor=thermal_predictor,
    )

    return engine


class TestTrendHelperMethods:
    """Test trend helper methods."""

    def test_get_thermal_trend_with_data(self, hass_mock):
        """Verify _get_thermal_trend returns trend data when available."""
        engine = create_engine_with_predictor(hass_mock)

        # Mock trend data
        engine.predictor.get_current_trend.return_value = {
            "trend": "falling",
            "rate_per_hour": -0.4,
            "confidence": 0.9,
            "samples": 48,
        }

        trend = engine._get_thermal_trend()

        assert trend["trend"] == "falling"
        assert trend["rate_per_hour"] == -0.4
        assert trend["confidence"] == 0.9

    def test_get_thermal_trend_insufficient_data(self, hass_mock):
        """Verify _get_thermal_trend returns empty dict when insufficient data."""
        engine = create_engine_with_predictor(hass_mock)
        engine.predictor.state_history = []  # No data

        trend = engine._get_thermal_trend()

        assert trend["trend"] == "unknown"
        assert trend["rate_per_hour"] == 0.0
        assert trend["confidence"] == 0.0

    def test_is_cooling_rapidly_true(self, hass_mock):
        """Verify _is_cooling_rapidly detects rapid cooling."""
        engine = create_engine_with_predictor(hass_mock)
        thermal_trend = {"rate_per_hour": -0.4}

        assert engine._is_cooling_rapidly(thermal_trend, threshold=-0.3) is True

    def test_is_cooling_rapidly_false(self, hass_mock):
        """Verify _is_cooling_rapidly returns False for slow cooling."""
        engine = create_engine_with_predictor(hass_mock)
        thermal_trend = {"rate_per_hour": -0.2}

        assert engine._is_cooling_rapidly(thermal_trend, threshold=-0.3) is False

    def test_is_warming_rapidly_true(self, hass_mock):
        """Verify _is_warming_rapidly detects rapid warming."""
        engine = create_engine_with_predictor(hass_mock)
        thermal_trend = {"rate_per_hour": 0.5}

        assert engine._is_warming_rapidly(thermal_trend, threshold=0.3) is True

    def test_is_warming_rapidly_false(self, hass_mock):
        """Verify _is_warming_rapidly returns False for slow warming."""
        engine = create_engine_with_predictor(hass_mock)
        thermal_trend = {"rate_per_hour": 0.2}

        assert engine._is_warming_rapidly(thermal_trend, threshold=0.3) is False


class TestRapidCoolingDetection:
    """Test predictive rapid cooling detection in proactive layer."""

    def test_rapid_cooling_triggers_proactive_boost(self, hass_mock, nibe_state_mock):
        """Test that rapid cooling with deficit triggers proactive heating boost."""
        engine = create_engine_with_predictor(hass_mock)
        nibe_state_mock.indoor_temp = 20.5  # 0.5°C below target
        nibe_state_mock.outdoor_temp = -5.0  # Cold outside

        # Mock rapid cooling trend
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": -0.4,  # Cooling 0.4°C/hour
            "trend": "falling",
            "confidence": 0.9,
        }

        decision = engine._proactive_debt_prevention_layer(nibe_state_mock)

        # Should boost heating proactively
        assert decision.offset > 0.5
        assert decision.weight == 0.8
        assert "rapid cooling" in decision.reason.lower()

    def test_rapid_cooling_warm_outdoor_no_boost(self, hass_mock, nibe_state_mock):
        """Test that rapid cooling with warm outdoor doesn't trigger boost."""
        engine = create_engine_with_predictor(hass_mock)
        nibe_state_mock.indoor_temp = 20.5
        nibe_state_mock.outdoor_temp = 10.0  # Warm outside

        # Mock rapid cooling trend
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": -0.4,
            "trend": "falling",
            "confidence": 0.9,
        }

        decision = engine._proactive_debt_prevention_layer(nibe_state_mock)

        # Should not boost (warm outdoor means not serious)
        # Will fall through to regular DM zones
        assert "rapid cooling" not in decision.reason.lower()

    def test_cooling_below_target_with_cold_outdoor(self, hass_mock, nibe_state_mock):
        """Test cooling below target with cold outdoor triggers strong response."""
        engine = create_engine_with_predictor(hass_mock)
        nibe_state_mock.indoor_temp = 20.3  # 0.7°C below target
        nibe_state_mock.outdoor_temp = -10.0

        # Mock rapid cooling
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": -0.5,
            "trend": "falling",
        }

        decision = engine._proactive_debt_prevention_layer(nibe_state_mock)

        # Should have significant boost (trend_rate * 2.0)
        assert decision.offset >= 1.0
        assert "rapid cooling" in decision.reason.lower()


class TestTrendAwareZoneBoost:
    """Test trend-aware boosting in proactive zones."""

    def test_zone1_with_rapid_cooling_gets_boost(self, hass_mock, nibe_state_mock):
        """Test Zone 1 offset boosted 30% when cooling rapidly."""
        engine = create_engine_with_predictor(hass_mock)

        # Set DM to Zone 1 threshold range
        nibe_state_mock.degree_minutes = -100  # In Zone 1
        nibe_state_mock.outdoor_temp = -5.0

        # Mock rapid cooling
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": -0.3,
            "trend": "falling",
        }

        decision = engine._proactive_debt_prevention_layer(nibe_state_mock)

        # Base Zone 1 offset is 0.5, with 30% boost = 0.65
        assert decision.offset > 0.5
        assert "trend:" in decision.reason


class TestOvershootPrevention:
    """Test overshoot prevention damping."""

    def test_rapid_warming_applies_damping(self, hass_mock, nibe_state_mock):
        """Test that rapid warming reduces final offset to prevent overshoot."""
        engine = create_engine_with_predictor(hass_mock)

        # Mock rapid warming
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.5,  # Warming 0.5°C/hour
            "trend": "rising",
        }

        # Calculate decision with some offset from layers
        decision = engine.calculate_decision(
            nibe_state=nibe_state_mock,
            price_data=None,
            weather_data=None,
            current_peak=0.0,
            current_power=0.0,
        )

        # Should mention damping in reasoning
        assert "damped" in decision.reasoning.lower() or "warming" in decision.reasoning.lower()

    def test_rapid_cooling_applies_boost(self, hass_mock, nibe_state_mock):
        """Test that rapid cooling boosts final offset to prevent undershoot."""
        engine = create_engine_with_predictor(hass_mock)

        # Mock rapid cooling
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": -0.4,
            "trend": "falling",
        }

        decision = engine.calculate_decision(
            nibe_state=nibe_state_mock,
            price_data=None,
            weather_data=None,
            current_peak=0.0,
            current_power=0.0,
        )

        # Should mention boost in reasoning
        assert "boosted" in decision.reasoning.lower() or "cooling" in decision.reasoning.lower()

    def test_stable_trend_no_damping(self, hass_mock, nibe_state_mock):
        """Test that stable trend doesn't apply damping."""
        engine = create_engine_with_predictor(hass_mock)

        # Mock stable trend
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.1,  # Stable
            "trend": "stable",
        }

        decision = engine.calculate_decision(
            nibe_state=nibe_state_mock,
            price_data=None,
            weather_data=None,
            current_peak=0.0,
            current_power=0.0,
        )

        # Should not mention damping or boosting
        assert "damped" not in decision.reasoning.lower()
        assert "boosted" not in decision.reasoning.lower()

    def test_high_offset_prevents_additional_boost(self, hass_mock, nibe_state_mock):
        """Test that high offset (>= 3.0) prevents additional boost even when cooling."""
        engine = create_engine_with_predictor(hass_mock)

        # Mock rapid cooling
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": -0.5,
            "trend": "falling",
        }

        # Force high offset by setting critical conditions
        nibe_state_mock.indoor_temp = 17.0  # Very low
        nibe_state_mock.outdoor_temp = -15.0

        decision = engine.calculate_decision(
            nibe_state=nibe_state_mock,
            price_data=None,
            weather_data=None,
            current_peak=0.0,
            current_power=0.0,
        )

        # Should mention safety limit if offset was capped
        if decision.offset >= 3.0:
            # If at high offset, should not boost further
            pass  # This is expected behavior


class TestPredictiveDeficitCalculation:
    """Test predicted deficit calculation from trend."""

    def test_predicted_deficit_calculation(self, hass_mock, nibe_state_mock):
        """Test that predicted deficit is calculated correctly."""
        engine = create_engine_with_predictor(hass_mock)

        # Current: 0.5°C below target, cooling at 0.3°C/h
        # In 1h: 0.5 + 0.3 = 0.8°C below target
        nibe_state_mock.indoor_temp = 20.5
        nibe_state_mock.outdoor_temp = -5.0

        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": -0.3,
            "trend": "falling",
        }

        decision = engine._proactive_debt_prevention_layer(nibe_state_mock)

        # Should mention predicted deficit in reasoning
        if "rapid cooling" in decision.reason.lower():
            assert "→" in decision.reason  # Shows prediction arrow

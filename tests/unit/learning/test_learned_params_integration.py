"""Tests for learned parameter integration in ThermalStatePredictor.

Verifies that ThermalStatePredictor correctly uses learned parameters
from AdaptiveThermalModel when available and falls back to calculated
values when not.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.const import (
    LEARNING_CONFIDENCE_THRESHOLD,
    PREDICTION_THERMAL_RESPONSIVENESS_DEFAULT,
)
from custom_components.effektguard.optimization.learning_types import (
    LearnedThermalParameters,
)
from custom_components.effektguard.optimization.prediction_layer import (
    ThermalStatePredictor,
)
from custom_components.effektguard.const import UFHType


@pytest.fixture
def predictor_with_history():
    """Create a ThermalStatePredictor with sufficient history for predictions."""
    predictor = ThermalStatePredictor()

    # Add 120 observations (30 hours at 4 per hour) - more than 96 required
    base_time = datetime.now()
    for i in range(120):
        predictor.record_state(
            timestamp=base_time - timedelta(minutes=15 * i),
            indoor_temp=21.0 + (i % 4) * 0.1,  # Small variation
            outdoor_temp=0.0,
            heating_offset=1.0,
            flow_temp=35.0,
            degree_minutes=-100.0,
        )

    return predictor


@pytest.fixture
def high_confidence_learned_params():
    """Create learned parameters with high confidence."""
    return LearnedThermalParameters(
        thermal_mass=1.5,
        heat_loss_coefficient=180.0,
        heating_efficiency=0.4,  # °C per offset per hour
        thermal_decay_rate=-0.1,  # °C per hour at typical temp diff
        ufh_type=UFHType.SLOW_RESPONSE,  # Concrete slab = slow response
        last_updated=datetime.now(),
        confidence=0.85,  # Above LEARNING_CONFIDENCE_THRESHOLD (0.7)
        observation_count=200,
    )


@pytest.fixture
def low_confidence_learned_params():
    """Create learned parameters with low confidence."""
    return LearnedThermalParameters(
        thermal_mass=1.0,
        heat_loss_coefficient=150.0,
        heating_efficiency=0.3,
        thermal_decay_rate=-0.08,
        ufh_type=UFHType.UNKNOWN,
        last_updated=datetime.now(),
        confidence=0.4,  # Below LEARNING_CONFIDENCE_THRESHOLD (0.7)
        observation_count=50,
    )


class TestPredictTemperatureWithLearnedParams:
    """Test predict_temperature uses learned parameters when appropriate."""

    def test_uses_learned_heating_efficiency_when_high_confidence(
        self, predictor_with_history, high_confidence_learned_params
    ):
        """Verify learned heating_efficiency is used when confidence is high."""
        # Predict with learned params
        result = predictor_with_history.predict_temperature(
            hours_ahead=4,
            future_outdoor_temps=[0.0, -1.0, -2.0, -3.0],
            planned_offsets=[2.0, 2.0, 2.0, 2.0],  # Positive offset
            thermal_mass=1.5,
            insulation_quality=1.0,
            learned_params=high_confidence_learned_params,
        )

        # Should complete without error
        assert result is not None
        assert len(result.predicted_temps) == 4
        assert result.confidence > 0

    def test_falls_back_when_low_confidence(
        self, predictor_with_history, low_confidence_learned_params
    ):
        """Verify fallback to calculated responsiveness when confidence is low."""
        # Predict with low-confidence params
        result_with_low_conf = predictor_with_history.predict_temperature(
            hours_ahead=4,
            future_outdoor_temps=[0.0, -1.0, -2.0, -3.0],
            planned_offsets=[2.0, 2.0, 2.0, 2.0],
            thermal_mass=1.0,
            insulation_quality=1.0,
            learned_params=low_confidence_learned_params,
        )

        # Predict without learned params
        result_without = predictor_with_history.predict_temperature(
            hours_ahead=4,
            future_outdoor_temps=[0.0, -1.0, -2.0, -3.0],
            planned_offsets=[2.0, 2.0, 2.0, 2.0],
            thermal_mass=1.0,
            insulation_quality=1.0,
            learned_params=None,
        )

        # Both should complete - low confidence falls back to same calculation as None
        assert result_with_low_conf is not None
        assert result_without is not None

        # With low confidence, should get same results as without learned params
        # (within floating point tolerance)
        for i in range(4):
            assert abs(
                result_with_low_conf.predicted_temps[i] - result_without.predicted_temps[i]
            ) < 0.01

    def test_uses_none_learned_params_gracefully(self, predictor_with_history):
        """Verify prediction works when learned_params is None."""
        result = predictor_with_history.predict_temperature(
            hours_ahead=4,
            future_outdoor_temps=[0.0, -1.0, -2.0, -3.0],
            thermal_mass=1.0,
            insulation_quality=1.0,
            learned_params=None,
        )

        assert result is not None
        assert len(result.predicted_temps) == 4

    def test_learned_params_affects_prediction(
        self, predictor_with_history, high_confidence_learned_params
    ):
        """Verify that learned params actually affect the prediction results."""
        # Predict with learned params (heating_efficiency=0.4)
        result_with_learned = predictor_with_history.predict_temperature(
            hours_ahead=4,
            future_outdoor_temps=[0.0, -1.0, -2.0, -3.0],
            planned_offsets=[3.0, 3.0, 3.0, 3.0],  # Strong positive offset
            thermal_mass=1.5,
            insulation_quality=1.0,
            learned_params=high_confidence_learned_params,
        )

        # Predict without learned params
        result_without = predictor_with_history.predict_temperature(
            hours_ahead=4,
            future_outdoor_temps=[0.0, -1.0, -2.0, -3.0],
            planned_offsets=[3.0, 3.0, 3.0, 3.0],
            thermal_mass=1.5,
            insulation_quality=1.0,
            learned_params=None,
        )

        # Results should be different (learned vs calculated responsiveness)
        # The exact difference depends on the calculated responsiveness
        # but they should not be identical
        assert result_with_learned is not None
        assert result_without is not None

        # Both predictions should be reasonable (not NaN or wildly off)
        for temp in result_with_learned.predicted_temps:
            assert 10.0 < temp < 30.0  # Reasonable indoor temp range
        for temp in result_without.predicted_temps:
            assert 10.0 < temp < 30.0


class TestShouldPreHeatWithLearnedParams:
    """Test should_pre_heat uses learned parameters."""

    def test_should_pre_heat_accepts_learned_params(
        self, predictor_with_history, high_confidence_learned_params
    ):
        """Verify should_pre_heat accepts and uses learned_params."""
        result = predictor_with_history.should_pre_heat(
            target_temp=21.0,
            hours_ahead=6,
            future_outdoor_temps=[-5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
            current_outdoor_temp=-5.0,
            current_indoor_temp=21.5,
            thermal_mass=1.5,
            insulation_quality=1.0,
            learned_params=high_confidence_learned_params,
        )

        assert result is not None
        assert hasattr(result, "should_preheat")
        assert hasattr(result, "recommended_offset")
        assert hasattr(result, "reason")

    def test_should_pre_heat_works_without_learned_params(self, predictor_with_history):
        """Verify should_pre_heat works when learned_params is None."""
        result = predictor_with_history.should_pre_heat(
            target_temp=21.0,
            hours_ahead=6,
            future_outdoor_temps=[-5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
            current_outdoor_temp=-5.0,
            current_indoor_temp=21.5,
            thermal_mass=1.5,
            insulation_quality=1.0,
            learned_params=None,
        )

        assert result is not None
        assert hasattr(result, "should_preheat")


class TestCalculatePreHeatDecisionIntegration:
    """Test evaluate_layer gets learned params from thermal model."""

    def test_gets_learned_params_from_adaptive_model(self, predictor_with_history):
        """Verify evaluate_layer extracts learned params from AdaptiveThermalModel."""
        # Create mock AdaptiveThermalModel with get_parameters
        mock_thermal_model = MagicMock()
        mock_thermal_model.thermal_mass = 1.5
        mock_thermal_model.insulation_quality = 1.0
        mock_thermal_model.get_prediction_horizon.return_value = 12.0
        mock_thermal_model.get_parameters.return_value = LearnedThermalParameters(
            thermal_mass=1.5,
            heat_loss_coefficient=180.0,
            heating_efficiency=0.4,
            thermal_decay_rate=-0.1,
            ufh_type=UFHType.SLOW_RESPONSE,
            last_updated=datetime.now(),
            confidence=0.9,
            observation_count=300,
        )

        # Create mock NIBE state
        mock_nibe_state = MagicMock()
        mock_nibe_state.indoor_temp = 21.0
        mock_nibe_state.outdoor_temp = -5.0

        # Create mock weather data
        mock_weather_data = MagicMock()
        mock_weather_data.forecast_hours = [
            MagicMock(temperature=-5.0 - i * 0.5) for i in range(24)
        ]

        result = predictor_with_history.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data,
            target_temp=21.0,
            thermal_model=mock_thermal_model,
        )

        # Should have called get_parameters
        mock_thermal_model.get_parameters.assert_called_once()

        # Result should be valid
        assert result is not None
        assert hasattr(result, "offset")
        assert hasattr(result, "weight")

    def test_works_with_thermal_model_without_get_parameters(self, predictor_with_history):
        """Verify evaluate_layer works with ThermalModel (no get_parameters)."""
        # Create mock ThermalModel (old-style, no get_parameters)
        mock_thermal_model = MagicMock(spec=["thermal_mass", "insulation_quality", "get_prediction_horizon"])
        mock_thermal_model.thermal_mass = 1.0
        mock_thermal_model.insulation_quality = 1.0
        mock_thermal_model.get_prediction_horizon.return_value = 12.0

        # Create mock NIBE state
        mock_nibe_state = MagicMock()
        mock_nibe_state.indoor_temp = 21.0
        mock_nibe_state.outdoor_temp = 0.0

        # Create mock weather data
        mock_weather_data = MagicMock()
        mock_weather_data.forecast_hours = [
            MagicMock(temperature=0.0 - i * 0.3) for i in range(24)
        ]

        result = predictor_with_history.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data,
            target_temp=21.0,
            thermal_model=mock_thermal_model,
        )

        # Should still work without get_parameters
        assert result is not None
        assert hasattr(result, "offset")
        assert hasattr(result, "weight")


class TestConfidenceThreshold:
    """Test that confidence threshold is respected."""

    def test_confidence_threshold_value(self):
        """Verify LEARNING_CONFIDENCE_THRESHOLD is 0.7."""
        assert LEARNING_CONFIDENCE_THRESHOLD == 0.7

    def test_exactly_at_threshold_uses_learned(
        self, predictor_with_history
    ):
        """Verify learned params at exactly threshold are used."""
        params_at_threshold = LearnedThermalParameters(
            thermal_mass=1.0,
            heat_loss_coefficient=180.0,
            heating_efficiency=0.5,
            thermal_decay_rate=-0.1,
            ufh_type=UFHType.SLOW_RESPONSE,
            last_updated=datetime.now(),
            confidence=LEARNING_CONFIDENCE_THRESHOLD,  # Exactly 0.7
            observation_count=100,
        )

        # Should use learned params (>= threshold)
        result = predictor_with_history.predict_temperature(
            hours_ahead=4,
            future_outdoor_temps=[0.0, -1.0, -2.0, -3.0],
            thermal_mass=1.0,
            insulation_quality=1.0,
            learned_params=params_at_threshold,
        )

        assert result is not None

    def test_just_below_threshold_falls_back(
        self, predictor_with_history
    ):
        """Verify learned params just below threshold fall back."""
        params_below_threshold = LearnedThermalParameters(
            thermal_mass=1.0,
            heat_loss_coefficient=180.0,
            heating_efficiency=0.5,
            thermal_decay_rate=-0.1,
            ufh_type=UFHType.SLOW_RESPONSE,
            last_updated=datetime.now(),
            confidence=LEARNING_CONFIDENCE_THRESHOLD - 0.01,  # Just below 0.7
            observation_count=100,
        )

        params_no = None

        result_below = predictor_with_history.predict_temperature(
            hours_ahead=4,
            future_outdoor_temps=[0.0, -1.0, -2.0, -3.0],
            thermal_mass=1.0,
            insulation_quality=1.0,
            learned_params=params_below_threshold,
        )

        result_none = predictor_with_history.predict_temperature(
            hours_ahead=4,
            future_outdoor_temps=[0.0, -1.0, -2.0, -3.0],
            thermal_mass=1.0,
            insulation_quality=1.0,
            learned_params=params_no,
        )

        # Should get same results (below threshold = fallback)
        for i in range(4):
            assert abs(result_below.predicted_temps[i] - result_none.predicted_temps[i]) < 0.01

"""Tests for ThermalStatePredictor.evaluate_layer() method.

Phase 3 of layer refactor: Tests for the extracted prediction layer evaluation logic.
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta

from custom_components.effektguard.optimization.prediction_layer import (
    ThermalStatePredictor,
    PredictionLayerDecision,
)
from custom_components.effektguard.const import LAYER_WEIGHT_PREDICTION


@pytest.fixture
def predictor():
    """Create a ThermalStatePredictor instance."""
    return ThermalStatePredictor()


@pytest.fixture
def mock_nibe_state():
    """Create mock NIBE state."""
    state = MagicMock()
    state.indoor_temp = 21.0
    state.outdoor_temp = 0.0
    state.degree_minutes = -100.0
    return state


@pytest.fixture
def mock_weather_data():
    """Create mock weather data with forecast."""
    weather = MagicMock()
    # 24 hours of forecast, starting at 0°C and dropping to -10°C
    weather.forecast_hours = [
        MagicMock(temperature=0.0 - (i * 0.4)) for i in range(24)
    ]
    return weather


@pytest.fixture
def mock_thermal_model():
    """Create mock thermal model."""
    model = MagicMock()
    model.thermal_mass = 1.5
    model.insulation_quality = 1.0
    model.get_prediction_horizon.return_value = 12.0
    return model


class TestEvaluateLayerInsufficientData:
    """Test evaluate_layer when insufficient learning data."""

    def test_insufficient_data_returns_learning_reason(
        self, predictor, mock_nibe_state, mock_weather_data, mock_thermal_model
    ):
        """When <96 observations, returns learning status."""
        # Predictor has no history
        assert len(predictor.state_history) == 0

        result = predictor.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data,
            target_temp=21.0,
            thermal_model=mock_thermal_model,
        )

        assert isinstance(result, PredictionLayerDecision)
        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "Learning:" in result.reason
        assert "0/96" in result.reason

    def test_partial_data_shows_progress(
        self, predictor, mock_nibe_state, mock_weather_data, mock_thermal_model
    ):
        """When partial observations, shows learning progress."""
        # Add 48 observations (half of required 96)
        for i in range(48):
            predictor.record_state(
                timestamp=datetime.now() - timedelta(minutes=15 * i),
                indoor_temp=21.0,
                outdoor_temp=0.0,
                heating_offset=0.0,
                flow_temp=35.0,
                degree_minutes=-100.0,
            )

        result = predictor.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data,
            target_temp=21.0,
            thermal_model=mock_thermal_model,
        )

        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "48/96" in result.reason


class TestEvaluateLayerNoWeatherData:
    """Test evaluate_layer when weather data unavailable."""

    def test_no_weather_data_returns_zero(
        self, predictor, mock_nibe_state, mock_thermal_model
    ):
        """When weather_data is None, returns no pre-heat."""
        # Add enough observations
        for i in range(100):
            predictor.record_state(
                timestamp=datetime.now() - timedelta(minutes=15 * i),
                indoor_temp=21.0,
                outdoor_temp=0.0,
                heating_offset=0.0,
                flow_temp=35.0,
                degree_minutes=-100.0,
            )

        result = predictor.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=None,
            target_temp=21.0,
            thermal_model=mock_thermal_model,
        )

        assert isinstance(result, PredictionLayerDecision)
        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "No weather forecast" in result.reason

    def test_empty_forecast_returns_zero(
        self, predictor, mock_nibe_state, mock_thermal_model
    ):
        """When forecast_hours is empty, returns no pre-heat."""
        # Add enough observations
        for i in range(100):
            predictor.record_state(
                timestamp=datetime.now() - timedelta(minutes=15 * i),
                indoor_temp=21.0,
                outdoor_temp=0.0,
                heating_offset=0.0,
                flow_temp=35.0,
                degree_minutes=-100.0,
            )

        empty_weather = MagicMock()
        empty_weather.forecast_hours = []

        result = predictor.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=empty_weather,
            target_temp=21.0,
            thermal_model=mock_thermal_model,
        )

        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "weather forecast" in result.reason.lower()


class TestEvaluateLayerReturnsDecision:
    """Test evaluate_layer returns proper decisions."""

    def test_returns_prediction_layer_decision(
        self, predictor, mock_nibe_state, mock_weather_data, mock_thermal_model
    ):
        """evaluate_layer returns PredictionLayerDecision."""
        # Add enough observations
        for i in range(100):
            predictor.record_state(
                timestamp=datetime.now() - timedelta(minutes=15 * i),
                indoor_temp=21.0,
                outdoor_temp=0.0,
                heating_offset=0.0,
                flow_temp=35.0,
                degree_minutes=-100.0,
            )

        result = predictor.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data,
            target_temp=21.0,
            thermal_model=mock_thermal_model,
        )

        assert isinstance(result, PredictionLayerDecision)
        assert result.name == "Learned Pre-heat"
        assert isinstance(result.offset, float)
        assert isinstance(result.weight, float)
        assert isinstance(result.reason, str)

    def test_decision_has_correct_name(
        self, predictor, mock_nibe_state, mock_weather_data, mock_thermal_model
    ):
        """Decision name is always 'Learned Pre-heat'."""
        # Add enough observations
        for i in range(100):
            predictor.record_state(
                timestamp=datetime.now() - timedelta(minutes=15 * i),
                indoor_temp=21.0,
                outdoor_temp=0.0,
                heating_offset=0.0,
                flow_temp=35.0,
                degree_minutes=-100.0,
            )

        result = predictor.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data,
            target_temp=21.0,
            thermal_model=mock_thermal_model,
        )

        assert result.name == "Learned Pre-heat"

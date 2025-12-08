"""Tests for WeatherPredictionLayer.evaluate_layer() method.

Phase 4 of layer refactor: Tests for the extracted weather prediction layer evaluation logic.
"""

import pytest
from unittest.mock import MagicMock

from custom_components.effektguard.optimization.weather_layer import (
    WeatherPredictionLayer,
    WeatherLayerDecision,
)
from custom_components.effektguard.const import (
    LAYER_WEIGHT_WEATHER_PREDICTION,
    WEATHER_GENTLE_OFFSET,
    WEATHER_WEIGHT_CAP,
)


@pytest.fixture
def weather_layer():
    """Create a WeatherPredictionLayer instance with default thermal mass."""
    return WeatherPredictionLayer(thermal_mass=1.0)


@pytest.fixture
def heavy_thermal_mass_layer():
    """Create a WeatherPredictionLayer with heavy thermal mass (concrete slab)."""
    return WeatherPredictionLayer(thermal_mass=1.5)


@pytest.fixture
def mock_nibe_state():
    """Create mock NIBE state."""
    state = MagicMock()
    state.indoor_temp = 21.0
    state.outdoor_temp = 0.0
    state.degree_minutes = -100.0
    return state


@pytest.fixture
def mock_weather_data_dropping():
    """Create mock weather data with temperature drop forecast."""
    weather = MagicMock()
    # Temperature dropping 6°C over forecast period
    weather.forecast_hours = [
        MagicMock(temperature=0.0),
        MagicMock(temperature=-1.0),
        MagicMock(temperature=-2.0),
        MagicMock(temperature=-3.0),
        MagicMock(temperature=-4.0),
        MagicMock(temperature=-5.0),
        MagicMock(temperature=-6.0),  # -6°C drop
    ]
    return weather


@pytest.fixture
def mock_weather_data_stable():
    """Create mock weather data with stable forecast."""
    weather = MagicMock()
    # Temperature stable around 0°C
    weather.forecast_hours = [
        MagicMock(temperature=0.0),
        MagicMock(temperature=0.5),
        MagicMock(temperature=-0.5),
        MagicMock(temperature=0.0),
    ]
    return weather


class TestEvaluateLayerDisabled:
    """Test evaluate_layer when weather prediction is disabled."""

    def test_disabled_returns_zero_offset(
        self, weather_layer, mock_nibe_state, mock_weather_data_dropping
    ):
        """When disabled, returns offset=0.0, weight=0.0."""
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = weather_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data_dropping,
            thermal_trend=thermal_trend,
            enable_weather_prediction=False,
        )

        assert isinstance(result, WeatherLayerDecision)
        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "Disabled by user" in result.reason

    def test_disabled_returns_correct_name(
        self, weather_layer, mock_nibe_state, mock_weather_data_dropping
    ):
        """Disabled layer still returns proper name."""
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = weather_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data_dropping,
            thermal_trend=thermal_trend,
            enable_weather_prediction=False,
        )

        assert result.name == "Weather Pre-heat"


class TestEvaluateLayerNoData:
    """Test evaluate_layer when weather data unavailable."""

    def test_no_weather_data_returns_zero(self, weather_layer, mock_nibe_state):
        """When weather_data is None, returns zero."""
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = weather_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=None,
            thermal_trend=thermal_trend,
            enable_weather_prediction=True,
        )

        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "No weather data" in result.reason

    def test_empty_forecast_returns_zero(self, weather_layer, mock_nibe_state):
        """When forecast_hours is empty, returns zero."""
        empty_weather = MagicMock()
        empty_weather.forecast_hours = []
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = weather_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=empty_weather,
            thermal_trend=thermal_trend,
            enable_weather_prediction=True,
        )

        assert result.offset == 0.0
        assert result.weight == 0.0
        # Could be "No weather data" or "No forecast data" depending on check order
        assert "no" in result.reason.lower() or "data" in result.reason.lower()


class TestEvaluateLayerForecastTrigger:
    """Test evaluate_layer forecast-triggered pre-heating."""

    def test_large_temp_drop_triggers_preheat(
        self, weather_layer, mock_nibe_state, mock_weather_data_dropping
    ):
        """Temperature drop >= 5°C triggers pre-heating."""
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = weather_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data_dropping,
            thermal_trend=thermal_trend,
            enable_weather_prediction=True,
        )

        assert result.offset == WEATHER_GENTLE_OFFSET  # +0.5°C
        assert result.weight > 0
        assert "proactive" in result.reason.lower() or "drop" in result.reason.lower()

    def test_stable_weather_no_preheat(
        self, weather_layer, mock_nibe_state, mock_weather_data_stable
    ):
        """Stable weather doesn't trigger pre-heating."""
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = weather_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data_stable,
            thermal_trend=thermal_trend,
            enable_weather_prediction=True,
        )

        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "stable" in result.reason.lower()


class TestEvaluateLayerIndoorCoolingTrigger:
    """Test evaluate_layer indoor cooling confirmation trigger."""

    def test_indoor_cooling_triggers_preheat(
        self, weather_layer, mock_nibe_state, mock_weather_data_stable
    ):
        """Indoor cooling >= 0.5°C/h triggers pre-heating even with stable forecast."""
        # Rapid indoor cooling confirms heat loss
        thermal_trend = {"rate_per_hour": -0.6, "confidence": 0.8}

        result = weather_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data_stable,
            thermal_trend=thermal_trend,
            enable_weather_prediction=True,
        )

        assert result.offset == WEATHER_GENTLE_OFFSET
        assert result.weight > 0
        assert "cooling" in result.reason.lower()


class TestEvaluateLayerThermalMassWeight:
    """Test weight scaling by thermal mass."""

    def test_heavy_thermal_mass_higher_weight(
        self, heavy_thermal_mass_layer, mock_nibe_state, mock_weather_data_dropping
    ):
        """Heavy thermal mass (1.5) gets higher weight."""
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = heavy_thermal_mass_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data_dropping,
            thermal_trend=thermal_trend,
            enable_weather_prediction=True,
        )

        # Weight = LAYER_WEIGHT_WEATHER_PREDICTION * 1.5 (capped at WEATHER_WEIGHT_CAP)
        expected_weight = min(LAYER_WEIGHT_WEATHER_PREDICTION * 1.5, WEATHER_WEIGHT_CAP)
        assert result.weight == expected_weight

    def test_light_thermal_mass_lower_weight(
        self, mock_nibe_state, mock_weather_data_dropping
    ):
        """Light thermal mass (0.5) gets lower weight."""
        light_layer = WeatherPredictionLayer(thermal_mass=0.5)
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = light_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data_dropping,
            thermal_trend=thermal_trend,
            enable_weather_prediction=True,
        )

        expected_weight = LAYER_WEIGHT_WEATHER_PREDICTION * 0.5
        assert result.weight == expected_weight

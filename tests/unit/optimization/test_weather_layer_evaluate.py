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
    # Twelve hours as now, then a day 10 °C colder: the daily mean drops 5 °C within 12 h
    weather.forecast_hours = [MagicMock(temperature=t) for t in [0.0] * 12 + [-10.0] * 36]
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


@pytest.mark.parametrize("past_day_mean", [None, 12.0])
def test_a_cold_night_is_not_a_cold_snap(weather_layer, mock_nibe_state, past_day_mean):
    """May: 16 °C afternoons, 8 °C nights. The night is 8 °C below now, but no day is colder."""
    mock_nibe_state.outdoor_temp = 16.0
    weather = MagicMock()
    weather.forecast_hours = [
        MagicMock(temperature=16.0 if 8 <= hour % 24 < 20 else 8.0) for hour in range(48)
    ]

    result = weather_layer.evaluate_layer(
        nibe_state=mock_nibe_state,
        weather_data=weather,
        thermal_trend={"rate_per_hour": 0.0, "confidence": 0.8},
        enable_weather_prediction=True,
        past_day_mean=past_day_mean,
    )

    assert result.offset == 0.0, f"pre-heated for a cold night: {result.reason}"


def test_a_front_hours_away_is_a_cold_snap_for_a_radiator_house(mock_nibe_state):
    """A fast house still pre-heats for a front three hours out, measured against the past day."""
    radiators = WeatherPredictionLayer(thermal_mass=0.5)
    weather = MagicMock()
    weather.forecast_hours = [MagicMock(temperature=t) for t in [0.0] * 3 + [-6.0] * 45]

    result = radiators.evaluate_layer(
        nibe_state=mock_nibe_state,
        weather_data=weather,
        thermal_trend={"rate_per_hour": 0.0, "confidence": 0.8},
        enable_weather_prediction=True,
        past_day_mean=0.0,
    )

    assert result.offset == WEATHER_GENTLE_OFFSET, result.reason


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

    def test_light_thermal_mass_lower_weight(self, mock_nibe_state, mock_weather_data_dropping):
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


def test_the_past_day_is_measured_once_a_day_of_history_exists():
    from custom_components.effektguard.const import SAMPLES_PER_HOUR
    from custom_components.effektguard.optimization.decision_engine import DecisionEngine

    engine = DecisionEngine(MagicMock(), MagicMock(), MagicMock(), config={})
    day = 24 * SAMPLES_PER_HOUR
    engine.predictor = MagicMock(state_history=[MagicMock(outdoor_temp=5.0)] * (day - 1))
    assert engine._outdoor_mean_past_day() is None
    engine.predictor.state_history = [MagicMock(outdoor_temp=-9.0)] + [
        MagicMock(outdoor_temp=5.0)
    ] * day
    assert engine._outdoor_mean_past_day() == 5.0

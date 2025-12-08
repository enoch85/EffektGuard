"""Tests for weather prediction layer (Phase 2 critical fix).

Tests the proactive pre-heating logic that ensures pre-heating starts 
before cold arrives, based on forecast drops or indoor cooling trends.
"""

import pytest
from unittest.mock import MagicMock

from custom_components.effektguard.optimization.weather_layer import (
    WeatherPredictionLayer,
)
from custom_components.effektguard.const import (
    WEATHER_FORECAST_DROP_THRESHOLD,
    WEATHER_INDOOR_COOLING_CONFIRMATION,
    WEATHER_GENTLE_OFFSET,
    LAYER_WEIGHT_WEATHER_PREDICTION,
    WEATHER_WEIGHT_CAP,
    WEATHER_FORECAST_HORIZON,
)


@pytest.fixture
def weather_layer():
    """Create weather prediction layer with default thermal mass."""
    return WeatherPredictionLayer(thermal_mass=1.0)


@pytest.fixture
def nibe_state_mock():
    """Create mocked NIBE state."""
    state = MagicMock()
    state.outdoor_temp = 0.0
    state.indoor_temp = 22.0
    state.degree_minutes = -100.0
    return state


@pytest.fixture
def weather_data_mock():
    """Create mocked weather data."""
    weather = MagicMock()
    # Default stable forecast
    weather.forecast_hours = [
        MagicMock(temperature=0.0) for _ in range(int(WEATHER_FORECAST_HORIZON) + 2)
    ]
    return weather


class TestWeatherPredictionLayer:
    """Test WeatherPredictionLayer logic."""

    def test_stable_conditions_no_action(
        self, weather_layer, nibe_state_mock, weather_data_mock
    ):
        """Stable forecast and indoor temp -> no pre-heating."""
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        decision = weather_layer.evaluate_layer(
            nibe_state_mock, weather_data_mock, thermal_trend
        )

        assert decision.offset == 0.0
        assert decision.weight == 0.0
        assert "stable" in decision.reason.lower()

    def test_forecast_drop_triggers_preheat(
        self, weather_layer, nibe_state_mock, weather_data_mock
    ):
        """Forecast drop >= threshold triggers pre-heating."""
        # Create a drop in forecast
        drop_temp = nibe_state_mock.outdoor_temp + WEATHER_FORECAST_DROP_THRESHOLD - 1.0
        # Set forecast to have a drop
        weather_data_mock.forecast_hours = [
            MagicMock(temperature=drop_temp) for _ in range(int(WEATHER_FORECAST_HORIZON))
        ]
        
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        decision = weather_layer.evaluate_layer(
            nibe_state_mock, weather_data_mock, thermal_trend
        )

        assert decision.offset == pytest.approx(WEATHER_GENTLE_OFFSET)
        assert decision.weight > 0.0
        assert "forecast" in decision.reason.lower()
        assert "drop" in decision.reason.lower()

    def test_indoor_cooling_triggers_preheat(
        self, weather_layer, nibe_state_mock, weather_data_mock
    ):
        """Indoor cooling >= threshold triggers pre-heating (confirmation)."""
        # Stable forecast
        thermal_trend = {
            "rate_per_hour": WEATHER_INDOOR_COOLING_CONFIRMATION - 0.1, # Cooling faster than threshold (negative)
            "confidence": 0.8
        }

        decision = weather_layer.evaluate_layer(
            nibe_state_mock, weather_data_mock, thermal_trend
        )

        assert decision.offset == pytest.approx(WEATHER_GENTLE_OFFSET)
        assert decision.weight > 0.0
        assert "indoor cooling" in decision.reason.lower()

    def test_combined_triggers_preheat(
        self, weather_layer, nibe_state_mock, weather_data_mock
    ):
        """Both forecast drop and indoor cooling trigger pre-heating."""
        # Forecast drop
        drop_temp = nibe_state_mock.outdoor_temp + WEATHER_FORECAST_DROP_THRESHOLD - 1.0
        weather_data_mock.forecast_hours = [
            MagicMock(temperature=drop_temp) for _ in range(int(WEATHER_FORECAST_HORIZON))
        ]
        
        # Indoor cooling
        thermal_trend = {
            "rate_per_hour": WEATHER_INDOOR_COOLING_CONFIRMATION - 0.1,
            "confidence": 0.8
        }

        decision = weather_layer.evaluate_layer(
            nibe_state_mock, weather_data_mock, thermal_trend
        )

        assert decision.offset == pytest.approx(WEATHER_GENTLE_OFFSET)
        assert decision.weight > 0.0
        assert "confirmed" in decision.reason.lower()

    def test_thermal_mass_scaling_concrete(
        self, nibe_state_mock, weather_data_mock
    ):
        """Test weight scaling for concrete slab (high thermal mass)."""
        layer = WeatherPredictionLayer(thermal_mass=1.5) # Concrete
        
        # Trigger pre-heat
        drop_temp = nibe_state_mock.outdoor_temp + WEATHER_FORECAST_DROP_THRESHOLD - 1.0
        weather_data_mock.forecast_hours = [
            MagicMock(temperature=drop_temp) for _ in range(int(WEATHER_FORECAST_HORIZON))
        ]
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        decision = layer.evaluate_layer(
            nibe_state_mock, weather_data_mock, thermal_trend
        )

        expected_weight = min(LAYER_WEIGHT_WEATHER_PREDICTION * 1.5, WEATHER_WEIGHT_CAP)
        assert decision.weight == pytest.approx(expected_weight)

    def test_thermal_mass_scaling_radiators(
        self, nibe_state_mock, weather_data_mock
    ):
        """Test weight scaling for radiators (low thermal mass)."""
        layer = WeatherPredictionLayer(thermal_mass=0.5) # Radiators
        
        # Trigger pre-heat
        drop_temp = nibe_state_mock.outdoor_temp + WEATHER_FORECAST_DROP_THRESHOLD - 1.0
        weather_data_mock.forecast_hours = [
            MagicMock(temperature=drop_temp) for _ in range(int(WEATHER_FORECAST_HORIZON))
        ]
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        decision = layer.evaluate_layer(
            nibe_state_mock, weather_data_mock, thermal_trend
        )

        expected_weight = min(LAYER_WEIGHT_WEATHER_PREDICTION * 0.5, WEATHER_WEIGHT_CAP)
        assert decision.weight == pytest.approx(expected_weight)

    def test_disabled_layer(
        self, weather_layer, nibe_state_mock, weather_data_mock
    ):
        """Test disabled layer returns 0."""
        # Trigger condition present
        drop_temp = nibe_state_mock.outdoor_temp + WEATHER_FORECAST_DROP_THRESHOLD - 1.0
        weather_data_mock.forecast_hours = [
            MagicMock(temperature=drop_temp) for _ in range(int(WEATHER_FORECAST_HORIZON))
        ]
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        decision = weather_layer.evaluate_layer(
            nibe_state_mock, weather_data_mock, thermal_trend, enable_weather_prediction=False
        )

        assert decision.offset == 0.0
        assert decision.weight == 0.0
        assert "disabled" in decision.reason.lower()

    def test_no_weather_data(
        self, weather_layer, nibe_state_mock
    ):
        """Test missing weather data handles gracefully."""
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        decision = weather_layer.evaluate_layer(
            nibe_state_mock, None, thermal_trend
        )

        assert decision.offset == 0.0
        assert decision.weight == 0.0
        assert "no weather data" in decision.reason.lower()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

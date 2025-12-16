"""Tests for WeatherCompensationLayer.evaluate_layer() method.

Phase 5 of layer refactoring: Mathematical weather compensation extraction.
"""

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.const import (
    DEFAULT_WEATHER_COMPENSATION_WEIGHT,
    WEATHER_COMP_DEFER_DM_CRITICAL,
    WEATHER_COMP_DEFER_DM_LIGHT,
    WEATHER_COMP_DEFER_WEIGHT_CRITICAL,
)
from custom_components.effektguard.optimization.weather_layer import (
    AdaptiveClimateSystem,
    WeatherCompensationCalculator,
    WeatherCompensationLayer,
    WeatherCompensationLayerDecision,
)


@dataclass
class MockNibeState:
    """Mock NIBE state for testing."""

    outdoor_temp: float = 0.0
    indoor_temp: float = 21.0
    flow_temp: float = 35.0
    degree_minutes: float = 0.0
    is_hot_water: bool = False  # DHW/lux heating status


@dataclass
class MockForecastHour:
    """Mock forecast hour for testing."""

    temperature: float


@dataclass
class MockWeatherData:
    """Mock weather data for testing."""

    forecast_hours: list = None


@dataclass
class MockUnusualWeather:
    """Mock unusual weather result."""

    is_unusual: bool = False
    severity: str = "moderate"
    deviation_from_typical: float = 0.0
    recommendation: str = ""


class TestWeatherCompensationLayerEvaluate:
    """Test suite for WeatherCompensationLayer.evaluate_layer()."""

    def _create_layer(
        self,
        heat_loss_coefficient: float = 200.0,
        heating_type: str = "radiator",
        latitude: float = 59.33,  # Stockholm
        weather_learner=None,
        weather_comp_weight: float = 1.0,
    ) -> WeatherCompensationLayer:
        """Create a WeatherCompensationLayer for testing."""
        weather_comp = WeatherCompensationCalculator(
            heat_loss_coefficient=heat_loss_coefficient,
            heating_type=heating_type,
        )
        climate_system = AdaptiveClimateSystem(latitude=latitude)

        return WeatherCompensationLayer(
            weather_comp=weather_comp,
            climate_system=climate_system,
            weather_learner=weather_learner,
            weather_comp_weight=weather_comp_weight,
        )

    def test_disabled_returns_zero(self):
        """Test that disabled feature returns zero offset/weight."""
        layer = self._create_layer()
        nibe_state = MockNibeState()
        weather_data = MockWeatherData(
            forecast_hours=[MockForecastHour(temperature=-5.0)]
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=False,
        )

        assert result.name == "Math WC"
        assert result.offset == 0.0
        assert result.weight == 0.0
        assert result.reason == "Disabled"

    def test_no_weather_data_returns_zero(self):
        """Test that missing weather data returns zero offset/weight."""
        layer = self._create_layer()
        nibe_state = MockNibeState()

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
            enable_weather_compensation=True,
        )

        assert result.name == "Math WC"
        assert result.offset == 0.0
        assert result.weight == 0.0
        assert result.reason == "No weather data"

    def test_empty_forecast_returns_zero(self):
        """Test that empty forecast returns zero offset/weight."""
        layer = self._create_layer()
        nibe_state = MockNibeState()
        weather_data = MockWeatherData(forecast_hours=[])

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
        )

        assert result.name == "Math WC"
        assert result.offset == 0.0
        assert result.weight == 0.0
        assert result.reason == "No weather data"

    def test_returns_weather_compensation_layer_decision(self):
        """Test that result is WeatherCompensationLayerDecision with diagnostic fields."""
        layer = self._create_layer()
        nibe_state = MockNibeState(outdoor_temp=-5.0, flow_temp=35.0)
        weather_data = MockWeatherData(
            forecast_hours=[MockForecastHour(temperature=-5.0)]
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
        )

        assert isinstance(result, WeatherCompensationLayerDecision)
        assert result.name == "Math WC"
        assert hasattr(result, "optimal_flow_temp")
        assert hasattr(result, "adjusted_flow_temp")
        assert hasattr(result, "safety_margin")
        assert hasattr(result, "unusual_weather")
        assert hasattr(result, "defer_factor")

    def test_calculates_offset_based_on_flow_temp_difference(self):
        """Test that offset is calculated from optimal vs current flow temp."""
        layer = self._create_layer()
        # Low outdoor temp should calculate higher optimal flow temp
        nibe_state = MockNibeState(outdoor_temp=-10.0, flow_temp=30.0)
        weather_data = MockWeatherData(
            forecast_hours=[MockForecastHour(temperature=-10.0)]
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
        )

        # Optimal flow should be higher than current 30°C at -10°C outdoor
        # So offset should be positive
        assert result.offset > 0.0
        assert result.optimal_flow_temp > 30.0

    def test_weight_increased_during_cold_weather(self):
        """Test that weight is higher during cold weather."""
        layer = self._create_layer()
        nibe_state_cold = MockNibeState(outdoor_temp=-15.0, flow_temp=35.0)
        nibe_state_warm = MockNibeState(outdoor_temp=10.0, flow_temp=25.0)
        weather_cold = MockWeatherData(
            forecast_hours=[MockForecastHour(temperature=-15.0)]
        )
        weather_warm = MockWeatherData(
            forecast_hours=[MockForecastHour(temperature=10.0)]
        )

        result_cold = layer.evaluate_layer(
            nibe_state=nibe_state_cold,
            weather_data=weather_cold,
            target_temp=21.0,
            enable_weather_compensation=True,
        )
        result_warm = layer.evaluate_layer(
            nibe_state=nibe_state_warm,
            weather_data=weather_warm,
            target_temp=21.0,
            enable_weather_compensation=True,
        )

        # Weight should be higher in cold conditions
        assert result_cold.weight > result_warm.weight


class TestWeatherCompensationDeferral:
    """Test thermal debt deferral logic."""

    def _create_layer(self) -> WeatherCompensationLayer:
        """Create a layer with default settings."""
        weather_comp = WeatherCompensationCalculator(
            heat_loss_coefficient=200.0,
            heating_type="radiator",
        )
        climate_system = AdaptiveClimateSystem(latitude=59.33)

        return WeatherCompensationLayer(
            weather_comp=weather_comp,
            climate_system=climate_system,
            weather_learner=None,
            weather_comp_weight=1.0,
        )

    def test_no_deferral_when_no_thermal_debt(self):
        """Test that no deferral occurs when DM is positive."""
        layer = self._create_layer()
        nibe_state = MockNibeState(degree_minutes=0.0, outdoor_temp=-5.0, flow_temp=35.0)
        weather_data = MockWeatherData(
            forecast_hours=[MockForecastHour(temperature=-5.0)]
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
        )

        assert result.defer_factor == 1.0
        assert "Deferred" not in result.reason

    def test_light_deferral_at_light_threshold(self):
        """Test light deferral when DM crosses light threshold."""
        layer = self._create_layer()
        nibe_state = MockNibeState(
            degree_minutes=WEATHER_COMP_DEFER_DM_LIGHT - 10,
            outdoor_temp=-5.0,
            flow_temp=35.0,
        )
        weather_data = MockWeatherData(
            forecast_hours=[MockForecastHour(temperature=-5.0)]
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
        )

        # Should have light deferral
        assert result.defer_factor < 1.0
        assert "Light debt" in result.reason

    def test_critical_deferral_at_critical_threshold(self):
        """Test critical deferral when DM crosses critical threshold."""
        layer = self._create_layer()
        nibe_state = MockNibeState(
            degree_minutes=WEATHER_COMP_DEFER_DM_CRITICAL - 10,
            outdoor_temp=-5.0,
            flow_temp=35.0,
        )
        weather_data = MockWeatherData(
            forecast_hours=[MockForecastHour(temperature=-5.0)]
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
        )

        # Critical deferral should reduce weight significantly
        expected_factor = WEATHER_COMP_DEFER_WEIGHT_CRITICAL / DEFAULT_WEATHER_COMPENSATION_WEIGHT
        assert result.defer_factor == pytest.approx(expected_factor, rel=0.01)
        assert "Critical debt" in result.reason

    def test_deferral_reduces_weight(self):
        """Test that deferral actually reduces the final weight."""
        layer = self._create_layer()
        weather_data = MockWeatherData(
            forecast_hours=[MockForecastHour(temperature=-5.0)]
        )

        # No debt
        nibe_no_debt = MockNibeState(degree_minutes=0.0, outdoor_temp=-5.0, flow_temp=35.0)
        result_no_debt = layer.evaluate_layer(
            nibe_state=nibe_no_debt,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
        )

        # Critical debt
        nibe_critical = MockNibeState(
            degree_minutes=WEATHER_COMP_DEFER_DM_CRITICAL - 10,
            outdoor_temp=-5.0,
            flow_temp=35.0,
        )
        result_critical = layer.evaluate_layer(
            nibe_state=nibe_critical,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
        )

        # Weight with critical debt should be much lower
        assert result_critical.weight < result_no_debt.weight


class TestWeatherCompensationUnusualWeather:
    """Test unusual weather detection integration."""

    def test_unusual_weather_increases_safety_margin(self):
        """Test that unusual weather detection increases safety margin."""
        # Create mock weather learner
        mock_learner = MagicMock()
        mock_unusual = MockUnusualWeather(
            is_unusual=True,
            severity="extreme",
            deviation_from_typical=5.0,
            recommendation="Prepare for cold snap",
        )
        mock_learner.detect_unusual_weather.return_value = mock_unusual

        weather_comp = WeatherCompensationCalculator(
            heat_loss_coefficient=200.0,
            heating_type="radiator",
        )
        climate_system = AdaptiveClimateSystem(latitude=59.33)
        layer = WeatherCompensationLayer(
            weather_comp=weather_comp,
            climate_system=climate_system,
            weather_learner=mock_learner,
            weather_comp_weight=1.0,
        )

        nibe_state = MockNibeState(outdoor_temp=-5.0, flow_temp=35.0)
        weather_data = MockWeatherData(
            forecast_hours=[MockForecastHour(temperature=-5.0)]
        )

        # Provide a callable for get_current_datetime to avoid dt_util import
        mock_datetime = datetime(2024, 1, 15, 12, 0, 0)
        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
            get_current_datetime=lambda: mock_datetime,
        )

        # Should have detected unusual weather
        assert result.unusual_weather is True
        assert result.safety_margin > 0.0
        assert "Unusual weather" in result.reason

    def test_normal_weather_no_extra_margin(self):
        """Test that normal weather doesn't add unusual weather margin."""
        # Create mock weather learner that returns normal weather
        mock_learner = MagicMock()
        mock_normal = MockUnusualWeather(
            is_unusual=False,
            severity="none",
            deviation_from_typical=0.5,
            recommendation="Normal conditions",
        )
        mock_learner.detect_unusual_weather.return_value = mock_normal

        weather_comp = WeatherCompensationCalculator(
            heat_loss_coefficient=200.0,
            heating_type="radiator",
        )
        climate_system = AdaptiveClimateSystem(latitude=59.33)
        layer = WeatherCompensationLayer(
            weather_comp=weather_comp,
            climate_system=climate_system,
            weather_learner=mock_learner,
            weather_comp_weight=1.0,
        )

        nibe_state = MockNibeState(outdoor_temp=-5.0, flow_temp=35.0)
        weather_data = MockWeatherData(
            forecast_hours=[MockForecastHour(temperature=-5.0)]
        )

        mock_datetime = datetime(2024, 1, 15, 12, 0, 0)
        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
            get_current_datetime=lambda: mock_datetime,
        )

        assert result.unusual_weather is False
        assert "Unusual weather" not in result.reason


class TestWeatherCompensationUserWeight:
    """Test user-configured weight adjustment."""

    def test_weight_scaled_by_user_config(self):
        """Test that user weight config scales final weight."""
        # Half weight
        layer_half = WeatherCompensationLayer(
            weather_comp=WeatherCompensationCalculator(heat_loss_coefficient=200.0),
            climate_system=AdaptiveClimateSystem(latitude=59.33),
            weather_learner=None,
            weather_comp_weight=0.5,
        )

        # Full weight
        layer_full = WeatherCompensationLayer(
            weather_comp=WeatherCompensationCalculator(heat_loss_coefficient=200.0),
            climate_system=AdaptiveClimateSystem(latitude=59.33),
            weather_learner=None,
            weather_comp_weight=1.0,
        )

        nibe_state = MockNibeState(outdoor_temp=-5.0, flow_temp=35.0)
        weather_data = MockWeatherData(
            forecast_hours=[MockForecastHour(temperature=-5.0)]
        )

        result_half = layer_half.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
        )

        result_full = layer_full.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
        )

        # Half weight should be approximately half of full weight
        assert result_half.weight == pytest.approx(result_full.weight * 0.5, rel=0.01)

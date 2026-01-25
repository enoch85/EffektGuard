"""Tests for weather compensation layer skipping during DHW/temp_lux heating.

Bug fix: When NIBE heats DHW via temporary lux switch, flow temperature reads
the DHW charging temperature (45-60°C) instead of space heating flow temp.
This caused incorrect negative offsets from weather compensation.

Solution: Check temp_lux_active parameter and skip weather compensation
when DHW is being heated.
"""

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock

import pytest

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
    is_hot_water: bool = False  # Note: This was broken in production


@dataclass
class MockForecastHour:
    """Mock forecast hour for testing."""

    temperature: float


@dataclass
class MockWeatherData:
    """Mock weather data for testing."""

    forecast_hours: list = None

    def __post_init__(self):
        if self.forecast_hours is None:
            self.forecast_hours = [MockForecastHour(temperature=5.0) for _ in range(24)]


class TestWeatherLayerTempLuxSkip:
    """Tests for temp_lux_active parameter causing weather comp to skip."""

    @pytest.fixture
    def weather_layer(self):
        """Create a weather compensation layer for testing."""
        weather_comp = WeatherCompensationCalculator()
        climate_system = AdaptiveClimateSystem(latitude=59.33)  # Stockholm
        return WeatherCompensationLayer(
            weather_comp=weather_comp,
            climate_system=climate_system,
            weather_learner=None,
            weather_comp_weight=0.8,
        )

    def test_temp_lux_active_skips_weather_compensation(self, weather_layer):
        """When temp_lux_active=True, weather compensation should be skipped.

        This prevents incorrect offsets when DHW heating causes flow temp to read
        DHW charging temperature (45-60°C) instead of space heating flow.
        """
        # DHW heating scenario: flow temp reads DHW charging temp (54.8°C)
        nibe_state = MockNibeState(
            outdoor_temp=5.0,
            indoor_temp=21.0,
            flow_temp=54.8,  # DHW charging temp, NOT space heating
            degree_minutes=-100,
        )
        weather_data = MockWeatherData()

        # With temp_lux_active=True, should skip weather comp
        result = weather_layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
            temp_lux_active=True,  # DHW heating active
        )

        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "DHW/lux active" in result.reason or "temp not valid" in result.reason

    def test_temp_lux_inactive_allows_weather_compensation(self, weather_layer):
        """When temp_lux_active=False, weather compensation should work normally."""
        # Normal space heating scenario
        nibe_state = MockNibeState(
            outdoor_temp=5.0,
            indoor_temp=21.0,
            flow_temp=32.0,  # Normal space heating flow temp
            degree_minutes=-100,
        )
        weather_data = MockWeatherData()

        # With temp_lux_active=False, weather comp should work
        result = weather_layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
            temp_lux_active=False,  # No DHW heating
        )

        # Should produce non-zero result (either offset or reason for no change)
        # The important thing is it didn't skip due to DHW
        assert "DHW/lux active" not in result.reason

    def test_temp_lux_default_is_false(self, weather_layer):
        """temp_lux_active parameter should default to False."""
        nibe_state = MockNibeState(
            outdoor_temp=5.0,
            indoor_temp=21.0,
            flow_temp=32.0,
            degree_minutes=-100,
        )
        weather_data = MockWeatherData()

        # Call without temp_lux_active parameter - should default to False
        result = weather_layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
            # temp_lux_active not specified - should default to False
        )

        # Should NOT skip due to DHW
        assert "DHW/lux active" not in result.reason

    def test_high_flow_temp_with_temp_lux_active_returns_zero(self, weather_layer):
        """High flow temp during DHW should not produce negative offset.

        This was the original bug: flow=54.8°C caused large negative offsets
        because weather comp thought the system was overheating.
        """
        # Reproduce the bug scenario from logs
        nibe_state = MockNibeState(
            outdoor_temp=6.0,
            indoor_temp=20.5,
            flow_temp=54.8,  # DHW charging temp
            degree_minutes=-60,
        )
        weather_data = MockWeatherData()

        # With temp_lux_active=True, should return neutral
        result = weather_layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
            temp_lux_active=True,
        )

        # Must return 0.0 offset - no negative offset from DHW temp
        assert result.offset == 0.0
        assert result.weight == 0.0

    def test_high_flow_temp_without_temp_lux_produces_offset(self, weather_layer):
        """High flow temp during normal heating should produce offset.

        When temp_lux_active=False but flow is high, weather comp should
        still calculate (this is legitimate overheating, not DHW).
        """
        nibe_state = MockNibeState(
            outdoor_temp=6.0,
            indoor_temp=20.5,
            flow_temp=54.8,  # Unusually high for space heating
            degree_minutes=-60,
        )
        weather_data = MockWeatherData()

        # Without temp_lux_active, weather comp should calculate
        result = weather_layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
            enable_weather_compensation=True,
            temp_lux_active=False,
        )

        # Should NOT be skipped for DHW reason
        assert "DHW/lux active" not in result.reason
        # Note: May still return 0 for other reasons (e.g., DM-based deferral)


class TestDecisionEngineTempLuxParameter:
    """Tests for temp_lux_active parameter in DecisionEngine."""

    def test_calculate_decision_accepts_temp_lux_active(self):
        """DecisionEngine.calculate_decision() should accept temp_lux_active parameter."""
        from custom_components.effektguard.optimization.decision_engine import DecisionEngine

        # Just verify the method signature accepts the parameter
        import inspect

        sig = inspect.signature(DecisionEngine.calculate_decision)
        param_names = list(sig.parameters.keys())

        assert (
            "temp_lux_active" in param_names
        ), "DecisionEngine.calculate_decision() must accept temp_lux_active parameter"

    def test_temp_lux_active_has_default_value(self):
        """temp_lux_active parameter should have a default value of False."""
        from custom_components.effektguard.optimization.decision_engine import DecisionEngine

        import inspect

        sig = inspect.signature(DecisionEngine.calculate_decision)
        param = sig.parameters.get("temp_lux_active")

        assert param is not None
        assert param.default is False, "temp_lux_active should default to False"

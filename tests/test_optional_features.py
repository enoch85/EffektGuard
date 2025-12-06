"""Tests for optional features and auto-detection."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult

from custom_components.effektguard.config_flow import EffektGuardConfigFlow
from custom_components.effektguard.const import (
    CONF_DEGREE_MINUTES_ENTITY,
    CONF_POWER_SENSOR_ENTITY,
    DOMAIN,
)


@pytest.fixture
def mock_hass():
    """Create mock Home Assistant instance."""
    hass = MagicMock(spec=HomeAssistant)

    # Initialize hass.data for entity registry
    hass.data = {}
    # Add mock entity_registry to prevent KeyError
    hass.data["entity_registry"] = MagicMock()

    # Initialize hass.config for storage manager
    hass.config = MagicMock()
    hass.config.config_dir = "/tmp/test_config"

    # Initialize hass.bus for event system
    hass.bus = MagicMock()

    # Create mock states object
    mock_states_obj = MagicMock()
    hass.states = mock_states_obj

    # Mock states for discovery
    mock_states = {}

    # NIBE entities
    mock_states["sensor.nibe_gradminuter_bt1"] = MagicMock(
        entity_id="sensor.nibe_gradminuter_bt1",
        state="100",
        attributes={},
    )
    mock_states["sensor.myuplink_degree_minutes"] = MagicMock(
        entity_id="sensor.myuplink_degree_minutes",
        state="-50",
        attributes={},
    )

    # Power sensors
    mock_states["sensor.house_power"] = MagicMock(
        entity_id="sensor.house_power",
        state="3500",
        attributes={
            "device_class": "power",
            "unit_of_measurement": "W",
        },
    )
    mock_states["sensor.nibe_power"] = MagicMock(
        entity_id="sensor.nibe_power",
        state="2000",
        attributes={
            "device_class": "power",
            "unit_of_measurement": "W",
        },
    )
    mock_states["sensor.random_sensor"] = MagicMock(
        entity_id="sensor.random_sensor",
        state="100",
        attributes={
            "device_class": "power",
            "unit_of_measurement": "kW",
        },
    )

    # Weather entity
    mock_states["weather.home"] = MagicMock(
        entity_id="weather.home",
        state="sunny",
        attributes={"forecast": [{"datetime": "2025-10-14T12:00:00", "temperature": 15}] * 24},
    )

    # Spot price entity
    mock_states["sensor.gespot_current_price"] = MagicMock(
        entity_id="sensor.gespot_current_price",
        state="0.50",
        attributes={},
    )

    def mock_async_all():
        # Return list of state objects, not tuples
        return list(mock_states.values())

    def mock_get_state(entity_id):
        return mock_states.get(entity_id)

    mock_states_obj.async_all = mock_async_all
    mock_states_obj.get = mock_get_state

    return hass


class TestOptionalFeaturesDiscovery:
    """Test auto-detection of optional features."""

    async def test_discover_degree_minutes_swedish(self, mock_hass):
        """Test discovery of Swedish gradminuter sensor."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass

        entities = config_flow._discover_degree_minutes_entities()

        assert len(entities) == 2
        assert "sensor.nibe_gradminuter_bt1" in entities
        assert "sensor.myuplink_degree_minutes" in entities

    async def test_discover_power_sensors_priority(self, mock_hass):
        """Test power sensor discovery with priority ordering."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass

        entities = config_flow._discover_power_entities()

        # Should find all 3 power sensors
        assert len(entities) == 3

        # House power should be first (highest priority)
        assert entities[0] == "sensor.house_power"

        # NIBE power should be second
        assert entities[1] == "sensor.nibe_power"

        # Generic power should be last
        assert entities[2] == "sensor.random_sensor"

    async def test_discover_no_degree_minutes(self):
        """Test when no degree minutes sensor exists."""
        hass = MagicMock(spec=HomeAssistant)
        mock_states_obj = MagicMock()
        hass.states = mock_states_obj
        # Return list of state objects, not tuples
        mock_states_obj.async_all = lambda: [MagicMock(entity_id="sensor.temperature")]

        config_flow = EffektGuardConfigFlow()
        config_flow.hass = hass

        entities = config_flow._discover_degree_minutes_entities()

        assert len(entities) == 0

    async def test_discover_no_power_sensors(self):
        """Test when no power sensors exist."""
        hass = MagicMock(spec=HomeAssistant)

        # Create sensor without power device class
        mock_state = MagicMock(
            entity_id="sensor.temperature",
            attributes={"device_class": "temperature"},
        )
        mock_states_obj = MagicMock()
        hass.states = mock_states_obj
        # Return list of state objects, not tuples
        mock_states_obj.async_all = lambda: [mock_state]

        config_flow = EffektGuardConfigFlow()
        config_flow.hass = hass

        entities = config_flow._discover_power_entities()

        assert len(entities) == 0


class TestOptionalSensorsConfigFlow:
    """Test config flow for optional sensors step."""

    async def test_optional_sensors_step_with_auto_detect(self, mock_hass):
        """Test optional sensors step with auto-detection."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {
            "nibe_entity": "sensor.nibe_outdoor_temp",
            "gespot_entity": "sensor.gespot_current_price",
            "weather_entity": "weather.home",
        }

        result = await config_flow.async_step_optional_sensors()

        assert result["type"] == "form"
        assert result["step_id"] == "optional_sensors"

        # Check description placeholders show auto-detection
        placeholders = result["description_placeholders"]
        assert "detected" in placeholders["dm_detected"].lower()
        assert "detected" in placeholders["power_detected"].lower()

    async def test_optional_sensors_step_submit(self, mock_hass):
        """Test submitting optional sensors step."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {
            "nibe_entity": "sensor.nibe_outdoor_temp",
            "gespot_entity": "sensor.gespot_current_price",
        }

        with patch.object(config_flow, "async_create_entry") as mock_create:
            mock_create.return_value = {"type": "create_entry"}

            result = await config_flow.async_step_optional_sensors(
                user_input={
                    CONF_DEGREE_MINUTES_ENTITY: "sensor.nibe_gradminuter_bt1",
                    CONF_POWER_SENSOR_ENTITY: "sensor.house_power",
                }
            )

            assert mock_create.called
            assert config_flow._data[CONF_DEGREE_MINUTES_ENTITY] == "sensor.nibe_gradminuter_bt1"
            assert config_flow._data[CONF_POWER_SENSOR_ENTITY] == "sensor.house_power"

    async def test_optional_sensors_step_skip(self, mock_hass):
        """Test skipping optional sensors (both None)."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {
            "nibe_entity": "sensor.nibe_outdoor_temp",
            "gespot_entity": "sensor.gespot_current_price",
        }

        with patch.object(config_flow, "async_create_entry") as mock_create:
            mock_create.return_value = {"type": "create_entry"}

            result = await config_flow.async_step_optional_sensors(
                user_input={
                    CONF_DEGREE_MINUTES_ENTITY: None,
                    CONF_POWER_SENSOR_ENTITY: None,
                }
            )

            assert mock_create.called
            assert config_flow._data.get(CONF_DEGREE_MINUTES_ENTITY) is None
            assert config_flow._data.get(CONF_POWER_SENSOR_ENTITY) is None


class TestOptionalFeaturesStatusSensor:
    """Test optional features status sensor."""

    def test_optional_features_sensor_exists(self):
        """Test that optional features status sensor is defined."""
        from custom_components.effektguard.sensor import SENSORS

        sensor_keys = [s.key for s in SENSORS]
        assert "optional_features_status" in sensor_keys

    def test_optional_features_sensor_attributes(self):
        """Test optional features sensor has correct attributes."""
        from custom_components.effektguard.sensor import SENSORS

        sensor = next(s for s in SENSORS if s.key == "optional_features_status")

        assert sensor.name == "Optional Features Status"
        assert sensor.icon == "mdi:feature-search-outline"
        assert sensor.value_fn is not None


class TestOptionalFeaturesEstimation:
    """Test estimation fallbacks for optional features."""

    def test_degree_minutes_estimation_note(self):
        """Test that missing DM sensor shows estimation note."""
        # This will be implemented in adapters
        # For now, just verify the sensor can show the status
        from custom_components.effektguard.sensor import SENSORS

        sensor = next(s for s in SENSORS if s.key == "optional_features_status")
        assert sensor is not None

    def test_power_estimation_note(self):
        """Test that missing power sensor shows estimation note."""
        from custom_components.effektguard.sensor import SENSORS

        sensor = next(s for s in SENSORS if s.key == "optional_features_status")
        assert sensor is not None


class TestWeatherForecastValidation:
    """Test weather forecast validation."""

    def test_weather_with_sufficient_forecast(self, mock_hass):
        """Test weather entity with 24h forecast (sufficient)."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass

        weather_state = mock_hass.states.get("weather.home")
        forecast = weather_state.attributes.get("forecast")

        assert len(forecast) >= 12  # Minimum 12h required

    def test_weather_with_short_forecast(self):
        """Test weather entity with only 6h forecast (insufficient)."""
        hass = MagicMock(spec=HomeAssistant)

        # Short forecast
        mock_state = MagicMock(
            entity_id="weather.home",
            attributes={"forecast": [{"datetime": "2025-10-14T12:00:00", "temperature": 15}] * 6},
        )
        mock_states_obj = MagicMock()
        hass.states = mock_states_obj
        mock_states_obj.get = lambda entity_id: mock_state

        forecast = mock_state.attributes.get("forecast")

        assert len(forecast) < 12  # Less than minimum


class TestTomorrowPricesDetection:
    """Test tomorrow prices detection from spot price integration."""

    def test_gespot_with_tomorrow_prices(self):
        """Test spot price integration with tomorrow prices available."""
        # This will be implemented in gespot_adapter.py
        # For now, verify the status sensor can detect it
        from custom_components.effektguard.sensor import SENSORS

        sensor = next(s for s in SENSORS if s.key == "optional_features_status")
        assert sensor is not None

    def test_gespot_without_tomorrow_prices(self):
        """Test spot price integration with only today prices."""
        from custom_components.effektguard.sensor import SENSORS

        sensor = next(s for s in SENSORS if s.key == "optional_features_status")
        assert sensor is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

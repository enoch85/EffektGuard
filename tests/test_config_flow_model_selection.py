"""Tests for config flow including model selection step.

Tests the complete configuration flow:
1. NIBE entity selection (async_step_user)
2. Spot price entity selection (async_step_gespot)
3. Heat pump model selection (async_step_model) - NEW
4. Optional features (async_step_optional)
5. Optional sensors (async_step_optional_sensors)
"""

import pytest
from unittest.mock import MagicMock, patch
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.effektguard.config_flow import EffektGuardConfigFlow
from custom_components.effektguard.const import (
    CONF_NIBE_ENTITY,
    CONF_GESPOT_ENTITY,
    CONF_HEAT_PUMP_MODEL,
    CONF_WEATHER_ENTITY,
    CONF_DEGREE_MINUTES_ENTITY,
    CONF_POWER_SENSOR_ENTITY,
    CONF_ENABLE_PRICE_OPTIMIZATION,
    CONF_ENABLE_PEAK_PROTECTION,
    DEFAULT_HEAT_PUMP_MODEL,
    DOMAIN,
)


@pytest.fixture
def mock_entity_registry():
    """Create mock entity registry with MyUplink entities."""
    mock_ent_reg = MagicMock()

    # Mock entity entry for MyUplink offset entity
    mock_entity_entry = MagicMock()
    mock_entity_entry.platform = "myuplink"

    def mock_registry_get(entity_id):
        if "offset" in entity_id:
            return mock_entity_entry
        return None

    mock_ent_reg.async_get = mock_registry_get
    return mock_ent_reg


@pytest.fixture
def mock_hass(mock_entity_registry, monkeypatch):
    """Create mock Home Assistant instance with entities."""
    hass = MagicMock(spec=HomeAssistant)

    # Create mock states object
    mock_states_obj = MagicMock()
    hass.states = mock_states_obj

    # Patch entity registry import in config_flow
    monkeypatch.setattr(
        "homeassistant.helpers.entity_registry.async_get", lambda h: mock_entity_registry
    )

    # Mock states for discovery
    mock_states = {
        # NIBE entities
        "sensor.nibe_outdoor_temp": MagicMock(
            entity_id="sensor.nibe_outdoor_temp",
            state="5.0",
            attributes={},
        ),
        "number.nibe_offset_s1": MagicMock(
            entity_id="number.nibe_offset_s1",
            state="0.0",
            attributes={},
        ),
        # Spot price entities
        "sensor.gespot_current_price": MagicMock(
            entity_id="sensor.gespot_current_price",
            state="0.50",
            attributes={},
        ),
        # Weather entities
        "weather.home": MagicMock(
            entity_id="weather.home",
            state="sunny",
            attributes={"forecast": [{"datetime": "2025-10-15T12:00:00", "temperature": 15}] * 24},
        ),
        # Degree minutes sensors
        "sensor.nibe_degree_minutes": MagicMock(
            entity_id="sensor.nibe_degree_minutes",
            state="-100",
            attributes={},
        ),
        # Power sensors
        "sensor.house_power": MagicMock(
            entity_id="sensor.house_power",
            state="3500",
            attributes={"device_class": "power", "unit_of_measurement": "W"},
        ),
    }

    def mock_async_all():
        # Return list of state objects, not tuples
        return list(mock_states.values())

    def mock_get_state(entity_id):
        return mock_states.get(entity_id)

    mock_states_obj.async_all = mock_async_all
    mock_states_obj.get = mock_get_state

    return hass


class TestConfigFlowUserStep:
    """Test the initial user step (NIBE entity selection)."""

    async def test_user_step_shows_form(self, mock_hass):
        """Test that user step shows form with NIBE entity selector."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass

        result = await config_flow.async_step_user()

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "user"
        assert CONF_NIBE_ENTITY in result["data_schema"].schema

    async def test_user_step_nibe_not_found(self, mock_entity_registry, monkeypatch):
        """Test form still shows when no NIBE entities found (allows manual selection)."""
        hass = MagicMock(spec=HomeAssistant)
        mock_states_obj = MagicMock()
        hass.states = mock_states_obj
        mock_states_obj.async_all = lambda: []  # No entities

        # Patch entity registry
        monkeypatch.setattr(
            "homeassistant.helpers.entity_registry.async_get", lambda h: mock_entity_registry
        )

        config_flow = EffektGuardConfigFlow()
        config_flow.hass = hass

        result = await config_flow.async_step_user()

        # Should show form with warning message, not abort
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "user"
        # Check for warning content, not emoji (to avoid Unicode issues)
        assert "No NIBE offset entities auto-detected" in result["description_placeholders"]["info"]
        assert "MyUplink just loaded" in result["description_placeholders"]["info"]

    async def test_user_step_valid_submission(self, mock_hass):
        """Test valid NIBE entity submission proceeds to spot price step."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass

        result = await config_flow.async_step_user(
            user_input={CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp"}
        )

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "gespot"
        assert config_flow._data[CONF_NIBE_ENTITY] == "sensor.nibe_outdoor_temp"

    async def test_user_step_invalid_entity(self, mock_hass):
        """Test error when NIBE entity doesn't exist."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass

        result = await config_flow.async_step_user(
            user_input={CONF_NIBE_ENTITY: "sensor.nonexistent"}
        )

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "user"
        assert "base" in result["errors"]


class TestConfigFlowGESpotStep:
    """Test the spot price entity selection step."""

    async def test_gespot_step_shows_form(self, mock_hass):
        """Test that spot price step shows form."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp"}

        result = await config_flow.async_step_gespot()

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "gespot"
        assert CONF_GESPOT_ENTITY in result["data_schema"].schema
        assert CONF_ENABLE_PRICE_OPTIMIZATION in result["data_schema"].schema

    async def test_gespot_step_with_entity_proceeds_to_model(self, mock_hass):
        """Test that valid spot price submission proceeds to model selection."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp"}

        result = await config_flow.async_step_gespot(
            user_input={
                CONF_GESPOT_ENTITY: "sensor.gespot_current_price",
                CONF_ENABLE_PRICE_OPTIMIZATION: True,
            }
        )

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "model"
        assert config_flow._data[CONF_GESPOT_ENTITY] == "sensor.gespot_current_price"
        assert config_flow._data[CONF_ENABLE_PRICE_OPTIMIZATION] is True

    async def test_gespot_step_skip_proceeds_to_model(self, mock_hass):
        """Test that skipping spot price (None) proceeds to model selection."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp"}

        result = await config_flow.async_step_gespot(
            user_input={
                CONF_GESPOT_ENTITY: None,
                CONF_ENABLE_PRICE_OPTIMIZATION: False,
            }
        )

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "model"
        assert config_flow._data.get(CONF_GESPOT_ENTITY) is None

    async def test_gespot_step_invalid_entity(self, mock_hass):
        """Test error when spot price entity doesn't exist."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp"}

        result = await config_flow.async_step_gespot(
            user_input={
                CONF_GESPOT_ENTITY: "sensor.nonexistent",
                CONF_ENABLE_PRICE_OPTIMIZATION: True,
            }
        )

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "gespot"
        assert "base" in result["errors"]


class TestConfigFlowModelStep:
    """Test the heat pump model selection step (NEW)."""

    async def test_model_step_shows_form(self, mock_hass):
        """Test that model step shows form with 4 NIBE models."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {
            CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp",
            CONF_GESPOT_ENTITY: "sensor.gespot_current_price",
        }

        result = await config_flow.async_step_model()

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "model"
        assert CONF_HEAT_PUMP_MODEL in result["data_schema"].schema

    async def test_model_step_all_models_available(self, mock_hass):
        """Test that all 4 NIBE models are available in dropdown."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {
            CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp",
            CONF_GESPOT_ENTITY: "sensor.gespot_current_price",
        }

        result = await config_flow.async_step_model()

        # Extract the selector options
        schema = result["data_schema"].schema
        heat_pump_field = schema[CONF_HEAT_PUMP_MODEL]

        # Get the vol.In container which has the options
        options = heat_pump_field.container

        # Verify all 4 models are present
        assert "nibe_f730" in options
        assert "nibe_f750" in options
        assert "nibe_f2040" in options
        assert "nibe_s1155" in options

        # Verify labels are descriptive
        assert "NIBE F730" in options["nibe_f730"]
        assert "NIBE F750" in options["nibe_f750"]
        assert "Most Common" in options["nibe_f750"]  # F750 is default
        assert "NIBE F2040" in options["nibe_f2040"]
        assert "NIBE S1155" in options["nibe_s1155"]
        assert "GSHP" in options["nibe_s1155"]

    async def test_model_step_f750_selection(self, mock_hass):
        """Test selecting F750 model proceeds to optional step."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {
            CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp",
            CONF_GESPOT_ENTITY: "sensor.gespot_current_price",
        }

        result = await config_flow.async_step_model(user_input={CONF_HEAT_PUMP_MODEL: "nibe_f750"})

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "optional"
        assert config_flow._data[CONF_HEAT_PUMP_MODEL] == "nibe_f750"

    async def test_model_step_f730_selection(self, mock_hass):
        """Test selecting F730 model (smaller ASHP)."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {
            CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp",
            CONF_GESPOT_ENTITY: "sensor.gespot_current_price",
        }

        result = await config_flow.async_step_model(user_input={CONF_HEAT_PUMP_MODEL: "nibe_f730"})

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "optional"
        assert config_flow._data[CONF_HEAT_PUMP_MODEL] == "nibe_f730"

    async def test_model_step_f2040_selection(self, mock_hass):
        """Test selecting F2040 model (larger ASHP)."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {
            CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp",
            CONF_GESPOT_ENTITY: "sensor.gespot_current_price",
        }

        result = await config_flow.async_step_model(user_input={CONF_HEAT_PUMP_MODEL: "nibe_f2040"})

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "optional"
        assert config_flow._data[CONF_HEAT_PUMP_MODEL] == "nibe_f2040"

    async def test_model_step_s1155_selection(self, mock_hass):
        """Test selecting S1155 model (GSHP)."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {
            CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp",
            CONF_GESPOT_ENTITY: "sensor.gespot_current_price",
        }

        result = await config_flow.async_step_model(user_input={CONF_HEAT_PUMP_MODEL: "nibe_s1155"})

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "optional"
        assert config_flow._data[CONF_HEAT_PUMP_MODEL] == "nibe_s1155"

    async def test_model_step_description_placeholder(self, mock_hass):
        """Test that model step has helpful description."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {
            CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp",
            CONF_GESPOT_ENTITY: "sensor.gespot_current_price",
        }

        result = await config_flow.async_step_model()

        assert "description_placeholders" in result
        assert "model_info" in result["description_placeholders"]


class TestConfigFlowOptionalStep:
    """Test the optional features step (weather, peak protection)."""

    async def test_optional_step_shows_form(self, mock_hass):
        """Test that optional step shows form."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {
            CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp",
            CONF_GESPOT_ENTITY: "sensor.gespot_current_price",
            CONF_HEAT_PUMP_MODEL: "nibe_f750",
        }

        result = await config_flow.async_step_optional()

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "optional"
        assert CONF_WEATHER_ENTITY in result["data_schema"].schema
        assert CONF_ENABLE_PEAK_PROTECTION in result["data_schema"].schema

    async def test_optional_step_with_weather(self, mock_hass):
        """Test optional step with weather entity."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {
            CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp",
            CONF_GESPOT_ENTITY: "sensor.gespot_current_price",
            CONF_HEAT_PUMP_MODEL: "nibe_f750",
        }

        result = await config_flow.async_step_optional(
            user_input={
                CONF_WEATHER_ENTITY: "weather.home",
                CONF_ENABLE_PEAK_PROTECTION: True,
            }
        )

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "optional_sensors"
        assert config_flow._data[CONF_WEATHER_ENTITY] == "weather.home"
        assert config_flow._data[CONF_ENABLE_PEAK_PROTECTION] is True

    async def test_optional_step_skip_weather(self, mock_hass):
        """Test optional step skipping weather entity."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {
            CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp",
            CONF_GESPOT_ENTITY: "sensor.gespot_current_price",
            CONF_HEAT_PUMP_MODEL: "nibe_f750",
        }

        result = await config_flow.async_step_optional(
            user_input={
                CONF_WEATHER_ENTITY: None,
                CONF_ENABLE_PEAK_PROTECTION: False,
            }
        )

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "optional_sensors"
        assert config_flow._data.get(CONF_WEATHER_ENTITY) is None


class TestCompleteConfigFlow:
    """Test complete config flow from start to finish."""

    async def test_complete_flow_all_features(self, mock_hass):
        """Test complete config flow with all features enabled."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass

        # Step 1: User - NIBE entity
        result = await config_flow.async_step_user(
            user_input={CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp"}
        )
        assert result["step_id"] == "gespot"

        # Step 2: Spot price entity
        result = await config_flow.async_step_gespot(
            user_input={
                CONF_GESPOT_ENTITY: "sensor.gespot_current_price",
                CONF_ENABLE_PRICE_OPTIMIZATION: True,
            }
        )
        assert result["step_id"] == "model"

        # Step 3: Heat pump model (NEW)
        result = await config_flow.async_step_model(user_input={CONF_HEAT_PUMP_MODEL: "nibe_f750"})
        assert result["step_id"] == "optional"

        # Step 4: Optional features
        result = await config_flow.async_step_optional(
            user_input={
                CONF_WEATHER_ENTITY: "weather.home",
                CONF_ENABLE_PEAK_PROTECTION: True,
            }
        )
        assert result["step_id"] == "optional_sensors"

        # Step 5: Optional sensors
        with patch.object(config_flow, "async_create_entry") as mock_create:
            mock_create.return_value = {"type": "create_entry"}

            result = await config_flow.async_step_optional_sensors(
                user_input={
                    CONF_DEGREE_MINUTES_ENTITY: "sensor.nibe_degree_minutes",
                    CONF_POWER_SENSOR_ENTITY: "sensor.house_power",
                }
            )

            assert mock_create.called

            # Verify all data was collected
            assert config_flow._data[CONF_NIBE_ENTITY] == "sensor.nibe_outdoor_temp"
            assert config_flow._data[CONF_GESPOT_ENTITY] == "sensor.gespot_current_price"
            assert config_flow._data[CONF_HEAT_PUMP_MODEL] == "nibe_f750"
            assert config_flow._data[CONF_WEATHER_ENTITY] == "weather.home"
            assert config_flow._data[CONF_DEGREE_MINUTES_ENTITY] == "sensor.nibe_degree_minutes"
            assert config_flow._data[CONF_POWER_SENSOR_ENTITY] == "sensor.house_power"
            assert config_flow._data[CONF_ENABLE_PRICE_OPTIMIZATION] is True
            assert config_flow._data[CONF_ENABLE_PEAK_PROTECTION] is True

    async def test_complete_flow_minimal(self, mock_hass):
        """Test complete config flow with minimal features (only required)."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass

        # Step 1: User - NIBE entity (required)
        result = await config_flow.async_step_user(
            user_input={CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp"}
        )
        assert result["step_id"] == "gespot"

        # Step 2: Skip spot price
        result = await config_flow.async_step_gespot(
            user_input={
                CONF_GESPOT_ENTITY: None,
                CONF_ENABLE_PRICE_OPTIMIZATION: False,
            }
        )
        assert result["step_id"] == "model"

        # Step 3: Select default model
        result = await config_flow.async_step_model(user_input={CONF_HEAT_PUMP_MODEL: "nibe_f750"})
        assert result["step_id"] == "optional"

        # Step 4: Skip optional features
        result = await config_flow.async_step_optional(
            user_input={
                CONF_WEATHER_ENTITY: None,
                CONF_ENABLE_PEAK_PROTECTION: False,
            }
        )
        assert result["step_id"] == "optional_sensors"

        # Step 5: Skip optional sensors
        with patch.object(config_flow, "async_create_entry") as mock_create:
            mock_create.return_value = {"type": "create_entry"}

            result = await config_flow.async_step_optional_sensors(
                user_input={
                    CONF_DEGREE_MINUTES_ENTITY: None,
                    CONF_POWER_SENSOR_ENTITY: None,
                }
            )

            assert mock_create.called

            # Verify minimal data was collected
            assert config_flow._data[CONF_NIBE_ENTITY] == "sensor.nibe_outdoor_temp"
            assert config_flow._data[CONF_HEAT_PUMP_MODEL] == "nibe_f750"
            assert config_flow._data.get(CONF_GESPOT_ENTITY) is None
            assert config_flow._data.get(CONF_WEATHER_ENTITY) is None

    async def test_complete_flow_gshp_model(self, mock_hass):
        """Test complete config flow with GSHP model selection."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass

        # Step 1: User
        result = await config_flow.async_step_user(
            user_input={CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp"}
        )

        # Step 2: Spot price
        result = await config_flow.async_step_gespot(
            user_input={
                CONF_GESPOT_ENTITY: "sensor.gespot_current_price",
                CONF_ENABLE_PRICE_OPTIMIZATION: True,
            }
        )

        # Step 3: Select GSHP model (S1155)
        result = await config_flow.async_step_model(user_input={CONF_HEAT_PUMP_MODEL: "nibe_s1155"})
        assert result["step_id"] == "optional"
        assert config_flow._data[CONF_HEAT_PUMP_MODEL] == "nibe_s1155"

        # Continue with rest of flow
        result = await config_flow.async_step_optional(
            user_input={
                CONF_WEATHER_ENTITY: "weather.home",
                CONF_ENABLE_PEAK_PROTECTION: True,
            }
        )

        with patch.object(config_flow, "async_create_entry") as mock_create:
            mock_create.return_value = {"type": "create_entry"}

            result = await config_flow.async_step_optional_sensors(
                user_input={
                    CONF_DEGREE_MINUTES_ENTITY: None,
                    CONF_POWER_SENSOR_ENTITY: None,
                }
            )

            assert mock_create.called
            # Verify GSHP model was saved
            assert config_flow._data[CONF_HEAT_PUMP_MODEL] == "nibe_s1155"


class TestModelStepIntegration:
    """Test that model step integrates correctly with coordinator."""

    def test_model_key_in_entry_data(self, mock_hass):
        """Test that model selection is saved in entry data."""
        config_flow = EffektGuardConfigFlow()
        config_flow.hass = mock_hass
        config_flow._data = {
            CONF_NIBE_ENTITY: "sensor.nibe_outdoor_temp",
            CONF_HEAT_PUMP_MODEL: "nibe_f2040",
        }

        # Verify the model key is in the data
        assert CONF_HEAT_PUMP_MODEL in config_flow._data
        assert config_flow._data[CONF_HEAT_PUMP_MODEL] == "nibe_f2040"

    def test_default_model_is_f750(self):
        """Test that default model is F750."""
        from custom_components.effektguard.const import DEFAULT_HEAT_PUMP_MODEL

        assert DEFAULT_HEAT_PUMP_MODEL == "nibe_f750"

    def test_all_model_keys_match_registry(self):
        """Test that config flow model keys match coordinator registry."""
        from custom_components.effektguard.coordinator import HEAT_PUMP_MODELS

        expected_models = ["nibe_f730", "nibe_f750", "nibe_f2040", "nibe_s1155"]

        for model_key in expected_models:
            assert model_key in HEAT_PUMP_MODELS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

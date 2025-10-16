"""Config flow for EffektGuard integration."""

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_ADDITIONAL_INDOOR_SENSORS,
    CONF_DEGREE_MINUTES_ENTITY,
    CONF_ENABLE_PEAK_PROTECTION,
    CONF_ENABLE_PRICE_OPTIMIZATION,
    CONF_GESPOT_ENTITY,
    CONF_HEAT_PUMP_MODEL,
    CONF_INDOOR_TEMP_METHOD,
    CONF_INSULATION_QUALITY,
    CONF_NIBE_ENTITY,
    CONF_OPTIMIZATION_MODE,
    CONF_POWER_SENSOR_ENTITY,
    CONF_TARGET_TEMPERATURE,
    CONF_THERMAL_MASS,
    CONF_TOLERANCE,
    CONF_WEATHER_ENTITY,
    DEFAULT_HEAT_PUMP_MODEL,
    DEFAULT_INDOOR_TEMP_METHOD,
    DEFAULT_INSULATION_QUALITY,
    DEFAULT_OPTIMIZATION_MODE,
    DEFAULT_TARGET_TEMP,
    DEFAULT_THERMAL_MASS,
    DEFAULT_TOLERANCE,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


class EffektGuardConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for EffektGuard."""

    VERSION = 1

    def __init__(self):
        """Initialize config flow."""
        self._data: dict[str, Any] = {}

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the initial step - NIBE integration selection."""
        errors = {}

        if user_input is not None:
            # Store NIBE entity
            self._data[CONF_NIBE_ENTITY] = user_input[CONF_NIBE_ENTITY]

            # Validate NIBE entity exists
            nibe_entity = user_input[CONF_NIBE_ENTITY]
            if not self.hass.states.get(nibe_entity):
                errors["base"] = "nibe_entity_not_found"
            else:
                return await self.async_step_gespot()

        # Discover NIBE entities
        nibe_entities = self._discover_nibe_entities()

        if not nibe_entities:
            return self.async_abort(reason="nibe_not_found")

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_NIBE_ENTITY): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=["sensor", "number"],
                        )
                    ),
                }
            ),
            errors=errors,
            description_placeholders={"nibe_count": str(len(nibe_entities))},
        )

    async def async_step_gespot(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Configure GE-Spot integration."""
        errors = {}

        if user_input is not None:
            # Store GE-Spot settings
            self._data[CONF_GESPOT_ENTITY] = user_input.get(CONF_GESPOT_ENTITY)
            self._data[CONF_ENABLE_PRICE_OPTIMIZATION] = user_input.get(
                CONF_ENABLE_PRICE_OPTIMIZATION, True
            )

            # Validate if provided
            if self._data[CONF_GESPOT_ENTITY]:
                gespot_entity = self._data[CONF_GESPOT_ENTITY]
                if not self.hass.states.get(gespot_entity):
                    errors["base"] = "gespot_entity_not_found"
                else:
                    return await self.async_step_model()
            else:
                return await self.async_step_model()

        # Discover GE-Spot entities
        gespot_entities = self._discover_gespot_entities()

        return self.async_show_form(
            step_id="gespot",
            data_schema=vol.Schema(
                {
                    vol.Optional(CONF_GESPOT_ENTITY): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor",
                        )
                    ),
                    vol.Optional(
                        CONF_ENABLE_PRICE_OPTIMIZATION, default=True
                    ): selector.BooleanSelector(),
                }
            ),
            errors=errors,
            description_placeholders={"gespot_count": str(len(gespot_entities))},
        )

    async def async_step_model(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle heat pump model selection."""
        errors = {}

        if user_input is not None:
            self._data[CONF_HEAT_PUMP_MODEL] = user_input[CONF_HEAT_PUMP_MODEL]
            return await self.async_step_optional()  # Continue to existing optional step

        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_HEAT_PUMP_MODEL,
                        default=DEFAULT_HEAT_PUMP_MODEL,
                    ): vol.In(
                        {
                            "nibe_f730": "NIBE F730 (6kW ASHP)",
                            "nibe_f750": "NIBE F750 (8kW ASHP - Most Common)",
                            "nibe_f2040": "NIBE F2040 (12-16kW ASHP)",
                            "nibe_s1155": "NIBE S1155 (GSHP)",
                        }
                    ),
                }
            ),
            errors=errors,
            description_placeholders={
                "model_info": "Select your heat pump model for optimized control"
            },
        )

    async def async_step_optional(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Configure optional features."""
        if user_input is not None:
            # Store optional settings
            self._data[CONF_WEATHER_ENTITY] = user_input.get(CONF_WEATHER_ENTITY)
            self._data[CONF_ENABLE_PEAK_PROTECTION] = user_input.get(
                CONF_ENABLE_PEAK_PROTECTION, True
            )

            # Move to optional sensors step
            return await self.async_step_optional_sensors()

        # Discover weather entities
        weather_entities = self._discover_weather_entities()

        return self.async_show_form(
            step_id="optional",
            data_schema=vol.Schema(
                {
                    vol.Optional(CONF_WEATHER_ENTITY): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="weather",
                        )
                    ),
                    vol.Optional(
                        CONF_ENABLE_PEAK_PROTECTION, default=True
                    ): selector.BooleanSelector(),
                }
            ),
            description_placeholders={"weather_count": str(len(weather_entities))},
        )

    async def async_step_optional_sensors(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure optional sensors (degree minutes, power meter, extra temp sensors)."""
        if user_input is not None:
            # Store optional sensor settings
            self._data[CONF_DEGREE_MINUTES_ENTITY] = user_input.get(CONF_DEGREE_MINUTES_ENTITY)
            self._data[CONF_POWER_SENSOR_ENTITY] = user_input.get(CONF_POWER_SENSOR_ENTITY)
            self._data[CONF_ADDITIONAL_INDOOR_SENSORS] = user_input.get(
                CONF_ADDITIONAL_INDOOR_SENSORS, []
            )
            self._data[CONF_INDOOR_TEMP_METHOD] = user_input.get(
                CONF_INDOOR_TEMP_METHOD, DEFAULT_INDOOR_TEMP_METHOD
            )

            # Create entry
            return self.async_create_entry(
                title="EffektGuard",
                data=self._data,
            )

        # Auto-detect degree minutes sensor
        dm_entities = self._discover_degree_minutes_entities()
        auto_detected_dm = dm_entities[0] if dm_entities else None

        # Auto-detect power sensor
        power_entities = self._discover_power_entities()
        auto_detected_power = power_entities[0] if power_entities else None

        schema_dict: dict[Any, Any] = {}

        # Degree minutes sensor (optional)
        if auto_detected_dm:
            schema_dict[vol.Optional(CONF_DEGREE_MINUTES_ENTITY, default=auto_detected_dm)] = (
                selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain="sensor",
                    )
                )
            )
        else:
            schema_dict[vol.Optional(CONF_DEGREE_MINUTES_ENTITY)] = selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor",
                )
            )

        # Power meter sensor (optional)
        if auto_detected_power:
            schema_dict[vol.Optional(CONF_POWER_SENSOR_ENTITY, default=auto_detected_power)] = (
                selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain="sensor",
                        device_class="power",
                    )
                )
            )
        else:
            schema_dict[vol.Optional(CONF_POWER_SENSOR_ENTITY)] = selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor",
                    device_class="power",
                )
            )

        # Additional indoor temperature sensors (optional, multi-select)
        schema_dict[vol.Optional(CONF_ADDITIONAL_INDOOR_SENSORS)] = selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class="temperature",
                multiple=True,
            )
        )

        # Indoor temperature calculation method
        schema_dict[vol.Optional(CONF_INDOOR_TEMP_METHOD, default=DEFAULT_INDOOR_TEMP_METHOD)] = (
            selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=["median", "average"],
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            )
        )

        return self.async_show_form(
            step_id="optional_sensors",
            data_schema=vol.Schema(schema_dict),
            description_placeholders={
                "dm_detected": "✓ Auto-detected" if auto_detected_dm else "❌ Not found",
                "power_detected": "✓ Auto-detected" if auto_detected_power else "❌ Not found",
                "dm_note": "Degree minutes sensor improves thermal debt tracking (estimated if not provided)",
                "power_note": "Power meter enables accurate peak tracking (estimated from heat pump data if not provided)",
                "temp_sensors_note": "Add extra temperature sensors (living room, bedroom, etc.) for whole-house averaging. Median recommended (more robust to outliers).",
            },
        )

    def _discover_nibe_entities(self) -> list[str]:
        """Discover NIBE entities."""
        entities = []
        for state in self.hass.states.async_all():
            # Look for NIBE-related entities
            if "nibe" in state.entity_id.lower() or "myuplink" in state.entity_id.lower():
                entities.append(state.entity_id)
        return entities

    def _discover_gespot_entities(self) -> list[str]:
        """Discover GE-Spot price entities."""
        entities = []
        for state in self.hass.states.async_all():
            if state.entity_id.startswith("sensor.ge") or "gespot" in state.entity_id.lower():
                entities.append(state.entity_id)
        return entities

    def _discover_weather_entities(self) -> list[str]:
        """Discover weather entities."""
        entities = []
        for state in self.hass.states.async_all():
            if state.entity_id.startswith("weather."):
                entities.append(state.entity_id)
        return entities

    def _discover_degree_minutes_entities(self) -> list[str]:
        """Discover NIBE degree minutes sensors.

        Auto-detect common patterns:
        - sensor.*gradminuter* (Swedish)
        - sensor.*degree_minutes*
        - sensor.*gm_* (abbreviation)
        - NIBE specific entity patterns
        """
        entities = []
        search_terms = [
            "gradminuter",
            "degree_minutes",
            "degree_minute",
            "_gm_",
            "_dm_",
        ]

        for state in self.hass.states.async_all():
            entity_id = state.entity_id
            if not entity_id.startswith("sensor."):
                continue

            entity_lower = entity_id.lower()

            # Check for any search term in entity ID
            for term in search_terms:
                if term in entity_lower:
                    # Additional validation: check if NIBE-related
                    if "nibe" in entity_lower or "myuplink" in entity_lower:
                        entities.append(entity_id)
                        break
                    # Or if it's explicitly a degree minutes sensor
                    elif "gradminuter" in entity_lower or "degree_minute" in entity_lower:
                        entities.append(entity_id)
                        break

        return entities

    def _discover_power_entities(self) -> list[str]:
        """Discover power meter sensors.

        Priority order:
        1. House/home power meters
        2. Heat pump power sensors
        3. Generic power sensors
        """
        entities = []
        house_power = []
        heatpump_power = []
        generic_power = []

        for state in self.hass.states.async_all():
            entity_id = state.entity_id
            if not entity_id.startswith("sensor."):
                continue

            # Check device class
            if state.attributes.get("device_class") != "power":
                continue

            # Check unit of measurement (should be W or kW)
            unit = state.attributes.get("unit_of_measurement", "")
            if unit.lower() not in ["w", "kw"]:
                continue

            entity_lower = entity_id.lower()

            # Categorize by priority
            if any(
                term in entity_lower
                for term in ["house", "home", "total", "grid", "main", "hus", "hem"]
            ):
                house_power.append(entity_id)
            elif any(
                term in entity_lower
                for term in ["nibe", "heat_pump", "heatpump", "myuplink", "värmepump"]
            ):
                heatpump_power.append(entity_id)
            else:
                generic_power.append(entity_id)

        # Return in priority order
        entities = house_power + heatpump_power + generic_power
        return entities

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Get the options flow for this handler."""
        return OptionsFlowHandler(config_entry)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for EffektGuard."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_TARGET_TEMPERATURE,
                        default=self.config_entry.options.get(
                            CONF_TARGET_TEMPERATURE, DEFAULT_TARGET_TEMP
                        ),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=18,
                            max=26,
                            step=0.5,
                            mode=selector.NumberSelectorMode.SLIDER,
                            unit_of_measurement="°C",
                        )
                    ),
                    vol.Optional(
                        CONF_TOLERANCE,
                        default=self.config_entry.options.get(CONF_TOLERANCE, DEFAULT_TOLERANCE),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1,
                            max=10,
                            step=1,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_OPTIMIZATION_MODE,
                        default=self.config_entry.options.get(
                            CONF_OPTIMIZATION_MODE, DEFAULT_OPTIMIZATION_MODE
                        ),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=["basic", "price", "advanced", "expert"],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_THERMAL_MASS,
                        default=self.config_entry.options.get(
                            CONF_THERMAL_MASS, DEFAULT_THERMAL_MASS
                        ),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.5,
                            max=2.0,
                            step=0.1,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_INSULATION_QUALITY,
                        default=self.config_entry.options.get(
                            CONF_INSULATION_QUALITY, DEFAULT_INSULATION_QUALITY
                        ),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.5,
                            max=2.0,
                            step=0.1,
                            mode=selector.NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_DEGREE_MINUTES_ENTITY,
                        default=self.config_entry.data.get(CONF_DEGREE_MINUTES_ENTITY),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor",
                        )
                    ),
                    vol.Optional(
                        CONF_POWER_SENSOR_ENTITY,
                        default=self.config_entry.data.get(CONF_POWER_SENSOR_ENTITY),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor",
                            device_class="power",
                        )
                    ),
                    vol.Optional(
                        CONF_WEATHER_ENTITY,
                        default=self.config_entry.data.get(CONF_WEATHER_ENTITY),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="weather",
                        )
                    ),
                    vol.Optional(
                        CONF_ADDITIONAL_INDOOR_SENSORS,
                        default=self.config_entry.data.get(CONF_ADDITIONAL_INDOOR_SENSORS, []),
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain="sensor",
                            device_class="temperature",
                            multiple=True,
                        )
                    ),
                }
            ),
        )

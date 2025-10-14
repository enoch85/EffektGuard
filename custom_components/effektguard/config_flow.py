"""Config flow for EffektGuard integration."""

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_ENABLE_PEAK_PROTECTION,
    CONF_ENABLE_PRICE_OPTIMIZATION,
    CONF_GESPOT_ENTITY,
    CONF_INSULATION_QUALITY,
    CONF_NIBE_ENTITY,
    CONF_OPTIMIZATION_MODE,
    CONF_TARGET_TEMPERATURE,
    CONF_THERMAL_MASS,
    CONF_TOLERANCE,
    CONF_WEATHER_ENTITY,
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
                    return await self.async_step_optional()
            else:
                return await self.async_step_optional()

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

    async def async_step_optional(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Configure optional features."""
        if user_input is not None:
            # Store optional settings
            self._data[CONF_WEATHER_ENTITY] = user_input.get(CONF_WEATHER_ENTITY)
            self._data[CONF_ENABLE_PEAK_PROTECTION] = user_input.get(
                CONF_ENABLE_PEAK_PROTECTION, True
            )

            # Create entry
            return self.async_create_entry(
                title="EffektGuard",
                data=self._data,
            )

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

    def _discover_nibe_entities(self) -> list[str]:
        """Discover NIBE entities."""
        entities = []
        for entity_id, state in self.hass.states.async_all():
            # Look for NIBE-related entities
            if "nibe" in entity_id.lower() or "myuplink" in entity_id.lower():
                entities.append(entity_id)
        return entities

    def _discover_gespot_entities(self) -> list[str]:
        """Discover GE-Spot price entities."""
        entities = []
        for entity_id, state in self.hass.states.async_all():
            if entity_id.startswith("sensor.ge") or "gespot" in entity_id.lower():
                entities.append(entity_id)
        return entities

    def _discover_weather_entities(self) -> list[str]:
        """Discover weather entities."""
        entities = []
        for entity_id, state in self.hass.states.async_all():
            if entity_id.startswith("weather."):
                entities.append(entity_id)
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

    def __init__(self, config_entry: config_entries.ConfigEntry):
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
                }
            ),
        )

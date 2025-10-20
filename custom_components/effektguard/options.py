"""Options flow for EffektGuard integration."""

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult, section
from homeassistant.helpers import selector

from .const import (
    CONF_INSULATION_QUALITY,
    CONF_OPTIMIZATION_MODE,
    CONF_THERMAL_MASS,
    CONF_TOLERANCE,
    DEFAULT_INSULATION_QUALITY,
    DEFAULT_OPTIMIZATION_MODE,
    DEFAULT_THERMAL_MASS,
    DEFAULT_TOLERANCE,
    OPTIMIZATION_MODE_BALANCED,
    OPTIMIZATION_MODE_COMFORT,
    OPTIMIZATION_MODE_SAVINGS,
)

_LOGGER = logging.getLogger(__name__)


class EffektGuardOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for EffektGuard."""

    def _validate_and_convert_dhw_config(self, user_input: dict) -> dict:
        """Validate and convert DHW configuration types.

        Sections create nested dicts, so we need to flatten them.
        """
        validated = {}

        # Flatten section structures
        if "optimization_settings" in user_input:
            validated.update(user_input["optimization_settings"])
        if "building_characteristics" in user_input:
            validated.update(user_input["building_characteristics"])
        if "domestic_hot_water" in user_input:
            validated.update(user_input["domestic_hot_water"])

        # Also handle flat structure (backward compatibility)
        for key, value in user_input.items():
            if key not in [
                "optimization_settings",
                "building_characteristics",
                "domestic_hot_water",
            ]:
                validated[key] = value

        # Convert dhw_schedules checkboxes to individual boolean flags
        if "dhw_schedules" in validated:
            schedules = validated.pop("dhw_schedules")
            validated["dhw_morning_enabled"] = "morning" in schedules
            validated["dhw_evening_enabled"] = "evening" in schedules

        # Validate DHW target temperature
        if "dhw_target_temp" in validated:
            try:
                dhw_target = float(validated["dhw_target_temp"])
                if dhw_target < 45.0 or dhw_target > 60.0:
                    raise vol.Invalid("DHW target temperature must be between 45-60°C")
                validated["dhw_target_temp"] = dhw_target
            except (TypeError, ValueError) as e:
                raise vol.Invalid(f"Invalid DHW target temperature: {e}")

        # Convert hour values to int (SelectSelector returns string)
        for hour_field in ["dhw_morning_hour", "dhw_evening_hour"]:
            if hour_field in validated:
                try:
                    validated[hour_field] = int(validated[hour_field])
                except (TypeError, ValueError) as e:
                    raise vol.Invalid(f"Invalid hour value for {hour_field}: {e}")

        return validated

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage runtime options."""
        if user_input is not None:
            validated_input = self._validate_and_convert_dhw_config(user_input)
            return self.async_create_entry(title="", data=validated_input)

        # Get current values for defaults
        morning_hour = self.config_entry.options.get("dhw_morning_hour", 6)
        evening_hour = self.config_entry.options.get("dhw_evening_hour", 18)
        default_schedules = []
        if self.config_entry.options.get("dhw_morning_enabled", True):
            default_schedules.append("morning")
        if self.config_entry.options.get("dhw_evening_enabled", True):
            default_schedules.append("evening")

        schema_dict = {
            # Optimization Settings Section
            vol.Required("optimization_settings"): section(
                vol.Schema(
                    {
                        vol.Optional(
                            CONF_OPTIMIZATION_MODE,
                            default=self.config_entry.options.get(
                                CONF_OPTIMIZATION_MODE, DEFAULT_OPTIMIZATION_MODE
                            ),
                        ): selector.SelectSelector(
                            selector.SelectSelectorConfig(
                                options=[
                                    OPTIMIZATION_MODE_COMFORT,
                                    OPTIMIZATION_MODE_BALANCED,
                                    OPTIMIZATION_MODE_SAVINGS,
                                ],
                                mode=selector.SelectSelectorMode.DROPDOWN,
                            )
                        ),
                        vol.Optional(
                            CONF_TOLERANCE,
                            default=self.config_entry.options.get(
                                CONF_TOLERANCE, DEFAULT_TOLERANCE
                            ),
                        ): selector.NumberSelector(
                            selector.NumberSelectorConfig(
                                min=0.5, max=3.0, step=0.1, mode=selector.NumberSelectorMode.SLIDER
                            )
                        ),
                    }
                ),
                {"collapsed": False},
            ),
            # Building Characteristics Section
            vol.Required("building_characteristics"): section(
                vol.Schema(
                    {
                        vol.Optional(
                            CONF_THERMAL_MASS,
                            default=self.config_entry.options.get(
                                CONF_THERMAL_MASS, DEFAULT_THERMAL_MASS
                            ),
                        ): selector.NumberSelector(
                            selector.NumberSelectorConfig(
                                min=0.5, max=2.0, step=0.1, mode=selector.NumberSelectorMode.SLIDER
                            )
                        ),
                        vol.Optional(
                            CONF_INSULATION_QUALITY,
                            default=self.config_entry.options.get(
                                CONF_INSULATION_QUALITY, DEFAULT_INSULATION_QUALITY
                            ),
                        ): selector.NumberSelector(
                            selector.NumberSelectorConfig(
                                min=0.5, max=2.0, step=0.1, mode=selector.NumberSelectorMode.SLIDER
                            )
                        ),
                    }
                ),
                {"collapsed": False},
            ),
            # Domestic Hot Water Section
            vol.Required("domestic_hot_water"): section(
                vol.Schema(
                    {
                        vol.Optional(
                            "dhw_target_temp",
                            default=self.config_entry.options.get("dhw_target_temp", 50.0),
                        ): selector.NumberSelector(
                            selector.NumberSelectorConfig(
                                min=45.0,
                                max=60.0,
                                step=1.0,
                                mode=selector.NumberSelectorMode.SLIDER,
                                unit_of_measurement="°C",
                            )
                        ),
                        vol.Optional(
                            "dhw_schedules",
                            default=default_schedules,
                        ): selector.SelectSelector(
                            selector.SelectSelectorConfig(
                                options=[
                                    {"value": "morning", "label": "Morning heating"},
                                    {"value": "evening", "label": "Evening heating"},
                                ],
                                mode=selector.SelectSelectorMode.LIST,
                                multiple=True,
                            )
                        ),
                        vol.Optional(
                            "dhw_morning_hour",
                            default=str(morning_hour),
                        ): selector.SelectSelector(
                            selector.SelectSelectorConfig(
                                options=[
                                    {"value": str(h), "label": f"{h:02d}:00"} for h in range(24)
                                ],
                                mode=selector.SelectSelectorMode.DROPDOWN,
                            )
                        ),
                        vol.Optional(
                            "dhw_evening_hour",
                            default=str(evening_hour),
                        ): selector.SelectSelector(
                            selector.SelectSelectorConfig(
                                options=[
                                    {"value": str(h), "label": f"{h:02d}:00"} for h in range(24)
                                ],
                                mode=selector.SelectSelectorMode.DROPDOWN,
                            )
                        ),
                    }
                ),
                {"collapsed": False},
            ),
        }

        return self.async_show_form(step_id="init", data_schema=vol.Schema(schema_dict))

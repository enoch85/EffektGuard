"""Options flow for EffektGuard integration."""

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult, section
from homeassistant.helpers import selector

from .const import (
    AIRFLOW_DEFAULT_ENHANCED,
    AIRFLOW_DEFAULT_STANDARD,
    CONF_AIRFLOW_ENHANCED_RATE,
    CONF_AIRFLOW_STANDARD_RATE,
    CONF_DHW_MIN_AMOUNT,
    CONF_HEAT_PUMP_MODEL,
    CONF_INSULATION_QUALITY,
    CONF_OPTIMIZATION_MODE,
    CONF_THERMAL_MASS,
    CONF_TOLERANCE,
    DEFAULT_HEAT_PUMP_MODEL,
    DEFAULT_INSULATION_QUALITY,
    DEFAULT_OPTIMIZATION_MODE,
    DEFAULT_THERMAL_MASS,
    DEFAULT_TOLERANCE,
    DHW_MIN_AMOUNT_DEFAULT,
    DHW_MIN_AMOUNT_MAX,
    DHW_MIN_AMOUNT_MIN,
    OPTIMIZATION_MODE_BALANCED,
    OPTIMIZATION_MODE_COMFORT,
    OPTIMIZATION_MODE_SAVINGS,
)
from .models import HeatPumpModelRegistry

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
        if "airflow_optimization" in user_input:
            validated.update(user_input["airflow_optimization"])

        # Handle direct keys (when input comes without section wrappers)
        for key, value in user_input.items():
            if key not in [
                "optimization_settings",
                "building_characteristics",
                "domestic_hot_water",
                "airflow_optimization",
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

        # Validate DHW min amount (minutes of hot water)
        if CONF_DHW_MIN_AMOUNT in validated:
            try:
                dhw_min = int(validated[CONF_DHW_MIN_AMOUNT])
                if dhw_min < DHW_MIN_AMOUNT_MIN or dhw_min > DHW_MIN_AMOUNT_MAX:
                    raise vol.Invalid(
                        f"DHW min amount must be {DHW_MIN_AMOUNT_MIN}-{DHW_MIN_AMOUNT_MAX} minutes"
                    )
                validated[CONF_DHW_MIN_AMOUNT] = dhw_min
            except (TypeError, ValueError) as e:
                raise vol.Invalid(f"Invalid DHW min amount: {e}")

        # Convert hour values to int (SelectSelector returns string)
        for hour_field in ["dhw_morning_hour", "dhw_evening_hour"]:
            if hour_field in validated:
                try:
                    validated[hour_field] = int(validated[hour_field])
                except (TypeError, ValueError) as e:
                    raise vol.Invalid(f"Invalid hour value for {hour_field}: {e}")

        # Validate airflow rates
        if CONF_AIRFLOW_STANDARD_RATE in validated:
            try:
                validated[CONF_AIRFLOW_STANDARD_RATE] = float(validated[CONF_AIRFLOW_STANDARD_RATE])
            except (TypeError, ValueError) as e:
                raise vol.Invalid(f"Invalid standard airflow rate: {e}")

        if CONF_AIRFLOW_ENHANCED_RATE in validated:
            try:
                validated[CONF_AIRFLOW_ENHANCED_RATE] = float(validated[CONF_AIRFLOW_ENHANCED_RATE])
            except (TypeError, ValueError) as e:
                raise vol.Invalid(f"Invalid enhanced airflow rate: {e}")

        return validated

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage runtime options."""
        if user_input is not None:
            validated_input = self._validate_and_convert_dhw_config(user_input)

            # Preserve ALL existing options not in the form
            # This is critical for values set by other entities (e.g., target_indoor_temp from climate)
            # and for options from entry.data that should persist
            for key, value in self.config_entry.options.items():
                if key not in validated_input:
                    _LOGGER.debug("Preserving option %s = %s", key, value)
                    validated_input[key] = value

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
                            CONF_DHW_MIN_AMOUNT,
                            default=self.config_entry.options.get(
                                CONF_DHW_MIN_AMOUNT, DHW_MIN_AMOUNT_DEFAULT
                            ),
                        ): selector.NumberSelector(
                            selector.NumberSelectorConfig(
                                min=DHW_MIN_AMOUNT_MIN,
                                max=DHW_MIN_AMOUNT_MAX,
                                step=1,
                                mode=selector.NumberSelectorMode.SLIDER,
                                unit_of_measurement="min",
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

        # Only show airflow section for exhaust air heat pumps (F750, F730)
        model_id = self.config_entry.data.get(CONF_HEAT_PUMP_MODEL, DEFAULT_HEAT_PUMP_MODEL)
        supports_airflow = False
        try:
            model_profile = HeatPumpModelRegistry.get_model(model_id)
            supports_airflow = getattr(model_profile, "supports_exhaust_airflow", False)
        except ValueError:
            _LOGGER.debug("Unknown heat pump model: %s", model_id)

        if supports_airflow:
            schema_dict[vol.Required("airflow_optimization")] = section(
                vol.Schema(
                    {
                        vol.Optional(
                            CONF_AIRFLOW_STANDARD_RATE,
                            default=self.config_entry.options.get(
                                CONF_AIRFLOW_STANDARD_RATE, AIRFLOW_DEFAULT_STANDARD
                            ),
                        ): selector.NumberSelector(
                            selector.NumberSelectorConfig(
                                min=100.0,
                                max=300.0,
                                step=10.0,
                                mode=selector.NumberSelectorMode.SLIDER,
                                unit_of_measurement="m³/h",
                            )
                        ),
                        vol.Optional(
                            CONF_AIRFLOW_ENHANCED_RATE,
                            default=self.config_entry.options.get(
                                CONF_AIRFLOW_ENHANCED_RATE, AIRFLOW_DEFAULT_ENHANCED
                            ),
                        ): selector.NumberSelector(
                            selector.NumberSelectorConfig(
                                min=150.0,
                                max=400.0,
                                step=10.0,
                                mode=selector.NumberSelectorMode.SLIDER,
                                unit_of_measurement="m³/h",
                            )
                        ),
                    }
                ),
                {"collapsed": False},
            )

        return self.async_show_form(step_id="init", data_schema=vol.Schema(schema_dict))

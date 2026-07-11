"""Config flow for EffektGuard integration."""

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.config_entries import ConfigFlowResult
from homeassistant.core import callback
from homeassistant.helpers import entity_registry as er, selector

from .options import EffektGuardOptionsFlow
from .const import (
    CONF_ADDITIONAL_INDOOR_SENSORS,
    CONF_DEGREE_MINUTES_ENTITY,
    CONF_ENABLE_PEAK_PROTECTION,
    CONF_ENABLE_PRICE_OPTIMIZATION,
    CONF_GESPOT_ENTITY,
    CONF_HEAT_PUMP_MODEL,
    CONF_INDOOR_TEMP_METHOD,
    CONF_NIBE_ENTITY,
    CONF_NIBE_TEMP_LUX_ENTITY,
    CONF_POWER_SENSOR_ENTITY,
    CONF_WEATHER_ENTITY,
    DEFAULT_HEAT_PUMP_MODEL,
    DEFAULT_INDOOR_TEMP_METHOD,
    DOMAIN,
    NIBE_DISCOVERY_PATTERNS,
    NIBE_MANUAL_OVERRIDE_KEYS,
)

_LOGGER = logging.getLogger(__name__)

# Entity fields users may set, clear, or leave on auto-discovery.
# Selectors allow sensor/number/input_number because local NIBE sources
# (nibe_heatpump, MyUplink) expose writable values as NUMBER entities and
# generic Modbus setups wrap registers in template numbers (issue #18).
SOURCE_ENTITY_DOMAINS = ["sensor", "number", "input_number"]

# Manual override fields shown in optional_sensors and reconfigure
# (single source of truth: the adapter's override map in const.py)
MANUAL_SENSOR_OVERRIDE_FIELDS: tuple[str, ...] = tuple(NIBE_MANUAL_OVERRIDE_KEYS.values())

# Heat pump model choices (setup step "model" and reconfigure)
HEAT_PUMP_MODEL_OPTIONS: dict[str, str] = {
    "nibe_f730": "NIBE F730 (6kW ASHP)",
    "nibe_f750": "NIBE F750 (8kW ASHP - Most Common)",
    "nibe_f1155": "NIBE F1155 (GSHP)",
    "nibe_f2040": "NIBE F2040 (12-16kW ASHP)",
    "nibe_s1155": "NIBE S1155 (GSHP)",
}


def _optional_entity_field(
    schema_dict: dict[Any, Any],
    conf_key: str,
    entity_selector: selector.EntitySelector,
    suggested: str | list[str] | None,
) -> None:
    """Add an optional entity selector that can be cleared by the user.

    Uses suggested_value instead of default: a voluptuous default is re-applied
    when the field is emptied, which makes stored entities impossible to clear.
    """
    if suggested:
        key = vol.Optional(conf_key, description={"suggested_value": suggested})
    else:
        key = vol.Optional(conf_key)
    schema_dict[key] = entity_selector


def _manual_override_schema(schema_dict: dict[Any, Any], current: dict[str, Any]) -> None:
    """Add the manual sensor-override fields to a schema dict.

    Overrides beat name-pattern discovery in the adapter. Domain filtering is
    deliberately loose (no device_class requirement): Modbus setups often lack
    device_class metadata, and these fields exist precisely for such setups.
    """
    for conf_key in MANUAL_SENSOR_OVERRIDE_FIELDS:
        _optional_entity_field(
            schema_dict,
            conf_key,
            selector.EntitySelector(selector.EntitySelectorConfig(domain=SOURCE_ENTITY_DOMAINS)),
            current.get(conf_key),
        )


class EffektGuardConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for EffektGuard."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize config flow."""
        self._data: dict[str, Any] = {}

    def is_matching(self, other_flow: "EffektGuardConfigFlow") -> bool:
        """Return True if this flow matches another flow in progress."""
        # Only one EffektGuard instance is supported
        return True

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
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

        # Discover NIBE entities (but don't abort if none found - entities may still load)
        nibe_entities = self._discover_nibe_entities()

        # Build helpful description based on discovery results
        if nibe_entities:
            info_msg = f"Found {len(nibe_entities)} NIBE offset entity(ies). Select the heating curve offset control."
        else:
            info_msg = (
                "No NIBE offset entities auto-detected. "
                "If your NIBE integration (MyUplink, nibe_heatpump, Modbus) just "
                "loaded, wait a moment and try again. Otherwise, manually select "
                "the number entity that controls the heating curve offset "
                "(register 47011 on F-series)."
            )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_NIBE_ENTITY): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=["number"],
                            multiple=False,
                        )
                    ),
                }
            ),
            errors=errors,
            description_placeholders={
                "nibe_count": str(len(nibe_entities)),
                "info": info_msg,
            },
        )

    async def async_step_gespot(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Configure spot price integration."""
        errors = {}

        if user_input is not None:
            # Store spot price settings
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

        # Auto-detect spot price entities (prioritizes current_price sensors)
        gespot_entities = self._discover_gespot_entities()
        auto_detected_gespot = gespot_entities[0] if gespot_entities else None

        # Build schema with auto-detected default
        schema_dict: dict[Any, Any] = {}

        if auto_detected_gespot:
            schema_dict[vol.Optional(CONF_GESPOT_ENTITY, default=auto_detected_gespot)] = (
                selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain="sensor",
                    )
                )
            )
        else:
            schema_dict[vol.Optional(CONF_GESPOT_ENTITY)] = selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor",
                )
            )

        schema_dict[vol.Optional(CONF_ENABLE_PRICE_OPTIMIZATION, default=True)] = (
            selector.BooleanSelector()
        )

        # Create description message
        if gespot_entities:
            detection_msg = f"Auto-detected {len(gespot_entities)} spot price sensor(s)"
            if auto_detected_gespot:
                detection_msg += f". Selected: {auto_detected_gespot.split('.')[-1]}"
        else:
            detection_msg = (
                "No spot price sensors detected. Please configure a spot price integration first."
            )

        return self.async_show_form(
            step_id="gespot",
            data_schema=vol.Schema(schema_dict),
            errors=errors,
            description_placeholders={
                "gespot_count": str(len(gespot_entities)),
                "detection_info": detection_msg,
            },
        )

    async def async_step_model(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
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
                    ): vol.In(HEAT_PUMP_MODEL_OPTIONS),
                }
            ),
            errors=errors,
            description_placeholders={
                "model_info": "Select your heat pump model for optimized control"
            },
        )

    async def async_step_optional(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
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
    ) -> ConfigFlowResult:
        """Configure optional sensors (degree minutes, power meter, extra temp sensors)."""
        if user_input is not None:
            # Store optional sensor settings
            self._data[CONF_DEGREE_MINUTES_ENTITY] = user_input.get(CONF_DEGREE_MINUTES_ENTITY)
            self._data[CONF_POWER_SENSOR_ENTITY] = user_input.get(CONF_POWER_SENSOR_ENTITY)
            self._data[CONF_NIBE_TEMP_LUX_ENTITY] = user_input.get(CONF_NIBE_TEMP_LUX_ENTITY)
            self._data[CONF_ADDITIONAL_INDOOR_SENSORS] = user_input.get(
                CONF_ADDITIONAL_INDOOR_SENSORS, []
            )
            self._data[CONF_INDOOR_TEMP_METHOD] = user_input.get(
                CONF_INDOOR_TEMP_METHOD, DEFAULT_INDOOR_TEMP_METHOD
            )
            for conf_key in MANUAL_SENSOR_OVERRIDE_FIELDS:
                self._data[conf_key] = user_input.get(conf_key)

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

        # Auto-detect temporary lux switch (DHW control)
        temp_lux_entities = self._discover_temp_lux_entities()
        auto_detected_temp_lux = temp_lux_entities[0] if temp_lux_entities else None

        schema_dict: dict[Any, Any] = {}

        # Degree minutes (optional) - a NUMBER entity on nibe_heatpump/MyUplink
        # (writable register), a sensor on generic Modbus setups (issue #18)
        _optional_entity_field(
            schema_dict,
            CONF_DEGREE_MINUTES_ENTITY,
            selector.EntitySelector(selector.EntitySelectorConfig(domain=SOURCE_ENTITY_DOMAINS)),
            auto_detected_dm,
        )

        # Power meter sensor (optional)
        _optional_entity_field(
            schema_dict,
            CONF_POWER_SENSOR_ENTITY,
            selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor",
                    device_class="power",
                )
            ),
            auto_detected_power,
        )

        # Temporary lux switch (DHW control, optional)
        _optional_entity_field(
            schema_dict,
            CONF_NIBE_TEMP_LUX_ENTITY,
            selector.EntitySelector(selector.EntitySelectorConfig(domain="switch")),
            auto_detected_temp_lux,
        )

        # Additional indoor temperature sensors (optional, multi-select)
        schema_dict[vol.Optional(CONF_ADDITIONAL_INDOOR_SENSORS)] = selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class="temperature",
                multiple=True,
            )
        )

        # Manual overrides for core NIBE sensors - required when name-pattern
        # discovery cannot find them (generic Modbus, renamed entities)
        _manual_override_schema(schema_dict, self._data)

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
                "temp_lux_detected": (
                    "✓ Auto-detected" if auto_detected_temp_lux else "❌ Not found"
                ),
                "dm_note": "Degree minutes sensor improves thermal debt tracking (estimated if not provided)",
                "power_note": "Power meter enables accurate peak tracking (estimated from heat pump data if not provided)",
                "temp_lux_note": "Temporary lux switch enables intelligent DHW scheduling and thermal debt protection (DHW optimization disabled if not provided)",
                "temp_sensors_note": "Add extra temperature sensors (living room, bedroom, etc.) for whole-house averaging. Median recommended (more robust to outliers).",
            },
        )

    def _discover_nibe_entities(self) -> list[str]:
        """Discover NIBE heating curve offset entities.

        Filters for writable number entities matching the shared offset
        patterns (const.NIBE_DISCOVERY_PATTERNS). Source-neutral: MyUplink,
        nibe_heatpump, and template numbers wrapping Modbus writes all qualify
        (issue #18).
        """
        entities = []
        # Count only NIBE-specific patterns: the bare "offset" pattern would
        # count unrelated calibration numbers (TRV/Tado temperature offsets)
        # and mislead the "Found N" message. The selector itself stays open.
        offset_patterns = [p for p in NIBE_DISCOVERY_PATTERNS["offset"] if p != "offset"]

        for state in self.hass.states.async_all("number"):
            entity_id = state.entity_id
            entity_lower = entity_id.lower()

            # Must be related to NIBE offset/curve control
            if not any(pattern in entity_lower for pattern in offset_patterns):
                continue

            # Exclude entities with translation errors
            friendly_name = state.attributes.get("friendly_name", "")
            if "text not found" in friendly_name.lower():
                continue

            entities.append(entity_id)

        return entities

    def _discover_gespot_entities(self) -> list[str]:
        """Discover spot price entities.

        Auto-detect spot price integration sensors:
        - sensor.gespot_current_price_* (preferred - real-time 15-min prices)
        - sensor.gespot_average_price_*
        - sensor.gespot_peak_price_*
        - sensor.gespot_off_peak_price_*
        - sensor.gespot_next_interval_price_*

        Prioritizes current_price sensor as it's most relevant for optimization.
        """
        entities = []
        current_price_entities = []

        for state in self.hass.states.async_all():
            entity_id = state.entity_id
            entity_lower = entity_id.lower()

            # Match spot price sensors
            if not entity_id.startswith("sensor."):
                continue

            # Check for spot price patterns
            is_gespot = (
                "gespot" in entity_lower
                or entity_id.startswith("sensor.ge_spot")
                or entity_id.startswith("sensor.ge-spot")
            )

            if not is_gespot:
                continue

            # Prioritize current_price sensors (real-time 15-min interval data)
            if "current_price" in entity_lower or "current-price" in entity_lower:
                current_price_entities.append(entity_id)
            else:
                entities.append(entity_id)

        # Return current_price sensors first, then others
        return current_price_entities + entities

    def _discover_weather_entities(self) -> list[str]:
        """Discover weather entities."""
        entities = []
        for state in self.hass.states.async_all():
            if state.entity_id.startswith("weather."):
                entities.append(state.entity_id)
        return entities

    def _discover_degree_minutes_entities(self) -> list[str]:
        """Discover NIBE degree minutes entities.

        Scans sensor AND number domains: nibe_heatpump and MyUplink expose
        degree minutes as a writable NUMBER entity (register 43005/40940),
        generic Modbus setups as a sensor (issue #18).

        Auto-detect common patterns:
        - *gradminuter* (Swedish)
        - *degree_minutes*
        - *_gm_* / *_dm_* (abbreviations, requires NIBE-related entity id)
        """
        entities = []
        search_terms = [
            "gradminuter",
            "degree_minutes",
            "degree_minute",
            "_gm_",
            "_dm_",
        ]

        ent_reg = er.async_get(self.hass)

        for state in self.hass.states.async_all(("sensor", "number")):
            entity_id = state.entity_id
            entity_lower = entity_id.lower()

            # Check for any search term in entity ID
            for term in search_terms:
                if term in entity_lower:
                    # Explicit degree-minutes name is always accepted
                    if "gradminuter" in entity_lower or "degree_minute" in entity_lower:
                        entities.append(entity_id)
                        break
                    # Abbreviations (_gm_/_dm_) need a NIBE source to avoid
                    # false positives. MyUplink ids carry the DEVICE name, not
                    # "myuplink", so check the registry platform, with the
                    # name substring as fallback for renamed entities.
                    entity_entry = ent_reg.async_get(entity_id)
                    if (
                        entity_entry and entity_entry.platform in ("myuplink", "nibe_heatpump")
                    ) or ("nibe" in entity_lower):
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

    def _discover_temp_lux_entities(self) -> list[str]:
        """Discover NIBE temporary lux switch for DHW control.

        Searches for switch entities with patterns:
        - switch.*temporary*lux*
        - switch.*temp*lux*
        - switch.*50004* (NIBE parameter ID)
        - Related to NIBE/MyUplink integration
        """
        entities = []
        ent_reg = er.async_get(self.hass)

        for state in self.hass.states.async_all():
            entity_id = state.entity_id

            # Must be a switch entity
            if not entity_id.startswith("switch."):
                continue

            entity_lower = entity_id.lower()

            # Check for temporary lux patterns
            is_temp_lux = (
                ("temporary" in entity_lower and "lux" in entity_lower)
                or ("temp" in entity_lower and "lux" in entity_lower)
                or "50004" in entity_lower  # MyUplink parameter ID
                or "48132" in entity_lower  # F-series Modbus register
            )

            if not is_temp_lux:
                continue

            # Verify it's from a NIBE source integration or NIBE-related by name
            entity_entry = ent_reg.async_get(entity_id)
            if entity_entry and entity_entry.platform in ("myuplink", "nibe_heatpump"):
                entities.append(entity_id)
            elif any(
                term in entity_lower
                for term in [
                    "nibe",
                    "myuplink",
                    "f750",
                    "f470",
                    "f1145",
                    "f1155",
                    "f1245",
                    "f1255",
                    "s1155",
                    "48132",
                ]
            ):
                # Entity ID contains NIBE model numbers, register, or source name
                entities.append(entity_id)

        return entities

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle reconfiguration of entity selections.

        Allows users to change entity selections (offset control, weather,
        power sensor, manual sensor overrides, etc.) without recreating the
        entire integration. Fields left empty are cleared (back to
        auto-discovery); the offset entity keeps its previous value when left
        empty because the integration cannot work without one.
        """
        errors = {}
        entry = self._get_reconfigure_entry()

        if user_input is not None:
            # Explicit None for absent optional fields so clearing works;
            # async_update_reload_and_abort merges data_updates over entry.data
            data_updates: dict[str, Any] = {
                CONF_WEATHER_ENTITY: user_input.get(CONF_WEATHER_ENTITY),
                CONF_DEGREE_MINUTES_ENTITY: user_input.get(CONF_DEGREE_MINUTES_ENTITY),
                CONF_POWER_SENSOR_ENTITY: user_input.get(CONF_POWER_SENSOR_ENTITY),
                CONF_NIBE_TEMP_LUX_ENTITY: user_input.get(CONF_NIBE_TEMP_LUX_ENTITY),
                CONF_ADDITIONAL_INDOOR_SENSORS: user_input.get(CONF_ADDITIONAL_INDOOR_SENSORS, []),
            }
            for conf_key in MANUAL_SENSOR_OVERRIDE_FIELDS:
                data_updates[conf_key] = user_input.get(conf_key)

            # Model and offset entity are required - keep old values if empty
            if user_input.get(CONF_HEAT_PUMP_MODEL):
                data_updates[CONF_HEAT_PUMP_MODEL] = user_input[CONF_HEAT_PUMP_MODEL]

            nibe_entity = user_input.get(CONF_NIBE_ENTITY)
            if nibe_entity:
                if not self.hass.states.get(nibe_entity):
                    errors["base"] = "nibe_entity_not_found"
                else:
                    data_updates[CONF_NIBE_ENTITY] = nibe_entity

            if not errors:
                return self.async_update_reload_and_abort(
                    entry,
                    data_updates=data_updates,
                )

        schema_dict: dict[Any, Any] = {}

        _optional_entity_field(
            schema_dict,
            CONF_NIBE_ENTITY,
            selector.EntitySelector(selector.EntitySelectorConfig(domain=["number"])),
            entry.data.get(CONF_NIBE_ENTITY),
        )
        # Model uses default= (not suggested_value): it must never be cleared,
        # an empty submit keeps the stored model
        schema_dict[
            vol.Optional(
                CONF_HEAT_PUMP_MODEL,
                default=entry.data.get(CONF_HEAT_PUMP_MODEL, DEFAULT_HEAT_PUMP_MODEL),
            )
        ] = vol.In(HEAT_PUMP_MODEL_OPTIONS)
        _optional_entity_field(
            schema_dict,
            CONF_WEATHER_ENTITY,
            selector.EntitySelector(selector.EntitySelectorConfig(domain="weather")),
            entry.data.get(CONF_WEATHER_ENTITY),
        )
        _optional_entity_field(
            schema_dict,
            CONF_DEGREE_MINUTES_ENTITY,
            selector.EntitySelector(selector.EntitySelectorConfig(domain=SOURCE_ENTITY_DOMAINS)),
            entry.data.get(CONF_DEGREE_MINUTES_ENTITY),
        )
        _optional_entity_field(
            schema_dict,
            CONF_POWER_SENSOR_ENTITY,
            selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor",
                    device_class="power",
                )
            ),
            entry.data.get(CONF_POWER_SENSOR_ENTITY),
        )
        _optional_entity_field(
            schema_dict,
            CONF_NIBE_TEMP_LUX_ENTITY,
            selector.EntitySelector(selector.EntitySelectorConfig(domain="switch")),
            entry.data.get(CONF_NIBE_TEMP_LUX_ENTITY),
        )
        _optional_entity_field(
            schema_dict,
            CONF_ADDITIONAL_INDOOR_SENSORS,
            selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor",
                    device_class="temperature",
                    multiple=True,
                )
            ),
            entry.data.get(CONF_ADDITIONAL_INDOOR_SENSORS, []),
        )
        _manual_override_schema(schema_dict, dict(entry.data))

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=vol.Schema(schema_dict),
            errors=errors,
            description_placeholders={
                "info": "Change entity selections. Integration will reload automatically."
            },
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return EffektGuardOptionsFlow()

"""Climate entity for EffektGuard.

Main user interface for the integration. Displays optimization status
and allows manual control.
"""

import logging
from typing import Any

from homeassistant.components.climate import (
    ClimateEntity,
    ClimateEntityFeature,
    HVACMode,
)
from homeassistant.components.climate.const import (
    PRESET_AWAY,
    PRESET_COMFORT,
    PRESET_ECO,
    PRESET_NONE,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    CONF_OPTIMIZATION_MODE,
    CONF_TARGET_INDOOR_TEMP,
    DEFAULT_INDOOR_TEMP,
    DOMAIN,
    MAX_INDOOR_TEMP,
    MIN_INDOOR_TEMP,
    OPTIMIZATION_MODE_BALANCED,
    OPTIMIZATION_MODE_COMFORT,
    OPTIMIZATION_MODE_SAVINGS,
    TEMP_STEP,
)
from .coordinator import EffektGuardCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up EffektGuard climate entity from a config entry."""
    coordinator: EffektGuardCoordinator = hass.data[DOMAIN][entry.entry_id]

    async_add_entities([EffektGuardClimate(coordinator, entry)])


class EffektGuardClimate(CoordinatorEntity, RestoreEntity, ClimateEntity):
    """Climate entity for EffektGuard.

    Main user interface displaying current optimization status and allowing
    manual control of target temperature and optimization mode via presets.
    """

    _attr_has_entity_name = True
    _attr_name = None
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.OFF]
    _attr_preset_modes = [PRESET_NONE, PRESET_ECO, PRESET_AWAY, PRESET_COMFORT]
    _attr_supported_features = (
        ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.PRESET_MODE
    )
    _attr_min_temp = MIN_INDOOR_TEMP
    _attr_max_temp = MAX_INDOOR_TEMP
    _attr_target_temperature_step = TEMP_STEP

    def __init__(
        self,
        coordinator: EffektGuardCoordinator,
        entry: ConfigEntry,
    ):
        """Initialize climate entity."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_climate"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="EffektGuard",
            manufacturer="EffektGuard",
            model="Heat Pump Optimizer",
        )
        self._entry = entry
        self._attr_hvac_mode = HVACMode.HEAT
        self._attr_preset_mode = PRESET_NONE

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass.

        Restore previous state to maintain HVAC mode across restarts.
        """
        await super().async_added_to_hass()

        # Restore previous state if available
        if (last_state := await self.async_get_last_state()) is not None:
            # Restore HVAC mode if valid
            if last_state.state in [mode.value for mode in self._attr_hvac_modes]:
                try:
                    self._attr_hvac_mode = HVACMode(last_state.state)
                    _LOGGER.debug(
                        "Restored HVAC mode: %s from previous state", self._attr_hvac_mode
                    )
                except ValueError:
                    _LOGGER.warning(
                        "Invalid HVAC mode '%s' in restored state, using default HEAT",
                        last_state.state,
                    )
                    self._attr_hvac_mode = HVACMode.HEAT
            else:
                _LOGGER.debug("No valid HVAC mode to restore, using default HEAT")
                self._attr_hvac_mode = HVACMode.HEAT

    @property
    def current_temperature(self) -> float | None:
        """Return current indoor temperature from NIBE."""
        if self.coordinator.data and "nibe" in self.coordinator.data:
            nibe_state = self.coordinator.data["nibe"]
            if nibe_state and hasattr(nibe_state, "indoor_temp"):
                return nibe_state.indoor_temp
        return None

    @property
    def target_temperature(self) -> float | None:
        """Return target temperature from config options."""
        # Check options first (preferred), fall back to data for migration
        return self._entry.options.get(
            CONF_TARGET_INDOOR_TEMP,
            self._entry.data.get(CONF_TARGET_INDOOR_TEMP, DEFAULT_INDOOR_TEMP),
        )

    @property
    def preset_mode(self) -> str:
        """Return current preset mode."""
        # Check options first (preferred), fall back to data for migration
        optimization_mode = self._entry.options.get(
            CONF_OPTIMIZATION_MODE,
            self._entry.data.get(CONF_OPTIMIZATION_MODE, OPTIMIZATION_MODE_BALANCED),
        )
        # Map optimization mode to preset
        preset_map = {
            OPTIMIZATION_MODE_COMFORT: PRESET_COMFORT,
            OPTIMIZATION_MODE_BALANCED: PRESET_NONE,
            OPTIMIZATION_MODE_SAVINGS: PRESET_ECO,
        }
        return preset_map.get(optimization_mode, PRESET_NONE)

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set target temperature.

        Updates the target temperature in config entry options.
        Optimization engine will use new target in next update cycle.
        """
        if (temperature := kwargs.get(ATTR_TEMPERATURE)) is None:
            return

        temperature = max(self._attr_min_temp, min(temperature, self._attr_max_temp))

        _LOGGER.info("Setting target temperature to %.1f°C", temperature)

        # Update config entry options (triggers update listener)
        # This follows Home Assistant best practices for ClimateEntity
        new_options = dict(self._entry.options)
        new_options[CONF_TARGET_INDOOR_TEMP] = temperature

        # Update the config entry - this automatically triggers the update listener
        # which calls async_update_config() on the coordinator
        self.hass.config_entries.async_update_entry(self._entry, options=new_options)

        # Update this entity's state to show new target temperature immediately
        # This is safe because we're only updating the climate entity's own state,
        # not triggering a coordinator data refresh that would affect other entities
        self.async_write_ha_state()

    async def async_set_hvac_mode(self, hvac_mode: HVACMode | str) -> None:
        """Set HVAC mode.

        HEAT: Optimization active
        OFF: Optimization disabled (safety monitoring only)
        """
        # Convert string to HVACMode if needed
        if isinstance(hvac_mode, str):
            hvac_mode = HVACMode(hvac_mode)

        _LOGGER.info("Setting HVAC mode to %s", hvac_mode)
        self._attr_hvac_mode = hvac_mode

        if hvac_mode == HVACMode.OFF:
            # Disable optimization, reset offset to neutral
            await self.coordinator.set_optimization_enabled(False)
        else:
            # Enable optimization
            await self.coordinator.set_optimization_enabled(True)

        self.async_write_ha_state()

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set preset mode.

        Maps preset to optimization mode:
        - COMFORT: Minimize temperature deviation, accept higher costs
        - NONE/BALANCED: Balance comfort and savings
        - ECO: Maximize savings, wider temperature tolerance
        - AWAY: Reduce temperature, maximum savings
        """
        _LOGGER.info("Setting preset mode to %s", preset_mode)

        # Map preset to optimization mode
        mode_map = {
            PRESET_COMFORT: OPTIMIZATION_MODE_COMFORT,
            PRESET_NONE: OPTIMIZATION_MODE_BALANCED,
            PRESET_ECO: OPTIMIZATION_MODE_SAVINGS,
            PRESET_AWAY: OPTIMIZATION_MODE_SAVINGS,  # Same as ECO but could differ
        }

        optimization_mode = mode_map.get(preset_mode, OPTIMIZATION_MODE_BALANCED)

        # Update config entry options (triggers update listener)
        # This follows Home Assistant best practices for ClimateEntity preset changes
        new_options = dict(self._entry.options)
        new_options[CONF_OPTIMIZATION_MODE] = optimization_mode

        self.hass.config_entries.async_update_entry(self._entry, options=new_options)
        # This automatically calls async_reload_entry() which updates coordinator config
        # via async_update_config() - no need for explicit refresh

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional state attributes."""
        attrs = {}

        if self.coordinator.data:
            # Current offset applied
            if "decision" in self.coordinator.data:
                decision = self.coordinator.data["decision"]
                attrs["current_offset"] = decision.offset
                attrs["optimization_reasoning"] = decision.reasoning

            # NIBE state
            if "nibe" in self.coordinator.data:
                nibe_state = self.coordinator.data["nibe"]
                if nibe_state:
                    attrs["outdoor_temp"] = getattr(nibe_state, "outdoor_temp", None)
                    attrs["supply_temp"] = getattr(nibe_state, "supply_temp", None)
                    attrs["degree_minutes"] = getattr(nibe_state, "degree_minutes", None)

            # Price info
            if "price" in self.coordinator.data:
                price_data = self.coordinator.data["price"]
                if price_data and hasattr(price_data, "current_price"):
                    attrs["current_price"] = price_data.current_price

            # Peak tracking
            attrs["monthly_peak"] = self.coordinator.current_peak

        return attrs

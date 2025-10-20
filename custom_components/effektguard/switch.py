"""Switch entities for EffektGuard.

Feature enable/disable switches for optimization features.
"""

import logging
from dataclasses import dataclass
from typing import Any

from homeassistant.components.switch import SwitchEntity, SwitchEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    CONF_ENABLE_HOT_WATER_OPTIMIZATION,
    CONF_ENABLE_OPTIMIZATION,
    CONF_ENABLE_PEAK_PROTECTION,
    CONF_ENABLE_PRICE_OPTIMIZATION,
    CONF_ENABLE_WEATHER_PREDICTION,
    DOMAIN,
)
from .coordinator import EffektGuardCoordinator

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EffektGuardSwitchEntityDescription(SwitchEntityDescription):
    """Describes EffektGuard switch entity."""

    config_key: str | None = None


SWITCHES: tuple[EffektGuardSwitchEntityDescription, ...] = (
    EffektGuardSwitchEntityDescription(
        key="enable_optimization",
        name="Enable Optimization",
        translation_key="enable_optimization",
        icon="mdi:power",
        config_key=CONF_ENABLE_OPTIMIZATION,
    ),
    EffektGuardSwitchEntityDescription(
        key="price_optimization",
        name="Price Optimization",
        translation_key="price_optimization",
        icon="mdi:cash",
        config_key=CONF_ENABLE_PRICE_OPTIMIZATION,
    ),
    EffektGuardSwitchEntityDescription(
        key="peak_protection",
        name="Peak Protection",
        translation_key="peak_protection",
        icon="mdi:shield-alert",
        config_key=CONF_ENABLE_PEAK_PROTECTION,
    ),
    EffektGuardSwitchEntityDescription(
        key="weather_prediction",
        name="Weather Prediction",
        translation_key="weather_prediction",
        icon="mdi:weather-partly-cloudy",
        config_key=CONF_ENABLE_WEATHER_PREDICTION,
    ),
    EffektGuardSwitchEntityDescription(
        key="hot_water_optimization",
        name="Hot Water Optimization",
        translation_key="hot_water_optimization",
        icon="mdi:water-boiler",
        config_key=CONF_ENABLE_HOT_WATER_OPTIMIZATION,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up EffektGuard switch entities from a config entry."""
    coordinator: EffektGuardCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities = [EffektGuardSwitch(coordinator, entry, description) for description in SWITCHES]

    async_add_entities(entities)


class EffektGuardSwitch(CoordinatorEntity, SwitchEntity):
    """EffektGuard feature toggle switch entity."""

    entity_description: EffektGuardSwitchEntityDescription
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: EffektGuardCoordinator,
        entry: ConfigEntry,
        description: EffektGuardSwitchEntityDescription,
    ):
        """Initialize switch entity."""
        super().__init__(coordinator)
        self.entity_description = description
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_{description.key}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="EffektGuard",
            manufacturer="EffektGuard",
            model="Heat Pump Optimizer",
        )

    @property
    def is_on(self) -> bool:
        """Return True if the switch is on."""
        config_key = self.entity_description.config_key
        if not config_key:
            return False

        # Default values based on config key
        defaults = {
            CONF_ENABLE_OPTIMIZATION: True,  # Master switch on by default
            CONF_ENABLE_PRICE_OPTIMIZATION: True,
            CONF_ENABLE_PEAK_PROTECTION: True,
            CONF_ENABLE_WEATHER_PREDICTION: True,
            CONF_ENABLE_HOT_WATER_OPTIMIZATION: False,  # Experimental, off by default
        }

        return self._entry.data.get(config_key, defaults.get(config_key, False))

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the switch on."""
        config_key = self.entity_description.config_key
        if not config_key:
            return

        _LOGGER.info("Enabling %s", config_key)

        # Update config entry data (triggers update listener)
        # This follows Home Assistant best practices for SwitchEntity
        new_data = dict(self._entry.data)
        new_data[config_key] = True

        self.hass.config_entries.async_update_entry(self._entry, data=new_data)
        # This automatically calls async_reload_entry() which updates coordinator config
        # via async_update_config() - no need for explicit refresh

        # Update this entity's state immediately (standard HA pattern)
        # User sees new value instantly, coordinator applies it on next cycle
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the switch off."""
        config_key = self.entity_description.config_key
        if not config_key:
            return

        _LOGGER.info("Disabling %s", config_key)

        # Update config entry data (triggers update listener)
        # This follows Home Assistant best practices for SwitchEntity
        new_data = dict(self._entry.data)
        new_data[config_key] = False

        self.hass.config_entries.async_update_entry(self._entry, data=new_data)
        # This automatically calls async_reload_entry() which updates coordinator config
        # via async_update_config() - no need for explicit refresh

        # Update this entity's state immediately (standard HA pattern)
        # User sees new value instantly, coordinator applies it on next cycle
        self.async_write_ha_state()

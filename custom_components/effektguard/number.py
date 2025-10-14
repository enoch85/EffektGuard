"""Number entities for EffektGuard.

Configuration number entities for runtime adjustment of thermal parameters
and comfort settings.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from homeassistant.components.number import NumberEntity, NumberEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfPower, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    CONF_INSULATION_QUALITY,
    CONF_PEAK_PROTECTION_MARGIN,
    CONF_TARGET_INDOOR_TEMP,
    CONF_THERMAL_MASS,
    CONF_TOLERANCE,
    DEFAULT_INSULATION_QUALITY,
    DEFAULT_PEAK_PROTECTION_MARGIN,
    DEFAULT_TARGET_TEMP,
    DEFAULT_THERMAL_MASS,
    DEFAULT_TOLERANCE,
    DOMAIN,
    MAX_TARGET_TEMP,
    MIN_TARGET_TEMP,
)
from .coordinator import EffektGuardCoordinator

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EffektGuardNumberEntityDescription(NumberEntityDescription):
    """Describes EffektGuard number entity."""

    config_key: str | None = None
    set_value_fn: Callable[[HomeAssistant, ConfigEntry, float], None] | None = None


NUMBERS: tuple[EffektGuardNumberEntityDescription, ...] = (
    EffektGuardNumberEntityDescription(
        key="target_temperature",
        name="Target Temperature",
        icon="mdi:thermometer",
        native_min_value=MIN_TARGET_TEMP,
        native_max_value=MAX_TARGET_TEMP,
        native_step=0.5,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        config_key=CONF_TARGET_INDOOR_TEMP,
    ),
    EffektGuardNumberEntityDescription(
        key="tolerance",
        name="Temperature Tolerance",
        icon="mdi:thermometer-lines",
        native_min_value=0.2,
        native_max_value=2.0,
        native_step=0.1,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        config_key=CONF_TOLERANCE,
    ),
    EffektGuardNumberEntityDescription(
        key="thermal_mass",
        name="Thermal Mass",
        icon="mdi:home-thermometer",
        native_min_value=0.5,
        native_max_value=2.0,
        native_step=0.1,
        config_key=CONF_THERMAL_MASS,
    ),
    EffektGuardNumberEntityDescription(
        key="insulation_quality",
        name="Building Insulation Quality",
        icon="mdi:home-thermometer-outline",
        native_min_value=0.5,
        native_max_value=2.0,
        native_step=0.1,
        config_key=CONF_INSULATION_QUALITY,
    ),
    EffektGuardNumberEntityDescription(
        key="peak_protection_margin",
        name="Peak Protection Margin",
        icon="mdi:shield-alert-outline",
        native_min_value=0.0,
        native_max_value=2.0,
        native_step=0.1,
        native_unit_of_measurement="kW",
        config_key=CONF_PEAK_PROTECTION_MARGIN,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up EffektGuard number entities from a config entry."""
    coordinator: EffektGuardCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities = [EffektGuardNumber(coordinator, entry, description) for description in NUMBERS]

    async_add_entities(entities)


class EffektGuardNumber(CoordinatorEntity, NumberEntity):
    """EffektGuard configuration number entity."""

    entity_description: EffektGuardNumberEntityDescription
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: EffektGuardCoordinator,
        entry: ConfigEntry,
        description: EffektGuardNumberEntityDescription,
    ):
        """Initialize number entity."""
        super().__init__(coordinator)
        self.entity_description = description
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_{description.key}"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "EffektGuard",
            "manufacturer": "EffektGuard",
            "model": "Heat Pump Optimizer",
        }

    @property
    def native_value(self) -> float:
        """Return the current value."""
        config_key = self.entity_description.config_key
        if not config_key:
            return 0.0

        # Get default value based on config key
        defaults = {
            CONF_TARGET_INDOOR_TEMP: DEFAULT_TARGET_TEMP,
            CONF_TOLERANCE: DEFAULT_TOLERANCE,
            CONF_THERMAL_MASS: DEFAULT_THERMAL_MASS,
            CONF_INSULATION_QUALITY: DEFAULT_INSULATION_QUALITY,
            CONF_PEAK_PROTECTION_MARGIN: DEFAULT_PEAK_PROTECTION_MARGIN,
        }

        return self._entry.data.get(config_key, defaults.get(config_key, 1.0))

    async def async_set_native_value(self, value: float) -> None:
        """Update the value."""
        config_key = self.entity_description.config_key
        if not config_key:
            return

        _LOGGER.info("Setting %s to %.2f", config_key, value)

        # Update config entry data
        new_data = dict(self._entry.data)
        new_data[config_key] = value

        self.hass.config_entries.async_update_entry(self._entry, data=new_data)

        # Request coordinator refresh to apply new value
        await self.coordinator.async_request_refresh()

        self.async_write_ha_state()

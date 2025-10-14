"""Select entities for EffektGuard.

Configuration select entities for choosing system type, UFH type,
and optimization modes.
"""

import logging
from dataclasses import dataclass

from homeassistant.components.select import SelectEntity, SelectEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    CONF_OPTIMIZATION_MODE,
    DEFAULT_OPTIMIZATION_MODE,
    DOMAIN,
    OPTIMIZATION_MODE_BALANCED,
    OPTIMIZATION_MODE_COMFORT,
    OPTIMIZATION_MODE_SAVINGS,
)
from .coordinator import EffektGuardCoordinator

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EffektGuardSelectEntityDescription(SelectEntityDescription):
    """Describes EffektGuard select entity."""

    config_key: str | None = None


SELECTS: tuple[EffektGuardSelectEntityDescription, ...] = (
    EffektGuardSelectEntityDescription(
        key="optimization_mode",
        name="Optimization Mode",
        icon="mdi:tune",
        options=[
            OPTIMIZATION_MODE_COMFORT,
            OPTIMIZATION_MODE_BALANCED,
            OPTIMIZATION_MODE_SAVINGS,
        ],
        config_key=CONF_OPTIMIZATION_MODE,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up EffektGuard select entities from a config entry."""
    coordinator: EffektGuardCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities = [EffektGuardSelect(coordinator, entry, description) for description in SELECTS]

    async_add_entities(entities)


class EffektGuardSelect(CoordinatorEntity, SelectEntity):
    """EffektGuard configuration select entity."""

    entity_description: EffektGuardSelectEntityDescription
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: EffektGuardCoordinator,
        entry: ConfigEntry,
        description: EffektGuardSelectEntityDescription,
    ):
        """Initialize select entity."""
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
    def current_option(self) -> str:
        """Return the current option."""
        config_key = self.entity_description.config_key
        if not config_key:
            return self.entity_description.options[0] if self.entity_description.options else ""

        return self._entry.data.get(config_key, DEFAULT_OPTIMIZATION_MODE)

    async def async_select_option(self, option: str) -> None:
        """Change the selected option."""
        if option not in self.entity_description.options:
            _LOGGER.error("Invalid option: %s", option)
            return

        config_key = self.entity_description.config_key
        if not config_key:
            return

        _LOGGER.info("Setting %s to %s", config_key, option)

        # Update config entry data
        new_data = dict(self._entry.data)
        new_data[config_key] = option

        self.hass.config_entries.async_update_entry(self._entry, data=new_data)

        # Request coordinator refresh to apply new mode
        await self.coordinator.async_request_refresh()

        self.async_write_ha_state()

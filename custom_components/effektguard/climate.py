"""Climate entity for EffektGuard.

Main user interface for the integration. Displays optimization status
and allows manual control.
"""

import logging

from homeassistant.components.climate import ClimateEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
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


class EffektGuardClimate(CoordinatorEntity, ClimateEntity):
    """Climate entity for EffektGuard.

    This will be fully implemented in Phase 4.
    """

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self,
        coordinator: EffektGuardCoordinator,
        entry: ConfigEntry,
    ):
        """Initialize climate entity."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_climate"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "EffektGuard",
            "manufacturer": "EffektGuard",
            "model": "Heat Pump Optimizer",
        }

    # Full implementation will be added in Phase 4

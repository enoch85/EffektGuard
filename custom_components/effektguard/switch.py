"""Switch entities for EffektGuard.

Feature enable/disable switches for optimization features.
"""

import logging
from dataclasses import dataclass
from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    CONF_ENABLE_AIRFLOW_OPTIMIZATION,
    CONF_ENABLE_HOT_WATER_OPTIMIZATION,
    CONF_ENABLE_OPTIMIZATION,
    CONF_ENABLE_PEAK_PROTECTION,
    CONF_ENABLE_PRICE_OPTIMIZATION,
    CONF_ENABLE_WEATHER_PREDICTION,
    CONF_HEAT_PUMP_MODEL,
    DEFAULT_HEAT_PUMP_MODEL,
    DOMAIN,
)
from .coordinator import EffektGuardCoordinator
from .models import HeatPumpModelRegistry

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EffektGuardSwitchEntityDescription:
    """Custom switch entity description for EffektGuard.

    Uses composition instead of inheritance from SwitchEntityDescription
    to avoid Pylance/Pylint type-checking issues with Home Assistant's FrozenOrThawed metaclass.
    All fields that SwitchEntity reads from entity_description are included.
    """

    # Required field
    key: str

    # Optional fields matching what SwitchEntity reads from entity_description
    name: str | None = None
    icon: str | None = None
    device_class: str | None = None
    entity_category: EntityCategory | None = None
    translation_key: str | None = None
    translation_placeholders: dict[str, str] | None = None
    has_entity_name: bool = False
    entity_registry_enabled_default: bool = True
    entity_registry_visible_default: bool = True
    force_update: bool = False

    # EffektGuard-specific field
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
    EffektGuardSwitchEntityDescription(
        key="airflow_optimization",
        name="Airflow Optimization",
        translation_key="airflow_optimization",
        icon="mdi:fan",
        config_key=CONF_ENABLE_AIRFLOW_OPTIMIZATION,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up EffektGuard switch entities from a config entry."""
    coordinator: EffektGuardCoordinator = hass.data[DOMAIN][entry.entry_id]

    # Check if heat pump supports exhaust airflow optimization
    model_id = entry.data.get(CONF_HEAT_PUMP_MODEL, DEFAULT_HEAT_PUMP_MODEL)
    supports_airflow = False
    try:
        model_profile = HeatPumpModelRegistry.get_model(model_id)
        supports_airflow = getattr(model_profile, "supports_exhaust_airflow", False)
    except ValueError:
        _LOGGER.warning("Unknown heat pump model: %s, airflow optimization disabled", model_id)

    # Filter switches based on heat pump capabilities
    entities = []
    for description in SWITCHES:
        # Only show airflow optimization switch for exhaust air heat pumps
        if description.key == "airflow_optimization" and not supports_airflow:
            _LOGGER.debug(
                "Hiding airflow optimization switch - model %s does not support exhaust airflow",
                model_id,
            )
            continue
        entities.append(EffektGuardSwitch(coordinator, entry, description))

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
            CONF_ENABLE_AIRFLOW_OPTIMIZATION: False,  # Experimental, off by default
        }

        return self._entry.data.get(config_key, defaults.get(config_key, False))

    def turn_on(self, **kwargs: Any) -> None:
        """Turn the switch on (sync wrapper)."""
        raise NotImplementedError("Use async_turn_on instead")

    def turn_off(self, **kwargs: Any) -> None:
        """Turn the switch off (sync wrapper)."""
        raise NotImplementedError("Use async_turn_off instead")

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

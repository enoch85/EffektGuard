"""Sensor entities for EffektGuard.

Diagnostic sensors for monitoring optimization status, peak tracking,
thermal debt, and system performance.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfEnergy, UnitOfPower, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import EffektGuardCoordinator

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EffektGuardSensorEntityDescription(SensorEntityDescription):
    """Describes EffektGuard sensor entity."""

    value_fn: Callable[[EffektGuardCoordinator], Any] | None = None


SENSORS: tuple[EffektGuardSensorEntityDescription, ...] = (
    EffektGuardSensorEntityDescription(
        key="current_offset",
        name="Current Offset",
        icon="mdi:thermometer-lines",
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: (
            coordinator.data.get("decision").offset
            if coordinator.data and "decision" in coordinator.data
            else 0.0
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="degree_minutes",
        name="Degree Minutes",
        icon="mdi:timer-outline",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: (
            coordinator.data.get("nibe").degree_minutes
            if coordinator.data and "nibe" in coordinator.data and coordinator.data["nibe"]
            else None
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="supply_temperature",
        name="Supply Temperature",
        icon="mdi:thermometer",
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: (
            coordinator.data.get("nibe").supply_temp
            if coordinator.data and "nibe" in coordinator.data and coordinator.data["nibe"]
            else None
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="outdoor_temperature",
        name="Outdoor Temperature",
        icon="mdi:thermometer",
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: (
            coordinator.data.get("nibe").outdoor_temp
            if coordinator.data and "nibe" in coordinator.data and coordinator.data["nibe"]
            else None
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="current_price",
        name="Current Electricity Price",
        icon="mdi:currency-eur",
        device_class=SensorDeviceClass.MONETARY,
        native_unit_of_measurement="SEK/kWh",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: (
            coordinator.data.get("price").current_price
            if coordinator.data
            and "price" in coordinator.data
            and coordinator.data["price"]
            and hasattr(coordinator.data["price"], "current_price")
            else None
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="peak_today",
        name="Peak Today",
        icon="mdi:transmission-tower",
        device_class=SensorDeviceClass.POWER,
        native_unit_of_measurement=UnitOfPower.KILO_WATT,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: coordinator.peak_today,
    ),
    EffektGuardSensorEntityDescription(
        key="peak_this_month",
        name="Peak This Month",
        icon="mdi:transmission-tower-export",
        device_class=SensorDeviceClass.POWER,
        native_unit_of_measurement=UnitOfPower.KILO_WATT,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: coordinator.peak_this_month,
    ),
    EffektGuardSensorEntityDescription(
        key="optimization_reasoning",
        name="Optimization Reasoning",
        icon="mdi:brain",
        value_fn=lambda coordinator: (
            coordinator.data.get("decision").reasoning
            if coordinator.data and "decision" in coordinator.data
            else "No decision yet"
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="quarter_of_day",
        name="Quarter of Day",
        icon="mdi:clock-outline",
        value_fn=lambda coordinator: (
            coordinator.data.get("price").current_quarter
            if coordinator.data
            and "price" in coordinator.data
            and coordinator.data["price"]
            and hasattr(coordinator.data["price"], "current_quarter")
            else None
        ),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up EffektGuard sensor entities from a config entry."""
    coordinator: EffektGuardCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities = [EffektGuardSensor(coordinator, entry, description) for description in SENSORS]

    async_add_entities(entities)


class EffektGuardSensor(CoordinatorEntity, SensorEntity):
    """EffektGuard diagnostic sensor."""

    entity_description: EffektGuardSensorEntityDescription
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: EffektGuardCoordinator,
        entry: ConfigEntry,
        description: EffektGuardSensorEntityDescription,
    ):
        """Initialize sensor."""
        super().__init__(coordinator)
        self.entity_description = description
        self._attr_unique_id = f"{entry.entry_id}_{description.key}"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "EffektGuard",
            "manufacturer": "EffektGuard",
            "model": "Heat Pump Optimizer",
        }

    @property
    def native_value(self) -> Any:
        """Return the state of the sensor."""
        if self.entity_description.value_fn:
            try:
                return self.entity_description.value_fn(self.coordinator)
            except (AttributeError, KeyError, TypeError) as err:
                _LOGGER.debug("Error getting value for %s: %s", self.entity_description.key, err)
                return None
        return None

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
        # Note: monetary device_class doesn't support state_class
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
    EffektGuardSensorEntityDescription(
        key="hour_classification",
        name="Hour Classification",
        icon="mdi:chart-timeline-variant",
        value_fn=lambda coordinator: (
            coordinator.data.get("price").current_classification
            if coordinator.data
            and "price" in coordinator.data
            and coordinator.data["price"]
            and hasattr(coordinator.data["price"], "current_classification")
            else "unknown"
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="peak_status",
        name="Peak Status",
        icon="mdi:alert-circle-outline",
        value_fn=lambda coordinator: (
            coordinator.data.get("decision").peak_status
            if coordinator.data
            and "decision" in coordinator.data
            and coordinator.data["decision"]
            and hasattr(coordinator.data["decision"], "peak_status")
            else "unknown"
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="temperature_trend",
        name="Temperature Trend",
        icon="mdi:trending-up",
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement="°C/h",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: (
            coordinator.data.get("thermal").temperature_trend
            if coordinator.data
            and "thermal" in coordinator.data
            and coordinator.data["thermal"]
            and hasattr(coordinator.data["thermal"], "temperature_trend")
            else None
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="savings_estimate",
        name="Estimated Monthly Savings",
        icon="mdi:cash-multiple",
        device_class=SensorDeviceClass.MONETARY,
        native_unit_of_measurement="SEK",
        state_class=SensorStateClass.TOTAL,
        value_fn=lambda coordinator: (
            coordinator.data.get("savings").monthly_estimate
            if coordinator.data
            and "savings" in coordinator.data
            and coordinator.data["savings"]
            and hasattr(coordinator.data["savings"], "monthly_estimate")
            else None
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="optional_features_status",
        name="Optional Features Status",
        icon="mdi:feature-search-outline",
        value_fn=lambda coordinator: ("active" if coordinator.data else "initializing"),
    ),
    EffektGuardSensorEntityDescription(
        key="heat_pump_model",
        name="Heat Pump Model",
        icon="mdi:heat-pump",
        value_fn=lambda coordinator: (
            coordinator.heat_pump_model.model_name
            if hasattr(coordinator, "heat_pump_model") and coordinator.heat_pump_model
            else "Unknown"
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

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional state attributes based on sensor type."""
        attrs = {}

        if not self.coordinator.data:
            return attrs

        # Add sensor-specific attributes
        key = self.entity_description.key

        if key == "current_offset":
            # Show layer breakdown for calculated offset
            if "decision" in self.coordinator.data:
                decision = self.coordinator.data["decision"]
                if decision and hasattr(decision, "layers"):
                    attrs["layer_votes"] = [
                        {
                            "name": layer.name,
                            "offset": layer.offset,
                            "weight": layer.weight,
                            "reason": layer.reason,
                        }
                        for layer in decision.layers
                    ]

        elif key == "hour_classification":
            # Show today's full classification
            if "price" in self.coordinator.data:
                price_data = self.coordinator.data["price"]
                if price_data and hasattr(price_data, "today_classifications"):
                    attrs["today_classifications"] = price_data.today_classifications
                if price_data and hasattr(price_data, "today_prices"):
                    attrs["today_min"] = min(price_data.today_prices)
                    attrs["today_max"] = max(price_data.today_prices)
                    attrs["today_average"] = sum(price_data.today_prices) / len(
                        price_data.today_prices
                    )

        elif key == "peak_status":
            # Show margin to peak and top peaks
            if "decision" in self.coordinator.data:
                decision = self.coordinator.data["decision"]
                if decision and hasattr(decision, "peak_margin"):
                    attrs["margin_to_peak"] = decision.peak_margin
                if decision and hasattr(decision, "current_power"):
                    attrs["current_power"] = decision.current_power
            attrs["monthly_peak"] = self.coordinator.peak_this_month
            attrs["daily_peak"] = self.coordinator.peak_today

        elif key == "temperature_trend":
            # Show prediction
            if "thermal" in self.coordinator.data:
                thermal_data = self.coordinator.data["thermal"]
                if thermal_data and hasattr(thermal_data, "prediction_3h"):
                    attrs["prediction_3h"] = thermal_data.prediction_3h
                if "weather" in self.coordinator.data:
                    weather = self.coordinator.data["weather"]
                    if weather and hasattr(weather, "forecast_hours"):
                        attrs["forecast"] = [
                            {"time": f.datetime, "temp": f.temperature}
                            for f in weather.forecast_hours[:3]
                        ]

        elif key == "savings_estimate":
            # Show breakdown of savings
            if "savings" in self.coordinator.data:
                savings = self.coordinator.data["savings"]
                if savings:
                    if hasattr(savings, "effect_savings"):
                        attrs["effect_savings"] = savings.effect_savings
                    if hasattr(savings, "spot_savings"):
                        attrs["spot_savings"] = savings.spot_savings
                    if hasattr(savings, "baseline_cost"):
                        attrs["baseline_cost"] = savings.baseline_cost
                    if hasattr(savings, "optimized_cost"):
                        attrs["optimized_cost"] = savings.optimized_cost

        elif key == "optimization_reasoning":
            # Already has reasoning in value, add decision timestamp
            if "decision" in self.coordinator.data:
                decision = self.coordinator.data["decision"]
                if decision and hasattr(decision, "timestamp"):
                    attrs["decision_timestamp"] = decision.timestamp.isoformat()
                if decision and hasattr(decision, "offset"):
                    attrs["applied_offset"] = decision.offset

        elif key == "optional_features_status":
            # Show detailed status of all optional features
            config = self.coordinator.config_entry.data

            # Degree minutes status
            dm_entity = config.get("degree_minutes_entity")
            if dm_entity:
                dm_state = self.coordinator.hass.states.get(dm_entity)
                attrs["degree_minutes"] = {
                    "status": "detected" if dm_state else "configured_but_unavailable",
                    "entity": dm_entity,
                    "value": dm_state.state if dm_state else None,
                }
            else:
                attrs["degree_minutes"] = {
                    "status": "estimated",
                    "entity": None,
                    "note": "Using thermal model estimation",
                }

            # Power meter status
            power_entity = config.get("power_sensor_entity")
            if power_entity:
                power_state = self.coordinator.hass.states.get(power_entity)
                attrs["power_meter"] = {
                    "status": "detected" if power_state else "configured_but_unavailable",
                    "entity": power_entity,
                    "value": power_state.state if power_state else None,
                    "unit": (
                        power_state.attributes.get("unit_of_measurement") if power_state else None
                    ),
                }
            else:
                attrs["power_meter"] = {
                    "status": "estimated",
                    "entity": None,
                    "note": "Estimating from heat pump data",
                }

            # Tomorrow prices status (from GE-Spot)
            if "price" in self.coordinator.data:
                price_data = self.coordinator.data["price"]
                has_tomorrow = (
                    price_data
                    and hasattr(price_data, "tomorrow_prices")
                    and price_data.tomorrow_prices
                    and len(price_data.tomorrow_prices) > 0
                )
                attrs["tomorrow_prices"] = {
                    "status": "available" if has_tomorrow else "today_only",
                    "count": len(price_data.tomorrow_prices) if has_tomorrow else 0,
                    "note": (
                        "Can optimize further ahead"
                        if has_tomorrow
                        else "Optimizing with today only"
                    ),
                }
            else:
                attrs["tomorrow_prices"] = {
                    "status": "unknown",
                    "count": 0,
                }

            # Weather forecast status
            weather_entity = config.get("weather_entity")
            if weather_entity:
                weather_state = self.coordinator.hass.states.get(weather_entity)
                if weather_state and "forecast" in weather_state.attributes:
                    forecast = weather_state.attributes["forecast"]
                    forecast_hours = len(forecast)
                    attrs["weather_forecast"] = {
                        "status": "available",
                        "entity": weather_entity,
                        "hours": forecast_hours,
                        "minimum_required": 12,
                        "note": (
                            f"{forecast_hours}h forecast available"
                            if forecast_hours >= 12
                            else f"Only {forecast_hours}h (12h minimum recommended)"
                        ),
                    }
                else:
                    attrs["weather_forecast"] = {
                        "status": "configured_but_no_forecast",
                        "entity": weather_entity,
                        "hours": 0,
                        "minimum_required": 12,
                    }
            else:
                attrs["weather_forecast"] = {
                    "status": "not_configured",
                    "entity": None,
                    "note": "Weather prediction layer disabled",
                }

        elif key == "heat_pump_model":
            # Show detailed model specifications
            if hasattr(self.coordinator, "heat_pump_model") and self.coordinator.heat_pump_model:
                model = self.coordinator.heat_pump_model
                attrs["manufacturer"] = model.manufacturer
                attrs["model_type"] = model.model_type
                attrs["electrical_range_kw"] = (
                    f"{model.typical_electrical_range_kw[0]}-{model.typical_electrical_range_kw[1]}"
                )
                attrs["heat_output_range_kw"] = (
                    f"{model.typical_heat_output_range_kw[0]}-{model.typical_heat_output_range_kw[1]}"
                )
                attrs["cop_range"] = f"{model.cop_range[0]}-{model.cop_range[1]}"
                attrs["optimal_flow_delta"] = model.optimal_flow_delta
                if hasattr(model, "compressor_type"):
                    attrs["compressor_type"] = model.compressor_type
                if hasattr(model, "refrigerant"):
                    attrs["refrigerant"] = model.refrigerant

        return attrs

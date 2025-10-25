"""Sensor entities for EffektGuard.

Diagnostic sensors for monitoring optimization status, peak tracking,
thermal debt, and system performance.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
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
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

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
            coordinator.data["decision"].offset
            if coordinator.data and coordinator.data.get("decision")
            else 0.0
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="degree_minutes",
        name="Degree Minutes",
        icon="mdi:timer-outline",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda coordinator: (
            coordinator.data["nibe"].degree_minutes
            if coordinator.data and coordinator.data.get("nibe")
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
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda coordinator: (
            coordinator.data["nibe"].supply_temp
            if coordinator.data and coordinator.data.get("nibe")
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
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda coordinator: (
            coordinator.data["nibe"].outdoor_temp
            if coordinator.data and coordinator.data.get("nibe")
            else None
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="indoor_temperature",
        name="Indoor Temperature",
        icon="mdi:home-thermometer",
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda coordinator: (
            coordinator.data["nibe"].indoor_temp
            if coordinator.data and coordinator.data.get("nibe")
            else None
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="current_price",
        name="Current Electricity Price",
        icon="mdi:currency-eur",
        device_class=SensorDeviceClass.MONETARY,
        # Unit dynamically set from GE-Spot entity in native_unit_of_measurement property
        # Note: monetary device_class doesn't support state_class
        value_fn=lambda coordinator: (
            coordinator.data["price"].current_price
            if coordinator.data
            and coordinator.data.get("price")
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
        key="nibe_power",
        name="NIBE Power",
        icon="mdi:heat-pump",
        device_class=SensorDeviceClass.POWER,
        native_unit_of_measurement=UnitOfPower.KILO_WATT,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda coordinator: (
            coordinator.nibe.calculate_power_from_currents(
                coordinator.data["nibe"].phase1_current,
                coordinator.data["nibe"].phase2_current,
                coordinator.data["nibe"].phase3_current,
            )
            if coordinator.data
            and coordinator.data.get("nibe")
            and coordinator.data["nibe"].phase1_current is not None
            else None
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="compressor_frequency",
        name="Compressor Frequency",
        icon="mdi:engine",
        native_unit_of_measurement="Hz",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda coordinator: (
            coordinator.data["nibe"].compressor_hz
            if coordinator.data
            and coordinator.data.get("nibe")
            and coordinator.data["nibe"].compressor_hz is not None
            else None
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="compressor_health",
        name="Compressor Health Status",
        icon="mdi:engine-outline",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda coordinator: (
            coordinator.compressor_monitor.assess_risk(coordinator.data["compressor_stats"])[0]
            if coordinator.data and coordinator.data.get("compressor_stats")
            else "unknown"
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="optimization_reasoning",
        name="Optimization Reasoning",
        icon="mdi:brain",
        value_fn=lambda coordinator: (
            # Truncate to 255 chars for Home Assistant state limit
            # Full reasoning available in attributes
            coordinator.data["decision"].reasoning[:252] + "..."
            if coordinator.data
            and coordinator.data.get("decision")
            and len(coordinator.data["decision"].reasoning) > 255
            else (
                coordinator.data["decision"].reasoning
                if coordinator.data and coordinator.data.get("decision")
                else "No decision yet"
            )
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="quarter_of_day",
        name="Quarter of Day",
        icon="mdi:clock-outline",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda coordinator: (
            coordinator.data.get("current_quarter") if coordinator.data else None
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="hour_classification",
        name="Hour Classification",
        icon="mdi:chart-timeline-variant",
        value_fn=lambda coordinator: (
            coordinator.data.get("current_classification") if coordinator.data else "unknown"
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="temperature_trend",
        name="Indoor Temperature Trend",
        icon="mdi:trending-up",
        # No device_class - this is a rate of change, not a temperature
        native_unit_of_measurement="°C/h",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=2,  # Display as "0.05 °C/h" for clarity
        value_fn=lambda coordinator: (
            coordinator.data["thermal_trend"]["rate_per_hour"]
            if coordinator.data
            and coordinator.data.get("thermal_trend")
            and isinstance(coordinator.data["thermal_trend"], dict)
            and "rate_per_hour" in coordinator.data["thermal_trend"]
            else None
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="outdoor_temperature_trend",
        name="Outdoor Temperature Trend",
        icon="mdi:weather-partly-cloudy",
        # No device_class - this is a rate of change, not a temperature
        native_unit_of_measurement="°C/h",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=2,  # Display as "0.05 °C/h" for clarity
        value_fn=lambda coordinator: (
            coordinator.data["outdoor_trend"]["rate_per_hour"]
            if coordinator.data
            and coordinator.data.get("outdoor_trend")
            and isinstance(coordinator.data["outdoor_trend"], dict)
            and "rate_per_hour" in coordinator.data["outdoor_trend"]
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
            coordinator.data["savings"].monthly_estimate
            if coordinator.data
            and coordinator.data.get("savings")
            and hasattr(coordinator.data["savings"], "monthly_estimate")
            else None
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="optional_features_status",
        name="Optional Features Status",
        icon="mdi:feature-search-outline",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda coordinator: ("active" if coordinator.data else "initializing"),
    ),
    EffektGuardSensorEntityDescription(
        key="heat_pump_model",
        name="Heat Pump Model",
        icon="mdi:heat-pump",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda coordinator: (
            coordinator.heat_pump_model.model_name
            if hasattr(coordinator, "heat_pump_model") and coordinator.heat_pump_model
            else "Unknown"
        ),
    ),
    # DHW (Domestic Hot Water) sensors
    EffektGuardSensorEntityDescription(
        key="dhw_status",
        name="DHW Status",
        icon="mdi:water-boiler",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda coordinator: (
            coordinator.data.get("dhw_status", "unknown") if coordinator.data else "unknown"
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="dhw_recommendation",
        name="DHW Recommendation",
        icon="mdi:water-boiler-auto",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda coordinator: (
            coordinator.data.get("dhw_recommendation", "No recommendation")
            if coordinator.data
            else "No recommendation"
        ),
    ),
    EffektGuardSensorEntityDescription(
        key="dhw_next_boost_time",
        name="DHW Next Boost Time",
        icon="mdi:clock-outline",
        device_class=SensorDeviceClass.TIMESTAMP,
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda coordinator: (
            coordinator.data.get("dhw_next_boost") if coordinator.data else None
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


class EffektGuardSensor(CoordinatorEntity, SensorEntity, RestoreEntity):
    """EffektGuard diagnostic sensor with state restoration."""

    entity_description: EffektGuardSensorEntityDescription
    _attr_has_entity_name = True
    _restored_value: Any = None

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
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="EffektGuard",
            manufacturer="EffektGuard",
            model="Heat Pump Optimizer",
        )

    async def async_added_to_hass(self) -> None:
        """Restore last state when entity is added to hass."""
        await super().async_added_to_hass()

        # Restore last state for specific sensors that benefit from persistence
        should_restore = self.entity_description.key in {
            "peak_today",
            "peak_this_month",
            "savings_estimate",
            "compressor_health",
        }

        if should_restore:
            last_state = await self.async_get_last_state()
            if last_state and last_state.state not in ("unknown", "unavailable"):
                try:
                    self._restored_value = float(last_state.state)
                    _LOGGER.debug(
                        "Restored %s: %.2f", self.entity_description.key, self._restored_value
                    )
                except (ValueError, TypeError):
                    _LOGGER.debug(
                        "Could not restore %s, will use coordinator value",
                        self.entity_description.key,
                    )

    @property
    def native_value(self) -> Any:
        """Return the state of the sensor."""
        # For restorable sensors, use restored value if coordinator not ready
        if self._restored_value is not None and not self.coordinator.data:
            return self._restored_value

        if self.entity_description.value_fn:
            try:
                value = self.entity_description.value_fn(self.coordinator)

                # Log if reasoning was truncated
                if (
                    self.entity_description.key == "optimization_reasoning"
                    and isinstance(value, str)
                    and value.endswith("...")
                ):
                    _LOGGER.debug("Reasoning truncated to 255 chars, full reasoning in attributes")

                # Clear restored value once we have real data
                if value is not None and self._restored_value is not None:
                    self._restored_value = None

                return value
            except (AttributeError, KeyError, TypeError) as err:
                _LOGGER.warning(
                    "Error getting value for %s: %s (type: %s)",
                    self.entity_description.key,
                    err,
                    type(err).__name__,
                )
                # Fall back to restored value if available
                return self._restored_value
        return self._restored_value

    @property
    def native_unit_of_measurement(self) -> str | None:
        """Return the unit of measurement dynamically for price sensor."""
        # For current_price sensor, get unit from GE-Spot entity
        if self.entity_description.key == "current_price":
            try:
                gespot_entity_id = self.coordinator.entry.data.get("gespot_entity")
                if gespot_entity_id:
                    gespot_state = self.coordinator.hass.states.get(gespot_entity_id)
                    if gespot_state:
                        return gespot_state.attributes.get("unit_of_measurement", "öre/kWh")
            except (AttributeError, KeyError):
                pass
            # Fallback to öre/kWh if GE-Spot not available
            return "öre/kWh"

        # For all other sensors, use description's unit
        return self.entity_description.native_unit_of_measurement

    def _add_weather_forecast_to_attrs(self, attrs: dict[str, Any], hours: int = 3) -> None:
        """Add weather forecast to attributes if available.

        Helper method to avoid code duplication across multiple sensors.

        Args:
            attrs: Attribute dictionary to add forecast to
            hours: Number of forecast hours to include (default: 3)
        """
        weather_data = self.coordinator.data.get("weather")
        if weather_data and hasattr(weather_data, "forecast_hours") and weather_data.forecast_hours:
            attrs["forecast"] = [
                {"time": f.datetime, "temp": f.temperature}
                for f in weather_data.forecast_hours[:hours]
            ]

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

        elif key == "dhw_recommendation":
            # Add human-readable planning summary
            if self.coordinator.data and "dhw_planning_summary" in self.coordinator.data:
                attrs["planning_summary"] = self.coordinator.data["dhw_planning_summary"]

            # Add DHW planning attributes (machine-readable details)
            if self.coordinator.data and "dhw_planning" in self.coordinator.data:
                planning = self.coordinator.data.get("dhw_planning", {})

                # Core decision info
                if "should_heat" in planning:
                    attrs["should_heat"] = planning["should_heat"]
                if "priority_reason" in planning:
                    attrs["priority_reason"] = planning["priority_reason"]

                # Temperature info
                if "current_temperature" in planning:
                    attrs["current_temperature"] = planning["current_temperature"]
                if "target_temperature" in planning:
                    attrs["target_temperature"] = planning["target_temperature"]

                # Thermal debt info
                if "thermal_debt" in planning:
                    attrs["thermal_debt"] = planning["thermal_debt"]
                if "thermal_debt_threshold_block" in planning:
                    attrs["thermal_debt_threshold_block"] = planning["thermal_debt_threshold_block"]
                if "thermal_debt_threshold_abort" in planning:
                    attrs["thermal_debt_threshold_abort"] = planning["thermal_debt_threshold_abort"]
                if "thermal_debt_status" in planning:
                    attrs["thermal_debt_status"] = planning["thermal_debt_status"]

                # Heating demand and conditions
                if "space_heating_demand_kw" in planning:
                    attrs["space_heating_demand_kw"] = planning["space_heating_demand_kw"]
                if "current_price_classification" in planning:
                    attrs["current_price_classification"] = planning["current_price_classification"]
                if "outdoor_temperature" in planning:
                    attrs["outdoor_temperature"] = planning["outdoor_temperature"]
                if "indoor_temperature" in planning:
                    attrs["indoor_temperature"] = planning["indoor_temperature"]
                if "climate_zone" in planning:
                    attrs["climate_zone"] = planning["climate_zone"]

                # Weather opportunity
                if "weather_opportunity" in planning:
                    attrs["weather_opportunity"] = planning["weather_opportunity"]

        elif key == "temperature_trend":
            # Show prediction and trend details for INDOOR temperature
            if "thermal_trend" in self.coordinator.data:
                trend_data = self.coordinator.data["thermal_trend"]
                if trend_data and isinstance(trend_data, dict):
                    # Add trend information
                    if "trend" in trend_data:
                        attrs["trend_direction"] = trend_data["trend"]
                    if "confidence" in trend_data:
                        attrs["confidence"] = trend_data["confidence"]
                    if "samples" in trend_data:
                        attrs["samples"] = trend_data["samples"]

            # Add indoor temperature prediction from thermal model if available
            if "thermal" in self.coordinator.data:
                thermal_data = self.coordinator.data["thermal"]
                if thermal_data and hasattr(thermal_data, "prediction_3h"):
                    attrs["prediction_3h"] = thermal_data.prediction_3h

            # NOTE: Weather forecast (outdoor temps) removed - only relevant for outdoor_temperature_trend
            # Indoor temperature trend should only show indoor predictions from thermal model

        elif key == "outdoor_temperature_trend":
            # Show outdoor trend details
            if "outdoor_trend" in self.coordinator.data:
                trend_data = self.coordinator.data["outdoor_trend"]
                if trend_data and isinstance(trend_data, dict):
                    # Add trend information
                    if "trend" in trend_data:
                        attrs["trend_direction"] = trend_data["trend"]
                    if "confidence" in trend_data:
                        attrs["confidence"] = trend_data["confidence"]
                    if "samples" in trend_data:
                        attrs["samples"] = trend_data["samples"]
                    if "temp_change_2h" in trend_data:
                        attrs["temp_change_2h"] = trend_data["temp_change_2h"]

            # Add current outdoor temp for reference
            if "nibe" in self.coordinator.data and self.coordinator.data["nibe"]:
                attrs["current_outdoor_temp"] = self.coordinator.data["nibe"].outdoor_temp

            # Add weather forecast (outdoor temperature predictions)
            # Only for outdoor_temperature_trend sensor - not for other sensors
            self._add_weather_forecast_to_attrs(attrs)

        elif key == "compressor_health":
            # Show detailed compressor statistics and health data
            if "compressor_stats" in self.coordinator.data:
                stats = self.coordinator.data["compressor_stats"]
                if stats:
                    # Current state
                    attrs["current_hz"] = stats.current_hz
                    attrs["avg_1h"] = round(stats.avg_1h, 1)
                    attrs["avg_6h"] = round(stats.avg_6h, 1)
                    attrs["avg_24h"] = round(stats.avg_24h, 1)

                    # Maximums
                    attrs["max_hz_1h"] = stats.max_hz_1h
                    attrs["max_hz_24h"] = stats.max_hz_24h

                    # Stress tracking
                    attrs["time_above_80hz_minutes"] = round(
                        stats.time_above_80hz.total_seconds() / 60, 1
                    )
                    attrs["time_above_100hz_minutes"] = round(
                        stats.time_above_100hz.total_seconds() / 60, 1
                    )

                    # Risk assessment
                    risk_level, risk_reason = self.coordinator.compressor_monitor.assess_risk(stats)
                    attrs["risk_level"] = risk_level
                    attrs["risk_reason"] = risk_reason

                    # Data quality
                    attrs["samples_count"] = stats.samples_count

                    # Operating ranges for reference (NIBE F750/F2040)
                    attrs["hz_range_idle"] = "20-30"
                    attrs["hz_range_normal"] = "40-70"
                    attrs["hz_range_high"] = "70-90"
                    attrs["hz_range_very_high"] = "90-110"
                    attrs["hz_range_emergency"] = "110-120"

            # Add current outdoor temperature from NIBE
            if "nibe" in self.coordinator.data:
                nibe_data = self.coordinator.data["nibe"]
                if nibe_data and hasattr(nibe_data, "outdoor_temp"):
                    attrs["current_outdoor_temp"] = nibe_data.outdoor_temp

            # Add weather forecast to compressor health for context
            # Helps understand if high Hz operation is expected due to incoming cold weather
            self._add_weather_forecast_to_attrs(attrs)

        elif key == "indoor_temperature":
            # Show all temperature sources and calculation method
            if "nibe" in self.coordinator.data:
                nibe_data = self.coordinator.data["nibe"]
                if nibe_data and hasattr(nibe_data, "indoor_temp"):
                    attrs["nibe_bt50"] = nibe_data.indoor_temp

                    # Show if multi-sensor averaging is being used
                    try:
                        config = self.coordinator.config_entry.data
                        additional_sensors = config.get("additional_indoor_sensors", [])

                        if additional_sensors:
                            # User has configured additional sensors
                            attrs["calculation_method"] = config.get("indoor_temp_method", "median")
                            attrs["sensor_count"] = len(additional_sensors) + 1  # +1 for NIBE

                            # Show each additional sensor value
                            sensor_temps = {}
                            for i, entity_id in enumerate(additional_sensors, 1):
                                state = self.coordinator.hass.states.get(entity_id)
                                if state and state.state not in ["unknown", "unavailable"]:
                                    try:
                                        temp = float(state.state)
                                        sensor_temps[f"sensor_{i}"] = {
                                            "entity_id": entity_id,
                                            "temperature": temp,
                                            "name": state.attributes.get(
                                                "friendly_name", entity_id
                                            ),
                                        }
                                    except (ValueError, TypeError):
                                        sensor_temps[f"sensor_{i}"] = {
                                            "entity_id": entity_id,
                                            "temperature": None,
                                            "status": "invalid",
                                        }

                            attrs["additional_sensors"] = sensor_temps
                            attrs["all_sensors"] = {
                                "nibe_bt50": nibe_data.indoor_temp,
                                **{
                                    f"sensor_{i+1}": sensor_temps[f"sensor_{i+1}"]["temperature"]
                                    for i in range(len(sensor_temps))
                                    if f"sensor_{i+1}" in sensor_temps
                                },
                            }
                        else:
                            # Only NIBE sensor
                            attrs["calculation_method"] = "single_sensor"
                            attrs["sensor_count"] = 1
                            attrs["note"] = (
                                "Using NIBE BT50 sensor only. Add additional room sensors in configuration for better whole-house temperature."
                            )
                    except AttributeError:
                        # Mock/test environment - just show basic info
                        attrs["calculation_method"] = "single_sensor"
                        attrs["sensor_count"] = 1

            # Add indoor temperature trend data
            if "thermal_trend" in self.coordinator.data:
                trend_data = self.coordinator.data["thermal_trend"]
                if trend_data and isinstance(trend_data, dict):
                    # Add trend information
                    if "trend" in trend_data:
                        attrs["trend_direction"] = trend_data["trend"]
                    if "rate_per_hour" in trend_data:
                        attrs["rate_per_hour"] = trend_data["rate_per_hour"]
                    if "confidence" in trend_data:
                        attrs["confidence"] = trend_data["confidence"]
                    if "samples" in trend_data:
                        attrs["samples"] = trend_data["samples"]

        elif key == "outdoor_temperature":
            # Show outdoor temperature details and trend
            if "nibe" in self.coordinator.data:
                nibe_data = self.coordinator.data["nibe"]
                if nibe_data and hasattr(nibe_data, "outdoor_temp"):
                    attrs["nibe_bt1"] = nibe_data.outdoor_temp

            # Add outdoor temperature trend data
            if "outdoor_trend" in self.coordinator.data:
                trend_data = self.coordinator.data["outdoor_trend"]
                if trend_data and isinstance(trend_data, dict):
                    # Add trend information
                    if "trend" in trend_data:
                        attrs["trend_direction"] = trend_data["trend"]
                    if "rate_per_hour" in trend_data:
                        attrs["rate_per_hour"] = trend_data["rate_per_hour"]
                    if "confidence" in trend_data:
                        attrs["confidence"] = trend_data["confidence"]
                    if "samples" in trend_data:
                        attrs["samples"] = trend_data["samples"]
                    if "temp_change_2h" in trend_data:
                        attrs["temp_change_2h"] = trend_data["temp_change_2h"]

            # Add weather forecast to outdoor temperature sensor
            self._add_weather_forecast_to_attrs(attrs)

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

        elif key == "peak_today":
            # Peak tracking metadata - when, how, and context
            # Most common user questions: When did it happen? Is it real data? Does it affect my bill?

            # When did today's peak occur?
            if self.coordinator.peak_today_time:
                attrs["peak_time"] = self.coordinator.peak_today_time.isoformat()
                attrs["peak_hour"] = self.coordinator.peak_today_time.hour

                # Calculate time since peak for context
                now = dt_util.now()
                time_since = now - self.coordinator.peak_today_time
                hours = int(time_since.total_seconds() // 3600)
                minutes = int((time_since.total_seconds() % 3600) // 60)
                if hours > 0:
                    attrs["time_since_peak"] = f"{hours}h {minutes}m ago"
                else:
                    attrs["time_since_peak"] = f"{minutes}m ago"
            else:
                attrs["peak_time"] = None
                attrs["time_since_peak"] = "No peak recorded today"

            # Which 15-minute quarter (Swedish effect tariff)
            if self.coordinator.peak_today_quarter is not None:
                attrs["peak_quarter"] = self.coordinator.peak_today_quarter
                # Convert quarter to human-readable time
                hour = self.coordinator.peak_today_quarter // 4
                minute = (self.coordinator.peak_today_quarter % 4) * 15
                attrs["peak_quarter_time"] = f"{hour:02d}:{minute:02d}"
            else:
                attrs["peak_quarter"] = None
                attrs["peak_quarter_time"] = None

            # How was it measured? (Trust/accuracy)
            attrs["measurement_source"] = self.coordinator.peak_today_source

            # Human-readable source description
            source_descriptions = {
                "external_meter": "Whole-house power meter (best accuracy)",
                "nibe_currents": "NIBE phase currents (NIBE only)",
                "estimate": "Estimated from compressor (display only)",
                "unknown": "No measurement available yet",
            }
            attrs["measurement_description"] = source_descriptions.get(
                self.coordinator.peak_today_source, "Unknown source"
            )

            # Is this real measurement or estimate?
            attrs["is_real_measurement"] = self.coordinator.peak_today_source in [
                "external_meter",
                "nibe_currents",
            ]

            # Will this affect monthly billing?
            # Only real measurements from external meter affect effect tariff billing
            # NIBE currents measure only heat pump (missing other house loads)
            will_affect = (
                self.coordinator.peak_today_source == "external_meter"
                and self.coordinator.peak_today > self.coordinator.peak_this_month
            )
            attrs["will_affect_billing"] = will_affect

            if will_affect:
                attrs["billing_impact"] = (
                    f"New monthly peak: {self.coordinator.peak_today:.2f} kW "
                    f"(previous: {self.coordinator.peak_this_month:.2f} kW)"
                )
            elif self.coordinator.peak_today_source != "external_meter":
                attrs["billing_impact"] = "Not used for billing (no external meter configured)"
            else:
                attrs["billing_impact"] = (
                    f"Below monthly peak of {self.coordinator.peak_this_month:.2f} kW"
                )

            # Yesterday's peak for comparison (trends/learning)
            attrs["yesterday_peak"] = self.coordinator.yesterday_peak

            if self.coordinator.yesterday_peak > 0:
                # Calculate percentage change from yesterday
                change = self.coordinator.peak_today - self.coordinator.yesterday_peak
                change_pct = (change / self.coordinator.yesterday_peak) * 100
                attrs["change_vs_yesterday"] = change
                attrs["change_vs_yesterday_percent"] = round(change_pct, 1)

                if change > 0:
                    attrs["trend"] = f"↑ {abs(change):.2f} kW higher than yesterday"
                elif change < 0:
                    attrs["trend"] = f"↓ {abs(change):.2f} kW lower than yesterday"
                else:
                    attrs["trend"] = "Same as yesterday"
            else:
                attrs["change_vs_yesterday"] = None
                attrs["change_vs_yesterday_percent"] = None
                attrs["trend"] = "No yesterday data yet"

            # Monthly peak for context
            attrs["monthly_peak"] = self.coordinator.peak_this_month

        elif key == "peak_this_month":
            # Monthly peak tracking status and configuration help
            attrs["daily_peak"] = self.coordinator.peak_today

            # Check if user has real power measurement configured
            has_external_power = hasattr(self.coordinator.nibe, "_power_sensor_entity") and bool(
                self.coordinator.nibe._power_sensor_entity
            )
            has_phase_currents = False
            if "nibe" in self.coordinator.data and self.coordinator.data["nibe"]:
                nibe_data = self.coordinator.data["nibe"]
                has_phase_currents = getattr(nibe_data, "phase1_current", None) is not None

            attrs["has_power_meter"] = has_external_power
            attrs["has_phase_currents"] = has_phase_currents
            attrs["has_real_measurement"] = has_external_power or has_phase_currents

            # Provide helpful guidance if no real measurements
            if not has_external_power and not has_phase_currents:
                attrs["status"] = "No power measurement configured"
                attrs["help"] = (
                    "Monthly peak tracking requires either: "
                    "(1) External power meter sensor, or "
                    "(2) NIBE phase current sensors (BE1/BE2/BE3). "
                    "Configure a power sensor in EffektGuard settings to enable peak tracking."
                )
                attrs["billing_accuracy"] = "Not available - no measurement source"
            elif has_external_power:
                attrs["status"] = "Tracking with external power meter"
                attrs["measurement_source"] = "external_meter"
                attrs["billing_accuracy"] = "Whole-house measurement (best for billing)"
            elif has_phase_currents:
                attrs["status"] = "Tracking with NIBE phase currents"
                attrs["measurement_source"] = "nibe_currents"
                attrs["billing_accuracy"] = "NIBE heat pump only (other house loads not included)"

            # Monthly reset info
            now = dt_util.now()
            attrs["current_month"] = now.strftime("%B %Y")
            attrs["days_into_month"] = now.day
            days_in_month = (now.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(
                days=1
            )
            attrs["days_remaining"] = days_in_month.day - now.day

        elif key == "optimization_reasoning":
            # Add full reasoning in attributes (not limited to 255 chars)
            if "decision" in self.coordinator.data:
                decision = self.coordinator.data["decision"]
                if decision:
                    # Full reasoning in attribute (bypasses 255 char state limit)
                    if hasattr(decision, "reasoning"):
                        attrs["full_reasoning"] = decision.reasoning
                    if hasattr(decision, "timestamp"):
                        attrs["decision_timestamp"] = decision.timestamp.isoformat()
                    if hasattr(decision, "offset"):
                        attrs["applied_offset"] = decision.offset
                    # Add layer breakdown for detailed analysis
                    if hasattr(decision, "layers") and decision.layers:
                        attrs["layers"] = [
                            {
                                "reason": layer.reason,
                                "offset": layer.offset,
                                "weight": layer.weight,
                            }
                            for layer in decision.layers
                        ]

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
            # Check actual weather data from coordinator (not just entity attributes)
            # This works for both attribute-based (Met.no) and service-based (OpenWeatherMap) forecasts
            weather_entity = config.get("weather_entity") or self.coordinator.entry.data.get(
                "weather_entity"
            )
            if weather_entity:
                # Check if coordinator has actual weather data
                if "weather" in self.coordinator.data and self.coordinator.data["weather"]:
                    weather_data = self.coordinator.data["weather"]
                    if hasattr(weather_data, "forecast_hours") and weather_data.forecast_hours:
                        forecast_hours = len(weather_data.forecast_hours)
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
                            "note": "Forecast data not available from weather integration",
                        }
                else:
                    attrs["weather_forecast"] = {
                        "status": "configured_but_no_forecast",
                        "entity": weather_entity,
                        "hours": 0,
                        "minimum_required": 12,
                        "note": "Forecast data not available from weather integration",
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
                    f"{model.rated_power_kw[0]}-{model.rated_power_kw[1]}"
                )
                attrs["cop_range"] = f"{model.typical_cop_range[0]}-{model.typical_cop_range[1]}"
                attrs["optimal_flow_delta"] = model.optimal_flow_delta
                if hasattr(model, "compressor_type"):
                    attrs["compressor_type"] = model.compressor_type
                if hasattr(model, "refrigerant"):
                    attrs["refrigerant"] = model.refrigerant

        elif key == "dhw_status":
            # Add DHW-specific attributes
            # Current temperature from NIBE BT7 sensor
            if "nibe" in self.coordinator.data:
                nibe_data = self.coordinator.data["nibe"]
                if nibe_data and hasattr(nibe_data, "dhw_top_temp"):
                    attrs["current_temperature"] = nibe_data.dhw_top_temp
                    attrs["temperature_unit"] = "°C"

            # Last Legionella boost (hygiene cycle at 56°C)
            if self.coordinator.dhw_optimizer:
                last_boost = self.coordinator.dhw_optimizer.last_legionella_boost
                attrs["last_legionella_boost"] = last_boost.isoformat() if last_boost else None

            # Last heating cycle times (start and end)
            if self.coordinator.data and "dhw_heating_start" in self.coordinator.data:
                heating_start = self.coordinator.data.get("dhw_heating_start")
                if heating_start:
                    attrs["heating_start"] = (
                        heating_start.strftime("%H:%M:%S")
                        if hasattr(heating_start, "strftime")
                        else str(heating_start)
                    )

            if self.coordinator.data and "dhw_heating_end" in self.coordinator.data:
                heating_end = self.coordinator.data.get("dhw_heating_end")
                if heating_end:
                    attrs["heating_end"] = (
                        heating_end.strftime("%H:%M:%S")
                        if hasattr(heating_end, "strftime")
                        else str(heating_end)
                    )

                    # Calculate duration if we have both start and end
                    heating_start = self.coordinator.data.get("dhw_heating_start")
                    if heating_start and heating_end:
                        duration_minutes = (heating_end - heating_start).total_seconds() / 60
                        attrs["last_cycle_duration"] = f"{duration_minutes:.1f} min"

            # Last heated time (legacy - when any heating was detected)
            if self.coordinator.data and "dhw_last_heated" in self.coordinator.data:
                last_heated = self.coordinator.data.get("dhw_last_heated")
                if last_heated:
                    attrs["last_heated"] = (
                        last_heated.isoformat()
                        if hasattr(last_heated, "isoformat")
                        else str(last_heated)
                    )

        return attrs

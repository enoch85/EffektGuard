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
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import EntityCategory
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
        key="peak_status",
        name="Peak Status",
        icon="mdi:alert-circle-outline",
        value_fn=lambda coordinator: (
            coordinator.data["decision"].peak_status
            if coordinator.data
            and coordinator.data.get("decision")
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
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda coordinator: (
            coordinator.data["thermal"].temperature_trend
            if coordinator.data
            and coordinator.data.get("thermal")
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
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="EffektGuard",
            manufacturer="EffektGuard",
            model="Heat Pump Optimizer",
        )

    @property
    def native_value(self) -> Any:
        """Return the state of the sensor."""
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

                return value
            except (AttributeError, KeyError, TypeError) as err:
                _LOGGER.debug("Error getting value for %s: %s", self.entity_description.key, err)
                return None
        return None

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
                
                # Optimal heating windows (next 3 windows)
                if "optimal_heating_windows" in planning and planning["optimal_heating_windows"]:
                    windows = planning["optimal_heating_windows"]
                    attrs["optimal_windows_count"] = len(windows)
                    
                    # Format windows for display
                    for i, window in enumerate(windows[:3], 1):
                        prefix = f"window_{i}"
                        attrs[f"{prefix}_time"] = window.get("time_range", "Unknown")
                        attrs[f"{prefix}_price"] = window.get("price_classification", "Unknown")
                        attrs[f"{prefix}_duration_hours"] = window.get("duration_hours", 0)
                        attrs[f"{prefix}_thermal_debt_ok"] = window.get("thermal_debt_ok", False)
                    
                    # Next optimal window (most important)
                    if "next_optimal_window" in planning and planning["next_optimal_window"]:
                        next_window = planning["next_optimal_window"]
                        attrs["next_window_time"] = next_window.get("time_range", "Unknown")
                        attrs["next_window_price"] = next_window.get("price_classification", "Unknown")
                        attrs["next_window_duration"] = f"{next_window.get('duration_hours', 0):.1f}h"

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
            # Check both options and data for weather entity
            weather_entity = config.get("weather_entity") or self.coordinator.entry.data.get(
                "weather_entity"
            )
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

            # Next boost time (when heating is planned)
            if self.coordinator.data and "dhw_next_boost" in self.coordinator.data:
                next_boost = self.coordinator.data.get("dhw_next_boost")
                if next_boost:
                    attrs["next_boost_time"] = (
                        next_boost.isoformat()
                        if hasattr(next_boost, "isoformat")
                        else str(next_boost)
                    )

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

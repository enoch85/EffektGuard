"""NIBE Myuplink adapter for reading heat pump state.

This adapter reads NIBE heat pump data from existing Home Assistant entities
created by the NIBE Myuplink integration. It does not directly communicate
with the MyUplink API.

Data read includes:
- Indoor temperature (BT50 or room sensor)
- Outdoor temperature (BT1)
- Supply temperature (BT25)
- Return temperature (BT3)
- Degree minutes (GM/DM)
- Current heating curve offset
- Compressor status
- Hot water status
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from ..const import (
    CONF_ADDITIONAL_INDOOR_SENSORS,
    CONF_DEGREE_MINUTES_ENTITY,
    CONF_INDOOR_TEMP_METHOD,
    CONF_NIBE_ENTITY,
    CONF_POWER_SENSOR_ENTITY,
    DEBUG_FORCE_OUTDOOR_TEMP,
    DEFAULT_BASE_POWER,
    DEFAULT_INDOOR_TEMP,
    DEFAULT_INDOOR_TEMP_METHOD,
    TEMP_FACTOR_MAX,
    TEMP_FACTOR_MIN,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class NibeState:
    """Current state of NIBE heat pump.

    All temperatures in °C, power in kW.
    """

    outdoor_temp: float
    indoor_temp: float
    supply_temp: float  # BT25 - Flow/supply temperature
    return_temp: float | None
    degree_minutes: float
    current_offset: float
    is_heating: bool
    is_hot_water: bool
    timestamp: datetime
    dhw_top_temp: float | None = None  # BT7 - Hot water top (40013) - optional
    dhw_charging_temp: float | None = None  # BT6 - Hot water charging/bottom (40014) - optional

    @property
    def flow_temp(self) -> float:
        """Alias for supply_temp (flow temperature = supply temperature)."""
        return self.supply_temp
    
    @property
    def dhw_temp(self) -> float | None:
        """Primary DHW temperature for heating decisions (BT6 charging sensor).
        
        Returns BT6 (charging/bottom) if available, falls back to BT7 (top) if not.
        
        BT6 is preferred because:
        - Drops faster when hot water is used (bottom of tank cools first)
        - Provides early warning that heating is needed
        - BT7 stays hot until tank is nearly empty (too late for proactive heating)
        
        BT7 is useful for Legionella detection (monitors peak temperatures),
        but BT6 is better for triggering DHW heating cycles.
        """
        return self.dhw_charging_temp if self.dhw_charging_temp is not None else self.dhw_top_temp


class NibeAdapter:
    """Adapter for reading NIBE Myuplink entities."""

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]):
        """Initialize NIBE adapter.

        Args:
            hass: Home Assistant instance
            config: Configuration dictionary with entity IDs
        """
        self.hass = hass
        self._nibe_base_entity = config.get(CONF_NIBE_ENTITY)
        self._degree_minutes_entity = config.get(CONF_DEGREE_MINUTES_ENTITY)  # Optional
        self._power_sensor_entity = config.get(CONF_POWER_SENSOR_ENTITY)  # Optional
        self._additional_indoor_sensors = config.get(CONF_ADDITIONAL_INDOOR_SENSORS, [])  # Optional
        self._indoor_temp_method = config.get(CONF_INDOOR_TEMP_METHOD, DEFAULT_INDOOR_TEMP_METHOD)
        self._last_write: datetime | None = None
        self._entity_cache: dict[str, str] = {}

    async def get_current_state(self) -> NibeState:
        """Read current NIBE heat pump state from entities.

        Returns:
            NibeState with current readings

        Raises:
            ValueError: If required entities are unavailable
        """
        # This will be implemented in Phase 1
        # For now, discover and read NIBE entities

        # Discover NIBE entities if not cached
        if not self._entity_cache:
            await self._discover_nibe_entities()

        # Read temperature sensors
        outdoor_temp = await self._read_entity_float(
            self._entity_cache.get("outdoor_temp"), default=0.0
        )

        # DEBUG: Override outdoor temperature if configured
        if DEBUG_FORCE_OUTDOOR_TEMP is not None:
            _LOGGER.warning(
                "🔧 DEBUG MODE: Forcing outdoor temp %.1f°C → %.1f°C (DEBUG_FORCE_OUTDOOR_TEMP)",
                outdoor_temp,
                DEBUG_FORCE_OUTDOOR_TEMP,
            )
            outdoor_temp = DEBUG_FORCE_OUTDOOR_TEMP

        indoor_temp = await self._read_entity_float(
            self._entity_cache.get("indoor_temp"), default=DEFAULT_INDOOR_TEMP
        )

        # Multi-sensor indoor temperature calculation
        if self._additional_indoor_sensors:
            indoor_temp = await self._calculate_multi_sensor_temperature(indoor_temp)

        supply_temp = await self._read_entity_float(
            self._entity_cache.get("supply_temp"), default=35.0
        )
        return_temp = await self._read_entity_float(
            self._entity_cache.get("return_temp"), default=None
        )

        # Read degree minutes (GM/DM)
        # First try optional configured sensor, then fall back to auto-discovery
        degree_minutes = None
        if self._degree_minutes_entity:
            degree_minutes = await self._read_entity_float(
                self._degree_minutes_entity, default=None
            )
            if degree_minutes is not None:
                _LOGGER.debug(
                    "Using configured degree minutes sensor: %s = %.1f",
                    self._degree_minutes_entity,
                    degree_minutes,
                )

        # Fall back to auto-discovered sensor
        if degree_minutes is None:
            degree_minutes = await self._read_entity_float(
                self._entity_cache.get("degree_minutes"), default=None
            )
            if degree_minutes is not None:
                _LOGGER.debug("Using auto-discovered degree minutes sensor: %.1f", degree_minutes)

        # If still None, estimate from thermal model (will be implemented in thermal_model.py)
        if degree_minutes is None:
            degree_minutes = self._estimate_degree_minutes(indoor_temp, supply_temp, outdoor_temp)
            _LOGGER.debug("Estimating degree minutes from thermal model: %.1f", degree_minutes)

        # Read current offset
        current_offset = await self._read_entity_float(
            self._entity_cache.get("offset"), default=0.0
        )

        # Read compressor status
        is_heating = await self._read_entity_bool(
            self._entity_cache.get("compressor_status"), default=False
        )

        # Read hot water status
        is_hot_water = await self._read_entity_bool(
            self._entity_cache.get("hot_water_status"), default=False
        )

        # Read DHW temperatures (optional - BT7 top, BT6 charging/bottom)
        dhw_top_temp = await self._read_entity_float(
            self._entity_cache.get("dhw_top_temp"), default=None
        )
        dhw_charging_temp = await self._read_entity_float(
            self._entity_cache.get("dhw_charging_temp"), default=None
        )

        # Log DHW sensor status
        if dhw_top_temp is not None:
            _LOGGER.debug("DHW top temperature (BT7): %.1f°C", dhw_top_temp)
        if dhw_charging_temp is not None:
            _LOGGER.debug("DHW charging temperature (BT6): %.1f°C", dhw_charging_temp)

        return NibeState(
            outdoor_temp=outdoor_temp,
            indoor_temp=indoor_temp,
            supply_temp=supply_temp,
            return_temp=return_temp,
            degree_minutes=degree_minutes,
            current_offset=current_offset,
            is_heating=is_heating,
            is_hot_water=is_hot_water,
            timestamp=dt_util.utcnow(),
            dhw_top_temp=dhw_top_temp,
            dhw_charging_temp=dhw_charging_temp,
        )

    async def set_curve_offset(self, offset: float) -> None:
        """Set heating curve offset via NIBE entity.

        Args:
            offset: Offset value in °C (-10 to +10)

        Note:
            Requires NIBE Myuplink Premium subscription for write access.
            Implements rate limiting to avoid excessive API calls.
        """
        from datetime import timedelta

        # Rate limiting - minimum 5 minutes between writes
        now = dt_util.utcnow()
        if self._last_write and now - self._last_write < timedelta(minutes=5):
            _LOGGER.debug("Skipping offset write, too soon since last write")
            return

        # Get offset entity
        offset_entity = self._entity_cache.get("offset")
        if not offset_entity:
            _LOGGER.error("No offset entity found")
            return

        # CHECK: Verify entity is available before service call
        # Prevents "Action number.set_value not found" errors during startup
        # when MyUplink entities are still initializing
        state = self.hass.states.get(offset_entity)
        if not state or state.state in ["unavailable", "unknown"]:
            _LOGGER.warning(
                "Offset entity %s not ready (state: %s), skipping write",
                offset_entity,
                state.state if state else "None",
            )
            return

        # Set value via number entity service (non-blocking)
        try:
            await self.hass.services.async_call(
                "number",
                "set_value",
                {
                    "entity_id": offset_entity,
                    "value": offset,
                },
                blocking=False,  # Non-blocking to avoid UI delays
            )
            self._last_write = now
            _LOGGER.info("Set NIBE offset to %.2f°C", offset)
        except Exception as err:
            _LOGGER.error("Failed to set NIBE offset: %s", err)
            # Don't raise - allow system to continue gracefully
            # Error logged for debugging, but not fatal

    async def _discover_nibe_entities(self) -> None:
        """Discover NIBE entities from entity registry.

        Populates _entity_cache with entity IDs for:
        - outdoor_temp (BT1)
        - indoor_temp (BT50 or room sensor)
        - supply_temp (BT25)
        - return_temp (BT3)
        - degree_minutes (GM/DM)
        - offset (S1 offset)
        - compressor_status
        - hot_water_status
        - dhw_top_temp (BT7 - hot water top)
        - dhw_charging_temp (BT6 - hot water charging/bottom)
        """
        registry = er.async_get(self.hass)

        # Search for NIBE entities
        # Patterns based on NIBE Myuplink integration entity naming
        # Use specific entity names from parameter IDs when possible
        patterns = {
            "outdoor_temp": ["bt1", "_outdoor", "40004"],  # BT1 / param 40004
            "indoor_temp": ["40033", "bt50"],  # param 40033 "Temperature" (BT50) / BT50 direct
            "supply_temp": ["bt25", "_supply", "40008"],  # BT25/BT63 / param 40008
            "return_temp": ["bt3", "_return", "40012"],  # BT3 / param 40012
            "degree_minutes": ["degree_minutes", "40941"],  # param 40941
            "offset": ["offset", "47011"],  # param 47011
            "compressor_status": ["compressor", "43427"],  # param 43427
            "hot_water_status": ["hot_water", "dhw"],
            "dhw_top_temp": ["bt7", "hw_top", "40013"],  # BT7 / param 40013 - hot water top
            "dhw_charging_temp": ["bt6", "hw_bottom", "hw_charging", "40014"],  # BT6 / param 40014
        }

        # Find entities
        for entity in registry.entities.values():
            if not entity.entity_id.startswith("sensor.") and not entity.entity_id.startswith(
                "number."
            ):
                continue

            entity_id_lower = entity.entity_id.lower()

            # Skip known non-temperature configuration parameters
            # 47394 = "control room sensor syst" (configuration, not temperature)
            if "47394" in entity_id_lower or "control_room_sensor" in entity_id_lower:
                continue

            # Match against patterns
            for key, patterns_list in patterns.items():
                if key not in self._entity_cache:
                    for pattern in patterns_list:
                        if pattern in entity_id_lower:
                            # Additional validation for temperature sensors
                            if key in [
                                "outdoor_temp",
                                "indoor_temp",
                                "supply_temp",
                                "return_temp",
                                "dhw_top_temp",
                                "dhw_charging_temp",
                            ]:
                                # Check if it's actually a temperature sensor
                                state = self.hass.states.get(entity.entity_id)
                                if state and state.attributes.get("device_class") != "temperature":
                                    _LOGGER.debug(
                                        "Skipping %s (not a temperature sensor): %s",
                                        key,
                                        entity.entity_id,
                                    )
                                    continue

                            self._entity_cache[key] = entity.entity_id
                            _LOGGER.debug("Found NIBE entity %s: %s", key, entity.entity_id)
                            break

        # Log all discovered entities for debugging
        _LOGGER.info("Discovered %d NIBE entities:", len(self._entity_cache))
        for key, entity_id in self._entity_cache.items():
            state = self.hass.states.get(entity_id)
            state_value = state.state if state else "unavailable"
            _LOGGER.debug("  - %s: %s = %s", key, entity_id, state_value)

    async def _read_entity_float(
        self,
        entity_id: str | None,
        default: float | None = None,
    ) -> float | None:
        """Read float value from entity.

        Args:
            entity_id: Entity ID to read
            default: Default value if entity unavailable

        Returns:
            Float value or default
        """
        if not entity_id:
            return default

        state = self.hass.states.get(entity_id)
        if not state or state.state in ["unknown", "unavailable"]:
            return default

        try:
            return float(state.state)
        except (ValueError, TypeError):
            _LOGGER.warning("Cannot parse float from %s: %s", entity_id, state.state)
            return default

    async def _read_entity_bool(
        self,
        entity_id: str | None,
        default: bool = False,
    ) -> bool:
        """Read boolean value from entity.

        Args:
            entity_id: Entity ID to read
            default: Default value if entity unavailable

        Returns:
            Boolean value or default
        """
        if not entity_id:
            return default

        state = self.hass.states.get(entity_id)
        if not state or state.state in ["unknown", "unavailable"]:
            return default

        return state.state in ["on", "true", "True", "ON", "1"]

    def _estimate_degree_minutes(
        self, indoor_temp: float, supply_temp: float, outdoor_temp: float
    ) -> float:
        """Estimate degree minutes from temperatures when sensor unavailable.

        Uses simplified thermal balance model:
        DM ≈ (actual_flow - target_flow) × time_factor

        This is a rough estimation. Real DM tracking from NIBE is much more accurate.

        Args:
            indoor_temp: Current indoor temperature (°C)
            supply_temp: Current supply/flow temperature (°C)
            outdoor_temp: Current outdoor temperature (°C)

        Returns:
            Estimated degree minutes (typically -500 to +500)

        Note:
            Negative DM = compressor needs to run (heat deficit)
            Positive DM = recent heating surplus
        """
        # Calculate target flow temp using simplified heating curve
        # Typical NIBE curve: Flow ≈ 20 + 1.5 × (20 - Outdoor)
        target_flow = 20.0 + 1.5 * (20.0 - outdoor_temp)

        # Calculate thermal imbalance
        flow_error = supply_temp - target_flow

        # Estimate DM based on flow error and indoor temp error
        target_indoor = DEFAULT_INDOOR_TEMP  # Assumed target when not configured
        indoor_error = indoor_temp - target_indoor

        # Simplified estimation
        # If too cold inside and flow too low → negative DM (needs heating)
        # If warm enough and flow adequate → near zero DM
        estimated_dm = flow_error * 10.0 + indoor_error * 50.0

        # Clamp to reasonable range
        estimated_dm = max(-800.0, min(estimated_dm, 500.0))

        return estimated_dm

    async def get_power_consumption(self) -> float | None:
        """Get current power consumption of heat pump.

        Tries in order:
        1. Configured power sensor (most accurate)
        2. Auto-discovered heat pump power sensor
        3. Estimation from supply temperature (least accurate)

        Returns:
            Power consumption in kW, or None if unavailable
        """
        # Try configured power sensor
        if self._power_sensor_entity:
            power = await self._read_entity_float(self._power_sensor_entity, default=None)
            if power is not None:
                # Convert W to kW if needed
                state = self.hass.states.get(self._power_sensor_entity)
                if state:
                    unit = state.attributes.get("unit_of_measurement", "").lower()
                    if unit == "w":
                        power = power / 1000.0
                _LOGGER.debug(
                    "Using configured power sensor: %s = %.2f kW",
                    self._power_sensor_entity,
                    power,
                )
                return power

        # Try auto-discovered power sensor
        power_entity = self._entity_cache.get("power")
        if power_entity:
            power = await self._read_entity_float(power_entity, default=None)
            if power is not None:
                state = self.hass.states.get(power_entity)
                if state:
                    unit = state.attributes.get("unit_of_measurement", "").lower()
                    if unit == "w":
                        power = power / 1000.0
                _LOGGER.debug("Using auto-discovered power sensor: %.2f kW", power)
                return power

        # Fall back to estimation from supply temperature
        supply_temp = await self._read_entity_float(
            self._entity_cache.get("supply_temp"), default=None
        )
        outdoor_temp = await self._read_entity_float(
            self._entity_cache.get("outdoor_temp"), default=None
        )

        if supply_temp is not None and outdoor_temp is not None:
            estimated_power = self._estimate_power_from_temps(supply_temp, outdoor_temp)
            _LOGGER.debug("Estimating power from temperatures: %.2f kW", estimated_power)
            return estimated_power

        return None

    async def _calculate_multi_sensor_temperature(self, nibe_temp: float) -> float:
        """Calculate indoor temperature from NIBE sensor + additional sensors.

        Combines NIBE BT50 with additional room sensors for more accurate
        whole-house temperature reading.

        Args:
            nibe_temp: Temperature from NIBE BT50 sensor

        Returns:
            Combined temperature using configured method (median/average)
        """
        # Start with NIBE sensor
        temps = [nibe_temp]

        # Read additional sensors
        for entity_id in self._additional_indoor_sensors:
            state = self.hass.states.get(entity_id)
            if state and state.state not in ["unknown", "unavailable"]:
                try:
                    temp = float(state.state)
                    # Sanity check (15-30°C range)
                    if 15.0 <= temp <= 30.0:
                        temps.append(temp)
                    else:
                        _LOGGER.warning(
                            "Ignoring out-of-range temperature from %s: %.1f°C",
                            entity_id,
                            temp,
                        )
                except (ValueError, TypeError) as err:
                    _LOGGER.debug("Failed to read sensor %s: %s", entity_id, err)

        # Calculate combined temperature
        if len(temps) == 1:
            # Only NIBE sensor available
            return nibe_temp

        if self._indoor_temp_method == "median":
            # Median is more robust to outliers (recommended)
            temps_sorted = sorted(temps)
            n = len(temps_sorted)
            if n % 2 == 0:
                result = (temps_sorted[n // 2 - 1] + temps_sorted[n // 2]) / 2
            else:
                result = temps_sorted[n // 2]
            method_name = "median"
        else:
            # Average (mean)
            result = sum(temps) / len(temps)
            method_name = "average"

        _LOGGER.info(
            "Multi-sensor indoor temp: %.1f°C (%s of %d sensors: %s)",
            result,
            method_name,
            len(temps),
            ", ".join(f"{t:.1f}°C" for t in temps),
        )

        return result

    def _estimate_power_from_temps(self, supply_temp: float, outdoor_temp: float) -> float:
        """Estimate heat pump power consumption from temperatures.

        Uses typical NIBE ASHP characteristics:
        - Higher flow temp = higher power
        - Lower outdoor temp = higher power
        - Typical residential heat pump: 2-8 kW range

        Args:
            supply_temp: Supply/flow temperature (°C)
            outdoor_temp: Outdoor temperature (°C)

        Returns:
            Estimated power consumption (kW)
        """
        # Base power from flow temperature (30-60°C typical range)
        # Higher flow = more power
        flow_factor = (supply_temp - 25.0) / 20.0  # 0.25 at 30°C, 1.75 at 60°C
        flow_factor = max(0.2, min(flow_factor, 2.0))

        # Outdoor temperature factor (more power needed when cold)
        # At +7°C: factor 1.0, At -20°C: factor ~2.5
        temp_factor = 1.0 + (7.0 - outdoor_temp) / 18.0
        temp_factor = max(TEMP_FACTOR_MIN, min(temp_factor, TEMP_FACTOR_MAX))

        # Base power for typical residential heat pump
        base_power = DEFAULT_BASE_POWER  # kW (typical average)

        estimated = base_power * flow_factor * temp_factor

        # Clamp to reasonable range for residential heat pumps
        return max(1.0, min(estimated, 12.0))

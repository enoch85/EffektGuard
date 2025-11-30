"""NIBE Myuplink adapter for reading heat pump state.

This adapter reads NIBE heat pump data from existing Home Assistant entities
created by the NIBE Myuplink integration. It does not directly communicate
with the MyUplink API.

Data read includes:
- Indoor temperature (BT50 or room sensor)
- Outdoor temperature (BT1)
- Supply temperature (BT25 on F2040, BT63 on F750)
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
    NIBE_OFFSET_HYSTERESIS_MARGIN,
    NIBE_POWER_FACTOR,
    NIBE_VOLTAGE_PER_PHASE,
    TEMP_FACTOR_MAX,
    TEMP_FACTOR_MIN,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class NibeState:
    """Current state of NIBE heat pump.

    All temperatures in °C, power in kW, currents in Amps.
    """

    outdoor_temp: float
    indoor_temp: float
    supply_temp: float  # BT25 (F2040) or BT63 (F750) - Flow/supply temperature
    return_temp: float | None
    degree_minutes: float
    current_offset: float
    is_heating: bool
    is_hot_water: bool
    timestamp: datetime
    dhw_top_temp: float | None = None  # BT7 - Hot water top (40013) - optional
    dhw_charging_temp: float | None = None  # BT6 - Hot water charging/bottom (40014) - optional
    phase1_current: float | None = None  # BE1 - Phase 1 current (43086) - optional
    phase2_current: float | None = None  # BE2 - Phase 2 current (43122) - optional
    phase3_current: float | None = None  # BE3 - Phase 3 current (43081) - optional
    compressor_hz: int | None = None  # Compressor frequency - optional

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
        # Accumulator for fractional offset changes (NIBE only accepts integers)
        self._fractional_accumulator: float = 0.0
        # Track if we've synced with actual NIBE offset since startup
        # On restart, accumulator resets but NIBE keeps its value - we need to verify sync
        self._startup_sync_done: bool = False

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

        # Read phase current sensors (optional - BE1, BE2, BE3)
        phase1_current = await self._read_entity_float(
            self._entity_cache.get("phase1_current"), default=None
        )
        phase2_current = await self._read_entity_float(
            self._entity_cache.get("phase2_current"), default=None
        )
        phase3_current = await self._read_entity_float(
            self._entity_cache.get("phase3_current"), default=None
        )
        compressor_hz = await self._read_entity_float(
            self._entity_cache.get("compressor_hz"), default=None
        )

        # Log current sensors if available
        if phase1_current is not None:
            _LOGGER.debug(
                "Phase currents (BE): L1=%.1fA, L2=%.1fA, L3=%.1fA",
                phase1_current,
                phase2_current or 0.0,
                phase3_current or 0.0,
            )

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
            phase1_current=phase1_current,
            phase2_current=phase2_current,
            phase3_current=phase3_current,
            compressor_hz=int(compressor_hz) if compressor_hz else None,
        )

    async def set_curve_offset(self, offset: float) -> bool:
        """Set heating curve offset via NIBE entity.

        Args:
            offset: Offset value in °C (-10 to +10)

        Returns:
            True if offset was written to NIBE, False if deferred/accumulated

        Note:
            Requires NIBE Myuplink Premium subscription for write access.
            Implements rate limiting to avoid excessive API calls.
        """
        from datetime import timedelta

        # Rate limiting - minimum 5 minutes between writes
        now = dt_util.utcnow()
        if self._last_write and now - self._last_write < timedelta(minutes=5):
            _LOGGER.debug("Skipping offset write, too soon since last write")
            return False

        # Get offset entity
        offset_entity = self._entity_cache.get("offset")
        if not offset_entity:
            _LOGGER.error("No offset entity found")
            return False

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
            return False

        # NIBE offset rounding with fractional accumulation
        # NIBE MyUplink entities have step=1 (integer only), but optimization
        # calculates fractional offsets (e.g., 0.35°C for gentle nudges).
        # Solution: Accumulate fractional parts and apply when they sum to ±1°C
        # This preserves gentle optimization while respecting NIBE constraints.

        original_offset = offset

        # Add fractional part to accumulator
        integer_part = int(offset)
        fractional_part = offset - integer_part

        _LOGGER.debug(
            "Offset calculation: original=%.2f°C, integer=%d°C, fractional=%.2f°C, "
            "accumulator_before=%.2f°C",
            original_offset,
            integer_part,
            fractional_part,
            self._fractional_accumulator,
        )

        self._fractional_accumulator += fractional_part

        # Check if accumulated fractional changes warrant an adjustment
        if abs(self._fractional_accumulator) >= 1.0:
            # Apply accumulated change
            accumulated_adjustment = int(self._fractional_accumulator)
            integer_part += accumulated_adjustment
            self._fractional_accumulator -= accumulated_adjustment
            _LOGGER.info(
                "✓ Accumulated fractional offset reached threshold: "
                "applying %+d°C adjustment (accumulator: %.2f°C → %.2f°C)",
                accumulated_adjustment,
                self._fractional_accumulator + accumulated_adjustment,
                self._fractional_accumulator,
            )
            _LOGGER.debug(
                "Accumulator triggered: total_accumulated=%.2f°C, adjustment=%+d°C, "
                "new_integer=%d°C, remaining=%.2f°C",
                self._fractional_accumulator + accumulated_adjustment,
                accumulated_adjustment,
                integer_part,
                self._fractional_accumulator,
            )

        offset_to_apply = integer_part

        # Read current NIBE offset to check if we need to actively reset to 0
        # This fixes the issue where NIBE stays at offset 1 when EffektGuard wants 0
        current_nibe_offset = None
        try:
            current_nibe_offset = float(state.state)
        except (ValueError, TypeError):
            _LOGGER.debug("Could not read current NIBE offset, will proceed with write")

        # STARTUP SYNC CHECK (Nov 30, 2025)
        # On restart, the fractional accumulator resets to 0 but NIBE keeps its value.
        # This can cause a mismatch where EffektGuard thinks NIBE is at 0 but it's at ±1.
        # On first write after restart, verify NIBE matches what we want to apply.
        if not self._startup_sync_done and current_nibe_offset is not None:
            self._startup_sync_done = True
            expected_nibe = offset_to_apply
            actual_nibe = int(current_nibe_offset)

            if actual_nibe != expected_nibe:
                _LOGGER.info(
                    "🔄 Startup sync: NIBE offset mismatch detected! "
                    "NIBE at %d°C but we want %d°C (calculated: %.2f°C). Forcing sync.",
                    actual_nibe,
                    expected_nibe,
                    original_offset,
                )
                # Force write by falling through (don't skip)
                # Reset accumulator to match actual state we're writing
                self._fractional_accumulator = original_offset - expected_nibe
                _LOGGER.debug(
                    "Startup sync: Reset accumulator to %.2f°C to track remaining fractional",
                    self._fractional_accumulator,
                )
            else:
                _LOGGER.info(
                    "✓ Startup sync: NIBE offset verified at %d°C (calculated: %.2f°C)",
                    actual_nibe,
                    original_offset,
                )
                # Initialize accumulator based on difference between calculated and actual
                self._fractional_accumulator = original_offset - actual_nibe
                _LOGGER.debug(
                    "Startup sync: Initialized accumulator to %.2f°C",
                    self._fractional_accumulator,
                )
                # If NIBE already matches, we can skip this write
                if abs(self._fractional_accumulator) < 1.0:
                    return False

        # Skip API call only if:
        # 1. Calculated offset rounds to 0, AND
        # 2. Accumulator hasn't reached threshold, AND
        # 3. NIBE is already at 0 (or we can't read current value safely)
        #
        # HYSTERESIS: To prevent offset oscillation (NIBE flipping 0↔1 repeatedly),
        # only reset NIBE downward when calculated is CLEARLY below threshold.
        # - To go from NIBE=1 to NIBE=0: calculated must be <= 0.3 (not just < 0.5)
        # - This creates a "dead zone" where values 0.3-0.7 don't trigger changes
        
        if offset_to_apply == 0 and abs(self._fractional_accumulator) < 1.0:
            if current_nibe_offset is not None and current_nibe_offset >= 1:
                # NIBE is at positive offset but we want 0
                # Only reset if calculated is clearly below (with hysteresis)
                if original_offset <= NIBE_OFFSET_HYSTERESIS_MARGIN:
                    _LOGGER.info(
                        "→ Resetting NIBE offset from %d°C to 0°C "
                        "(calculated: %.2f°C <= hysteresis %.1f)",
                        int(current_nibe_offset),
                        original_offset,
                        NIBE_OFFSET_HYSTERESIS_MARGIN,
                    )
                    # Fall through to write
                else:
                    _LOGGER.debug(
                        "⏸ Skipping reset: calculated %.2f°C > hysteresis %.1f, "
                        "keeping NIBE at %d°C",
                        original_offset,
                        NIBE_OFFSET_HYSTERESIS_MARGIN,
                        int(current_nibe_offset),
                    )
                    return False
            elif current_nibe_offset is not None and current_nibe_offset <= -1:
                # NIBE is at negative offset but we want 0
                # Only reset if calculated is clearly above (with hysteresis)
                if original_offset >= -NIBE_OFFSET_HYSTERESIS_MARGIN:
                    _LOGGER.info(
                        "→ Resetting NIBE offset from %d°C to 0°C "
                        "(calculated: %.2f°C >= hysteresis %.1f)",
                        int(current_nibe_offset),
                        original_offset,
                        -NIBE_OFFSET_HYSTERESIS_MARGIN,
                    )
                    # Fall through to write
                else:
                    _LOGGER.debug(
                        "⏸ Skipping reset: calculated %.2f°C < hysteresis %.1f, "
                        "keeping NIBE at %d°C",
                        original_offset,
                        -NIBE_OFFSET_HYSTERESIS_MARGIN,
                        int(current_nibe_offset),
                    )
                    return False
            else:
                _LOGGER.debug(
                    "⏸ Skipping NIBE write: offset rounds to 0°C and NIBE already at %s "
                    "(accumulator: %.2f°C, will apply when ≥±1.0°C)",
                    f"{int(current_nibe_offset)}°C" if current_nibe_offset is not None else "0°C",
                    self._fractional_accumulator,
                )
                return False

        # Log accumulator status
        if abs(fractional_part) > 0.01:
            if abs(self._fractional_accumulator) < 1.0:
                _LOGGER.debug(
                    "→ Deferring fractional offset %.2f°C → applying %d°C "
                    "(accumulator: %.2f°C, needs %.2f°C more to trigger)",
                    original_offset,
                    offset_to_apply,
                    self._fractional_accumulator,
                    1.0 - abs(self._fractional_accumulator),
                )
            else:
                _LOGGER.debug(
                    "→ Fractional offset %.2f°C → applying %d°C (after accumulator adjustment)",
                    original_offset,
                    offset_to_apply,
                )
        else:
            _LOGGER.debug(
                "→ Integer offset %.2f°C → applying %d°C (no accumulation needed)",
                original_offset,
                offset_to_apply,
            )

        # Set value via number entity service (non-blocking)
        try:
            await self.hass.services.async_call(
                "number",
                "set_value",
                {
                    "entity_id": offset_entity,
                    "value": offset_to_apply,
                },
                blocking=False,  # Non-blocking to avoid UI delays
            )
            self._last_write = now

            # Summary logging
            if abs(original_offset - offset_to_apply) > 0.01:
                _LOGGER.info(
                    "Set NIBE offset to %d°C (calculated: %.2f°C, accumulator: %.2f°C)",
                    offset_to_apply,
                    original_offset,
                    self._fractional_accumulator,
                )
            else:
                _LOGGER.info("Set NIBE offset to %d°C", offset_to_apply)

            _LOGGER.debug(
                "Offset write complete: entity=%s, value=%d°C, "
                "original=%.2f°C, accumulator=%.2f°C",
                offset_entity,
                offset_to_apply,
                original_offset,
                self._fractional_accumulator,
            )
            return True
        except (AttributeError, OSError, ValueError, TypeError) as err:
            _LOGGER.error("Failed to set NIBE offset: %s", err)
            # Don't raise - allow system to continue gracefully
            # Error logged for debugging, but not fatal
            return False

    async def _discover_nibe_entities(self) -> None:
        """Discover NIBE entities from entity registry.

        Populates _entity_cache with entity IDs for:
        - outdoor_temp (BT1)
        - indoor_temp (BT50 or room sensor)
        - supply_temp (BT25 on F2040, BT63 on F750)
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
        # CRITICAL: Use word boundaries (_btX, btX_) to prevent substring matches
        # e.g., "bt6" would match "bt63", "bt1" would match "bt10", etc.
        # This ensures we match the exact sensor, not similar named sensors
        patterns = {
            "outdoor_temp": ["_bt1", "bt1_", "outdoor_temp", "40004"],  # BT1 / param 40004
            "indoor_temp": [
                "_bt50",
                "bt50_",
                "room_temperature",
                "40033",
            ],  # BT50 / param 40033 "Temperature"
            "supply_temp": [
                "_bt25",
                "bt25_",
                "_bt63",
                "bt63_",
                "supply_temp",
                "heating_medium_supply",
                "40008",
            ],  # BT25 (F2040) or BT63 (F750) / param 40008
            "return_temp": ["_bt3", "bt3_", "return_temp", "40012"],  # BT3 / param 40012
            "degree_minutes": ["degree_minutes", "40941"],  # param 40941
            "offset": ["offset", "47011"],  # param 47011
            "compressor_status": ["compressor", "43427"],  # param 43427
            "hot_water_status": ["hot_water", "dhw"],
            "dhw_top_temp": [
                "_bt7",
                "bt7_",
                "hw_top",
                "40013",
            ],  # BT7 / param 40013 - hot water top
            "dhw_charging_temp": [
                "_bt6",
                "bt6_",
                "hw_bottom",
                "hw_charging",
                "40014",
            ],  # BT6 / param 40014
            "phase1_current": [
                "current_be1",
                "_be1",
                "phase_1_current",
                "43086",
            ],  # BE1 / param 43086
            "phase2_current": [
                "current_be2",
                "_be2",
                "phase_2_current",
                "43122",
            ],  # BE2 / param 43122
            "phase3_current": [
                "current_be3",
                "_be3",
                "phase_3_current",
                "electrical_addition",
                "43081",
            ],  # BE3 / param 43081
            "compressor_hz": [
                "compressor_frequency",
                "43136",
            ],  # Compressor frequency param 43136
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
                                if state:
                                    device_class = state.attributes.get("device_class")
                                    unit = state.attributes.get("unit_of_measurement")

                                    # Accept if device_class is temperature OR unit is °C/°F
                                    # This handles cases where device_class is not set but unit is
                                    is_temp_sensor = device_class == "temperature" or unit in [
                                        "°C",
                                        "°F",
                                        "C",
                                        "F",
                                    ]

                                    if not is_temp_sensor:
                                        _LOGGER.debug(
                                            "Skipping %s (not a temperature sensor, device_class=%s, unit=%s): %s",
                                            key,
                                            device_class,
                                            unit,
                                            entity.entity_id,
                                        )
                                        continue

                            self._entity_cache[key] = entity.entity_id
                            _LOGGER.debug("Found NIBE entity %s: %s", key, entity.entity_id)
                            break

        # Log all discovered entities for debugging
        _LOGGER.info("NIBE Entity Discovery: Found %d entities", len(self._entity_cache))
        for key, entity_id in self._entity_cache.items():
            state = self.hass.states.get(entity_id)
            state_value = state.state if state else "unavailable"
            unit = state.attributes.get("unit_of_measurement", "") if state else ""
            _LOGGER.info("  %s: %s = %s %s", key, entity_id, state_value, unit)

        # Warn if critical sensors are missing
        if "indoor_temp" not in self._entity_cache:
            _LOGGER.warning(
                "No indoor temperature sensor (BT50) found! "
                "Looking for entities with: bt50, room_temperature, or 40033. "
                "Will use default fallback temperature (21°C)."
            )
        if "outdoor_temp" not in self._entity_cache:
            _LOGGER.warning("No outdoor temperature sensor (BT1) found!")
        if "degree_minutes" not in self._entity_cache:
            _LOGGER.warning("No degree minutes sensor found, will estimate from thermal model")

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

    def calculate_power_from_currents(
        self,
        phase1_amps: float | None,
        phase2_amps: float | None = None,
        phase3_amps: float | None = None,
        voltage_per_phase: float = NIBE_VOLTAGE_PER_PHASE,
        power_factor: float = NIBE_POWER_FACTOR,
    ) -> float | None:
        """Calculate real power consumption from phase current sensors.

        Uses NIBE's built-in current sensors (BE1, BE2, BE3) to calculate
        actual power consumption instead of estimating.

        All NIBE heat pumps in Sweden are 3-phase systems (F730, F750, F2040, S1155).

        Args:
            phase1_amps: Phase 1 current (BE1) in Amps
            phase2_amps: Phase 2 current (BE2) in Amps (may be 0 when compressor off)
            phase3_amps: Phase 3 current (BE3) in Amps (may be 0 when compressor off)
            voltage_per_phase: Voltage per phase (from const.py: NIBE_VOLTAGE_PER_PHASE)
            power_factor: Motor power factor (from const.py: NIBE_POWER_FACTOR)

        Returns:
            Power consumption in kW, or None if no current data available

        Notes:
            - Swedish 3-phase: 400V between phases, 240V phase-to-neutral
            - All NIBE heat pumps are 3-phase (no single-phase models in Sweden)
            - Power factor from const.py (conservative 0.95, real likely 0.96-0.98)
            - Formula: P = V × (I1 + I2 + I3) × cos(φ) / 1000
        """
        if phase1_amps is None:
            return None

        # All NIBE heat pumps are 3-phase in Sweden
        total_amps = phase1_amps + (phase2_amps or 0.0) + (phase3_amps or 0.0)
        power_kw = voltage_per_phase * total_amps * power_factor / 1000

        _LOGGER.debug(
            "NIBE 3-phase power: %.1fV × (%.1f + %.1f + %.1f)A × %.2f = %.2f kW",
            voltage_per_phase,
            phase1_amps,
            phase2_amps or 0.0,
            phase3_amps or 0.0,
            power_factor,
            power_kw,
        )

        return power_kw

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

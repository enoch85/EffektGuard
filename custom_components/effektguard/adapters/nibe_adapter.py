"""NIBE adapter for reading heat pump state from Home Assistant entities.

This adapter reads NIBE heat pump data from existing Home Assistant entities.
It never calls external APIs. Supported sources (issue #18):
- myuplink (cloud): writable parameters are NUMBER entities
- nibe_heatpump (local NibeGW/Modbus): entity ids embed register numbers,
  writable coils are NUMBER entities
- generic modbus YAML sensors + template numbers wrapping modbus.write_register

Entities are found via manual config-flow overrides first (authoritative),
then name-pattern discovery over the state machine and entity registry
(NIBE_DISCOVERY_PATTERNS in const.py).

Data read includes:
- Indoor temperature (BT50 or room sensor)
- Outdoor temperature (BT1)
- Supply temperature (BT2/BT25, BT63 on MyUplink F750)
- Return temperature (BT3)
- Degree minutes (GM/DM)
- Current heating curve offset
- Compressor status
- Hot water status
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.update_coordinator import UpdateFailed
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import TemperatureConverter

from ..const import (
    CONF_ADDITIONAL_INDOOR_SENSORS,
    CONF_DEGREE_MINUTES_ENTITY,
    CONF_INDOOR_TEMP_METHOD,
    CONF_NIBE_ENTITY,
    CONF_POWER_SENSOR_ENTITY,
    DEFAULT_BASE_POWER,
    DEFAULT_INDOOR_TEMP,
    DEFAULT_INDOOR_TEMP_METHOD,
    DOMAIN,
    INDOOR_SENSOR_PLAUSIBLE_MAX,
    INDOOR_SENSOR_PLAUSIBLE_MIN,
    MAX_OFFSET,
    MIN_OFFSET,
    NIBE_COMPRESSOR_ACTIVE_HZ_THRESHOLD,
    NIBE_DEFAULT_SUPPLY_TEMP,
    NIBE_DISCOVERY_CORE_KEYS,
    NIBE_DISCOVERY_EXCLUDE,
    NIBE_DISCOVERY_MAX_ATTEMPTS,
    NIBE_DISCOVERY_PATTERNS,
    NIBE_DISCOVERY_RANK_LIVE,
    NIBE_DISCOVERY_RANK_LIVE_UNREGISTERED,
    NIBE_DISCOVERY_RANK_MANUAL,
    NIBE_DISCOVERY_RANK_REGISTRY_ONLY,
    NIBE_DISCOVERY_SLOW_RETRY_CYCLES,
    NIBE_FRACTIONAL_ACCUMULATOR_THRESHOLD,
    NIBE_MANUAL_OVERRIDE_KEYS,
    NIBE_OFFSET_RESYNC_MINUTES,
    NIBE_POWER_FACTOR,
    NIBE_PRIO_HEATING_STATES,
    NIBE_PRIO_HOT_WATER_STATES,
    NIBE_STATUS_ACTIVE_STATES,
    NIBE_SWITCH_DISCOVERY_PATTERNS,
    NIBE_TEMPERATURE_KEYS,
    NIBE_UNKNOWN_VALUE_MARKERS,
    NIBE_VOLTAGE_PER_PHASE,
    SERVICE_RATE_LIMIT_MINUTES,
    TEMP_FACTOR_MAX,
    TEMP_FACTOR_MIN,
)

if TYPE_CHECKING:
    from ..models.types import AdapterConfigDict

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
    dhw_amount_minutes: float | None = None  # Hot water amount available (minutes) - optional
    phase1_current: float | None = None  # BE1 - Phase 1 current (43086) - optional
    phase2_current: float | None = None  # BE2 - Phase 2 current (43122) - optional
    phase3_current: float | None = None  # BE3 - Phase 3 current (43081) - optional
    compressor_hz: int | None = None  # Compressor frequency - optional
    power_kw: float | None = None  # Total power consumption in kW - optional
    # False when no indoor sensor could be read and indoor_temp is DEFAULT_INDOOR_TEMP
    # rather than a measurement. A NIBE system without a room sensor (no BT50) is a
    # LEGITIMATE configuration - the pump runs on degree minutes and the heating curve
    # alone - so this is not an error. But any layer that reasons about comfort MUST
    # abstain rather than trust the placeholder: DEFAULT_INDOOR_TEMP equals the usual
    # target, which silently produces a temperature deviation of exactly 0.0.
    indoor_temp_valid: bool = True

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

    def __init__(self, hass: HomeAssistant, config: "AdapterConfigDict"):
        """Initialize NIBE adapter.

        Args:
            hass: Home Assistant instance
            config: Configuration dictionary with entity IDs
        """
        self.hass = hass
        self._degree_minutes_entity = config.get(CONF_DEGREE_MINUTES_ENTITY)  # Optional
        self._power_sensor_entity = config.get(CONF_POWER_SENSOR_ENTITY)  # Optional
        self._additional_indoor_sensors = config.get(CONF_ADDITIONAL_INDOOR_SENSORS, [])  # Optional
        self._indoor_temp_method = config.get(CONF_INDOOR_TEMP_METHOD, DEFAULT_INDOOR_TEMP_METHOD)
        # Manual entity overrides beat pattern discovery (issue #18: Modbus/renamed
        # entities cannot rely on name patterns). The offset entity the user picked
        # in the first config step is authoritative for the write path.
        self._manual_overrides: dict[str, str] = {}
        if config.get(CONF_NIBE_ENTITY):
            self._manual_overrides["offset"] = config[CONF_NIBE_ENTITY]
        if self._degree_minutes_entity:
            self._manual_overrides["degree_minutes"] = self._degree_minutes_entity
        for cache_key, conf_key in NIBE_MANUAL_OVERRIDE_KEYS.items():
            if config.get(conf_key):
                self._manual_overrides[cache_key] = config[conf_key]
        self._last_write: datetime | None = None
        self._last_ventilation_write: datetime | None = None
        # Rank of each cached discovery result, persisted across discovery
        # passes so a live entity found later still displaces a registry-only
        # entry cached during startup
        self._entity_ranks: dict[str, int] = {}
        # Seed overrides immediately: consumers like the coordinator's offset
        # sync read entity_cache before the first discovery pass runs
        self._entity_cache: dict[str, str] = dict(self._manual_overrides)
        for key in self._manual_overrides:
            self._entity_ranks[key] = NIBE_DISCOVERY_RANK_MANUAL
        self._discovery_attempts: int = 0
        self._cycles_since_discovery: int = 0
        # Fractional accumulator for precise offset tracking
        # NIBE only accepts integers, so we accumulate fractional parts
        # and apply them when they sum to ±1°C
        self._fractional_accumulator: float = 0.0
        # Track last integer offset sent to NIBE (to avoid redundant writes)
        self._last_nibe_offset: int | None = None

    @property
    def entity_cache(self) -> dict[str, str]:
        """Public access to discovered entity cache."""
        return self._entity_cache

    @property
    def power_sensor_entity(self) -> str | None:
        """Public access to configured power sensor entity."""
        return self._power_sensor_entity

    async def discover_entities(self) -> None:
        """Discover NIBE entities (public wrapper)."""
        await self._discover_nibe_entities()

    async def get_current_state(self) -> NibeState:
        """Read current NIBE heat pump state from entities.

        Missing entities are read as safe defaults - see the per-field
        defaults below and the discovery warnings in _discover_nibe_entities.

        Returns:
            NibeState with current readings
        """
        # Discover NIBE entities if not cached. Re-discover while core sensors
        # are missing OR only backed by registry entries without a live state
        # yet: source integrations (modbus, nibe_heatpump, myuplink) may create
        # their entities AFTER our first pass during HA startup, and a stale
        # registry id must not permanently block the live entity (issue #18).
        # Capped: a sensor still absent after NIBE_DISCOVERY_MAX_ATTEMPTS
        # cycles does not exist in this setup (e.g. no BT50 room sensor).
        core_unresolved = any(
            key not in self._entity_cache
            or self._entity_ranks.get(key, NIBE_DISCOVERY_RANK_LIVE)
            >= NIBE_DISCOVERY_RANK_REGISTRY_ONLY
            for key in NIBE_DISCOVERY_CORE_KEYS
        )
        self._cycles_since_discovery += 1
        if not self._entity_cache or (
            core_unresolved
            and (
                self._discovery_attempts < NIBE_DISCOVERY_MAX_ATTEMPTS
                or self._cycles_since_discovery >= NIBE_DISCOVERY_SLOW_RETRY_CYCLES
            )
        ):
            await self._discover_nibe_entities()

        # --- REQUIRED readings -------------------------------------------------------
        # These three drive every control decision. Never substitute a plausible constant for a
        # missing one: that makes a broken installation indistinguishable from a healthy one and
        # still writes a curve offset to the pump. Refuse, and let the coordinator degrade
        # (startup_pending before the first success, UpdateFailed after) - entities go
        # unavailable and nothing is written.
        outdoor_temp = await self._read_temperature(self._entity_cache.get("outdoor_temp"))
        supply_temp = await self._read_temperature(self._entity_cache.get("supply_temp"))

        # Degree minutes: configured sensor first, then auto-discovery. NEVER estimated -
        # DM is the primary thermal-debt safety signal and every NIBE exposes it
        # (register 40940 / 43005). Guessing it would drive the emergency layer on fiction.
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

        if degree_minutes is None:
            degree_minutes = await self._read_entity_float(
                self._entity_cache.get("degree_minutes"), default=None
            )
            if degree_minutes is not None:
                _LOGGER.debug("Using auto-discovered degree minutes sensor: %.1f", degree_minutes)

        missing = [
            name
            for name, value in (
                ("outdoor temperature (BT1)", outdoor_temp),
                ("supply/flow temperature (BT25/BT63)", supply_temp),
                ("degree minutes", degree_minutes),
            )
            if value is None
        ]
        if missing:
            raise UpdateFailed(
                "Cannot read required NIBE sensors: "
                + ", ".join(missing)
                + ". EffektGuard will not control the heat pump on incomplete data. "
                "Check that the source integration (myuplink / nibe_heatpump / modbus) is "
                "loaded and its entities are available, or set the matching manual entity "
                "overrides in the EffektGuard options."
            )

        # --- OPTIONAL readings -------------------------------------------------------
        # Indoor temperature: a NIBE without a room sensor (no BT50) is a legitimate
        # configuration - it runs on degree minutes and the heating curve. Keep the
        # placeholder for display, but mark it invalid so comfort-reasoning layers abstain
        # instead of reading a deviation of exactly 0.0 from a value that IS the target.
        measured_indoor = await self._read_temperature(self._entity_cache.get("indoor_temp"))
        indoor_temp_valid = measured_indoor is not None
        indoor_temp = measured_indoor if indoor_temp_valid else DEFAULT_INDOOR_TEMP

        # Multi-sensor indoor temperature calculation
        if self._additional_indoor_sensors:
            combined = await self._calculate_multi_sensor_temperature(indoor_temp)
            if combined is not None:
                indoor_temp = combined
                indoor_temp_valid = True

        return_temp = await self._read_temperature(self._entity_cache.get("return_temp"))

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

        # Priority (register 43086) identifies what the compressor is serving.
        # It overrides the broad status patterns: a DHW-priority run must not
        # count as space heating (nibe_heatpump reports compressor "Running"
        # during DHW production while no hot_water_status entity exists).
        prio = self._read_prio_state()
        if prio == "hot_water":
            is_hot_water = True
        elif prio == "heating":
            is_hot_water = False

        # Read DHW temperatures (optional - BT7 top, BT6 charging/bottom)
        dhw_top_temp = await self._read_temperature(self._entity_cache.get("dhw_top_temp"))
        dhw_charging_temp = await self._read_temperature(
            self._entity_cache.get("dhw_charging_temp")
        )

        # Read DHW amount (hot water minutes available) - NIBE calculates this
        dhw_amount_minutes = await self._read_entity_float(
            self._entity_cache.get("dhw_amount"), default=None
        )

        # Log DHW sensor status
        if dhw_top_temp is not None:
            _LOGGER.debug("DHW top temperature (BT7): %.1f°C", dhw_top_temp)
        if dhw_charging_temp is not None:
            _LOGGER.debug("DHW charging temperature (BT6): %.1f°C", dhw_charging_temp)
        if dhw_amount_minutes is not None:
            _LOGGER.debug("DHW amount available: %.1f minutes", dhw_amount_minutes)

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

        # Derive heating activity from compressor frequency when the status
        # entity is missing or unparseable (raw Modbus 43427 is a numeric enum
        # that _read_entity_bool cannot interpret). A DHW-only run also spins
        # the compressor, so hot-water production must not count as heating.
        if not is_heating and not is_hot_water and compressor_hz is not None:
            is_heating = compressor_hz > NIBE_COMPRESSOR_ACTIVE_HZ_THRESHOLD

        # Final priority override: whatever the status entities claimed, a
        # hot-water-priority compressor run is not space heating
        if prio == "hot_water":
            is_heating = False

        # Log current sensors if available
        if phase1_current is not None:
            _LOGGER.debug(
                "Phase currents (BE): L1=%.1fA, L2=%.1fA, L3=%.1fA",
                phase1_current,
                phase2_current or 0.0,
                phase3_current or 0.0,
            )

        # Read actual power consumption (if available)
        power_kw = await self.get_power_consumption()
        if power_kw is not None:
            _LOGGER.debug("Power consumption: %.2f kW", power_kw)

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
            dhw_amount_minutes=dhw_amount_minutes,
            phase1_current=phase1_current,
            phase2_current=phase2_current,
            phase3_current=phase3_current,
            compressor_hz=int(compressor_hz) if compressor_hz is not None else None,
            power_kw=power_kw,
            indoor_temp_valid=indoor_temp_valid,
        )

    async def set_curve_offset(self, offset: float) -> bool:
        """Set heating curve offset via NIBE entity with fractional accumulation.

        The NIBE offset register (47011 on F-series) is integer-only, but the
        optimization engine calculates precise fractional offsets (e.g., 0.35°C,
        -1.24°C).

        Solution: Accumulate fractional parts and apply when they sum to ±1°C.
        This preserves the precision of gentle optimization while respecting
        NIBE's integer-only constraint.

        Example:
            Cycle 1: Calculate 0.35°C → Send 0, accumulate +0.35
            Cycle 2: Calculate 0.42°C → Delta +0.07, accumulator = +0.42
            Cycle 3: Calculate 0.89°C → Delta +0.47, accumulator = +0.89
            Cycle 4: Calculate 1.12°C → Delta +0.23, accumulator = +1.12 → Send +1

        Args:
            offset: Calculated offset value in °C (e.g., -1.24, +0.87)

        Returns:
            True if offset was written to NIBE, False if deferred/accumulated

        Note:
            The target must be a writable number entity: a MyUplink offset
            number (cloud writes may require a valid myUplink subscription), a
            nibe_heatpump number (e.g. number.heat_offset_s1_47011), or a
            template number wrapping modbus.write_register.
        """
        # Rate limiting - minimum time between writes
        now = dt_util.utcnow()
        if self._last_write and now - self._last_write < timedelta(
            minutes=SERVICE_RATE_LIMIT_MINUTES
        ):
            _LOGGER.debug("Skipping offset write, too soon since last write")
            return False

        # Get offset entity
        offset_entity = self._entity_cache.get("offset")
        if not offset_entity:
            _LOGGER.error("No offset entity found")
            return False

        # Check entity is available
        state = self.hass.states.get(offset_entity)
        if not state or state.state in ["unavailable", "unknown"]:
            _LOGGER.warning(
                "Offset entity %s not ready (state: %s), skipping write",
                offset_entity,
                state.state if state else "None",
            )
            return False

        # Sync with the entity's actual value on first call, and re-sync when
        # the entity disagrees with our bookkeeping and we have not written
        # recently: the offset can change externally (pump display, another
        # automation) or a fire-and-forget write may have failed silently.
        entity_offset: int | None = None
        try:
            entity_offset = int(float(state.state))
        except (ValueError, TypeError):
            pass

        resync_window_passed = self._last_write is None or now - self._last_write >= timedelta(
            minutes=NIBE_OFFSET_RESYNC_MINUTES
        )
        if entity_offset is not None and (
            self._last_nibe_offset is None
            or (entity_offset != self._last_nibe_offset and resync_window_passed)
        ):
            self._last_nibe_offset = entity_offset
            self._fractional_accumulator = offset - self._last_nibe_offset
            _LOGGER.info(
                "✓ Synced with NIBE: current offset %d°C (calculated: %.2f°C, accumulator: %.2f°C)",
                self._last_nibe_offset,
                offset,
                self._fractional_accumulator,
            )
        elif self._last_nibe_offset is None:
            self._last_nibe_offset = 0
            self._fractional_accumulator = offset

        # Fractional accumulation logic
        # The accumulator tracks the total difference between what we've calculated
        # and what NIBE actually has.
        self._fractional_accumulator = offset - self._last_nibe_offset

        _LOGGER.debug(
            "Offset calculation: calculated=%.2f°C, NIBE_current=%d°C, accumulator=%.2f°C",
            offset,
            self._last_nibe_offset,
            self._fractional_accumulator,
        )

        # Determine what integer value to apply to NIBE
        # Only write when accumulator crosses threshold
        offset_to_apply = self._last_nibe_offset  # Start with current NIBE value

        if abs(self._fractional_accumulator) >= NIBE_FRACTIONAL_ACCUMULATOR_THRESHOLD:
            # Accumulator has crossed threshold, apply the integer part
            accumulated_adjustment = int(self._fractional_accumulator)
            offset_to_apply = self._last_nibe_offset + accumulated_adjustment

            # Clamp to NIBE's valid range (MIN_OFFSET to MAX_OFFSET)
            offset_to_apply = int(max(MIN_OFFSET, min(offset_to_apply, MAX_OFFSET)))

            _LOGGER.info(
                "✓ Accumulated fractional offset reached threshold: "
                "applying %+d°C adjustment (accumulator: %.2f°C, new_offset: %d°C)",
                accumulated_adjustment,
                self._fractional_accumulator,
                offset_to_apply,
            )
        else:
            # Accumulator hasn't crossed threshold, keep current NIBE offset
            _LOGGER.debug(
                "Accumulator below threshold (%.2f°C), keeping NIBE at %d°C",
                self._fractional_accumulator,
                offset_to_apply,
            )

        # Only write if integer part changed from last written value
        if offset_to_apply == self._last_nibe_offset:
            _LOGGER.debug(
                "Offset unchanged: %.2f°C → int(%d°C) = NIBE already at %d°C (accumulator: %.2f°C)",
                offset,
                offset_to_apply,
                self._last_nibe_offset,
                self._fractional_accumulator,
            )
            return False

        # Respect the target entity's own limits when it exposes them
        # (a template number left at default min 0/max 100 would otherwise
        # reject every negative offset in a background task we never see).
        # Defensive float() - a malformed entity must never crash the write.
        try:
            entity_min = float(state.attributes["min"])
            entity_max = float(state.attributes["max"])
        except (KeyError, TypeError, ValueError):
            entity_min = entity_max = None
        if entity_min is not None and entity_max is not None:
            clamped = int(max(entity_min, min(float(offset_to_apply), entity_max)))
            if clamped != offset_to_apply:
                _LOGGER.warning(
                    "Offset %d°C outside %s range [%s, %s], clamping to %d°C - "
                    "check the number entity's min/max configuration",
                    offset_to_apply,
                    offset_entity,
                    entity_min,
                    entity_max,
                    clamped,
                )
                offset_to_apply = clamped
                if offset_to_apply == self._last_nibe_offset:
                    return False

        # Store old value for logging before updating
        old_offset = self._last_nibe_offset

        # Write to NIBE. blocking=True so handler failures (out-of-range,
        # unavailable target, cloud errors) surface here instead of being
        # swallowed in a background task while we record a phantom success.
        try:
            await self.hass.services.async_call(
                "number",
                "set_value",
                {
                    "entity_id": offset_entity,
                    "value": offset_to_apply,
                },
                blocking=True,
            )
            self._last_write = now
            self._last_nibe_offset = offset_to_apply

            _LOGGER.info(
                "✓ Applied offset to NIBE: %d°C → %d°C (calculated: %.2f°C, accumulator: %.2f°C)",
                old_offset,
                offset_to_apply,
                offset,
                self._fractional_accumulator,
            )
            return True

        except (HomeAssistantError, AttributeError, OSError, ValueError, TypeError) as err:
            _LOGGER.error("Failed to set NIBE offset: %s", err)
            return False

    async def set_enhanced_ventilation(self, enabled: bool) -> bool:
        """Enable or disable enhanced ventilation for exhaust air heat pumps.

        NIBE F750/F730 "Increased Ventilation" is a switch entity that toggles
        between normal and enhanced airflow. When enabled, the exhaust fan runs
        at higher speed to extract more heat from indoor air.

        Includes rate limiting and redundant state check to minimize API calls.

        Entity pattern: switch.{device}_increased_ventilation

        Args:
            enabled: True to enable enhanced ventilation, False for normal

        Returns:
            True if ventilation was set, False if skipped/failed

        Note:
            Based on NIBE myuplink entity: switch.f750_cu_3x400v_increased_ventilation
        """
        # Get ventilation switch entity from cache
        ventilation_entity = self._entity_cache.get("increased_ventilation")

        if not ventilation_entity:
            _LOGGER.debug("Enhanced ventilation switch not found - exhaust air control unavailable")
            return False

        # Check entity is available and get current state
        state = self.hass.states.get(ventilation_entity)
        if not state or state.state in ["unavailable", "unknown"]:
            _LOGGER.warning(
                "Ventilation switch %s not ready (state: %s), skipping",
                ventilation_entity,
                state.state if state else "None",
            )
            return False

        # Check if already in desired state (avoid redundant API calls)
        current_is_on = state.state == "on"
        if current_is_on == enabled:
            _LOGGER.debug(
                "Ventilation already %s, skipping redundant call",
                "ENHANCED" if enabled else "NORMAL",
            )
            return False

        # Rate limiting - minimum time between writes
        now = dt_util.utcnow()
        if self._last_ventilation_write and now - self._last_ventilation_write < timedelta(
            minutes=SERVICE_RATE_LIMIT_MINUTES
        ):
            remaining = (
                timedelta(minutes=SERVICE_RATE_LIMIT_MINUTES) - (now - self._last_ventilation_write)
            ).total_seconds()
            _LOGGER.debug(
                "Ventilation rate limited, %d seconds remaining",
                int(remaining),
            )
            return False

        try:
            service = "turn_on" if enabled else "turn_off"
            await self.hass.services.async_call(
                "switch",
                service,
                {"entity_id": ventilation_entity},
                blocking=True,
            )

            self._last_ventilation_write = now
            status = "ENHANCED" if enabled else "NORMAL"
            _LOGGER.info("✓ Ventilation set to %s via %s", status, ventilation_entity)
            return True

        except (HomeAssistantError, AttributeError, OSError, ValueError, TypeError) as err:
            _LOGGER.error("Failed to set enhanced ventilation: %s", err)
            return False

    async def is_enhanced_ventilation_active(self) -> bool | None:
        """Check if enhanced ventilation is currently active.

        Returns:
            True if enhanced ventilation is on, False if normal, None if unavailable
        """
        ventilation_entity = self._entity_cache.get("increased_ventilation")

        if not ventilation_entity:
            return None

        state = self.hass.states.get(ventilation_entity)
        if state is None or state.state in ("unknown", "unavailable"):
            return None

        return state.state == "on"

    async def _discover_nibe_entities(self) -> None:
        """Discover NIBE source entities.

        Population order (first wins):
        1. Manual config-flow overrides (authoritative - issue #18 Modbus setups)
        2. Entities in the state machine matching NIBE_DISCOVERY_PATTERNS.
           Scanning states (not only the registry) is REQUIRED: generic Modbus
           YAML entities without unique_id never enter the entity registry.
        3. Enabled registry entries without a state yet (integration still
           starting) - lowest priority, never beats a live entity.

        Ranks persist across passes (self._entity_ranks) so a live entity
        appearing after startup still displaces a registry-only entry.

        Matching rules:
        - Keys are tried in NIBE_DISCOVERY_PATTERNS dict order; each entity can
          satisfy at most one key (specific temperature keys are ordered before
          broad status keys to prevent e.g. hot_water_status grabbing BT7).
        - Temperature keys require device_class temperature or a °C/°F unit.
        - The offset key requires a number entity (write path uses
          number.set_value; a read-only Modbus offset sensor must never win).
        - EffektGuard's own entities are excluded by registry platform.
        """
        self._discovery_attempts += 1
        self._cycles_since_discovery = 0
        previous_cache = dict(self._entity_cache)

        for key, entity_id in self._manual_overrides.items():
            self._entity_cache[key] = entity_id
            self._entity_ranks[key] = NIBE_DISCOVERY_RANK_MANUAL

        ranks = self._entity_ranks
        claimed: set[str] = set(self._entity_cache.values())

        registry = er.async_get(self.hass)
        live_ids = set()

        for state in self.hass.states.async_all(("sensor", "number")):
            live_ids.add(state.entity_id)
            # Never discover our own entities (e.g. sensor.effektguard_dhw_status
            # would match the hot_water_status patterns)
            registry_entry = registry.async_get(state.entity_id)
            if registry_entry and registry_entry.platform == DOMAIN:
                continue
            # Integration-backed entities outrank template/YAML lookalikes
            # without a registry entry (generic Modbus setups still win when
            # nothing registered matches)
            self._consider_candidate(
                state.entity_id,
                state.attributes.get("device_class"),
                state.attributes.get("unit_of_measurement"),
                (
                    NIBE_DISCOVERY_RANK_LIVE
                    if registry_entry
                    else NIBE_DISCOVERY_RANK_LIVE_UNREGISTERED
                ),
                ranks,
                claimed,
            )

        for entity in registry.entities.values():
            if entity.entity_id in live_ids or entity.disabled_by is not None:
                continue
            if entity.domain not in ("sensor", "number") or entity.platform == DOMAIN:
                continue
            self._consider_candidate(
                entity.entity_id,
                entity.device_class or entity.original_device_class,
                entity.unit_of_measurement,
                NIBE_DISCOVERY_RANK_REGISTRY_ONLY,
                ranks,
                claimed,
            )

        # Discover switch entities (separate domain scan for increased_ventilation).
        # Registry entries are included so a switch platform that loads after
        # our last discovery pass is still found (its entity id is registered
        # before its state exists).
        switch_ids = [state.entity_id for state in self.hass.states.async_all("switch")]
        switch_ids += [
            entity.entity_id
            for entity in registry.entities.values()
            if entity.domain == "switch"
            and entity.disabled_by is None
            and entity.platform != DOMAIN
            and entity.entity_id not in switch_ids
        ]
        for entity_id in switch_ids:
            entity_id_lower = entity_id.lower()
            for key, patterns_list in NIBE_SWITCH_DISCOVERY_PATTERNS.items():
                if key not in self._entity_cache and any(
                    pattern in entity_id_lower for pattern in patterns_list
                ):
                    self._entity_cache[key] = entity_id
                    _LOGGER.debug("Found NIBE switch %s: %s", key, entity_id)

        # Log results only when they changed (rediscovery runs during startup
        # while core sensors are missing - identical repeats are just noise)
        if self._entity_cache == previous_cache:
            return

        _LOGGER.info("NIBE Entity Discovery: Found %d entities", len(self._entity_cache))
        for key, entity_id in self._entity_cache.items():
            state = self.hass.states.get(entity_id)
            state_value = state.state if state else "unavailable"
            unit = state.attributes.get("unit_of_measurement", "") if state else ""
            _LOGGER.info("  %s: %s = %s %s", key, entity_id, state_value, unit)

        # Warn if sensors are missing.
        # Indoor is OPTIONAL (a system without a room sensor runs on DM + the heating
        # curve); the others are REQUIRED and get_current_state() refuses to run without
        # them rather than substituting a plausible constant.
        if "indoor_temp" not in self._entity_cache:
            _LOGGER.info(
                "No indoor temperature sensor (BT50) found. Comfort-based layers will "
                "abstain; optimization continues on degree minutes and the heating curve. "
                "Set the indoor temperature override in options if you do have one."
            )
        if "outdoor_temp" not in self._entity_cache:
            _LOGGER.error(
                "No outdoor temperature sensor (BT1) found - EffektGuard cannot optimize "
                "without it. Looking for entities with: bt1, outdoor_temp, or 40004."
            )
        if "supply_temp" not in self._entity_cache:
            _LOGGER.error(
                "No supply/flow temperature sensor (BT25/BT63) found - EffektGuard cannot "
                "optimize without it. Looking for: bt25, bt63, supply_temp, 40008, 40071."
            )
        if "degree_minutes" not in self._entity_cache and not self._degree_minutes_entity:
            _LOGGER.error(
                "No degree minutes sensor found - EffektGuard cannot protect against "
                "thermal debt without it and will not control the pump. Looking for: "
                "degree_minutes, gradminuter, 40940, 43005. Set the degree-minutes entity "
                "override in options if your sensor is named differently."
            )

    def _consider_candidate(
        self,
        entity_id: str,
        device_class: str | None,
        unit: str | None,
        rank: int,
        ranks: dict[str, int],
        claimed: set[str],
    ) -> None:
        """Match one entity against the pattern table and cache it if it wins.

        Args:
            entity_id: Candidate entity id
            device_class: Its device class (state attribute or registry)
            unit: Its unit of measurement (state attribute or registry)
            rank: Candidate quality (lower wins; live state beats registry-only)
            ranks: Current rank per cached key (mutated)
            claimed: Entity ids already assigned to a key (mutated)
        """
        if entity_id in claimed:
            # Already bound to a key - refresh its rank so a registry-only
            # binding whose state has appeared becomes a live binding
            # (otherwise it stays displaceable and core_unresolved never
            # settles)
            for key, cached_id in self._entity_cache.items():
                if cached_id == entity_id and rank < ranks.get(key, rank):
                    ranks[key] = rank
            return

        entity_id_lower = entity_id.lower()
        if any(excluded in entity_id_lower for excluded in NIBE_DISCOVERY_EXCLUDE):
            return

        for key, patterns_list in NIBE_DISCOVERY_PATTERNS.items():
            if not any(pattern in entity_id_lower for pattern in patterns_list):
                continue

            if key in NIBE_TEMPERATURE_KEYS:
                # Accept if device_class is temperature OR unit is °C/°F
                # (handles Modbus sensors where only the unit is configured)
                if device_class != "temperature" and unit not in ["°C", "°F", "C", "F"]:
                    _LOGGER.debug(
                        "Skipping %s (not a temperature sensor, device_class=%s, unit=%s): %s",
                        key,
                        device_class,
                        unit,
                        entity_id,
                    )
                    continue

            if key == "offset" and not entity_id.startswith("number."):
                # Write path calls number.set_value - a sensor can never work
                _LOGGER.debug("Skipping offset candidate %s (not a number entity)", entity_id)
                continue

            # The entity belongs to this key; cache it only if it outranks the
            # current holder, then stop either way (one key per entity).
            if key in self._entity_cache and ranks.get(key, NIBE_DISCOVERY_RANK_LIVE) <= rank:
                return

            previous = self._entity_cache.get(key)
            if previous:
                claimed.discard(previous)
            self._entity_cache[key] = entity_id
            ranks[key] = rank
            claimed.add(entity_id)
            _LOGGER.debug("Found NIBE entity %s: %s", key, entity_id)
            return

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
            value = float(state.state)
        except (ValueError, TypeError):
            _LOGGER.warning("Cannot parse float from %s: %s", entity_id, state.state)
            return default

        if value in NIBE_UNKNOWN_VALUE_MARKERS:
            # s16 unknown-value marker (MyUplink / disconnected Modbus sensor)
            _LOGGER.debug("Ignoring unknown-value marker %s from %s", value, entity_id)
            return default

        return value

    async def _read_temperature(
        self,
        entity_id: str | None,
        default: float | None = None,
    ) -> float | None:
        """Read a temperature entity and normalise it to °C.

        Every temperature in NibeState is documented as °C, and the whole optimization
        stack assumes it. But the unit was never checked: discovery ACCEPTS an entity whose
        unit is °F (see _consider_candidate) and the read path then passed the raw number
        straight through.

        Home Assistant presents a `temperature` device-class sensor in the USER'S preferred
        unit, so on an imperial install - or with a single entity overridden to °F - BT1
        reading 32 (0 °C) was taken as +32 °C and BT25 reading 95 (35 °C) as a 95 °C flow
        temperature. Weather compensation would then drive the offset to minimum in the
        middle of winter.

        Args:
            entity_id: Entity to read
            default: Value to return when the entity is missing or unreadable

        Returns:
            Temperature in °C, or `default`
        """
        if not entity_id:
            return default

        state = self.hass.states.get(entity_id)
        if not state or state.state in ["unknown", "unavailable"]:
            return default

        try:
            value = float(state.state)
        except (ValueError, TypeError):
            _LOGGER.warning("Cannot parse temperature from %s: %s", entity_id, state.state)
            return default

        # The unknown-value marker is a RAW sensor value - check it before converting.
        if value in NIBE_UNKNOWN_VALUE_MARKERS:
            _LOGGER.debug("Ignoring unknown-value marker %s from %s", value, entity_id)
            return default

        unit = state.attributes.get("unit_of_measurement")
        if unit is None or unit == UnitOfTemperature.CELSIUS:
            return value

        try:
            return TemperatureConverter.convert(value, unit, UnitOfTemperature.CELSIUS)
        except (HomeAssistantError, ValueError, TypeError):
            _LOGGER.warning(
                "Unrecognised temperature unit %r on %s - treating %.1f as °C",
                unit,
                entity_id,
                value,
            )
            return value

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

        # Lowercase comparison covers on/off switches plus mapped coil states
        # from nibe_heatpump/myuplink ("Running", "Starting")
        return state.state.lower() in NIBE_STATUS_ACTIVE_STATES

    def _read_prio_state(self) -> str | None:
        """Read the pump priority (register 43086) and classify it.

        Covers nibe_heatpump mapped strings ("Hot Water", "Heat"), raw Modbus
        numbers (20, 30), and MyUplink enum text. Unknown or unavailable
        values return None so the pattern-based status reads stay untouched
        (MyUplink priority ids/texts are not fully documented).

        Returns:
            "hot_water", "heating", "other", or None when unavailable/unknown
        """
        prio_entity = self._entity_cache.get("prio")
        if not prio_entity:
            return None

        state = self.hass.states.get(prio_entity)
        if not state or state.state in ["unknown", "unavailable"]:
            return None

        prio_lower = state.state.lower()
        if prio_lower in NIBE_PRIO_HOT_WATER_STATES:
            return "hot_water"
        if prio_lower in NIBE_PRIO_HEATING_STATES:
            return "heating"
        return "other"

    async def get_power_consumption(self) -> float | None:
        """Get current power consumption of heat pump.

        Tries in order:
        1. Configured power sensor (most accurate)
        2. Estimation from supply temperature (least accurate)

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

    async def _calculate_multi_sensor_temperature(self, nibe_temp: float | None) -> float | None:
        """Calculate indoor temperature from NIBE sensor + additional sensors.

        Combines NIBE BT50 with additional room sensors for more accurate
        whole-house temperature reading.

        During startup, MQTT sensors may not be available yet. This method:
        - Logs a warning when configured sensors aren't ready
        - Uses only available sensors for calculation
        - Logs when all sensors become available

        Args:
            nibe_temp: Temperature from NIBE BT50, or None when there is no room sensor.
                A placeholder must NEVER be passed here - seeding the median with
                DEFAULT_INDOOR_TEMP would drag the combined reading toward the target and
                mask a real deviation.

        Returns:
            Combined temperature using the configured method (median/average), or None
            when no sensor produced a usable reading.
        """
        # Start with the NIBE sensor, if there is one
        temps = [nibe_temp] if nibe_temp is not None else []

        # Read additional sensors. _read_temperature normalises each to °C first - these
        # are arbitrary user-chosen room sensors, so a °F one is entirely plausible, and
        # the plausibility band below would otherwise reject every Fahrenheit reading.
        for entity_id in self._additional_indoor_sensors:
            temp = await self._read_temperature(entity_id)
            if temp is None:
                continue

            if INDOOR_SENSOR_PLAUSIBLE_MIN <= temp <= INDOOR_SENSOR_PLAUSIBLE_MAX:
                temps.append(temp)
            else:
                _LOGGER.warning(
                    "Ignoring out-of-range temperature from %s: %.1f°C (expected %.0f-%.0f°C)",
                    entity_id,
                    temp,
                    INDOOR_SENSOR_PLAUSIBLE_MIN,
                    INDOOR_SENSOR_PLAUSIBLE_MAX,
                )

        # Calculate combined temperature
        if not temps:
            # Neither BT50 nor any additional sensor produced a reading
            return None

        if len(temps) == 1:
            return temps[0]

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

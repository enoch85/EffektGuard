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

from ..const import CONF_NIBE_ENTITY

_LOGGER = logging.getLogger(__name__)


@dataclass
class NibeState:
    """Current state of NIBE heat pump.

    All temperatures in °C, power in kW.
    """

    outdoor_temp: float
    indoor_temp: float
    supply_temp: float
    return_temp: float | None
    degree_minutes: float
    current_offset: float
    is_heating: bool
    is_hot_water: bool
    timestamp: datetime


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
        indoor_temp = await self._read_entity_float(
            self._entity_cache.get("indoor_temp"), default=21.0
        )
        supply_temp = await self._read_entity_float(
            self._entity_cache.get("supply_temp"), default=35.0
        )
        return_temp = await self._read_entity_float(
            self._entity_cache.get("return_temp"), default=None
        )

        # Read degree minutes (GM/DM)
        degree_minutes = await self._read_entity_float(
            self._entity_cache.get("degree_minutes"), default=0.0
        )

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

        # Set value via number entity service
        try:
            await self.hass.services.async_call(
                "number",
                "set_value",
                {
                    "entity_id": offset_entity,
                    "value": offset,
                },
                blocking=True,
            )
            self._last_write = now
            _LOGGER.info("Set NIBE offset to %.2f°C", offset)
        except Exception as err:
            _LOGGER.error("Failed to set NIBE offset: %s", err)
            raise

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
        """
        registry = er.async_get(self.hass)

        # Search for NIBE entities
        # Patterns based on NIBE Myuplink integration entity naming
        patterns = {
            "outdoor_temp": ["bt1", "outdoor"],
            "indoor_temp": ["bt50", "room_temp", "indoor"],
            "supply_temp": ["bt25", "supply"],
            "return_temp": ["bt3", "return"],
            "degree_minutes": ["degree_minutes", "gm", "dm"],
            "offset": ["offset", "s1", "47011"],
            "compressor_status": ["compressor", "eb100"],
            "hot_water_status": ["hot_water", "dhw"],
        }

        # Find entities
        for entity in registry.entities.values():
            if not entity.entity_id.startswith("sensor.") and not entity.entity_id.startswith(
                "number."
            ):
                continue

            entity_id_lower = entity.entity_id.lower()

            # Match against patterns
            for key, patterns_list in patterns.items():
                if key not in self._entity_cache:
                    for pattern in patterns_list:
                        if pattern in entity_id_lower:
                            self._entity_cache[key] = entity.entity_id
                            _LOGGER.debug("Found NIBE entity %s: %s", key, entity.entity_id)
                            break

        _LOGGER.info("Discovered %d NIBE entities", len(self._entity_cache))

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

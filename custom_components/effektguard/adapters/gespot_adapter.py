"""GE-Spot adapter for reading electricity prices.

This adapter reads electricity price data from the GE-Spot integration,
which provides native 15-minute price intervals perfectly aligned with
Swedish Effektavgift requirements.

GE-Spot features:
- Native 15-minute granularity (96 periods per day)
- Multiple data sources with automatic fallback
- Today + tomorrow prices
- Multi-currency support (SEK, EUR, etc.) with automatic conversion
- Flexible display units (main unit or subunit: SEK/öre, EUR/cents)

This adapter:
- Uses prices exactly as configured in GE-Spot (respects user's display preference)
- GE-Spot handles all currency and unit conversions
- No unit conversion performed - user's choice is respected
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from ..const import (
    CONF_GESPOT_ENTITY,
    DAYTIME_END_HOUR,
    DAYTIME_START_HOUR,
    NATIVE_DAY_QUARTER_COUNTS,
    QUARTER_INTERVAL_MINUTES,
)
from ..utils.time_utils import QUARTERS_PER_HOUR

if TYPE_CHECKING:
    from ..models.types import AdapterConfigDict

_LOGGER = logging.getLogger(__name__)

QUARTER_DURATION: Final = timedelta(minutes=QUARTER_INTERVAL_MINUTES)


@dataclass
class QuarterPeriod:
    """Single 15-minute period with price data.

    All time information derived from start_time (timezone-aware datetime from GE-Spot).
    Prices are in whatever unit the user configured in GE-Spot.
    """

    start_time: datetime  # Timezone-aware datetime from GE-Spot
    price: float  # Price in user's configured GE-Spot unit (öre/kWh, SEK/kWh, etc.)

    @property
    def quarter_of_day(self) -> int:
        """Wall-clock quarter number 0-95 for this period.

        Ambiguous during the repeated hour when daylight saving time ends;
        use list positions or timestamps to identify a period, and this
        number only for display and tariff bookkeeping.
        """
        return (self.start_time.hour * QUARTERS_PER_HOUR) + (
            self.start_time.minute // QUARTER_INTERVAL_MINUTES
        )

    @property
    def hour(self) -> int:
        """Hour 0-23."""
        return self.start_time.hour

    @property
    def minute(self) -> int:
        """Minute 0, 15, 30, or 45."""
        return self.start_time.minute

    @property
    def is_daytime(self) -> bool:
        """True if 06:00-22:00 (full effect tariff weight)."""
        return DAYTIME_START_HOUR <= self.start_time.hour < DAYTIME_END_HOUR


@dataclass
class PriceData:
    """Price data for optimization.

    Contains native 15-minute periods from GE-Spot.
    """

    today: list[QuarterPeriod]  # Native intervals; DST days contain 92 or 100 periods
    tomorrow: list[QuarterPeriod]  # Native intervals for tomorrow (if available)
    has_tomorrow: bool

    @staticmethod
    def _index_containing(periods: list[QuarterPeriod], when: datetime) -> int | None:
        """Return the index of the interval in ``periods`` containing ``when``.

        Timestamp containment, rather than wall-clock quarter numbers,
        preserves both occurrences of the repeated hour when daylight saving
        time ends. Mixed naive/aware timestamps resolve to None rather than
        raising - price lookups must never crash pump control.
        """
        try:
            for index, period in enumerate(periods):
                if period.start_time <= when < period.start_time + QUARTER_DURATION:
                    return index
        except TypeError:
            return None
        return None

    def get_period_index(self, when: datetime) -> int | None:
        """Return the index into ``today`` of the interval containing ``when``."""
        return self._index_containing(self.today, when)

    def get_tomorrow_period_index(self, when: datetime) -> int | None:
        """Return the index into ``tomorrow`` of the interval containing ``when``."""
        return self._index_containing(self.tomorrow, when)

    def get_period(self, when: datetime) -> QuarterPeriod | None:
        """Return today's native price interval containing ``when``."""
        index = self.get_period_index(when)
        return self.today[index] if index is not None else None

    @property
    def current_price(self) -> float | None:
        """Get the current interval's price.

        Returns:
            Current price in user's configured GE-Spot unit, or None if not available
        """
        if not self.today:
            return None

        period = self.get_period(dt_util.now())
        return period.price if period else None


class GESpotAdapter:
    """Adapter for reading GE-Spot price entities."""

    def __init__(self, hass: HomeAssistant, config: "AdapterConfigDict"):
        """Initialize GE-Spot adapter.

        Args:
            hass: Home Assistant instance
            config: Configuration dictionary with entity IDs
        """
        self.hass = hass
        self.price_unit: str | None = None
        self._gespot_entity = config.get(CONF_GESPOT_ENTITY)

    async def get_prices(self) -> PriceData:
        """Read price data from GE-Spot entities.

        GE-Spot provides native 15-minute intervals (96 per day), which
        perfectly match Swedish Effektavgift quarterly measurement requirements.

        Returns:
            PriceData with today's and tomorrow's 15-minute prices

        Raises:
            ValueError: If GE-Spot entity is unavailable
        """
        if not self._gespot_entity:
            raise ValueError("No GE-Spot entity configured")

        # Read GE-Spot entity state
        state = self.hass.states.get(self._gespot_entity)
        if not state or state.state in ["unknown", "unavailable"]:
            raise ValueError(f"GE-Spot entity {self._gespot_entity} unavailable")

        # Debug: Log all available GE-Spot attributes
        _LOGGER.debug("GE-Spot entity %s attributes:", self._gespot_entity)
        for attr_name, attr_value in state.attributes.items():
            if isinstance(attr_value, list):
                _LOGGER.debug("  - %s: list with %d items", attr_name, len(attr_value))
            else:
                _LOGGER.debug("  - %s: %s", attr_name, attr_value)

        # Get GE-Spot unit configuration. Prices are used as-is for
        # ranking/classification (unit-invariant); the unit is exposed so the
        # savings calculator can convert to the main currency unit correctly.
        unit_of_measurement = state.attributes.get("unit_of_measurement", "unknown")
        self.price_unit = unit_of_measurement

        _LOGGER.debug(
            "GE-Spot configured unit: %s (using prices as-is, no conversion)",
            unit_of_measurement,
        )

        # Parse today's prices from attributes
        # GE-Spot stores prices in attributes as list of dicts with:
        # - time: datetime string (ISO format with timezone)
        # - value: float (in user's configured display unit)
        # We use values exactly as GE-Spot provides them - no conversion
        today_raw = state.attributes.get("today_interval_prices", [])
        tomorrow_raw = state.attributes.get("tomorrow_interval_prices", [])

        # Convert to QuarterPeriod objects (no unit conversion, use as-is)
        today_periods = self._parse_periods(today_raw)
        tomorrow_periods = self._parse_periods(tomorrow_raw) if tomorrow_raw else []

        # Log availability with dynamic unit
        if tomorrow_periods:
            _LOGGER.info(
                "GE-Spot prices loaded: %d today, %d tomorrow (extended optimization available) in %s",
                len(today_periods),
                len(tomorrow_periods),
                unit_of_measurement,
            )
        else:
            _LOGGER.info(
                "GE-Spot prices loaded: %d today only (tomorrow not yet available) in %s",
                len(today_periods),
                unit_of_measurement,
            )

        return PriceData(
            today=today_periods,
            tomorrow=tomorrow_periods,
            has_tomorrow=len(tomorrow_periods) > 0,
        )

    def _parse_periods(self, raw_prices: list[dict[str, Any]]) -> list[QuarterPeriod]:
        """Parse raw price data into QuarterPeriod objects.

        Args:
            raw_prices: List of dicts with 'time' (datetime string) and 'value' (float)

        Returns:
            The day's native intervals sorted by absolute instant: 96 on a
            normal day, 92/100 on DST days. Nothing is fabricated for gaps;
            consumers locate intervals by timestamp or list position.

        Note:
            Prices are used exactly as GE-Spot provides them.
            No unit conversion is performed - we respect user's configured unit.
        """
        periods = []

        for item in raw_prices:
            try:
                # Parse timestamp (GE-Spot uses 'time' key with timezone-aware datetime)
                time_str = item.get("time")
                if isinstance(time_str, str):
                    start_time = dt_util.parse_datetime(time_str)
                elif isinstance(time_str, datetime):
                    start_time = time_str
                else:
                    continue

                # Parse price - use exactly as provided by GE-Spot (no conversion)
                price = float(item.get("value", 0.0))

                # Create period with just datetime and price (all else derived)
                periods.append(QuarterPeriod(start_time=start_time, price=price))

            except (ValueError, TypeError, KeyError) as err:
                _LOGGER.warning("Failed to parse price period: %s", err)
                continue

        # Sort by absolute instant. A local wall-clock sort cannot distinguish
        # the two 02:xx delivery hours when daylight saving time ends.
        periods.sort(key=lambda p: p.start_time.timestamp())

        if periods and len(periods) not in NATIVE_DAY_QUARTER_COUNTS:
            _LOGGER.warning(
                "Received %d native price intervals; preserving available "
                "timestamps without fabricating prices",
                len(periods),
            )

        return periods

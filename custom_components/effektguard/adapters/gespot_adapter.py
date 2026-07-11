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
from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from ..const import CONF_GESPOT_ENTITY, DAYTIME_END_HOUR, DAYTIME_START_HOUR
from ..utils.time_utils import get_current_quarter

if TYPE_CHECKING:
    from ..models.types import AdapterConfigDict

_LOGGER = logging.getLogger(__name__)


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
        """Quarter number 0-95 for this period."""
        return (self.start_time.hour * 4) + (self.start_time.minute // 15)

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

    today: list[QuarterPeriod]  # 96 quarters for today
    tomorrow: list[QuarterPeriod]  # 96 quarters for tomorrow (if available)
    has_tomorrow: bool

    @property
    def current_price(self) -> float | None:
        """Get current quarter's price.

        Returns:
            Current price in user's configured GE-Spot unit, or None if not available
        """
        if not self.today:
            return None

        # Find current quarter (0-95)
        current_quarter = get_current_quarter()

        # Find matching quarter period
        for period in self.today:
            period_quarter = (period.hour * 4) + (period.minute // 15)
            if period_quarter == current_quarter:
                return period.price

        return None

    @property
    def current_quarter(self) -> int | None:
        """Get current quarter of day (0-95).

        Returns:
            Quarter number 0-95, or None if not available
        """
        now = dt_util.now()
        return (now.hour * 4) + (now.minute // 15)


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
            List of 96 QuarterPeriod objects for the day

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

        # Sort by quarter
        periods.sort(key=lambda p: p.quarter_of_day)

        # Validate we have 96 periods
        if len(periods) != 96:
            _LOGGER.warning(
                "Expected 96 quarterly periods, got %d. Filling missing periods.",
                len(periods),
            )
            periods = self._fill_missing_periods(periods)

        return periods

    def _fill_missing_periods(
        self,
        periods: list[QuarterPeriod],
    ) -> list[QuarterPeriod]:
        """Normalize a day to a dense list of 96 wall-clock quarters.

        The coordinator indexes today[current_quarter] POSITIONALLY, so the
        list must stay dense: a shorter list would shift every price after a
        gap by the gap's length. Non-96 days are real, not errors:

        - Spring DST (92 periods): the skipped hour's quarters never occur on
          the wall clock, so the filled entries are never indexed that day;
          they only preserve alignment.
        - Autumn DST (100 periods): the repeated hour yields duplicate
          quarter_of_day values; the first occurrence wins, so the repeated
          wall-clock hour reads the first pass's prices.
        - Mid-day data gaps: filled from the nearest earlier real price
          (prices are step functions; a neighbor beats a day average and
          does not skew cheap/expensive classification toward the mean).
        """
        if not periods:
            # Nothing real to anchor on: an empty day is honest (all callers
            # guard on truthiness); 96 fabricated prices are not.
            return []

        period_dict: dict[int, QuarterPeriod] = {}
        for period in periods:
            # Keep the first occurrence (autumn DST duplicates)
            period_dict.setdefault(period.quarter_of_day, period)

        if len(period_dict) < len(periods):
            _LOGGER.info(
                "Dropped %d duplicate quarter(s) (autumn DST repeated hour)",
                len(periods) - len(period_dict),
            )

        # Derive the day from the real data, not from now(): this method also
        # normalizes tomorrow's list, whose filled entries must carry
        # tomorrow's date.
        base_date = periods[0].start_time.replace(hour=0, minute=0, second=0, microsecond=0)

        complete_periods = []
        last_real: QuarterPeriod | None = None
        for quarter in range(96):
            period = period_dict.get(quarter)
            if period is not None:
                last_real = period
                complete_periods.append(period)
                continue
            # Forward-fill from the nearest earlier real price; a leading gap
            # takes the first real price of the day.
            neighbor = last_real if last_real is not None else periods[0]
            complete_periods.append(
                QuarterPeriod(
                    start_time=base_date.replace(hour=quarter // 4, minute=(quarter % 4) * 15),
                    price=neighbor.price,
                )
            )

        return complete_periods

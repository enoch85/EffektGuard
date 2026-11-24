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
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Final, NotRequired, TypedDict

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from ..const import (
    CONF_GESPOT_ENTITY,
    DAYTIME_END_HOUR,
    DAYTIME_START_HOUR,
    NATIVE_DAY_QUARTER_COUNTS,
    MINUTES_PER_QUARTER,
)
from ..utils.time_utils import QUARTERS_PER_HOUR, seconds_between

if TYPE_CHECKING:
    from ..models.types import AdapterConfigDict

_LOGGER = logging.getLogger(__name__)

QUARTER_DURATION: Final = timedelta(minutes=MINUTES_PER_QUARTER)


def _follows(earlier: "QuarterPeriod", later: "QuarterPeriod") -> bool:
    """`later` is the quarter right after `earlier`, in real time (a DST night included)."""
    return seconds_between(earlier.start_time, later.start_time) == (
        QUARTER_DURATION.total_seconds()
    )


def _up_to_first_gap(periods: list["QuarterPeriod"]) -> list["QuarterPeriod"]:
    """The quarters before the first one missing, in the order they come."""
    for index in range(1, len(periods)):
        if not _follows(periods[index - 1], periods[index]):
            return periods[:index]
    return periods


class RawPricePeriod(TypedDict):
    """One interval as GE-Spot publishes it in `today_interval_prices`.

    `time` is a timezone-aware datetime on a live GE-Spot and an ISO string only when Home
    Assistant has restored the attribute from JSON across a restart; `_parse_periods` handles
    both. `value` is the billed price; `raw_value` (NotRequired) is the pre-VAT market price,
    which nothing reads and which must never be used as the price. The TypedDict enforces
    nothing at runtime - `_parse_periods` validates every field and drops the interval if it
    cannot.
    """

    time: datetime | str
    value: float
    raw_value: NotRequired[float]


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
            self.start_time.minute // MINUTES_PER_QUARTER
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


def index_containing(periods: list[QuarterPeriod], when: datetime) -> int | None:
    """The index of the interval in ``periods`` containing ``when``, or None.

    Containment is decided on the absolute timeline, not by comparing the local datetimes: two
    aware datetimes with the same tzinfo compare by wall clock (PEP 495 ignores ``fold``), so in
    the second 02:xx hour of the autumn clock change 02:05 CET matched the interval that began at
    02:00 CEST, an hour earlier, and the first hour's prices were served for the second. A naive
    timestamp against an aware one resolves to None rather than raising - price lookups must
    never crash pump control.
    """
    try:
        for index, period in enumerate(periods):
            if 0 <= seconds_between(period.start_time, when) < QUARTER_DURATION.total_seconds():
                return index
    except TypeError:
        return None
    return None


@dataclass
class PriceData:
    """Price data for optimization.

    Contains native 15-minute periods from GE-Spot.
    """

    today: list[QuarterPeriod]  # Native intervals; DST days contain 92 or 100 periods
    tomorrow: list[QuarterPeriod]  # Native intervals for tomorrow (if available)
    has_tomorrow: bool

    def get_period_index(self, when: datetime) -> int | None:
        """Return the index into ``today`` of the interval containing ``when``."""
        return index_containing(self.today, when)

    def get_tomorrow_period_index(self, when: datetime) -> int | None:
        """Return the index into ``tomorrow`` of the interval containing ``when``."""
        return index_containing(self.tomorrow, when)

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
        # The holes already reported - see _report_gap
        self._reported_gaps: set[tuple[str, datetime]] = set()

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
        today_periods = self._contiguous(self._parse_periods(today_raw), "today")
        tomorrow_periods = (
            self._contiguous(self._parse_periods(tomorrow_raw), "tomorrow") if tomorrow_raw else []
        )
        # GE-Spot dates each quarter when it writes its state, and writes the new day a random
        # 0-30 s after midnight. Read before that, its today is the day that has ended and its
        # tomorrow the day that has begun.
        now = dt_util.utcnow()
        if (
            index_containing(today_periods, now) is None
            and index_containing(tomorrow_periods, now) is not None
        ):
            today_periods, tomorrow_periods = tomorrow_periods, []
        if (
            today_periods
            and tomorrow_periods
            and not _follows(today_periods[-1], tomorrow_periods[0])
        ):
            self._report_gap("tomorrow", tomorrow_periods[0], "does not join today's last quarter")
            tomorrow_periods = []

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

    def _contiguous(self, periods: list[QuarterPeriod], day: str) -> list[QuarterPeriod]:
        """A day's quarters up to the first one missing.

        Every reader of the lists counts quarters by position - the price forecast's "N quarters
        away", the cheap-run search, the volatility runs - so a hole (an interval dropped as
        unparseable, or hours GE-Spot did not publish) moved every later quarter earlier by the
        hole's length: with 10:30-14:45 missing, a price at 15:00 read as 30 minutes away at 10:00.
        After the hole the day has no prices, as before it is published, and the price layer
        abstains for those quarters; tomorrow, which no longer joins today, waits for midnight.
        """
        kept = _up_to_first_gap(periods)
        if len(kept) < len(periods):
            self._report_gap(day, periods[len(kept)], "does not follow the quarter before it")
        return kept

    def _report_gap(self, day: str, period: QuarterPeriod, why: str) -> None:
        """Say once per hole, not every cycle it stays, that prices stop there."""
        key = (day, period.start_time)
        if key in self._reported_gaps:
            return
        self._reported_gaps.add(key)
        _LOGGER.warning(
            "GE-Spot's %s prices have a hole: the interval at %s %s. Using the quarters before it "
            "only, rather than counting later prices too early.",
            day,
            period.start_time.isoformat(),
            why,
        )

    def _parse_periods(self, raw_prices: list[RawPricePeriod]) -> list[QuarterPeriod]:
        """Parse raw price data into QuarterPeriod objects.

        Args:
            raw_prices: GE-Spot's published intervals. See RawPricePeriod: `time` is a
                timezone-aware datetime object on a live GE-Spot, and an ISO string when
                Home Assistant has restored the attribute from JSON across a restart.

        Returns:
            The day's native intervals sorted by absolute instant: 96 on a
            normal day, 92/100 on DST days. Nothing is fabricated for gaps;
            get_prices uses a day only up to its first hole (_contiguous).

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
                    if start_time is None:
                        # parse_datetime signals invalid input with None, not
                        # an exception; letting it through would break the
                        # whole day at sort time instead of one interval here
                        _LOGGER.warning("Skipping price period with invalid time: %s", time_str)
                        continue
                elif isinstance(time_str, datetime):
                    start_time = time_str
                else:
                    continue

                # Used exactly as GE-Spot provides it (no conversion). Deliberately NOT
                # `.get("value", 0.0)`: a missing price is not a price of zero. Zero is a real
                # Nordic price and the cheapest one possible, so a defaulted interval outranks
                # every genuine quarter of the day and wins the most aggressive pre-heating the
                # price layer can command. KeyError below drops the interval instead.
                price = float(item["value"])
                # float() accepts "nan" and "inf", and they are no more a price than a missing
                # value: a NaN in the look-ahead made min() return NaN, which nothing equals, and the
                # price layer's search raised StopIteration - an exception the decision does not
                # catch - so every cycle failed until the NaN left the window.
                if not math.isfinite(price):
                    raise ValueError(f"not a finite price: {price}")

                # Create period with just datetime and price (all else derived)
                periods.append(QuarterPeriod(start_time=start_time, price=price))

            except (ValueError, TypeError, KeyError) as err:
                # Drop it rather than substitute anything. The day is then used up to this hole
                # (_contiguous); if every interval drops, the empty day trips the no-price-source
                # repair issue.
                _LOGGER.warning("Dropping unparseable price period (%s): %s", err, item)
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

"""GE-Spot adapter for reading Swedish electricity prices.

This adapter reads electricity price data from the GE-Spot integration,
which provides native 15-minute price intervals perfectly aligned with
Swedish Effektavgift requirements.

GE-Spot features:
- Native 15-minute granularity (96 periods per day)
- Multiple data sources with automatic fallback
- Today + tomorrow prices
- Matches Swedish electricity market structure
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from ..const import CONF_GESPOT_ENTITY

_LOGGER = logging.getLogger(__name__)


@dataclass
class QuarterPeriod:
    """Single 15-minute period with price data.

    Quarter numbering: 0-95 per day
    - Quarter 0: 00:00-00:15
    - Quarter 24: 06:00-06:15 (daytime starts)
    - Quarter 87: 21:45-22:00 (daytime ends)
    - Quarter 95: 23:45-00:00
    """

    quarter_of_day: int  # 0-95
    hour: int  # 0-23
    minute: int  # 0, 15, 30, 45
    price: float  # SEK/kWh
    is_daytime: bool  # True if 06:00-22:00 (full effect tariff weight)


@dataclass
class PriceData:
    """Price data for optimization.

    Contains native 15-minute periods from GE-Spot.
    """

    today: list[QuarterPeriod]  # 96 quarters for today
    tomorrow: list[QuarterPeriod]  # 96 quarters for tomorrow (if available)
    has_tomorrow: bool


class GESpotAdapter:
    """Adapter for reading GE-Spot price entities."""

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]):
        """Initialize GE-Spot adapter.

        Args:
            hass: Home Assistant instance
            config: Configuration dictionary with entity IDs
        """
        self.hass = hass
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

        # Parse today's prices from attributes
        # GE-Spot stores prices in attributes as list of dicts with:
        # - start: datetime
        # - price: float (SEK/kWh)
        today_raw = state.attributes.get("prices_today", [])
        tomorrow_raw = state.attributes.get("prices_tomorrow", [])

        # Convert to QuarterPeriod objects
        today_periods = self._parse_periods(today_raw)
        tomorrow_periods = self._parse_periods(tomorrow_raw) if tomorrow_raw else []

        # Log availability
        if tomorrow_periods:
            _LOGGER.info(
                "GE-Spot prices loaded: %d today, %d tomorrow (extended optimization available)",
                len(today_periods),
                len(tomorrow_periods),
            )
        else:
            _LOGGER.info(
                "GE-Spot prices loaded: %d today only (tomorrow not yet available)",
                len(today_periods),
            )

        return PriceData(
            today=today_periods,
            tomorrow=tomorrow_periods,
            has_tomorrow=len(tomorrow_periods) > 0,
        )

    def _parse_periods(self, raw_prices: list[dict[str, Any]]) -> list[QuarterPeriod]:
        """Parse raw price data into QuarterPeriod objects.

        Args:
            raw_prices: List of dicts with 'start' (datetime) and 'price' (float)

        Returns:
            List of 96 QuarterPeriod objects for the day
        """
        periods = []

        for item in raw_prices:
            try:
                # Parse timestamp
                start = item.get("start")
                if isinstance(start, str):
                    start = dt_util.parse_datetime(start)
                elif not isinstance(start, datetime):
                    continue

                # Calculate quarter of day (0-95)
                hour = start.hour
                minute = start.minute
                quarter_of_day = (hour * 4) + (minute // 15)

                # Parse price
                price = float(item.get("price", 0.0))

                # Determine if daytime (06:00-22:00 = full effect tariff weight)
                is_daytime = 6 <= hour < 22

                periods.append(
                    QuarterPeriod(
                        quarter_of_day=quarter_of_day,
                        hour=hour,
                        minute=minute,
                        price=price,
                        is_daytime=is_daytime,
                    )
                )

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
        """Fill missing periods to ensure 96 quarters per day.

        Args:
            periods: Partial list of periods

        Returns:
            Complete list of 96 periods (missing filled with average price)
        """
        # Calculate average price from available periods
        avg_price = sum(p.price for p in periods) / len(periods) if periods else 1.0

        # Create lookup dict
        period_dict = {p.quarter_of_day: p for p in periods}

        # Fill all 96 quarters
        complete_periods = []
        for quarter in range(96):
            if quarter in period_dict:
                complete_periods.append(period_dict[quarter])
            else:
                # Fill missing with average
                hour = quarter // 4
                minute = (quarter % 4) * 15
                complete_periods.append(
                    QuarterPeriod(
                        quarter_of_day=quarter,
                        hour=hour,
                        minute=minute,
                        price=avg_price,
                        is_daytime=(6 <= hour < 22),
                    )
                )

        return complete_periods

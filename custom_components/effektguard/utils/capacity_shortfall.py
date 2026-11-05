"""Where this pump runs short of capacity, learned from its own immersion heater.

A NIBE pump that cannot keep up with its own heating curve holds its degree minutes at the start of
its immersion heater: the heater switches in where DM crosses the pump's setting and works it back
up, so DM sits at that start for hours. A pump with capacity to spare only gets there after a cut
of ours - the slab drains, and the return to the curve drops DM into the heater - and the
compressor then works it back up within hours.

So the pump is short at the outdoor temperature of a stretch in which its degree minutes stayed
within SHORTFALL_NEAR_HEATER_START_DM of its heater start for SHORTFALL_DWELL_MINUTES, outside hot
water and at or above its own curve (register >= 0), beginning at least SHORTFALL_AFTER_CUT_HOURS
after the last cut the layers asked for and after this memory started. The warmest such
temperature of the last SHORTFALL_MEMORY_DAYS is where the pump runs short. The heater start is one
the pump has switched its heater on at, or the model's factory value - not the depth it held with
the heater off, which says only that its start is deeper (HeaterStart.started_degree_minutes).

While the coldest of the outdoor temperature now and the next SHORTFALL_FORECAST_HOURS forecast
hours is at or below it, heat held back cannot come back from the compressor: only from the
immersion heater, or not at all. The decision engine then cuts nothing below the pump's own curve,
and its heater guard stands down (DecisionEngine.calculate_decision).
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta

from homeassistant.util import dt as dt_util

from ..const import (
    SHORTFALL_AFTER_CUT_HOURS,
    SHORTFALL_DWELL_MINUTES,
    SHORTFALL_FORECAST_HOURS,
    SHORTFALL_MEMORY_DAYS,
    SHORTFALL_NEAR_HEATER_START_DM,
)
from .time_utils import seconds_between

_LOGGER = logging.getLogger(__name__)


class CapacityShortfall:
    """The outdoor temperature at which this pump has been seen short of capacity."""

    def __init__(self) -> None:
        """Start with nothing seen."""
        self._started: datetime | None = None
        self._last_cut: datetime | None = None
        self._near_since: datetime | None = None
        self._counted = False
        self._seen: deque[tuple[datetime, float]] = deque()

    @property
    def temperature(self) -> float | None:
        """The warmest outdoor temperature at which the pump ran short, or None."""
        return max((outdoor for _, outdoor in self._seen), default=None)

    def update(self, nibe_state, heater_start: float | None) -> None:
        """Record one reading of an automatic cycle; with no heater start known, nothing is near it."""
        now = nibe_state.timestamp
        if self._started is None:
            self._started = now
        register = float(getattr(nibe_state, "current_offset", 0.0) or 0.0)
        near = (
            heater_start is not None
            and not nibe_state.is_hot_water
            and register >= 0
            and float(nibe_state.degree_minutes) <= heater_start + SHORTFALL_NEAR_HEATER_START_DM
        )
        if not near:
            self._near_since = None
        elif self._near_since is None:
            self._near_since = now
            self._counted = False

        quiet = timedelta(hours=SHORTFALL_AFTER_CUT_HOURS).total_seconds()
        if (
            self._near_since is not None
            and not self._counted
            and seconds_between(self._near_since, now)
            >= timedelta(minutes=SHORTFALL_DWELL_MINUTES).total_seconds()
            and seconds_between(self._started, self._near_since) >= quiet
            and (
                self._last_cut is None or seconds_between(self._last_cut, self._near_since) >= quiet
            )
        ):
            self._seen.append((now, float(nibe_state.outdoor_temp)))
            self._counted = True
            _LOGGER.info(
                "The pump has held its degree minutes at its heater start (%.0f) for %d minutes at "
                "%.1f°C outdoors without a cut of ours: it runs short of capacity at or below "
                "%.1f°C.",
                heater_start,
                SHORTFALL_DWELL_MINUTES,
                nibe_state.outdoor_temp,
                self.temperature,
            )

        memory = timedelta(days=SHORTFALL_MEMORY_DAYS).total_seconds()
        while self._seen and seconds_between(self._seen[0][0], now) > memory:
            self._seen.popleft()

    def record_ask(self, now: datetime, offset: float) -> None:
        """The offset the layers asked for this cycle; a cut below the curve restarts the wait."""
        if offset < 0:
            self._last_cut = now

    def is_short(self, nibe_state, weather_data) -> bool:
        """Whether the coldest of now and the next forecast hours is as cold as a shortfall seen."""
        temperature = self.temperature
        if temperature is None:
            return False
        hours = getattr(weather_data, "forecast_hours", None) or []
        coldest = min(
            [float(nibe_state.outdoor_temp)]
            + [hour.temperature for hour in hours[:SHORTFALL_FORECAST_HOURS]]
        )
        return coldest <= temperature

    def to_dict(self) -> dict:
        """The shortfalls seen, for storage across restarts."""
        return {"seen": [[at.isoformat(), outdoor] for at, outdoor in self._seen]}

    def restore(self, data: dict) -> None:
        """Take back the shortfalls a previous run saw; a malformed entry is skipped."""
        seen = list(self._seen)
        for entry in (data or {}).get("seen", []):
            try:
                at = dt_util.parse_datetime(entry[0])
                outdoor = float(entry[1])
            except (TypeError, ValueError, IndexError):
                continue
            if at is not None and at.tzinfo is not None:
                seen.append((at, outdoor))
        self._seen = deque(sorted(seen, key=lambda entry: entry[0].timestamp()))

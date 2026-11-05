"""Where this pump starts its immersion heater, learned from the heater's own power reading.

A NIBE pump starts its immersion heater for space heating where its degree minutes cross its own
setting (menu 4.9.3: "start addition", or "start compressor" less "start diff additional heat"),
and an installer can move that far from the factory value the model profile holds: the owner's
F750 starts at -1450, its profile says -700. The pump publishes the heater's power (43084 on the
F-series, 31028 on the S-series), so the start can be seen instead of set, in readings taken while
the compressor heats the house, each no more than HEATER_START_MAX_GAP_MINUTES after the one before:

- One with the heater off, the next with it on: the heater switched on between the two, so the
  second reading's degree minutes are at or past its start.
- Two with the heater off: the start is deeper than both. The guard keeps a room at or above its
  floor short of the start it assumes, so a pump whose heater starts deeper than the factory value
  may seldom get to switch it on; the depth it holds with the heater off shows the start is deeper.

Hot water is left out: the pump runs the same heater for the tank at any degree minutes, and so is
a reading whose priority cannot say which it serves. The pump's heater start is the shallowest of
the last HEATER_START_MEMORY starts seen; before the first, the deepest degree minutes it has held
with the heater off; never shallower than the model's factory value (HeaterStart.degree_minutes).
The capacity shortfall reads only a start the pump has switched its heater on at
(HeaterStart.started_degree_minutes).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from datetime import datetime, timedelta

from homeassistant.util import dt as dt_util

from ..const import HEATER_START_MAX_GAP_MINUTES, HEATER_START_MEMORY
from .time_utils import seconds_between

_LOGGER = logging.getLogger(__name__)


class HeaterStart:
    """The degree minutes at which this pump has been seen starting its immersion heater."""

    def __init__(self) -> None:
        """Start with nothing seen."""
        self._last: tuple[datetime, bool, float] | None = None
        self._starts: deque[tuple[datetime, float]] = deque(maxlen=HEATER_START_MEMORY)
        self._held_off: float | None = None

    @property
    def started(self) -> float | None:
        """The shallowest degree minutes of the starts remembered, or None before the first."""
        return max((degree_minutes for _, degree_minutes in self._starts), default=None)

    @property
    def seen(self) -> float | None:
        """Where the heater starts as this pump has shown it, or None before it has shown anything.

        The shallowest of the starts remembered; before the first, the deepest degree minutes held
        with the heater off.
        """
        started = self.started
        return self._held_off if started is None else started

    def degree_minutes(self, factory: float) -> float:
        """Where the heater starts: as the pump has shown it, never shallower than `factory`.

        A start seen at or above the factory value says nothing the profile does not, and a heater
        switched on for anything other than its degree minutes cannot move the start shallower
        than the profile's.
        """
        seen = self.seen
        return factory if seen is None else min(factory, seen)

    def started_degree_minutes(self, factory: float) -> float | None:
        """Where the heater starts as far as the pump has switched it on, never shallower than
        `factory`; None while the pump has held deeper than `factory` with its heater off but not yet
        switched it on - the factory value is not its start then, and its own is not known yet.

        This is the start a heater holds degree minutes at. The depth held with the heater off only
        says the start is deeper, and held there the pump is not held by its heater.
        """
        started = self.started
        if started is not None:
            return min(factory, started)
        if self._held_off is not None and self._held_off < factory:
            return None
        return factory

    def update(self, nibe_state) -> None:
        """Record one reading of the heater's power while the compressor heats the house."""
        now = nibe_state.timestamp
        last = self._last
        if last is not None and seconds_between(last[0], now) <= 0:
            return
        power = getattr(nibe_state, "heater_power_kw", None)
        if (
            not isinstance(power, (int, float))
            or nibe_state.is_hot_water is not False
            or not nibe_state.is_heating
        ):
            self._last = None
            return
        on = power > 0
        degree_minutes = float(nibe_state.degree_minutes)
        self._last = (now, on, degree_minutes)
        if (
            last is None
            or last[1]
            or seconds_between(last[0], now)
            > timedelta(minutes=HEATER_START_MAX_GAP_MINUTES).total_seconds()
        ):
            return

        if not on:
            held = max(last[2], degree_minutes)
            if self._held_off is None or held < self._held_off:
                self._held_off = held
            return

        self._starts.append((now, degree_minutes))
        _LOGGER.info(
            "The pump switched its immersion heater on at DM %.0f while heating the house; the "
            "shallowest start remembered is %.0f.",
            degree_minutes,
            self.seen,
        )

    def to_dict(self) -> dict:
        """What the pump has shown, for storage across restarts."""
        return {
            "starts": [[at.isoformat(), degree_minutes] for at, degree_minutes in self._starts],
            "held_off": self._held_off,
        }

    def restore(self, data: dict) -> None:
        """Take back what a previous run saw; a malformed entry is skipped."""
        data = data or {}
        starts = list(self._starts)
        for entry in data.get("starts", []):
            try:
                at = dt_util.parse_datetime(entry[0])
                degree_minutes = float(entry[1])
            except (TypeError, ValueError, IndexError):
                continue
            if at is not None and at.tzinfo is not None and math.isfinite(degree_minutes):
                starts.append((at, degree_minutes))
        starts.sort(key=lambda entry: entry[0].timestamp())
        self._starts = deque(starts, maxlen=HEATER_START_MEMORY)
        try:
            held_off = float(data.get("held_off"))
        except (TypeError, ValueError):
            return
        if math.isfinite(held_off) and (self._held_off is None or held_off < self._held_off):
            self._held_off = held_off

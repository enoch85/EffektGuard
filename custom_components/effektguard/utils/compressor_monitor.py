"""Compressor frequency (Hz) monitoring and health tracking.

This module tracks NIBE compressor frequency over time to:
1. Detect sustained high-Hz operation (health risk)
2. Identify undersized systems (continuous high Hz)
3. Diagnose thermal stress patterns
4. Provide diagnostic data for troubleshooting

CompressorHealthMonitor is data collection only. CompressorSaturation is the one control input: the
decision engine's wear guard reads it.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median
from typing import TypedDict

from homeassistant.util import dt as dt_util

from ..const import (
    ANTI_WINDUP_DM_DROPPING_RATE,
    COMPRESSOR_HZ_MAX,
    COMPRESSOR_HZ_MIN,
    COMPRESSOR_RISK_ELEVATED,
    COMPRESSOR_RISK_HIGH,
    COMPRESSOR_RISK_NOTABLE,
    COMPRESSOR_RISK_OK,
    COMPRESSOR_RISK_WATCH,
    COMPRESSOR_SATURATION_BAND_HZ,
    COMPRESSOR_SATURATION_MEMORY_HOURS,
    DM_RATE_WINDOW_MINUTES,
    NIBE_COMPRESSOR_ACTIVE_HZ_THRESHOLD,
)
from .degree_minutes import dm_rate_per_hour

_LOGGER = logging.getLogger(__name__)


class CompressorDiagnosticsDict(TypedDict):
    """Diagnostic information for compressor troubleshooting."""

    current_hz: int
    avg_1h: float
    avg_6h: float
    avg_24h: float
    max_hz_1h: int
    max_hz_24h: int
    time_above_80hz_minutes: float
    time_above_100hz_minutes: float
    risk_level: str
    risk_reason: str
    samples_count: int


@dataclass
class CompressorStats:
    """Compressor frequency statistics, for diagnostics.

    What the frequency reads on the owner's F750 (production statistics, Oct 2025 - Oct 2026): 0 or
    1 Hz while stopped, never steady below 20 Hz while running, about 80 Hz through a hot-water
    charge, 75-90 Hz flat out on normal ventilation (91 % of its saturation-like winter hours), and
    up to COMPRESSOR_HZ_MAX with the ventilation boosted - held at 120 Hz for 8 whole hours.
    docs/research cites no frequency or duration that harms a NIBE compressor.
    """

    current_hz: int
    """Current compressor frequency in Hz"""

    avg_1h: float
    """Rolling 1-hour average Hz"""

    avg_6h: float
    """Rolling 6-hour average Hz"""

    avg_24h: float
    """Rolling 24-hour average Hz"""

    time_above_80hz: timedelta
    """Continuous time at or above 80 Hz"""

    time_above_100hz: timedelta
    """Continuous time at or above 100 Hz"""

    max_hz_1h: int
    """Maximum Hz in last 1 hour"""

    max_hz_24h: int
    """Maximum Hz in last 24 hours"""

    samples_count: int
    """Number of samples in history"""


class CompressorHealthMonitor:
    """Monitor compressor frequency and track health indicators.

    Maintains rolling history of Hz readings and calculates statistics.
    Does NOT make heating decisions - only provides diagnostic data.
    """

    def __init__(self, max_history_hours: int = 24):
        """Initialize compressor monitor.

        Args:
            max_history_hours: Maximum hours of history to retain (default: 24)
        """
        # Store (timestamp, hz) tuples
        # At 1-minute intervals, 24 hours = 1440 samples
        max_samples = max_history_hours * 60
        self.hz_history: deque[tuple[datetime, int]] = deque(maxlen=max_samples)

        # Track continuous high-Hz periods
        self.high_hz_start: datetime | None = None  # Time when Hz first exceeded 80
        self.very_high_hz_start: datetime | None = None  # Time when Hz first exceeded 100

        _LOGGER.debug(
            "CompressorHealthMonitor initialized (max %d hours, %d samples)",
            max_history_hours,
            max_samples,
        )

    def update(
        self, hz: int, timestamp: datetime | None = None, heating_mode: str = "space"
    ) -> CompressorStats:
        """Update monitor with latest Hz reading.

        Args:
            hz: Current compressor frequency in Hz
            timestamp: Sample timestamp (defaults to now)
            heating_mode: "space" for space heating, "dhw" for DHW heating
                Affects logging behavior (DHW 80Hz is normal, space 80Hz is elevated)

        Returns:
            CompressorStats with current and historical data
        """
        if timestamp is None:
            timestamp = dt_util.now()

        # Clamp and warning message both bound by COMPRESSOR_HZ_MAX, so they cannot disagree.
        if hz < 0 or hz > COMPRESSOR_HZ_MAX:
            _LOGGER.warning(
                "Invalid compressor Hz reading: %d (expected 0-%d range)", hz, COMPRESSOR_HZ_MAX
            )
            hz = max(0, min(hz, COMPRESSOR_HZ_MAX))

        # Add to history
        self.hz_history.append((timestamp, hz))

        # Calculate rolling averages
        avg_1h = self._calculate_average(timestamp, timedelta(hours=1))
        avg_6h = self._calculate_average(timestamp, timedelta(hours=6))
        avg_24h = self._calculate_average(timestamp, timedelta(hours=24))

        # Calculate maximums
        max_hz_1h = self._calculate_maximum(timestamp, timedelta(hours=1))
        max_hz_24h = self._calculate_maximum(timestamp, timedelta(hours=24))

        # Track continuous high-Hz time
        time_above_80hz = self._track_continuous_time(hz, 80, timestamp)
        time_above_100hz = self._track_continuous_time(hz, 100, timestamp)

        stats = CompressorStats(
            current_hz=hz,
            avg_1h=avg_1h,
            avg_6h=avg_6h,
            avg_24h=avg_24h,
            time_above_80hz=time_above_80hz,
            time_above_100hz=time_above_100hz,
            max_hz_1h=max_hz_1h,
            max_hz_24h=max_hz_24h,
            samples_count=len(self.hz_history),
        )

        # Log significant events with heating mode context
        self._log_events(stats, heating_mode)

        return stats

    def _calculate_average(self, current_time: datetime, window: timedelta) -> float:
        """Calculate average Hz over time window.

        Args:
            current_time: Current timestamp
            window: Time window to average over

        Returns:
            Average Hz, or 0.0 if no samples
        """
        cutoff_time = current_time - window
        samples = [hz for ts, hz in self.hz_history if ts >= cutoff_time]

        if not samples:
            return 0.0

        return sum(samples) / len(samples)

    def _calculate_maximum(self, current_time: datetime, window: timedelta) -> int:
        """Calculate maximum Hz over time window.

        Args:
            current_time: Current timestamp
            window: Time window to check

        Returns:
            Maximum Hz, or 0 if no samples
        """
        cutoff_time = current_time - window
        samples = [hz for ts, hz in self.hz_history if ts >= cutoff_time]

        if not samples:
            return 0

        return max(samples)

    def _track_continuous_time(
        self, current_hz: int, threshold: int, timestamp: datetime
    ) -> timedelta:
        """Track continuous time above Hz threshold.

        Args:
            current_hz: Current Hz reading
            threshold: Hz threshold (80 or 100)
            timestamp: Current timestamp

        Returns:
            Time continuously above threshold
        """
        if threshold == 80:
            start_marker = self.high_hz_start
        elif threshold == 100:
            start_marker = self.very_high_hz_start
        else:
            return timedelta(0)

        if current_hz >= threshold:
            # Currently above threshold
            if start_marker is None:
                # Just crossed threshold
                if threshold == 80:
                    self.high_hz_start = timestamp
                else:
                    self.very_high_hz_start = timestamp
                return timedelta(0)
            else:
                # Still above threshold
                return timestamp - start_marker
        else:
            # Below threshold - reset marker
            if threshold == 80:
                self.high_hz_start = None
            else:
                self.very_high_hz_start = None
            return timedelta(0)

    def _log_events(self, stats: CompressorStats, heating_mode: str = "space") -> None:
        """Log significant compressor events with context awareness.

        Args:
            stats: Current compressor statistics
            heating_mode: "space" for space heating, "dhw" for DHW heating
                Different modes have different normal Hz ranges:
                - Space heating (25-35°C): 80 Hz elevated
                - DHW heating (50°C): 80 Hz normal, 95 Hz elevated
        """
        # Normalize mode to lowercase for consistent comparison
        mode = heating_mode.lower() if heating_mode else "space"

        # Always log >100 Hz (elevated regardless of mode)
        if stats.current_hz >= 100 and stats.time_above_100hz == timedelta(0):
            _LOGGER.warning(
                "Compressor operating at %d Hz (mode: %s)",
                stats.current_hz,
                mode,
            )
        # Context-aware logging for elevated operation
        elif mode == "dhw":
            # DHW: 80-95 Hz is normal operation, log at DEBUG level
            if stats.current_hz >= 80 and stats.time_above_80hz == timedelta(0):
                _LOGGER.debug(
                    "Compressor at %d Hz for DHW heating (normal operation, target 50°C)",
                    stats.current_hz,
                )
        else:  # space heating
            # Space heating: 80+ Hz is elevated, log at INFO level
            if stats.current_hz >= 80 and stats.time_above_80hz == timedelta(0):
                _LOGGER.info(
                    "Compressor at %d Hz for space heating",
                    stats.current_hz,
                )

        # Sustained warnings - context-aware durations
        if stats.time_above_100hz >= timedelta(minutes=15):
            # Always notable (any mode)
            minutes = stats.time_above_100hz.total_seconds() / 60
            _LOGGER.warning(
                "Compressor at >100 Hz for %.0f minutes "
                "(current: %d Hz, mode: %s, 1h avg: %.0f Hz)",
                minutes,
                stats.current_hz,
                mode,
                stats.avg_1h,
            )
        elif mode == "dhw":
            # DHW: Warn if >95 Hz for >30 minutes (unusual for DHW cycle)
            if stats.time_above_80hz >= timedelta(minutes=30) and stats.current_hz >= 95:
                minutes = stats.time_above_80hz.total_seconds() / 60
                _LOGGER.info(
                    "DHW compressor at %d Hz for %.0f minutes",
                    stats.current_hz,
                    minutes,
                )
        else:  # space heating
            # Space: Warn if >80 Hz for >2 hours (system running hard)
            if stats.time_above_80hz >= timedelta(hours=2):
                hours = stats.time_above_80hz.total_seconds() / 3600
                _LOGGER.info(
                    "Space heating compressor at >80 Hz for %.1f hours "
                    "(current: %d Hz, 6h avg: %.0f Hz)",
                    hours,
                    stats.current_hz,
                    stats.avg_6h,
                )

    def assess_risk(self, stats: CompressorStats) -> tuple[str, str]:
        """Assess compressor operating status.

        Status levels based on NIBE forum research and OEM recommendations:
        - HIGH: Extended high-frequency operation
        - ELEVATED: Sustained elevated operation
        - NOTABLE: Above-average operation
        - WATCH: Temporary high demand
        - OK: Normal operation

        Args:
            stats: Current compressor statistics

        Returns:
            (status_level, reason) tuple
        """
        # HIGH: Above 100 Hz for more than 15 minutes
        # This indicates compressor at maximum capacity for extended period
        if stats.time_above_100hz > timedelta(minutes=15):
            minutes = stats.time_above_100hz.total_seconds() / 60
            return (
                COMPRESSOR_RISK_HIGH,
                f"Compressor at {stats.current_hz} Hz for {minutes:.0f} min",
            )

        # ELEVATED: Above 80 Hz for more than 2 hours
        # Continuous high-Hz operation
        if stats.time_above_80hz > timedelta(hours=2):
            hours = stats.time_above_80hz.total_seconds() / 3600
            return (
                COMPRESSOR_RISK_ELEVATED,
                f"Sustained operation ({stats.avg_1h:.0f} Hz avg) for {hours:.1f}h",
            )

        # NOTABLE: 6-hour average above 70 Hz
        # System consistently operating at high capacity
        if stats.avg_6h > 70:
            return (
                COMPRESSOR_RISK_NOTABLE,
                f"6-hour average: {stats.avg_6h:.0f} Hz",
            )

        # WATCH: 1-hour average above 75 Hz
        # Temporary high demand (cold snap, recovery from setback)
        # Normal if brief, concerning if sustained
        if stats.avg_1h > 75:
            return (
                COMPRESSOR_RISK_WATCH,
                f"High demand period (1h avg: {stats.avg_1h:.0f} Hz, current: {stats.current_hz} Hz)",
            )

        # OK: Normal operation
        if stats.current_hz > 0:
            return (
                COMPRESSOR_RISK_OK,
                f"Normal operation ({stats.current_hz} Hz, 6h avg: {stats.avg_6h:.0f} Hz)",
            )
        else:
            return (COMPRESSOR_RISK_OK, "Compressor idle")

    def get_diagnostic_info(self, stats: CompressorStats) -> CompressorDiagnosticsDict:
        """Get diagnostic information for troubleshooting.

        Args:
            stats: Current compressor statistics

        Returns:
            Dictionary with diagnostic data
        """
        risk_level, risk_reason = self.assess_risk(stats)

        return {
            "current_hz": stats.current_hz,
            "avg_1h": round(stats.avg_1h, 1),
            "avg_6h": round(stats.avg_6h, 1),
            "avg_24h": round(stats.avg_24h, 1),
            "max_hz_1h": stats.max_hz_1h,
            "max_hz_24h": stats.max_hz_24h,
            "time_above_80hz_minutes": stats.time_above_80hz.total_seconds() / 60,
            "time_above_100hz_minutes": stats.time_above_100hz.total_seconds() / 60,
            "risk_level": risk_level,
            "risk_reason": risk_reason,
            "samples_count": stats.samples_count,
        }


class CompressorSaturation:
    """Whether the compressor has run out of frequency to give - the trigger of the wear guard.

    A NIBE inverter answers falling degree minutes by raising its frequency, until it can rise no
    further. The F750, F730 and F1155/F1255 publish how far that is - their current maximum
    frequency, register 43123 - and the S1155/S1255 theirs (31043). It moves: the owner's F750
    puts it anywhere from 70 to 120 Hz, often several times an hour. With that maximum the
    compressor is saturated when degree minutes fall at ANTI_WINDUP_DM_DROPPING_RATE or faster
    through a window in which every reading sits within COMPRESSOR_SATURATION_BAND_HZ of the maximum
    published with it, and stays saturated while the frequency stays at that maximum. Below its own
    maximum it has frequency to give.

    A pump that publishes no maximum (an F2040 behind its SMO or VVM) is judged on a ceiling learned
    from its frequency: degree minutes falling at ANTI_WINDUP_DM_DROPPING_RATE or faster through a
    window in which every reading sits at the highest level the frequency has held flat, within the
    band through a whole window, in the last COMPRESSOR_SATURATION_MEMORY_HOURS - or, whatever the
    memory holds, at COMPRESSOR_HZ_MAX without gaining on its degree minutes. It stays saturated
    while the frequency stays at that ceiling and stops climbing. A window that ramps - into or out
    of a hot-water run, say - is no ceiling. The ceiling has to be earned. It is remembered per
    airflow state, because enhanced ventilation lifts an exhaust-air pump's ceiling, and only counts
    once this run or the memory has shown the frequency span more than the band: a frozen reading or
    a run that began at the ceiling proves nothing. A level rising past the highest one an hour or
    more old is climbing, not stuck - without such a level there is no evidence yet. A learned
    ceiling goes stale wherever the real one moves: a level held hours ago hides the saturation at
    a lower one.

    Either way a compressor within the band of COMPRESSOR_HZ_MIN is never at its limit. Hot water
    pauses the run without ending it, and so does a reading that cannot say whether the compressor
    serves the tank, since a charge holds a high frequency while degree minutes fall, as saturation
    does: a verdict stands until the readings after the pause confirm or release it, and nothing in
    the pause counts. A stop (the pump reports 1 Hz, NIBE_COMPRESSOR_ACTIVE_HZ_THRESHOLD, while
    stopped), a dropout, an airflow change, the clock going back or a gap longer than half a window
    since the last reading ends the run.
    """

    def __init__(self) -> None:
        """Start with no history."""
        # (time, Hz, DM, run part, the maximum the pump published with it)
        self._window: deque[tuple[datetime, int, float, int, int | None]] = deque()
        # (time, level, whether the window was flat) per airflow state
        self._levels: dict[bool, deque[tuple[datetime, float, bool]]] = {
            False: deque(),
            True: deque(),
        }
        self._run_low: float | None = None
        self._seen: datetime | None = None
        self._part = 0
        self._paused = False
        self._enhanced = False
        self.saturated = False
        self.level: float | None = None
        self.ceiling: float | None = None

    def update(self, nibe_state, *, hot_water: bool | None, airflow_enhanced: bool) -> bool:
        """Record one reading of a pump-driving cycle and return whether it is saturated."""
        now = nibe_state.timestamp
        window = timedelta(minutes=DM_RATE_WINDOW_MINUTES)
        band = COMPRESSOR_SATURATION_BAND_HZ
        hz = nibe_state.compressor_hz
        self.level = self.ceiling = None
        stopped = hz is None or hz <= NIBE_COMPRESSOR_ACTIVE_HZ_THRESHOLD
        if (
            stopped
            or airflow_enhanced != self._enhanced
            or (self._seen and not timedelta(0) <= now - self._seen <= window / 2)
        ):
            self._end_run()
            self._enhanced = airflow_enhanced
            if stopped:
                return False
        self._seen = now
        if hot_water or hot_water is None:
            self._paused = True
            return self.saturated
        if self._paused:
            self._paused = False
            self._part += 1

        self._window.append(
            (now, hz, nibe_state.degree_minutes, self._part, nibe_state.compressor_hz_max)
        )
        while now - self._window[0][0] >= window:
            self._window.popleft()
        memory = self._levels[airflow_enhanced]
        while memory and now - memory[0][0] > timedelta(hours=COMPRESSOR_SATURATION_MEMORY_HOURS):
            memory.popleft()
        if now - self._window[0][0] < window / 2:
            return self.saturated  # too little evidence yet: only a verdict carried over hot water

        readings = [held for _, held, _, _, _ in self._window]
        level = median(readings)
        published = [top for *_, top in self._window]
        if None not in published:
            self.level, self.ceiling = level, published[-1]
            short = [top - held for held, top in zip(readings, published)]
            if self.saturated:
                self.saturated = level > COMPRESSOR_HZ_MIN + band and median(short) <= band
                return self.saturated
            rate = dm_rate_per_hour(
                ((at, dm, part) for at, _, dm, part, _ in self._window), window / 2
            )
            self.saturated = (
                level > COMPRESSOR_HZ_MIN + band
                and max(short) <= band
                and rate is not None
                and rate <= ANTI_WINDUP_DM_DROPPING_RATE
            )
            return self.saturated
        self._run_low = level if self._run_low is None else min(self._run_low, level)
        lowest = min([held for _, held, _ in memory] + [self._run_low])
        held_flat = [held for _, held, flat in memory if flat]
        ceiling = max(held_flat) if held_flat and max(held_flat) - lowest > band else None
        hour_old = max((held for at, held, _ in memory if now - at >= 2 * window), default=None)
        memory.append((now, level, max(readings) - min(readings) <= band))
        self.level, self.ceiling = level, ceiling

        at_max = all(held >= COMPRESSOR_HZ_MAX - band for held in readings)
        steady = hour_old is not None and level <= hour_old + band / 2
        if self.saturated:
            self.saturated = at_max or (
                ceiling is not None
                and abs(level - ceiling) <= band
                and (hour_old is None or steady)
            )
            return self.saturated
        rate = dm_rate_per_hour(((at, dm, part) for at, _, dm, part, _ in self._window), window / 2)
        at_ceiling = (
            ceiling is not None
            and level > COMPRESSOR_HZ_MIN + band
            and all(abs(held - ceiling) <= band for held in readings)
        )
        self.saturated = rate is not None and (
            (at_max and rate <= 0)
            or (at_ceiling and steady and rate <= ANTI_WINDUP_DM_DROPPING_RATE)
        )
        return self.saturated

    def _end_run(self) -> None:
        """A stop, a dropout, an airflow change or a gap: nothing in the run carries over."""
        self._window.clear()
        self._run_low = None
        self._seen = None
        self._paused = False
        self.saturated = False

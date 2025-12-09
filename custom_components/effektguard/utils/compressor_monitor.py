"""Compressor frequency (Hz) monitoring and health tracking.

This module tracks NIBE compressor frequency over time to:
1. Detect sustained high-Hz operation (health risk)
2. Identify undersized systems (continuous high Hz)
3. Diagnose thermal stress patterns
4. Provide diagnostic data for troubleshooting

Data collection only - decision-making logic can be added later.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypedDict

from homeassistant.util import dt as dt_util

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
    """Compressor frequency statistics.

    Tracks current and historical Hz readings to identify health risks.

    NIBE F750/F2040 Compressor Frequency Ranges:
    - Idle: 20-30 Hz (pump circulation only)
    - Normal: 40-70 Hz (typical heating)
    - High: 70-90 Hz (cold weather or high demand)
    - Very High: 90-110 Hz (peak capacity)
    - Emergency: 110-120 Hz (absolute maximum, short duration only)

    Sustained operation above 80 Hz indicates thermal stress.
    Continuous operation above 100 Hz risks compressor damage.
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
    """Continuous time above 80 Hz (high stress threshold)"""

    time_above_100hz: timedelta
    """Continuous time above 100 Hz (critical stress threshold)"""

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

        # Validate Hz reading
        if hz < 0 or hz > 150:
            _LOGGER.warning("Invalid compressor Hz reading: %d (expected 0-120 range)", hz)
            hz = max(0, min(hz, 150))

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
                "HIGH",
                f"Compressor at {stats.current_hz} Hz for {minutes:.0f} min",
            )

        # ELEVATED: Above 80 Hz for more than 2 hours
        # Continuous high-Hz operation
        if stats.time_above_80hz > timedelta(hours=2):
            hours = stats.time_above_80hz.total_seconds() / 3600
            return (
                "ELEVATED",
                f"Sustained operation ({stats.avg_1h:.0f} Hz avg) for {hours:.1f}h",
            )

        # NOTABLE: 6-hour average above 70 Hz
        # System consistently operating at high capacity
        if stats.avg_6h > 70:
            return (
                "NOTABLE",
                f"6-hour average: {stats.avg_6h:.0f} Hz",
            )

        # WATCH: 1-hour average above 75 Hz
        # Temporary high demand (cold snap, recovery from setback)
        # Normal if brief, concerning if sustained
        if stats.avg_1h > 75:
            return (
                "WATCH",
                f"High demand period (1h avg: {stats.avg_1h:.0f} Hz, current: {stats.current_hz} Hz)",
            )

        # OK: Normal operation
        if stats.current_hz > 0:
            return (
                "OK",
                f"Normal operation ({stats.current_hz} Hz, 6h avg: {stats.avg_6h:.0f} Hz)",
            )
        else:
            return ("OK", "Compressor idle")

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

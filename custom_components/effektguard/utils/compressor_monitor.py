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
from typing import Any

_LOGGER = logging.getLogger(__name__)


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

    def update(self, hz: int, timestamp: datetime | None = None) -> CompressorStats:
        """Update monitor with latest Hz reading.

        Args:
            hz: Current compressor frequency in Hz
            timestamp: Sample timestamp (defaults to now)

        Returns:
            CompressorStats with current and historical data
        """
        if timestamp is None:
            timestamp = datetime.now()

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

        # Log significant events
        self._log_events(stats)

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

    def _log_events(self, stats: CompressorStats) -> None:
        """Log significant compressor events.

        Args:
            stats: Current compressor statistics
        """
        # Log when crossing into high-stress zones
        if stats.current_hz >= 100 and stats.time_above_100hz == timedelta(0):
            _LOGGER.warning(
                "Compressor entered VERY HIGH zone: %d Hz (>100 Hz sustained can damage compressor)",
                stats.current_hz,
            )
        elif stats.current_hz >= 80 and stats.time_above_80hz == timedelta(0):
            _LOGGER.info(
                "Compressor entered HIGH zone: %d Hz (monitor for sustained operation)",
                stats.current_hz,
            )

        # Log sustained high-Hz warnings
        if stats.time_above_100hz >= timedelta(minutes=15):
            minutes = stats.time_above_100hz.total_seconds() / 60
            _LOGGER.error(
                "COMPRESSOR STRESS CRITICAL: >100 Hz for %.0f minutes (current: %d Hz, 1h avg: %.0f Hz)",
                minutes,
                stats.current_hz,
                stats.avg_1h,
            )
        elif stats.time_above_80hz >= timedelta(hours=2):
            hours = stats.time_above_80hz.total_seconds() / 3600
            _LOGGER.warning(
                "Compressor sustained high operation: >80 Hz for %.1f hours (current: %d Hz, 6h avg: %.0f Hz)",
                hours,
                stats.current_hz,
                stats.avg_6h,
            )

    def assess_risk(self, stats: CompressorStats) -> tuple[str, str]:
        """Assess compressor health risk level.

        Risk levels based on NIBE forum research and OEM recommendations:
        - CRITICAL: Immediate action required (hardware damage risk)
        - SEVERE: Sustained stress (reduce demand)
        - WARNING: Monitor closely (elevated operation)
        - WATCH: Note for diagnostics
        - OK: Normal operation

        Args:
            stats: Current compressor statistics

        Returns:
            (risk_level, reason) tuple
        """
        # CRITICAL: Above 100 Hz for more than 15 minutes
        # This indicates compressor at absolute maximum for extended period
        # Risk: Bearing damage, oil breakdown, motor overheating
        if stats.time_above_100hz > timedelta(minutes=15):
            minutes = stats.time_above_100hz.total_seconds() / 60
            return (
                "CRITICAL",
                f"Compressor at {stats.current_hz} Hz for {minutes:.0f} min - REDUCE DEMAND IMMEDIATELY",
            )

        # SEVERE: Above 80 Hz for more than 2 hours
        # Continuous high-Hz operation causes accelerated wear
        # Indicates system struggling to meet demand
        if stats.time_above_80hz > timedelta(hours=2):
            hours = stats.time_above_80hz.total_seconds() / 3600
            return (
                "SEVERE",
                f"Sustained high Hz ({stats.avg_1h:.0f} avg) for {hours:.1f}h - THERMAL STRESS",
            )

        # WARNING: 6-hour average above 70 Hz
        # System consistently operating at high capacity
        # May indicate undersized system or excessive heat loss
        if stats.avg_6h > 70:
            return (
                "WARNING",
                f"Sustained high operation (6h avg: {stats.avg_6h:.0f} Hz) - monitor closely",
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

    def get_diagnostic_info(self, stats: CompressorStats) -> dict[str, Any]:
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

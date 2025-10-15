"""Effect tariff manager for Swedish Effektavgift optimization.

Tracks 15-minute power consumption windows and manages monthly peak
avoidance to minimize effect tariff charges.

Swedish effect tariff rules:
- Measured in 15-minute windows (quarterly periods)
- Daytime (06:00-22:00): Full weight
- Nighttime (22:00-06:00): 50% weight
- Monthly charge based on top 3 peaks
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from ..const import PEAK_RECORDING_MINIMUM, STORAGE_KEY, STORAGE_VERSION

_LOGGER = logging.getLogger(__name__)


@dataclass
class PeakEvent:
    """Record of a 15-minute peak power event.

    Tracks both actual and effective power (accounting for day/night weighting).
    """

    timestamp: datetime
    quarter_of_day: int  # 0-95
    actual_power: float  # kW
    effective_power: float  # kW (with day/night weighting)
    is_daytime: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "quarter_of_day": self.quarter_of_day,
            "actual_power": self.actual_power,
            "effective_power": self.effective_power,
            "is_daytime": self.is_daytime,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PeakEvent":
        """Create from dictionary."""
        return cls(
            timestamp=dt_util.parse_datetime(data["timestamp"]),
            quarter_of_day=data["quarter_of_day"],
            actual_power=data["actual_power"],
            effective_power=data["effective_power"],
            is_daytime=data["is_daytime"],
        )


@dataclass
class PowerLimitDecision:
    """Decision on whether to limit power to avoid peak."""

    should_limit: bool
    severity: str  # "OK", "WARNING", "CRITICAL"
    reason: str
    recommended_offset: float  # Additional negative offset to reduce power


class EffectManager:
    """Manage effect tariff optimization with 15-minute granularity."""

    def __init__(self, hass: HomeAssistant):
        """Initialize effect manager.

        Args:
            hass: Home Assistant instance for storage
        """
        self.hass = hass
        self._store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._monthly_peaks: list[PeakEvent] = []  # Top 3 peaks this month
        self._current_peak: float = 0.0

    async def async_load(self) -> None:
        """Load persistent state from storage."""
        data = await self._store.async_load()
        if data:
            # Load monthly peaks
            peaks_data = data.get("peaks", [])
            self._monthly_peaks = [PeakEvent.from_dict(p) for p in peaks_data]

            # Clean old peaks (from previous months)
            self._clean_old_peaks()

            # Validate peaks - remove unrealistic values (standby/startup noise)
            # Based on NIBE heat pump typical consumption patterns
            invalid_peaks = [
                p for p in self._monthly_peaks if p.actual_power < PEAK_RECORDING_MINIMUM
            ]

            if invalid_peaks:
                _LOGGER.warning(
                    "Removing %d invalid peaks below %.1f kW threshold (standby/startup noise)",
                    len(invalid_peaks),
                    PEAK_RECORDING_MINIMUM,
                )
                self._monthly_peaks = [
                    p for p in self._monthly_peaks if p.actual_power >= PEAK_RECORDING_MINIMUM
                ]

            _LOGGER.info("Loaded %d monthly peaks", len(self._monthly_peaks))

    async def async_save(self) -> None:
        """Save persistent state to storage."""
        await self._store.async_save(
            {
                "peaks": [p.to_dict() for p in self._monthly_peaks],
            }
        )

    async def record_quarter_measurement(
        self,
        power_kw: float,
        quarter: int,
        timestamp: datetime,
    ) -> PeakEvent | None:
        """Record a 15-minute power measurement.

        Only records measurements above minimum threshold to avoid storing
        standby power or startup transients as legitimate peaks.

        Args:
            power_kw: Power consumption in kW
            quarter: Quarter of day (0-95)
            timestamp: Measurement timestamp

        Returns:
            PeakEvent if this creates a new monthly peak, None otherwise
        """
        # Don't record peaks below minimum threshold (standby/startup noise)
        # Typical NIBE: standby 0.05-0.1 kW, heating 2.5-6.0 kW
        if power_kw < PEAK_RECORDING_MINIMUM:
            _LOGGER.debug(
                "Skipping peak recording: %.2f kW below %.1f kW threshold",
                power_kw,
                PEAK_RECORDING_MINIMUM,
            )
            return None

        # Determine if daytime (06:00-22:00)
        is_daytime = 24 <= quarter <= 87  # Quarters 24-87 = 06:00-22:00

        # Calculate effective power (50% weight at night)
        effective_power = power_kw if is_daytime else power_kw * 0.5

        # Check if this is a new peak
        is_new_peak = False
        if len(self._monthly_peaks) < 3:
            # Haven't filled top 3 yet
            is_new_peak = True
        else:
            # Check if exceeds lowest of top 3
            lowest_peak = min(self._monthly_peaks, key=lambda p: p.effective_power)
            if effective_power > lowest_peak.effective_power:
                # Remove lowest peak
                self._monthly_peaks.remove(lowest_peak)
                is_new_peak = True

        if is_new_peak:
            # Create new peak event
            peak_event = PeakEvent(
                timestamp=timestamp,
                quarter_of_day=quarter,
                actual_power=power_kw,
                effective_power=effective_power,
                is_daytime=is_daytime,
            )
            self._monthly_peaks.append(peak_event)

            # Sort peaks by effective power (highest first)
            self._monthly_peaks.sort(key=lambda p: p.effective_power, reverse=True)

            _LOGGER.info(
                "New monthly peak #%d: %.2f kW effective (%.2f kW actual) at Q%d %s",
                len(self._monthly_peaks),
                effective_power,
                power_kw,
                quarter,
                "day" if is_daytime else "night",
            )

            return peak_event

        return None

    def should_limit_power(
        self,
        current_power: float,
        current_quarter: int,
    ) -> PowerLimitDecision:
        """Determine if power should be limited to avoid new 15-minute peak.

        Original algorithm for Swedish Effektavgift with native 15-minute periods.

        Args:
            current_power: Current household power draw (kW)
            current_quarter: Current quarter of day (0-95)

        Returns:
            Decision with limit recommendation and severity
        """
        # Calculate effective power
        is_daytime = 24 <= current_quarter <= 87  # 06:00-22:00
        effective_power = current_power if is_daytime else current_power * 0.5

        # If no peaks yet, no limit needed
        if not self._monthly_peaks:
            return PowerLimitDecision(
                should_limit=False,
                severity="OK",
                reason="No peaks recorded yet",
                recommended_offset=0.0,
            )

        # Get lowest of top 3 peaks (the threshold to beat)
        threshold_peak = min(self._monthly_peaks, key=lambda p: p.effective_power)
        threshold = threshold_peak.effective_power

        # Calculate margin
        margin = threshold - effective_power

        # Determine severity and recommendation
        if margin < 0:
            # Already exceeding threshold - critical
            return PowerLimitDecision(
                should_limit=True,
                severity="CRITICAL",
                reason=f"Exceeding peak by {-margin:.2f} kW",
                recommended_offset=-3.0,  # Aggressive reduction
            )
        elif margin < 0.5:
            # Within 0.5 kW - critical warning
            return PowerLimitDecision(
                should_limit=True,
                severity="CRITICAL",
                reason=f"Within 0.5 kW of peak (margin: {margin:.2f} kW)",
                recommended_offset=-2.0,
            )
        elif margin < 1.0:
            # Within 1.0 kW - warning
            return PowerLimitDecision(
                should_limit=True,
                severity="WARNING",
                reason=f"Within 1.0 kW of peak (margin: {margin:.2f} kW)",
                recommended_offset=-1.0,
            )
        else:
            # Safe margin
            return PowerLimitDecision(
                should_limit=False,
                severity="OK",
                reason=f"Safe margin: {margin:.2f} kW below peak",
                recommended_offset=0.0,
            )

    def get_peak_protection_offset(
        self,
        current_power: float,
        current_quarter: int,
        base_offset: float,
    ) -> float:
        """Calculate additional offset for 15-minute peak protection.

        Args:
            current_power: Current power draw (kW)
            current_quarter: Quarter of day (0-95)
            base_offset: Base offset from price optimization

        Returns:
            Additional negative offset if needed to reduce consumption
        """
        decision = self.should_limit_power(current_power, current_quarter)

        if decision.should_limit:
            # Return the recommended offset (already negative)
            return decision.recommended_offset
        else:
            # No additional offset needed
            return 0.0

    def _clean_old_peaks(self) -> None:
        """Remove peaks from previous months."""
        now = dt_util.now()
        current_month = (now.year, now.month)

        self._monthly_peaks = [
            peak
            for peak in self._monthly_peaks
            if (peak.timestamp.year, peak.timestamp.month) == current_month
        ]

    def reset_monthly_peaks(self) -> None:
        """Reset all monthly peak tracking.

        Used by reset_peak_tracking service (Phase 5).
        Call this at the start of a new billing period.
        """
        _LOGGER.info("Resetting monthly peak tracking")
        self._monthly_peaks = []
        self._current_peak = 0.0

    def get_monthly_peak_summary(self) -> dict[str, Any]:
        """Get summary of monthly peaks for display.

        Returns:
            Dictionary with peak information
        """
        if not self._monthly_peaks:
            return {
                "count": 0,
                "highest": 0.0,
                "peaks": [],
            }

        return {
            "count": len(self._monthly_peaks),
            "highest": self._monthly_peaks[0].effective_power,
            "peaks": [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "effective_power": p.effective_power,
                    "actual_power": p.actual_power,
                    "is_daytime": p.is_daytime,
                }
                for p in self._monthly_peaks
            ],
        }

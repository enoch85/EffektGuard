"""Effect tariff manager for Swedish Effektavgift optimization.

Tracks HOURLY mean power and manages monthly peak avoidance to minimise effect tariff charges.

Swedish effect tariff rules, as Ellevio actually publishes them:
- Measured as HOURLY MEAN POWER, not 15-minute windows (a quarter-hour mean overstates the billed
  peak by up to fourfold).
- Daytime (06:00-22:00): full weight
- Nighttime (22:00-06:00): "raknas bara halva effekttoppen" - half the peak counts
- Monthly charge on the mean of the three highest hours, at most one per day
- 81,25 kr/kW/month
  https://www.ellevio.se/abonnemang/ny-prismodell-baserad-pa-effekt/
  Energimarknadsinspektionen: "elnatsforetagen mater din elanvandning per timme."
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from ..const import (
    BILLABLE_POWER_SOURCES,
    COMPRESSOR_HZ_MIN,
    COMPRESSOR_HZ_RANGE,
    COMPRESSOR_POWER_MAX_KW,
    COMPRESSOR_POWER_MIN_KW,
    COMPRESSOR_POWER_RANGE_KW,
    COMPRESSOR_TEMP_COLD_THRESHOLD,
    COMPRESSOR_TEMP_COOL_THRESHOLD,
    COMPRESSOR_TEMP_EXTREME_COLD_THRESHOLD,
    COMPRESSOR_TEMP_FACTOR_COLD,
    COMPRESSOR_TEMP_FACTOR_COOL,
    COMPRESSOR_TEMP_FACTOR_EXTREME_COLD,
    COMPRESSOR_TEMP_FACTOR_MILD,
    DAYTIME_END_HOUR,
    DAYTIME_START_HOUR,
    DEFAULT_HEAT_PUMP_POWER_KW,
    EFFECT_MARGIN_PREDICTIVE,
    EFFECT_OFFSET_CRITICAL,
    EFFECT_OFFSET_PREDICTIVE,
    EFFECT_OFFSET_WARNING_RISING,
    EFFECT_OFFSET_WARNING_STABLE,
    EFFECT_PEAK_MARGIN_CRITICAL,
    EFFECT_PEAK_MARGIN_WARNING,
    EFFECT_PEAK_OFFSET_CRITICAL,
    EFFECT_PEAK_OFFSET_EXCEEDING,
    EFFECT_PEAK_OFFSET_WARNING,
    EFFECT_PREDICTIVE_MODERATE_COOLING_INCREASE,
    EFFECT_PREDICTIVE_RAPID_COOLING_INCREASE,
    EFFECT_PREDICTIVE_RAPID_COOLING_THRESHOLD,
    EFFECT_PREDICTIVE_WARMING_DECREASE,
    EFFECT_STORAGE_VERSION,
    EFFECT_WEIGHT_CRITICAL,
    EFFECT_WEIGHT_PREDICTIVE,
    EFFECT_WEIGHT_WARNING_RISING,
    EFFECT_WEIGHT_WARNING_STABLE,
    NIGHT_TARIFF_WEIGHT,
    PEAK_RECORDING_MAXIMUM,
    PEAK_RECORDING_MINIMUM,
    POWER_MULTIPLIER_COLD,
    POWER_MULTIPLIER_MILD,
    POWER_MULTIPLIER_VERY_COLD,
    POWER_SOURCE_EXTERNAL_METER,
    POWER_SOURCE_NONE,
    POWER_STANDBY_KW,
    POWER_TEMP_COLD_THRESHOLD,
    POWER_TEMP_VERY_COLD_THRESHOLD,
    STORAGE_KEY,
    THERMAL_CHANGE_MODERATE,
    THERMAL_CHANGE_MODERATE_COOLING,
)
from ..utils.time_utils import get_current_billing_period

_LOGGER = logging.getLogger(__name__)


def is_daytime_hour(hour: int) -> bool:
    """Whether this HOUR is billed at the full tariff rate. Ellevio's discount is 22:00-06:00."""
    return DAYTIME_START_HOUR <= hour < DAYTIME_END_HOUR


def effective_tariff_power_kw(power_kw: float, hour: int) -> float:
    """What the effect tariff will BILL this hour's mean power as. Night hours count half.

    THE ONE DEFINITION - everything that goes near a monthly peak comes through here. When the
    weighting was open-coded, the savings baseline compared an UNWEIGHTED peak against a weighted
    one and reported phantom savings (a night peak looked ~half off with the optimiser idle).
    """
    return power_kw if is_daytime_hour(hour) else power_kw * NIGHT_TARIFF_WEIGHT


class PeakEventDict(TypedDict):
    """Dictionary representation of a PeakEvent for serialization."""

    timestamp: str  # ISO format
    period_of_day: int
    actual_power: float
    effective_power: float
    is_daytime: bool
    source: str


class PeakSummaryPeakDict(TypedDict):
    """Individual peak in the monthly summary."""

    timestamp: str
    effective_power: float
    actual_power: float
    is_daytime: bool
    source: str
    billable: bool


class MonthlyPeakSummaryDict(TypedDict):
    """Summary of monthly peaks for display.

    `billable` is False as soon as ANY peak in the history came from something other than a
    whole-house meter. The tariff is charged on the top three HOURS together, so one pump-only
    hour in the set makes the whole figure something other than the bill - and the owner is told
    that rather than shown a number that looks like money.
    """

    count: int
    highest: float
    billable: bool
    peaks: list[PeakSummaryPeakDict]


@dataclass
class PeakEvent:
    """One billing period's mean power - an HOUR, which is what the tariff bills.

    Tracks both actual and effective power (accounting for the 22:00-06:00 half-price window).
    """

    timestamp: datetime
    period_of_day: int  # the billing HOUR, 0-23
    actual_power: float  # kW
    effective_power: float  # kW (with day/night weighting)
    is_daytime: bool
    # Where the number came from. A peak from NIBE's phase currents is a valid CONTROL threshold but
    # is not whole-house grid import, so it must never be reported as the month's billing peak.
    # Carrying provenance lets one history serve both purposes.
    source: str = POWER_SOURCE_EXTERNAL_METER

    @property
    def is_billable(self) -> bool:
        """Whether this peak may appear in a billing figure shown to the owner."""
        return self.source in BILLABLE_POWER_SOURCES

    def to_dict(self) -> PeakEventDict:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "period_of_day": self.period_of_day,
            "actual_power": self.actual_power,
            "effective_power": self.effective_power,
            "is_daytime": self.is_daytime,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: PeakEventDict) -> "PeakEvent":
        """Create from dictionary. Records older than this schema never reach here: the store
        migration (EffectStore) discards them, so every field is present."""
        return cls(
            timestamp=dt_util.parse_datetime(data["timestamp"]),
            period_of_day=data["period_of_day"],
            actual_power=data["actual_power"],
            effective_power=data["effective_power"],
            is_daytime=data["is_daytime"],
            source=data["source"],
        )


@dataclass
class PowerLimitDecision:
    """Decision on whether to limit power to avoid peak."""

    should_limit: bool
    severity: str  # "OK", "WARNING", "CRITICAL"
    reason: str
    recommended_offset: float  # Additional negative offset to reduce power


@dataclass
class EffectLayerDecision:
    """Decision from effect/peak protection layer.

    Used to communicate layer decisions back to decision engine.
    """

    name: str  # Layer name for display (e.g., "Peak")
    offset: float  # Proposed heating curve offset (°C)
    weight: float  # Layer weight/priority (0.0-1.0)
    reason: str  # Human-readable explanation


class EffectStore(Store):
    """Peak-history storage, with migration from the quarter-hour era.

    Version 1 recorded 15-minute quarter peaks (``quarter_of_day``). The effect tariff bills
    the HOURLY mean, so a quarter-hour record is not a billable quantity and cannot be
    converted into one - migration discards them and the month's top-3 restarts from live
    measurement. Parsing them instead is what broke setup for every upgrading install:
    ``PeakEvent.from_dict`` raised ``KeyError: 'period_of_day'`` inside ``async_setup_entry``.
    """

    async def _async_migrate_func(
        self,
        old_major_version: int,
        old_minor_version: int,
        old_data: dict | None,
    ) -> dict:
        """Migrate stored peak history to the current schema."""
        if old_major_version < EFFECT_STORAGE_VERSION:
            discarded = len(old_data.get("peaks", [])) if isinstance(old_data, dict) else 0
            if discarded:
                _LOGGER.warning(
                    "Discarding %d peak record(s) written by the 15-minute tariff model: the "
                    "effect tariff bills the hourly mean, and a quarter-hour mean is not "
                    "convertible to one. Peak tracking restarts from live measurement.",
                    discarded,
                )
            return {"peaks": []}
        return old_data if isinstance(old_data, dict) else {"peaks": []}


class EffectManager:
    """Manage effect tariff optimization on hourly billing periods."""

    def __init__(self, hass: HomeAssistant):
        """Initialize effect manager.

        Args:
            hass: Home Assistant instance for storage
        """
        self.hass = hass
        self._store = EffectStore(hass, EFFECT_STORAGE_VERSION, STORAGE_KEY)
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

    async def record_period_measurement(
        self,
        power_kw: float,
        period: int,
        timestamp: datetime,
        source: str = POWER_SOURCE_EXTERNAL_METER,
    ) -> PeakEvent | None:
        """Record one completed BILLING PERIOD - an HOUR, which is what the tariff bills.

        The tariff bills the mean power over a whole hour, so a 15-minute 9 kW hot-water cycle in an
        otherwise idle hour is a 3 kW billed peak, not 9 - recording the quarter-hour mean overstates
        it and throttles the pump to defend the difference.

        Args:
            power_kw: MEAN power over the hour, in kW
            period: the billing hour of the day (0-23)
            timestamp: Measurement timestamp
            source: Where the reading came from. A NIBE-currents peak is a valid CONTROL threshold
                but is not whole-house grid import, so it never reaches a billing figure.

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

        # A plausibility CEILING, for the same reason the floor exists. A peak is persisted for a
        # month and every later hour is judged against it, so one impossible reading (a mis-scaled
        # unit once put 5 000 000 kW here) makes every real hour look safe and disables peak
        # protection until the month rolls over. Nothing behind a domestic main fuse reaches
        # PEAK_RECORDING_MAXIMUM, so no real house is refused.
        if power_kw > PEAK_RECORDING_MAXIMUM:
            _LOGGER.warning(
                "Refusing to record %.0f kW as a tariff peak: no domestic supply can deliver it "
                "(ceiling %.0f kW), so this is a sensor fault or a unit-scaling error. Recording "
                "it would make every real hour look safe against it and disable peak protection "
                "for the rest of the month.",
                power_kw,
                PEAK_RECORDING_MAXIMUM,
            )
            return None

        is_daytime = is_daytime_hour(period)
        effective_power = effective_tariff_power_kw(power_kw, period)

        # AT MOST ONE PEAK PER DAY. Ellevio: the monthly charge is the mean of the three highest
        # hourly peaks, and "the three peaks must come from three different days" - only a day's
        # highest hour counts. Date-blind top-3 let one cold Saturday fill all three slots, which
        # overstates the bill and understates the margin the pump is then throttled against.
        # https://www.ellevio.se/abonnemang/elnatspriser/ny-prismodell-baserad-pa-effekt/
        same_day = next(
            (p for p in self._monthly_peaks if p.timestamp.date() == timestamp.date()),
            None,
        )
        if same_day is not None and effective_power <= same_day.effective_power:
            return None

        # Check if this is a new peak
        is_new_peak = False
        if same_day is not None:
            # The day's counted peak is its highest hour; this hour outbills it.
            self._monthly_peaks.remove(same_day)
            is_new_peak = True
        elif len(self._monthly_peaks) < 3:
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
                period_of_day=period,
                actual_power=power_kw,
                effective_power=effective_power,
                is_daytime=is_daytime,
                source=source,
            )
            self._monthly_peaks.append(peak_event)

            # Sort peaks by effective power (highest first)
            self._monthly_peaks.sort(key=lambda p: p.effective_power, reverse=True)

            _LOGGER.info(
                "New monthly peak #%d: %.2f kW effective (%.2f kW actual) in hour %02d, %s",
                len(self._monthly_peaks),
                effective_power,
                power_kw,
                period,
                "day" if is_daytime else "night",
            )

            return peak_event

        return None

    def should_limit_power(
        self,
        current_power: float,
        current_period: int,
    ) -> PowerLimitDecision:
        """Determine if power should be limited to avoid new 15-minute peak.

        Original algorithm for Swedish Effektavgift with native 15-minute periods.

        Args:
            current_power: Current household power draw (kW)
            current_period: the billing HOUR of the day (0-23)

        Returns:
            Decision with limit recommendation and severity
        """
        effective_power = effective_tariff_power_kw(current_power, current_period)

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
                recommended_offset=EFFECT_PEAK_OFFSET_EXCEEDING,
            )
        elif margin < EFFECT_PEAK_MARGIN_CRITICAL:
            # Within critical margin - critical warning
            return PowerLimitDecision(
                should_limit=True,
                severity="CRITICAL",
                reason=f"Within {EFFECT_PEAK_MARGIN_CRITICAL} kW of peak (margin: {margin:.2f} kW)",
                recommended_offset=EFFECT_PEAK_OFFSET_CRITICAL,
            )
        elif margin < EFFECT_PEAK_MARGIN_WARNING:
            # Within warning margin - warning
            return PowerLimitDecision(
                should_limit=True,
                severity="WARNING",
                reason=f"Within {EFFECT_PEAK_MARGIN_WARNING} kW of peak (margin: {margin:.2f} kW)",
                recommended_offset=EFFECT_PEAK_OFFSET_WARNING,
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
        current_period: int,
        base_offset: float,
    ) -> float:
        """Calculate additional offset for HOURLY peak protection.

        Args:
            current_power: Current power draw (kW)
            current_period: the billing HOUR of the day (0-23)
            base_offset: Base offset from price optimization

        Returns:
            Additional negative offset if needed to reduce consumption
        """
        decision = self.should_limit_power(current_power, current_period)

        if decision.should_limit:
            # Return the recommended offset (already negative)
            return decision.recommended_offset
        else:
            # No additional offset needed
            return 0.0

    def prune_peaks_for_current_month(self) -> None:
        """Drop peaks that belong to a previous month.

        The effect tariff bills a MONTHLY peak, so last month's peaks must not survive into
        this one. Must be reachable outside async_load(): an instance that stays up across a
        month boundary would otherwise carry the old month's top-3 forward, leaving the
        threshold, the peak_this_month sensor and the savings figure all stale.

        Called on the month-change branch of the coordinator's daily rollover.
        """
        self._clean_old_peaks()

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

    def get_monthly_peak_summary(self) -> MonthlyPeakSummaryDict:
        """Get summary of monthly peaks for display.

        Returns:
            Dictionary with peak information
        """
        if not self._monthly_peaks:
            return {
                "count": 0,
                "highest": 0.0,
                "billable": False,
                "peaks": [],
            }

        return {
            "count": len(self._monthly_peaks),
            "highest": self._monthly_peaks[0].effective_power,
            "billable": all(p.is_billable for p in self._monthly_peaks),
            "peaks": [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "effective_power": p.effective_power,
                    "actual_power": p.actual_power,
                    "is_daytime": p.is_daytime,
                    "source": p.source,
                    "billable": p.is_billable,
                }
                for p in self._monthly_peaks
            ],
        }

    def estimate_power_consumption(
        self,
        is_heating: bool,
        outdoor_temp: float,
    ) -> float:
        """Estimate heat pump power consumption from state.

        Moved from coordinator._estimate_power_consumption for shared reuse.
        Basic estimation based on heating status and outdoor temperature.

        Args:
            is_heating: Whether compressor is currently heating
            outdoor_temp: Current outdoor temperature (°C)

        Returns:
            Estimated power consumption in kW
        """
        if not is_heating:
            return POWER_STANDBY_KW

        # Adjust for outdoor temperature - colder = more power needed
        if outdoor_temp < POWER_TEMP_VERY_COLD_THRESHOLD:
            return DEFAULT_HEAT_PUMP_POWER_KW * POWER_MULTIPLIER_VERY_COLD
        elif outdoor_temp < POWER_TEMP_COLD_THRESHOLD:
            return DEFAULT_HEAT_PUMP_POWER_KW * POWER_MULTIPLIER_COLD
        else:
            return DEFAULT_HEAT_PUMP_POWER_KW * POWER_MULTIPLIER_MILD

    def estimate_power_from_compressor(
        self,
        compressor_hz: int,
        outdoor_temp: float,
    ) -> float:
        """Estimate heat pump power from compressor frequency and outdoor temperature.

        Moved from coordinator._estimate_power_from_compressor for shared reuse.
        More accurate estimation when we know compressor is running.
        Used for smart fallback when grid meter shows solar/battery offset.

        Based on typical NIBE F750 performance curves:
        - 20 Hz (minimum): ~1.5-2.0 kW
        - 50 Hz (mid): ~3.5-4.5 kW
        - 80 Hz (maximum): ~6.0-7.0 kW

        Args:
            compressor_hz: Current compressor frequency (Hz)
            outdoor_temp: Current outdoor temperature (°C)

        Returns:
            Estimated power consumption in kW
        """
        if compressor_hz == 0:
            return POWER_STANDBY_KW

        # Base power from frequency (linear approximation)
        # 20-80 Hz range maps to ~1.5-6.5 kW
        base_from_hz = COMPRESSOR_POWER_MIN_KW + (compressor_hz - COMPRESSOR_HZ_MIN) * (
            COMPRESSOR_POWER_RANGE_KW / COMPRESSOR_HZ_RANGE
        )
        base_from_hz = max(COMPRESSOR_POWER_MIN_KW, min(base_from_hz, COMPRESSOR_POWER_MAX_KW))

        # Temperature adjustment (colder = more power needed for same output)
        if outdoor_temp < COMPRESSOR_TEMP_EXTREME_COLD_THRESHOLD:
            temp_factor = COMPRESSOR_TEMP_FACTOR_EXTREME_COLD
        elif outdoor_temp < COMPRESSOR_TEMP_COLD_THRESHOLD:
            temp_factor = COMPRESSOR_TEMP_FACTOR_COLD
        elif outdoor_temp < COMPRESSOR_TEMP_COOL_THRESHOLD:
            temp_factor = COMPRESSOR_TEMP_FACTOR_COOL
        else:
            temp_factor = COMPRESSOR_TEMP_FACTOR_MILD

        estimated = base_from_hz * temp_factor

        _LOGGER.debug(
            "Power estimation: %d Hz at %.1f°C → %.2f kW (base: %.2f, temp_factor: %.2f)",
            compressor_hz,
            outdoor_temp,
            estimated,
            base_from_hz,
            temp_factor,
        )

        return estimated

    def evaluate_layer(
        self,
        current_peak: float,
        current_power: float,
        thermal_trend: dict,
        enable_peak_protection: bool = True,
    ) -> EffectLayerDecision:
        """Effect tariff protection with PREDICTIVE peak avoidance.

        Uses indoor temperature trend to predict heating demand in next 15 minutes.
        Acts BEFORE power spikes instead of reacting to them.

        PHILOSOPHY:
        Traditional reactive approach waits until power is high, then reduces.
        Predictive approach sees house cooling rapidly → knows compressor will ramp up soon
        → reduces offset NOW before spike occurs → smoother power profile.

        Args:
            current_peak: Current monthly peak (kW) - from peak_this_month sensor
            current_power: Current whole-house power consumption (kW)
            thermal_trend: Temperature trend data with 'rate_per_hour' key
            enable_peak_protection: Whether peak protection is enabled in config

        Returns:
            EffectLayerDecision with predictive peak protection

        References:
            MASTER_IMPLEMENTATION_PLAN.md: Phase 4 - Predictive Peak Avoidance
        """
        # Check if peak protection is enabled
        if not enable_peak_protection:
            return EffectLayerDecision(
                name="Peak",
                offset=0.0,
                weight=0.0,
                reason="Disabled by user",
            )

        # The billing period is the HOUR - see the module docstring and BILLING_PERIOD_MINUTES.
        current_period = get_current_billing_period()

        limit_decision = self.should_limit_power(current_power, current_period)

        # Get thermal trend for predictive analysis
        trend_rate = thermal_trend.get("rate_per_hour", 0.0)

        # Predict power change in next 15 minutes based on indoor temperature trend
        # When house cooling fast → heat pump will increase power soon
        # When house warming → heat pump may reduce power
        if trend_rate < EFFECT_PREDICTIVE_RAPID_COOLING_THRESHOLD:
            # Rapid cooling → Compressor will ramp up significantly
            predicted_power_increase = EFFECT_PREDICTIVE_RAPID_COOLING_INCREASE
            prediction_reason = "cooling rapidly"
        elif trend_rate < THERMAL_CHANGE_MODERATE_COOLING:
            # Moderate cooling → Slight power increase expected
            predicted_power_increase = EFFECT_PREDICTIVE_MODERATE_COOLING_INCREASE
            prediction_reason = "gentle cooling"
        elif trend_rate > THERMAL_CHANGE_MODERATE:
            # Rapid warming → Compressor may reduce
            predicted_power_increase = EFFECT_PREDICTIVE_WARMING_DECREASE
            prediction_reason = "warming"
        else:
            # Stable → No significant change expected
            predicted_power_increase = 0.0
            prediction_reason = "stable"

        predicted_power = current_power + predicted_power_increase
        predicted_margin = current_peak - predicted_power

        # Peak protection logic with predictive enhancement
        # Oct 19, 2025: All values now constants for test script reuse
        # Peak costs (effect tariff) are monthly charges that accumulate - worth protecting
        if limit_decision.severity == "CRITICAL":
            # Already at peak - immediate action
            return EffectLayerDecision(
                name="Peak",
                offset=EFFECT_OFFSET_CRITICAL,
                weight=EFFECT_WEIGHT_CRITICAL,
                reason=f"CRITICAL ({current_power:.1f}/{current_peak:.1f} kW)",
            )
        elif (
            current_peak > 0
            and predicted_margin < EFFECT_MARGIN_PREDICTIVE
            and predicted_power_increase > 0
        ):
            # PREDICTIVE: Will approach peak in next 15 min - act NOW to prevent the spike.
            #
            # The `current_peak > 0` guard is required: with no peak history current_peak is 0.0, so
            # `predicted_margin = 0.0 - predicted_power` is ALWAYS negative and this branch would fire
            # on every cooling house from day one, voting a heat-reducing offset that outranks T1/T2
            # recovery. Missing input must abstain, never vote to reduce heat.
            return EffectLayerDecision(
                name="Peak",
                offset=EFFECT_OFFSET_PREDICTIVE,
                weight=EFFECT_WEIGHT_PREDICTIVE,
                reason=f"Predictive avoidance ({prediction_reason})",
            )
        elif limit_decision.severity == "WARNING":
            # Close to peak - check if trend shows increasing demand
            if predicted_power_increase > 0:
                # Warning + rising demand = moderate reduction
                return EffectLayerDecision(
                    name="Peak",
                    offset=EFFECT_OFFSET_WARNING_RISING,
                    weight=EFFECT_WEIGHT_WARNING_RISING,
                    reason=f"WARNING + demand rising ({prediction_reason})",
                )
            else:
                # Warning but demand stable/falling = gentle reduction
                return EffectLayerDecision(
                    name="Peak",
                    offset=EFFECT_OFFSET_WARNING_STABLE,
                    weight=EFFECT_WEIGHT_WARNING_STABLE,
                    reason=f"WARNING + demand {prediction_reason}",
                )
        else:
            # Safe margin - no action needed
            return EffectLayerDecision(
                name="Peak",
                offset=0.0,
                weight=0.0,
                reason=f"Safe margin ({current_power:.1f}/{current_peak:.1f} kW)",
            )

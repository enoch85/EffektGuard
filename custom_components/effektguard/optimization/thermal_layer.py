"""Thermal model for building thermal behavior prediction.

Models heat storage and loss characteristics for predictive control.
Enables pre-heating and thermal energy banking strategies.
Includes emergency thermal debt response layer (Phase 6 refactor).
Includes proactive debt prevention layer (Phase 7 refactor).
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional, Protocol

from ..const import (
    ANTI_WINDUP_CAUSATION_WINDOW_MINUTES,
    ANTI_WINDUP_COOLDOWN_MINUTES,
    ANTI_WINDUP_DM_DROPPING_RATE,
    ANTI_WINDUP_DM_HISTORY_BASE_SIZE,
    ANTI_WINDUP_MIN_POSITIVE_OFFSET,
    ANTI_WINDUP_MIN_SAMPLES,
    ANTI_WINDUP_OFFSET_CAP_MULTIPLIER,
    ANTI_WINDUP_REDUCTION_MULTIPLIER,
    ANTI_WINDUP_REDUCTION_RATE_DIVISOR,
    ANTI_WINDUP_REDUCTION_THRESHOLD,
    DM_CRITICAL_T1_MARGIN,
    DM_CRITICAL_T1_OFFSET,
    DM_CRITICAL_T1_WEIGHT,
    DM_CRITICAL_T2_MARGIN,
    DM_CRITICAL_T2_OFFSET,
    DM_CRITICAL_T2_WEIGHT,
    DM_CRITICAL_T3_MARGIN,
    DM_CRITICAL_T3_MAX,
    DM_CRITICAL_T3_OFFSET,
    DM_CRITICAL_T3_WEIGHT,
    DM_THERMAL_MASS_BUFFER_CONCRETE,
    DM_THERMAL_MASS_BUFFER_RADIATOR,
    DM_THERMAL_MASS_BUFFER_TIMBER,
    DM_THRESHOLD_AUX_LIMIT,
    LAYER_WEIGHT_EMERGENCY,
    LAYER_WEIGHT_PROACTIVE_MIN,
    MULTIPLIER_BOOST_30_PERCENT,
    MULTIPLIER_REDUCTION_20_PERCENT,
    PROACTIVE_ZONE1_OFFSET,
    PROACTIVE_ZONE1_THRESHOLD_PERCENT,
    PROACTIVE_ZONE2_OFFSET,
    PROACTIVE_ZONE2_THRESHOLD_PERCENT,
    PROACTIVE_ZONE2_WEIGHT,
    PROACTIVE_ZONE3_OFFSET_MIN,
    PROACTIVE_ZONE3_OFFSET_RANGE,
    PROACTIVE_ZONE3_THRESHOLD_PERCENT,
    PROACTIVE_ZONE3_WEIGHT,
    PROACTIVE_ZONE4_OFFSET,
    PROACTIVE_ZONE4_THRESHOLD_PERCENT,
    PROACTIVE_ZONE4_WEIGHT,
    PROACTIVE_ZONE5_OFFSET,
    PROACTIVE_ZONE5_THRESHOLD_PERCENT,
    PROACTIVE_ZONE5_WEIGHT,
    PROACTIVE_WARM_HOUSE_THRESHOLD,
    PROACTIVE_WARM_HOUSE_WEIGHT_REDUCTION,
    QuarterClassification,
    RAPID_COOLING_BOOST_MAX,
    RAPID_COOLING_BOOST_MULTIPLIER,
    RAPID_COOLING_OUTDOOR_THRESHOLD,
    RAPID_COOLING_THRESHOLD,
    RAPID_COOLING_WEIGHT,
    SAFETY_EMERGENCY_OFFSET,
    THERMAL_CHANGE_MODERATE,
    THERMAL_CHANGE_MODERATE_COOLING,
    THERMAL_RECOVERY_DAMPING_FACTOR,
    THERMAL_RECOVERY_FORECAST_DROP_THRESHOLD,
    THERMAL_RECOVERY_FORECAST_HORIZON,
    THERMAL_RECOVERY_MIN_CONFIDENCE,
    THERMAL_RECOVERY_OUTDOOR_DROPPING_THRESHOLD,
    THERMAL_RECOVERY_OVERSHOOT_MILD_DAMPING,
    THERMAL_RECOVERY_OVERSHOOT_MILD_THRESHOLD,
    THERMAL_RECOVERY_OVERSHOOT_MODERATE_DAMPING,
    THERMAL_RECOVERY_OVERSHOOT_MODERATE_THRESHOLD,
    THERMAL_RECOVERY_OVERSHOOT_SEVERE_DAMPING,
    THERMAL_RECOVERY_OVERSHOOT_SEVERE_THRESHOLD,
    THERMAL_RECOVERY_RAPID_FACTOR,
    THERMAL_RECOVERY_RAPID_THRESHOLD,
    THERMAL_RECOVERY_T1_MIN_OFFSET,
    THERMAL_RECOVERY_T2_MIN_OFFSET,
    THERMAL_RECOVERY_T3_MIN_OFFSET,
    THERMAL_RECOVERY_WARMING_THRESHOLD,
    VOLATILE_WEIGHT_REDUCTION,
    WARNING_CAUTION_OFFSET,
    WARNING_CAUTION_WEIGHT,
    WARNING_DEVIATION_DIVISOR_MODERATE,
    WARNING_DEVIATION_DIVISOR_SEVERE,
    WARNING_DEVIATION_THRESHOLD,
    WARNING_OFFSET_MAX_MODERATE,
    WARNING_OFFSET_MAX_SEVERE,
    WARNING_OFFSET_MIN_MODERATE,
    WARNING_OFFSET_MIN_SEVERE,
    # DM recovery constants (shared with dhw_optimizer)
    DM_RECOVERY_MAX_HOURS,
    DM_RECOVERY_RATE_COLD,
    DM_RECOVERY_RATE_MILD,
    DM_RECOVERY_RATE_VERY_COLD,
)
from .climate_zones import ClimateZoneDetector
from ..utils.time_utils import get_current_quarter
from ..utils.volatile_helpers import should_skip_volatile_boost

_LOGGER = logging.getLogger(__name__)


def is_cooling_rapidly(thermal_trend: dict, threshold: float = RAPID_COOLING_THRESHOLD) -> bool:
    """Check if house is cooling rapidly.

    Args:
        thermal_trend: Trend data
        threshold: Cooling rate threshold (°C/h, negative)

    Returns:
        True if cooling faster than threshold
    """
    return thermal_trend.get("rate_per_hour", 0.0) < threshold


def is_warming_rapidly(thermal_trend: dict, threshold: float = THERMAL_CHANGE_MODERATE) -> bool:
    """Check if house is warming rapidly.

    Args:
        thermal_trend: Trend data
        threshold: Warming rate threshold (°C/h, positive)

    Returns:
        True if warming faster than threshold
    """
    return thermal_trend.get("rate_per_hour", 0.0) > threshold


def estimate_dm_recovery_time(current_dm: float, target_dm: float, outdoor_temp: float) -> float:
    """Estimate hours until DM recovers to target level.

    Shared function for use by both DHW optimizer and space heating logic.
    Uses simplified thermal model based on outdoor temperature.

    Args:
        current_dm: Current degree minutes value (negative)
        target_dm: Target DM to reach (less negative)
        outdoor_temp: Current outdoor temperature for recovery rate

    Returns:
        Estimated hours until recovery (minimum 0.5h, maximum 12h)

    References:
        - Thermal debt analysis: DM recovery varies with heating capacity
        - Conservative estimates prevent false promises to user
        - Moved from dhw_optimizer._estimate_dm_recovery_time for shared reuse
    """
    dm_deficit = target_dm - current_dm  # e.g., -350 - (-435) = 85

    # Recovery rate depends on outdoor temp (from const.py)
    if outdoor_temp > 5:
        recovery_rate = DM_RECOVERY_RATE_MILD
    elif outdoor_temp > 0:
        recovery_rate = DM_RECOVERY_RATE_COLD
    else:
        recovery_rate = DM_RECOVERY_RATE_VERY_COLD

    hours = dm_deficit / recovery_rate

    # Constrain to reasonable range
    # Min: 0.5h - minimum meaningful recovery period
    # Max: DM_RECOVERY_MAX_HOURS (12h)
    estimated_hours = max(0.5, min(hours, DM_RECOVERY_MAX_HOURS))

    _LOGGER.debug(
        "DM recovery estimate: %.0f DM deficit / %.0f DM/h = %.1fh "
        "(outdoor: %.1f°C, constrained: %.1fh)",
        dm_deficit,
        recovery_rate,
        hours,
        outdoor_temp,
        estimated_hours,
    )

    return estimated_hours


def get_thermal_debt_status(thermal_debt: float, dm_thresholds: dict) -> str:
    """Get human-readable thermal debt status.

    Moved from coordinator._get_thermal_debt_status for shared reuse.
    Provides consistent status descriptions across all components.

    Args:
        thermal_debt: Current thermal debt (DM)
        dm_thresholds: Thresholds for climate zone with keys:
            - "block": DHW blocking threshold (warning level)
            - "abort": DHW abort threshold (critical level)
            - "critical": Optional, alias for abort
            - "warning": Optional, alias for block

    Returns:
        Status string with margin information
    """
    # Support both naming conventions
    abort_threshold = dm_thresholds.get("abort") or dm_thresholds.get("critical", -1500)
    block_threshold = dm_thresholds.get("block") or dm_thresholds.get("warning", -700)

    if thermal_debt < abort_threshold:
        margin = abs(thermal_debt - abort_threshold)
        return f"CRITICAL - {margin:.0f} DM past abort threshold"
    elif thermal_debt < block_threshold:
        margin = abs(thermal_debt - block_threshold)
        return f"WARNING - {margin:.0f} DM from block threshold"
    else:
        margin = thermal_debt - block_threshold
        return f"OK - {margin:.0f} DM safety margin"


class PriceAnalyzerProtocol(Protocol):
    """Protocol for price analyzer interface."""

    def get_current_classification(self, quarter: int) -> QuarterClassification:
        """Get price classification for a quarter."""
        ...


@dataclass
class EmergencyLayerDecision:
    """Decision from the emergency thermal debt layer.

    Encapsulates the thermal debt response with context-aware thresholds.
    """

    name: str
    offset: float
    weight: float
    reason: str
    # Additional diagnostic fields
    tier: str = ""  # "T1", "T2", "T3", "WARNING", "CAUTION", "OK"
    degree_minutes: float = 0.0
    threshold_used: float = 0.0
    damping_applied: bool = False
    anti_windup_active: bool = False  # True if offset was capped due to anti-windup
    dm_rate: float = 0.0  # DM rate of change (DM/hour, negative = dropping)


@dataclass
class ProactiveLayerDecision:
    """Decision from the proactive debt prevention layer.

    Encapsulates the proactive intervention with zone-based thresholds.
    """

    name: str
    offset: float
    weight: float
    reason: str
    # Additional diagnostic fields
    zone: str = ""  # "Z1", "Z2", "Z3", "Z4", "Z5", "RAPID_COOLING", "NONE"
    degree_minutes: float = 0.0
    trend_rate: float = 0.0
    forecast_validated: bool = False


class ThermalModel:
    """Model building thermal characteristics for predictive control.

    Provides thermal mass and insulation quality parameters used by
    DecisionEngine for heat loss calculations and comfort predictions.
    """

    def __init__(
        self,
        thermal_mass: float = 1.0,  # Relative scale 0.5-2.0
        insulation_quality: float = 1.0,  # Relative scale 0.5-2.0
    ):
        """Initialize thermal model.

        Args:
            thermal_mass: Relative thermal mass (1.0 = normal)
                - 0.5 = low mass (timber frame)
                - 1.0 = normal mass (mixed construction)
                - 2.0 = high mass (concrete/masonry)
            insulation_quality: Relative insulation (1.0 = normal)
                - 0.5 = poor insulation
                - 1.0 = standard insulation
                - 2.0 = excellent insulation
        """
        self.thermal_mass = thermal_mass
        self.insulation_quality = insulation_quality

    def get_prediction_horizon(self) -> float:
        """Get prediction horizon for weather forecasting.

        Base implementation returns default 12 hours.
        AdaptiveThermalModel overrides this with UFH-type-specific values.

        Returns:
            Prediction horizon in hours (default 12.0)
        """
        return 12.0  # Default medium horizon


class EmergencyLayer:
    """Emergency layer: Smart context-aware thermal debt response.

    DESIGN PHILOSOPHY:
    Instead of fixed thresholds, this layer understands what's "normal" for
    current outdoor temperature. At -30°C in Kiruna, DM -1000 might be normal.
    At 0°C in Malmö, DM -1000 indicates a problem.

    The algorithm calculates expected DM range based on:
    - Outdoor temperature (colder = deeper normal DM)
    - Heat demand intensity
    - Distance from absolute safety limit (-1500)

    This automatically adapts to ANY climate without configuration:
    - Malmö (0°C): Tight tolerances, early warnings
    - Stockholm (-5°C): Moderate tolerances
    - Luleå/Kiruna (-30°C): Wide tolerances, expect deep DM

    Absolute maximum DM -1500 is ALWAYS enforced regardless of conditions.
    This is the hard safety limit validated by Swedish NIBE forums.
    """

    def __init__(
        self,
        climate_detector: ClimateZoneDetector,
        price_analyzer: Optional[PriceAnalyzerProtocol] = None,
        heating_type: str = "radiator",
        get_thermal_trend: Optional[Callable[[], dict]] = None,
        get_outdoor_trend: Optional[Callable[[], dict]] = None,
    ):
        """Initialize emergency layer.

        Args:
            climate_detector: ClimateZoneDetector for context-aware thresholds
            price_analyzer: Optional price analyzer for cheap period detection
            heating_type: Heating system type ("radiator", "concrete_ufh", "timber")
            get_thermal_trend: Callable returning thermal trend dict
            get_outdoor_trend: Callable returning outdoor trend dict
        """
        self.climate_detector = climate_detector
        self.price_analyzer = price_analyzer
        self.heating_type = heating_type
        self._get_thermal_trend = get_thermal_trend or (
            lambda: {"rate_per_hour": 0.0, "confidence": 0.0}
        )
        self._get_outdoor_trend = get_outdoor_trend or (
            lambda: {"rate_per_hour": 0.0, "confidence": 0.0}
        )

        # Anti-windup: Track DM history to detect "heat in transit" situations
        # When offset is positive but DM is still dropping, heat hasn't reached
        # the thermal mass yet - don't escalate offset further
        # History size varies by heating type to match thermal lag
        history_size = self._get_dm_history_size_for_heating_type(heating_type)
        self._dm_history: deque[tuple[datetime, float]] = deque(maxlen=history_size)

        # Anti-windup cooldown: After detecting spiral, wait before trying to raise again
        # Jan 2026: Raising offset when DM is dropping makes DM drop FASTER
        # (S1 increases but BT25 can't catch up). Wait for pump to stabilize.
        self._anti_windup_cooldown_until: Optional[datetime] = None

        # Track when we last raised offset (for causation detection)
        # Anti-windup should only trigger if we CAUSED the spiral by raising recently
        # Jan 2026: Distinguish self-induced spiral vs environmental DM drop
        self._last_offset_raise_time: Optional[datetime] = None
        self._last_offset_value: Optional[float] = None

    def _get_dm_history_size_for_heating_type(self, heating_type: str) -> int:
        """Get DM history size based on heating system thermal lag.

        Derives from ANTI_WINDUP_DM_HISTORY_BASE_SIZE scaled by thermal mass buffer.
        Longer thermal lag systems need more history to detect sustained trends.

        Args:
            heating_type: Heating system type

        Returns:
            Number of samples to track (at 5-min intervals)
        """
        # Use existing thermal mass buffer multipliers to scale history size
        # Higher thermal mass = longer history needed to detect trends
        if heating_type in ("concrete_ufh", "concrete_slab", "concrete"):
            multiplier = DM_THERMAL_MASS_BUFFER_CONCRETE  # 1.3
        elif heating_type in ("timber", "timber_ufh"):
            multiplier = DM_THERMAL_MASS_BUFFER_TIMBER  # 1.15
        else:
            multiplier = DM_THERMAL_MASS_BUFFER_RADIATOR  # 1.0

        # Scale base history size by thermal mass buffer
        # Base 6 samples × 1.3 = ~8 samples for concrete (40 min)
        # Base 6 samples × 1.15 = ~7 samples for timber (35 min)
        # Base 6 samples × 1.0 = 6 samples for radiator (30 min)
        return round(ANTI_WINDUP_DM_HISTORY_BASE_SIZE * multiplier)

    def _update_dm_history(self, dm: float, timestamp: datetime) -> None:
        """Update DM history for anti-windup tracking.

        Args:
            dm: Current degree minutes value
            timestamp: Current timestamp
        """
        self._dm_history.append((timestamp, dm))

    def _calculate_dm_rate(self) -> tuple[float, bool]:
        """Calculate DM rate of change from history.

        Returns:
            (dm_rate_per_hour, has_valid_data)
            dm_rate is negative when DM is dropping (going more negative)
        """
        if len(self._dm_history) < ANTI_WINDUP_MIN_SAMPLES:
            return 0.0, False

        oldest_time, oldest_dm = self._dm_history[0]
        newest_time, newest_dm = self._dm_history[-1]

        time_diff = (newest_time - oldest_time).total_seconds() / 3600  # hours
        if time_diff < 0.05:  # Less than 3 minutes of data
            return 0.0, False

        dm_change = newest_dm - oldest_dm  # Negative means DM dropped
        dm_rate = dm_change / time_diff  # DM/hour

        return dm_rate, True

    def _is_in_cooldown(self, timestamp: datetime) -> bool:
        """Check if still in cooldown from previous anti-windup activation.

        Args:
            timestamp: Current timestamp

        Returns:
            True if in cooldown period, False otherwise
        """
        if self._anti_windup_cooldown_until is None:
            return False
        return timestamp < self._anti_windup_cooldown_until

    def _start_cooldown(self, timestamp: datetime) -> None:
        """Start cooldown period after anti-windup activation.

        Prevents oscillation by waiting before trying to raise offset again.

        Args:
            timestamp: Current timestamp
        """
        self._anti_windup_cooldown_until = timestamp + timedelta(
            minutes=ANTI_WINDUP_COOLDOWN_MINUTES
        )
        _LOGGER.info(
            "Anti-windup cooldown started: no offset raises until %s",
            self._anti_windup_cooldown_until.strftime("%H:%M"),
        )

    def _track_offset_change(self, timestamp: datetime, new_offset: float) -> None:
        """Track offset changes to detect self-induced spirals.

        Only records RAISES (new_offset > previous). Decreases don't cause spirals.
        Called at the end of evaluate_layer to track the final offset decision.

        Args:
            timestamp: Current timestamp
            new_offset: The offset value being returned
        """
        if self._last_offset_value is None:
            self._last_offset_value = new_offset
            return

        if new_offset > self._last_offset_value:
            # This is a RAISE - record the time
            self._last_offset_raise_time = timestamp
            _LOGGER.debug(
                "Offset raised: %.1f → %.1f at %s",
                self._last_offset_value,
                new_offset,
                timestamp.strftime("%H:%M"),
            )

        self._last_offset_value = new_offset

    def _raised_offset_recently(self, timestamp: datetime) -> bool:
        """Check if we raised offset within the causation window.

        Returns True if we raised offset recently (potential self-induced spiral).
        Returns False if offset has been stable (likely environmental DM drop).

        Args:
            timestamp: Current timestamp

        Returns:
            True if offset was raised within ANTI_WINDUP_CAUSATION_WINDOW_MINUTES
        """
        if self._last_offset_raise_time is None:
            return False

        minutes_since_raise = (timestamp - self._last_offset_raise_time).total_seconds() / 60
        return minutes_since_raise <= ANTI_WINDUP_CAUSATION_WINDOW_MINUTES

    def _check_anti_windup(
        self, current_offset: float, dm_rate: float, has_valid_rate: bool
    ) -> tuple[bool, str]:
        """Check if anti-windup protection should be activated.

        Anti-windup prevents escalating offset when heat is "in transit" through
        thermal mass. This occurs when:
        1. Current offset is already positive (we're adding heat)
        2. DM is still dropping (heat hasn't arrived yet)

        Args:
            current_offset: Current NIBE offset value
            dm_rate: DM rate of change (negative = dropping)
            has_valid_rate: Whether we have enough data to trust the rate

        Returns:
            (anti_windup_active, reason_string)
        """
        if not has_valid_rate:
            return False, ""

        # Check if we're already adding heat (positive offset)
        if current_offset < ANTI_WINDUP_MIN_POSITIVE_OFFSET:
            return False, ""

        # Check if DM is dropping despite positive offset
        if dm_rate > ANTI_WINDUP_DM_DROPPING_RATE:
            return False, ""

        # Anti-windup triggered: heat is in transit, don't escalate
        return True, f"anti-windup: DM dropping {dm_rate:.0f}/h while offset +{current_offset:.0f}"

    def _apply_anti_windup_cap(
        self,
        calculated_offset: float,
        current_offset: float,
        anti_windup_active: bool,
        tier_name: str,
    ) -> tuple[float, str]:
        """Apply anti-windup offset capping if active.

        When anti-windup is triggered (heat in transit), cap the new offset to
        prevent escalation. Allow maintaining or slightly reducing offset.

        Args:
            calculated_offset: Offset calculated by tier logic
            current_offset: Current NIBE offset value
            anti_windup_active: Whether anti-windup is triggered
            tier_name: Tier name for logging

        Returns:
            (final_offset, cap_reason) - reason is empty if no cap applied
        """
        if not anti_windup_active:
            return calculated_offset, ""

        # Calculate capped offset: don't exceed current offset * cap multiplier
        # This prevents escalation while allowing gradual reduction
        max_offset = max(current_offset, current_offset * ANTI_WINDUP_OFFSET_CAP_MULTIPLIER)

        if calculated_offset > max_offset:
            _LOGGER.info(
                "%s anti-windup: capping offset %.2f°C → %.2f°C (heat in transit)",
                tier_name,
                calculated_offset,
                max_offset,
            )
            return max_offset, f"capped from {calculated_offset:.1f}°C"

        return calculated_offset, ""

    def evaluate_layer(
        self,
        nibe_state,
        weather_data,
        price_data,
        target_temp: float,
        tolerance_range: float,
        get_current_datetime: Optional[Callable] = None,
        is_volatile: bool = False,
    ) -> EmergencyLayerDecision:
        """Emergency layer: Smart context-aware thermal debt response.

        CRITICAL RULE (Nov 27, 2025):
        Never heat when indoor temperature exceeds target + tolerance.
        Thermal debt while overheating means "flow temp was correctly reduced".
        Let natural cooling bring temp down first, THEN recover DM during cheap periods.

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data (for damping decisions)
            price_data: Price data (for cheap period detection)
            target_temp: Target indoor temperature
            tolerance_range: Temperature tolerance range
            get_current_datetime: Callable returning current datetime (for testing)
            is_volatile: True if in volatile cheap period (brief duration)

        Returns:
            EmergencyLayerDecision with context-aware emergency response
        """
        degree_minutes = nibe_state.degree_minutes
        outdoor_temp = nibe_state.outdoor_temp
        indoor_temp = nibe_state.indoor_temp
        current_offset = getattr(nibe_state, "current_offset", 0.0)
        timestamp = getattr(nibe_state, "timestamp", datetime.now())

        # Track offset changes for causation detection (Jan 2026)
        # This records when offset was raised to distinguish self-induced spirals
        # from environmental DM drops (e.g., cold snap arriving hours later)
        self._track_offset_change(timestamp, current_offset)

        # Update DM history for anti-windup tracking
        self._update_dm_history(degree_minutes, timestamp)

        # Calculate DM rate and check anti-windup
        dm_rate, has_valid_rate = self._calculate_dm_rate()
        anti_windup_active, anti_windup_reason = self._check_anti_windup(
            current_offset, dm_rate, has_valid_rate
        )

        temp_deviation = indoor_temp - target_temp

        # ========================================
        # ANTI-WINDUP: Prevent offset raises that make DM worse (Jan 2026 fix)
        # ========================================
        # Physics: Raising offset increases S1 (target), but BT25 (actual) can't catch up
        # → Gap (BT25 - S1) becomes MORE negative → DM drops FASTER
        # Solution: If anti-windup detects this pattern, don't calculate higher offsets at all

        # Check 1: Are we in cooldown from a previous anti-windup activation?
        if self._is_in_cooldown(timestamp):
            return EmergencyLayerDecision(
                name="Anti-windup Cooldown",
                offset=current_offset,
                weight=0.5,
                reason=f"In cooldown ({ANTI_WINDUP_COOLDOWN_MINUTES}min) - not raising offset",
                tier="COOLDOWN",
                degree_minutes=degree_minutes,
                anti_windup_active=True,
                dm_rate=dm_rate,
            )

        # Check 2: Is anti-windup detecting a spiral (DM dropping while offset positive)?
        # Jan 2026: Only trigger if we CAUSED the spiral by raising offset recently
        # If offset has been stable for >90 min, DM drop is likely environmental
        if anti_windup_active and current_offset >= ANTI_WINDUP_MIN_POSITIVE_OFFSET:
            # Check causation window - did we raise offset recently?
            if not self._raised_offset_recently(timestamp):
                # Offset has been stable - this is likely environmental, not self-induced
                _LOGGER.info(
                    "Anti-windup skipped: offset stable for >%d min, "
                    "DM drop (%.0f/h) likely environmental",
                    ANTI_WINDUP_CAUSATION_WINDOW_MINUTES,
                    dm_rate,
                )
                # Continue to tier calculations - allow response to environmental change
            else:
                # We raised recently and DM is dropping - self-induced spiral
                # Start cooldown to prevent oscillation
                self._start_cooldown(timestamp)

                # Calculate reduction based on severity (Jan 2026 enhancement)
                # If dm_rate is worse than REDUCTION_THRESHOLD, actively reduce offset
                # Formula: reduction = 1.0°C × (|dm_rate| / 100)
                if dm_rate < ANTI_WINDUP_REDUCTION_THRESHOLD:
                    # Severe spiral - actively reduce offset
                    reduction = (
                        abs(dm_rate) / ANTI_WINDUP_REDUCTION_RATE_DIVISOR
                    ) * ANTI_WINDUP_REDUCTION_MULTIPLIER
                    new_offset = max(-10.0, current_offset - reduction)  # Floor at MIN_OFFSET
                    reason = (
                        f"DM dropping {dm_rate:.0f}/h - reducing offset by {reduction:.1f}°C "
                        f"(from +{current_offset:.0f}°C to {new_offset:.1f}°C)"
                    )
                    _LOGGER.info(
                        "Anti-windup REDUCTION: dm_rate=%d/h, reduction=%.1f°C, "
                        "offset %+.1f → %+.1f",
                        dm_rate,
                        reduction,
                        current_offset,
                        new_offset,
                    )
                else:
                    # Mild spiral - just keep current offset, don't raise
                    new_offset = current_offset
                    reason = (
                        f"DM dropping {dm_rate:.0f}/h while offset +{current_offset:.0f}°C - "
                        f"not raising"
                    )

                return EmergencyLayerDecision(
                    name="Anti-windup",
                    offset=new_offset,
                    weight=0.5,
                    reason=reason,
                    tier="ANTI_WINDUP",
                    degree_minutes=degree_minutes,
                    threshold_used=0.0,
                    anti_windup_active=True,
                    dm_rate=dm_rate,
                )

        # Case 1: Too warm (above tolerance)
        if temp_deviation > tolerance_range:
            return EmergencyLayerDecision(
                name="Thermal Debt",
                offset=0.0,
                weight=0.0,
                reason=f"DM {degree_minutes:.0f} but {temp_deviation:.1f}°C over target - let cool naturally",
                tier="OK",
                degree_minutes=degree_minutes,
            )

        # Case 2: At target + Not cheap (and not at absolute limit)
        # Use _is_price_cheap to check current price classification
        if temp_deviation >= 0 and degree_minutes > DM_THRESHOLD_AUX_LIMIT:
            if not self._is_price_cheap(price_data, get_current_datetime):
                return EmergencyLayerDecision(
                    name="Thermal Debt",
                    offset=0.0,
                    weight=0.0,
                    reason=f"At target & price not cheap - ignoring DM {degree_minutes:.0f}",
                    tier="OK",
                    degree_minutes=degree_minutes,
                )

        # HARD LIMIT: DM -1500 absolute maximum (never exceed)
        if degree_minutes <= DM_THRESHOLD_AUX_LIMIT:
            return EmergencyLayerDecision(
                name="Thermal Debt",
                offset=SAFETY_EMERGENCY_OFFSET,
                weight=1.0,
                reason=f"EMERGENCY: DM {degree_minutes:.0f} at aux limit -1500",
                tier="EMERGENCY",
                degree_minutes=degree_minutes,
                threshold_used=DM_THRESHOLD_AUX_LIMIT,
            )

        # Calculate context-aware thresholds based on outdoor temperature
        expected_dm_range = self.climate_detector.get_expected_dm_range(outdoor_temp)
        adjusted_dm_range = self._get_thermal_mass_adjusted_thresholds(expected_dm_range)
        expected_dm = {
            "normal": adjusted_dm_range["normal_max"],
            "warning": adjusted_dm_range["warning"],
        }

        # Distance from absolute maximum
        margin_to_limit = degree_minutes - DM_THRESHOLD_AUX_LIMIT

        # Calculate climate-aware tier thresholds
        warning_threshold = expected_dm["warning"]
        t1_threshold = warning_threshold - DM_CRITICAL_T1_MARGIN
        t2_threshold = warning_threshold - DM_CRITICAL_T2_MARGIN
        t3_threshold = max(
            warning_threshold - DM_CRITICAL_T3_MARGIN,
            DM_CRITICAL_T3_MAX,
        )

        # CRITICAL TIER 3
        if degree_minutes <= t3_threshold:
            base_offset = DM_CRITICAL_T3_OFFSET
            damped_offset, damping_reason = self._apply_thermal_recovery_damping(
                base_offset=base_offset,
                tier_name="T3",
                min_damped_offset=THERMAL_RECOVERY_T3_MIN_OFFSET,
                weather_data=weather_data,
                current_outdoor_temp=outdoor_temp,
                indoor_temp=indoor_temp,
                target_temp=target_temp,
            )

            # Apply anti-windup cap (T3 is critical, so we still apply but note it)
            final_offset, cap_reason = self._apply_anti_windup_cap(
                damped_offset, current_offset, anti_windup_active, "T3"
            )

            # Skip boost during volatile cheap period (shared logic)
            if should_skip_volatile_boost(is_volatile, final_offset):
                final_offset = 0.0
                volatile_suffix = "[volatile: skipped]"
            else:
                volatile_suffix = ""

            reason_parts = [
                f"DM {degree_minutes:.0f} near absolute max "
                f"(threshold: {t3_threshold:.0f}, margin: {margin_to_limit:.0f})"
            ]
            if damping_reason:
                reason_parts.append(f"[{damping_reason}]")
            if anti_windup_active:
                reason_parts.append(f"[{anti_windup_reason}]")
            if volatile_suffix:
                reason_parts.append(volatile_suffix.strip())

            # Apply weight reduction during volatile cheap periods
            weight = DM_CRITICAL_T3_WEIGHT
            if is_volatile:
                weight = weight * VOLATILE_WEIGHT_REDUCTION

            return EmergencyLayerDecision(
                name="T3",
                offset=final_offset,
                weight=weight,
                reason=" ".join(reason_parts),
                tier="T3",
                degree_minutes=degree_minutes,
                threshold_used=t3_threshold,
                damping_applied=bool(damping_reason),
                anti_windup_active=anti_windup_active,
                dm_rate=dm_rate,
            )

        # TIER 2
        if degree_minutes <= t2_threshold:
            base_offset = DM_CRITICAL_T2_OFFSET
            damped_offset, damping_reason = self._apply_thermal_recovery_damping(
                base_offset=base_offset,
                tier_name="T2",
                min_damped_offset=THERMAL_RECOVERY_T2_MIN_OFFSET,
                weather_data=weather_data,
                current_outdoor_temp=outdoor_temp,
                indoor_temp=indoor_temp,
                target_temp=target_temp,
            )

            # Apply anti-windup cap to prevent escalation
            final_offset, cap_reason = self._apply_anti_windup_cap(
                damped_offset, current_offset, anti_windup_active, "T2"
            )

            # Skip boost during volatile cheap period (shared logic)
            if should_skip_volatile_boost(is_volatile, final_offset):
                final_offset = 0.0
                volatile_suffix = "[volatile: skipped]"
            else:
                volatile_suffix = ""

            reason_parts = [
                f"DM {degree_minutes:.0f} approaching T3 "
                f"(threshold: {t2_threshold:.0f}, margin: {margin_to_limit:.0f})"
            ]
            if damping_reason:
                reason_parts.append(f"[{damping_reason}]")
            if anti_windup_active:
                reason_parts.append(f"[{anti_windup_reason}]")
            if volatile_suffix:
                reason_parts.append(volatile_suffix.strip())

            # Apply weight reduction during volatile cheap periods
            weight = DM_CRITICAL_T2_WEIGHT
            if is_volatile:
                weight = weight * VOLATILE_WEIGHT_REDUCTION

            return EmergencyLayerDecision(
                name="T2",
                offset=final_offset,
                weight=weight,
                reason=" ".join(reason_parts),
                tier="T2",
                degree_minutes=degree_minutes,
                threshold_used=t2_threshold,
                damping_applied=bool(damping_reason),
                anti_windup_active=anti_windup_active,
                dm_rate=dm_rate,
            )

        # TIER 1
        if degree_minutes <= t1_threshold:
            base_offset = DM_CRITICAL_T1_OFFSET
            damped_offset, damping_reason = self._apply_thermal_recovery_damping(
                base_offset=base_offset,
                tier_name="T1",
                min_damped_offset=THERMAL_RECOVERY_T1_MIN_OFFSET,
                weather_data=weather_data,
                current_outdoor_temp=outdoor_temp,
                indoor_temp=indoor_temp,
                target_temp=target_temp,
            )

            # Apply anti-windup cap to prevent escalation
            final_offset, cap_reason = self._apply_anti_windup_cap(
                damped_offset, current_offset, anti_windup_active, "T1"
            )

            # Skip boost during volatile cheap period (shared logic)
            if should_skip_volatile_boost(is_volatile, final_offset):
                final_offset = 0.0
                volatile_suffix = "[volatile: skipped]"
            else:
                volatile_suffix = ""

            reason_parts = [
                f"DM {degree_minutes:.0f} beyond expected for {outdoor_temp:.1f}°C "
                f"(threshold: {t1_threshold:.0f})"
            ]
            if damping_reason:
                reason_parts.append(f"[{damping_reason}]")
            if anti_windup_active:
                reason_parts.append(f"[{anti_windup_reason}]")
            if volatile_suffix:
                reason_parts.append(volatile_suffix.strip())

            # Apply weight reduction during volatile cheap periods
            weight = DM_CRITICAL_T1_WEIGHT
            if is_volatile:
                weight = weight * VOLATILE_WEIGHT_REDUCTION

            return EmergencyLayerDecision(
                name="T1",
                offset=final_offset,
                weight=weight,
                reason=" ".join(reason_parts),
                tier="T1",
                degree_minutes=degree_minutes,
                threshold_used=t1_threshold,
                damping_applied=bool(damping_reason),
                anti_windup_active=anti_windup_active,
                dm_rate=dm_rate,
            )

        # WARNING: Beyond expected range
        if degree_minutes < expected_dm["warning"]:
            deviation = expected_dm["warning"] - degree_minutes

            if deviation > WARNING_DEVIATION_THRESHOLD:
                offset = min(
                    WARNING_OFFSET_MAX_SEVERE,
                    WARNING_OFFSET_MIN_SEVERE + (deviation / WARNING_DEVIATION_DIVISOR_SEVERE),
                )
            else:
                offset = min(
                    WARNING_OFFSET_MAX_MODERATE,
                    WARNING_OFFSET_MIN_MODERATE + (deviation / WARNING_DEVIATION_DIVISOR_MODERATE),
                )

            # Apply anti-windup cap to WARNING tier too
            final_offset, cap_reason = self._apply_anti_windup_cap(
                offset, current_offset, anti_windup_active, "WARNING"
            )

            # Skip boost during volatile cheap period (shared logic)
            if should_skip_volatile_boost(is_volatile, final_offset):
                final_offset = 0.0
                volatile_suffix = " [volatile: skipped]"
            else:
                volatile_suffix = ""

            reason = (
                f"DM {degree_minutes:.0f} beyond expected for "
                f"{outdoor_temp:.1f}°C (expected: {expected_dm['normal']:.0f}, "
                f"warning: {expected_dm['warning']:.0f}, deviation: {deviation:.0f})"
            )
            if anti_windup_active:
                reason += f" [{anti_windup_reason}]"
            if volatile_suffix:
                reason += volatile_suffix

            # Apply weight reduction during volatile cheap periods
            weight = LAYER_WEIGHT_EMERGENCY
            if is_volatile:
                weight = weight * VOLATILE_WEIGHT_REDUCTION

            return EmergencyLayerDecision(
                name="Thermal Debt Warning",
                offset=final_offset,
                weight=weight,
                reason=reason,
                tier="WARNING",
                degree_minutes=degree_minutes,
                threshold_used=expected_dm["warning"],
                anti_windup_active=anti_windup_active,
                dm_rate=dm_rate,
            )

        # CAUTION: Approaching limits
        if degree_minutes < expected_dm["normal"]:
            return EmergencyLayerDecision(
                name="Thermal Debt",
                offset=WARNING_CAUTION_OFFSET,
                weight=WARNING_CAUTION_WEIGHT,
                reason=f"DM {degree_minutes:.0f} approaching limits - monitoring",
                tier="CAUTION",
                degree_minutes=degree_minutes,
                threshold_used=expected_dm["normal"],
                dm_rate=dm_rate,
            )

        # OK: Within normal range
        return EmergencyLayerDecision(
            name="Thermal Debt",
            offset=0.0,
            weight=0.0,
            reason=f"OK (DM: {degree_minutes:.0f})",
            tier="OK",
            degree_minutes=degree_minutes,
            dm_rate=dm_rate,
        )

    def _is_price_cheap(self, price_data, get_current_datetime: Optional[Callable] = None) -> bool:
        """Check if current price is classified as CHEAP or VERY_CHEAP."""
        if not price_data or not price_data.today or not self.price_analyzer:
            return False

        try:
            if get_current_datetime:
                now = get_current_datetime()
            else:
                now = None  # get_current_quarter will use dt_util.now()

            current_quarter = get_current_quarter(now)
            if current_quarter < len(price_data.today):
                classification = self.price_analyzer.get_current_classification(current_quarter)
                return classification in (
                    QuarterClassification.CHEAP,
                    QuarterClassification.VERY_CHEAP,
                )
        except (AttributeError, IndexError, TypeError):
            pass  # Price data unavailable or malformed - return False (default to not cheap)

        return False

    def _get_thermal_mass_adjusted_thresholds(self, base_thresholds: dict) -> dict:
        """Adjust DM thresholds based on thermal mass.

        High thermal mass systems need tighter thresholds because:
        - Long thermal lag (6+ hours for concrete slab)
        - Current DM doesn't immediately affect indoor temperature
        - Solar gain can mask underlying thermal debt accumulation

        Args:
            base_thresholds: Climate-aware thresholds from ClimateZoneDetector

        Returns:
            Adjusted thresholds with thermal mass buffer applied
        """
        if self.heating_type in ("concrete_ufh", "concrete_slab"):
            multiplier = DM_THERMAL_MASS_BUFFER_CONCRETE
        elif self.heating_type in ("timber", "timber_ufh"):
            multiplier = DM_THERMAL_MASS_BUFFER_TIMBER
        else:
            multiplier = DM_THERMAL_MASS_BUFFER_RADIATOR

        adjusted = {
            "normal_min": base_thresholds["normal_min"] * multiplier,
            "normal_max": base_thresholds["normal_max"] * multiplier,
            "warning": base_thresholds["warning"] * multiplier,
            "critical": base_thresholds["critical"],  # Never adjust absolute maximum
        }

        _LOGGER.debug(
            "Thermal mass adjusted thresholds: heating type '%s' (multiplier %.2f) "
            "→ warning %.0f (base: %.0f)",
            self.heating_type,
            multiplier,
            adjusted["warning"],
            base_thresholds["warning"],
        )

        return adjusted

    def _apply_thermal_recovery_damping(
        self,
        base_offset: float,
        tier_name: str,
        min_damped_offset: float,
        weather_data=None,
        current_outdoor_temp: Optional[float] = None,
        indoor_temp: Optional[float] = None,
        target_temp: Optional[float] = None,
    ) -> tuple[float, str]:
        """Apply thermal recovery damping when house warming naturally or overshooting.

        Prevents concrete slab thermal overshoot during recovery periods.

        Args:
            base_offset: Original recovery offset
            tier_name: Recovery tier name for logging
            min_damped_offset: Minimum allowed offset after damping
            weather_data: Weather forecast data
            current_outdoor_temp: Current outdoor temperature
            indoor_temp: Current indoor temperature
            target_temp: Target indoor temperature

        Returns:
            Tuple of (damped_offset, damping_reason_string)
        """
        thermal_trend = self._get_thermal_trend() or {}
        outdoor_trend = self._get_outdoor_trend() or {}

        # Calculate overshoot damping factor
        overshoot_factor = 1.0
        overshoot_reason = ""

        if indoor_temp is not None and target_temp is not None:
            overshoot = indoor_temp - target_temp
            if overshoot >= THERMAL_RECOVERY_OVERSHOOT_SEVERE_THRESHOLD:
                overshoot_factor = THERMAL_RECOVERY_OVERSHOOT_SEVERE_DAMPING
                overshoot_reason = f"severe overshoot +{overshoot:.1f}°C"
            elif overshoot >= THERMAL_RECOVERY_OVERSHOOT_MODERATE_THRESHOLD:
                overshoot_factor = THERMAL_RECOVERY_OVERSHOOT_MODERATE_DAMPING
                overshoot_reason = f"moderate overshoot +{overshoot:.1f}°C"
            elif overshoot >= THERMAL_RECOVERY_OVERSHOOT_MILD_THRESHOLD:
                overshoot_factor = THERMAL_RECOVERY_OVERSHOOT_MILD_DAMPING
                overshoot_reason = f"mild overshoot +{overshoot:.1f}°C"

        # Check trend confidence
        if thermal_trend.get("confidence", 0.0) < THERMAL_RECOVERY_MIN_CONFIDENCE:
            if overshoot_factor < 1.0:
                damped_offset = max(base_offset * overshoot_factor, min_damped_offset)
                _LOGGER.info(
                    "%s overshoot damping: %s, offset %.2f°C → %.2f°C",
                    tier_name,
                    overshoot_reason,
                    base_offset,
                    damped_offset,
                )
                return damped_offset, overshoot_reason
            return base_offset, ""

        warming_rate = thermal_trend.get("rate_per_hour", 0.0)
        outdoor_rate = outdoor_trend.get("rate_per_hour", 0.0)

        # Check weather forecast for incoming cold
        if weather_data and weather_data.forecast_hours and current_outdoor_temp is not None:
            forecast_horizon = int(THERMAL_RECOVERY_FORECAST_HORIZON)
            forecast_temps = [h.temperature for h in weather_data.forecast_hours[:forecast_horizon]]

            if forecast_temps:
                min_forecast_temp = min(forecast_temps)
                forecast_drop = min_forecast_temp - current_outdoor_temp

                if forecast_drop < THERMAL_RECOVERY_FORECAST_DROP_THRESHOLD:
                    _LOGGER.info(
                        "%s: Cold weather forecast (%.1f°C drop), maintaining full offset %.2f°C",
                        tier_name,
                        abs(forecast_drop),
                        base_offset,
                    )
                    return base_offset, ""

        # Apply damping if warming naturally
        if (
            warming_rate >= THERMAL_RECOVERY_WARMING_THRESHOLD
            and outdoor_rate >= THERMAL_RECOVERY_OUTDOOR_DROPPING_THRESHOLD
        ):
            if warming_rate >= THERMAL_RECOVERY_RAPID_THRESHOLD:
                warming_factor = THERMAL_RECOVERY_RAPID_FACTOR
                warming_reason = f"rapid warming {warming_rate:.2f}°C/h"
            else:
                warming_factor = THERMAL_RECOVERY_DAMPING_FACTOR
                warming_reason = f"warming {warming_rate:.2f}°C/h"

            combined_factor = warming_factor * overshoot_factor

            if overshoot_reason:
                damping_reason = f"{warming_reason} + {overshoot_reason}"
            else:
                damping_reason = warming_reason

            damped_offset = max(base_offset * combined_factor, min_damped_offset)

            _LOGGER.info(
                "%s thermal recovery damping: %s, offset %.2f°C → %.2f°C",
                tier_name,
                damping_reason,
                base_offset,
                damped_offset,
            )

            return damped_offset, damping_reason

        # Apply overshoot damping only if no warming detected
        if overshoot_factor < 1.0:
            damped_offset = max(base_offset * overshoot_factor, min_damped_offset)
            return damped_offset, overshoot_reason

        return base_offset, ""

    def should_block_dhw(self, degree_minutes: float, outdoor_temp: float) -> bool:
        """Determine if DHW should be blocked due to thermal debt.

        Provides shared logic for DHW optimizer to prevent DHW heating
        during critical thermal debt conditions.

        Args:
            degree_minutes: Current degree minutes value
            outdoor_temp: Current outdoor temperature

        Returns:
            True if DHW should be blocked, False otherwise
        """
        # Absolute limit always blocks DHW
        if degree_minutes <= DM_THRESHOLD_AUX_LIMIT:
            return True

        # Get climate-aware thresholds
        expected_dm_range = self.climate_detector.get_expected_dm_range(outdoor_temp)
        adjusted_dm_range = self._get_thermal_mass_adjusted_thresholds(expected_dm_range)

        # Block at T2 threshold or worse
        warning_threshold = adjusted_dm_range["warning"]
        t2_threshold = warning_threshold - DM_CRITICAL_T2_MARGIN

        if degree_minutes <= t2_threshold:
            _LOGGER.info(
                "DHW blocked: DM %.0f at T2 threshold %.0f (outdoor %.1f°C)",
                degree_minutes,
                t2_threshold,
                outdoor_temp,
            )
            return True

        return False

    def get_adjusted_dm_thresholds(self, outdoor_temp: float) -> dict:
        """Get thermal mass adjusted DM thresholds for DHW decisions.

        Provides consistent thresholds for DHW optimizer to use for both
        blocking AND abort conditions, ensuring no mismatch.

        Args:
            outdoor_temp: Current outdoor temperature

        Returns:
            Dict with 'warning', 'critical', 'normal_min', 'normal_max' keys,
            all adjusted for thermal mass.
        """
        expected_dm_range = self.climate_detector.get_expected_dm_range(outdoor_temp)
        return self._get_thermal_mass_adjusted_thresholds(expected_dm_range)


class ProactiveLayer:
    """Proactive thermal debt prevention with climate-aware thresholds.

    PHILOSOPHY:
    - Continuous modulation beats forced cycling
    - Thermal debt from forced stops worse than running low power
    - Prevent peaks by maintaining gentle background heating

    PREDICTIVE TREND ANALYSIS:
    Uses indoor temperature trend to detect problems 30-60 minutes before they occur.
    Rapid cooling (-0.3°C/h or faster) combined with outdoor cold and temperature deficit
    indicates thermal debt will develop soon - intervene proactively.

    CLIMATE-AWARE DESIGN:
    Thresholds calculated as percentages of climate-aware expected DM (normal_max):
    - Zone 1 (15%): Early warning, gentle nudge before compressor starts
    - Zone 2 (40%): Moderate action when compressor running
    - Zone 3 (80%): Strong action when approaching warning threshold
    """

    def __init__(
        self,
        climate_detector: ClimateZoneDetector,
        get_thermal_trend: Optional[Callable[[], dict]] = None,
    ):
        """Initialize proactive layer.

        Args:
            climate_detector: ClimateZoneDetector for context-aware thresholds
            get_thermal_trend: Callable returning thermal trend dict
        """
        self.climate_detector = climate_detector
        self._get_thermal_trend = get_thermal_trend or (
            lambda: {"rate_per_hour": 0.0, "confidence": 0.0}
        )

    def _is_house_warm(self, indoor_temp: float, target_temp: float) -> bool:
        """Check if house is warm enough to reduce proactive heating.

        When house is above target + threshold, DM recovery is less urgent
        because COAST (overshoot protection) should handle temperature.
        Reduces proactive layer weight to prevent fighting with COAST.

        Args:
            indoor_temp: Current indoor temperature (°C)
            target_temp: Target indoor temperature (°C)

        Returns:
            True if house is warm (above target + threshold)
        """
        return indoor_temp > target_temp + PROACTIVE_WARM_HOUSE_THRESHOLD

    def _apply_warm_house_reduction(
        self, weight: float, indoor_temp: float, target_temp: float, zone: str
    ) -> tuple[float, str]:
        """Apply weight reduction when house is warm.

        Z5 is exempt because at warning boundary, DM recovery takes priority.

        Args:
            weight: Original weight
            indoor_temp: Current indoor temperature (°C)
            target_temp: Target indoor temperature (°C)
            zone: Zone identifier (Z1-Z5, RAPID_COOLING)

        Returns:
            Tuple of (adjusted_weight, reason_suffix)
        """
        # Z5 exempt: at warning boundary, prioritize DM recovery
        if zone == "Z5":
            return weight, ""

        if self._is_house_warm(indoor_temp, target_temp):
            overshoot = indoor_temp - target_temp
            reduced_weight = weight * PROACTIVE_WARM_HOUSE_WEIGHT_REDUCTION
            return (
                reduced_weight,
                f" [warm house: +{overshoot:.1f}°C, weight {weight:.2f}→{reduced_weight:.2f}]",
            )

        return weight, ""

    def evaluate_layer(
        self,
        nibe_state,
        weather_data,
        target_temp: float,
        is_volatile: bool = False,
    ) -> ProactiveLayerDecision:
        """Proactive debt prevention with climate-aware thresholds and trend prediction.

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data (for forecast validation)
            target_temp: Target indoor temperature
            is_volatile: True if in volatile cheap period (brief duration)

        Returns:
            ProactiveLayerDecision with climate-aware proactive gentle heating
        """
        degree_minutes = nibe_state.degree_minutes
        outdoor_temp = nibe_state.outdoor_temp
        indoor_temp = nibe_state.indoor_temp

        # Get temperature trend for predictive intervention
        thermal_trend = self._get_thermal_trend()
        trend_rate = thermal_trend.get("rate_per_hour", 0.0)

        # Calculate current deficit
        deficit = target_temp - indoor_temp

        # Predict deficit in 1 hour if trend continues
        predicted_deficit_1h = deficit - trend_rate  # Negative rate increases deficit

        # ========================================
        # RAPID COOLING DETECTION (Predictive)
        # ========================================
        if is_cooling_rapidly(thermal_trend):
            if outdoor_temp < RAPID_COOLING_OUTDOOR_THRESHOLD and deficit > THERMAL_CHANGE_MODERATE:
                boost = min(
                    abs(trend_rate) * RAPID_COOLING_BOOST_MULTIPLIER,
                    RAPID_COOLING_BOOST_MAX,
                )

                # Validate against weather forecast
                forecast_reason = ""
                forecast_validated = False
                if weather_data and weather_data.forecast_hours:
                    next_3h_temps = [h.temperature for h in weather_data.forecast_hours[:3]]
                    if next_3h_temps:
                        min_forecast_temp = min(next_3h_temps)
                        temp_will_drop = min_forecast_temp < outdoor_temp - 1.0

                        if temp_will_drop:
                            boost *= MULTIPLIER_BOOST_30_PERCENT
                            forecast_reason = " (forecast confirms cooling)"
                            forecast_validated = True
                        else:
                            boost *= MULTIPLIER_REDUCTION_20_PERCENT
                            forecast_reason = " (forecast stable, likely temporary)"

                # Skip boost during volatile cheap period (shared logic)
                if should_skip_volatile_boost(is_volatile, boost):
                    boost = 0.0
                    volatile_suffix = " [volatile: skipped]"
                else:
                    volatile_suffix = ""

                # Apply warm house weight reduction
                weight = RAPID_COOLING_WEIGHT
                weight, warm_suffix = self._apply_warm_house_reduction(
                    weight, indoor_temp, target_temp, "RAPID_COOLING"
                )

                # Apply weight reduction during volatile cheap periods
                if is_volatile:
                    weight = weight * VOLATILE_WEIGHT_REDUCTION

                return ProactiveLayerDecision(
                    name="Proactive",
                    offset=boost,
                    weight=weight,
                    reason=(
                        f"Rapid cooling ({trend_rate:.2f}°C/h), "
                        f"deficit {deficit:.1f}°C → {predicted_deficit_1h:.1f}°C in 1h"
                        f"{forecast_reason}{warm_suffix}{volatile_suffix}"
                    ),
                    zone="RAPID_COOLING",
                    degree_minutes=degree_minutes,
                    trend_rate=trend_rate,
                    forecast_validated=forecast_validated,
                )

        # Get climate-aware expected DM range for current conditions
        expected_dm = self._calculate_expected_dm_for_temperature(outdoor_temp)

        # Climate-aware thresholds
        zone1_threshold = expected_dm["normal"] * PROACTIVE_ZONE1_THRESHOLD_PERCENT
        zone2_threshold = expected_dm["normal"] * PROACTIVE_ZONE2_THRESHOLD_PERCENT
        zone3_threshold = expected_dm["normal"] * PROACTIVE_ZONE3_THRESHOLD_PERCENT
        zone4_threshold = expected_dm["normal"] * PROACTIVE_ZONE4_THRESHOLD_PERCENT
        zone5_threshold = expected_dm["normal"] * PROACTIVE_ZONE5_THRESHOLD_PERCENT

        # PROACTIVE ZONE 1
        if zone2_threshold < degree_minutes <= zone1_threshold:
            offset = PROACTIVE_ZONE1_OFFSET

            if trend_rate < THERMAL_CHANGE_MODERATE_COOLING:
                offset *= MULTIPLIER_BOOST_30_PERCENT
                reason_suffix = f" (trend: {trend_rate:.2f}°C/h)"
            else:
                reason_suffix = ""

            # Skip boost during volatile cheap period (shared logic)
            if should_skip_volatile_boost(is_volatile, offset):
                offset = 0.0
                volatile_suffix = " [volatile: skipped]"
            else:
                volatile_suffix = ""

            # Apply warm house weight reduction
            weight = LAYER_WEIGHT_PROACTIVE_MIN
            weight, warm_suffix = self._apply_warm_house_reduction(
                weight, indoor_temp, target_temp, "Z1"
            )

            # Apply weight reduction during volatile cheap periods
            if is_volatile:
                weight = weight * VOLATILE_WEIGHT_REDUCTION

            return ProactiveLayerDecision(
                name="Z1",
                offset=offset,
                weight=weight,
                reason=f"DM {degree_minutes:.0f}, gentle heating prevents debt{reason_suffix}{warm_suffix}{volatile_suffix}",
                zone="Z1",
                degree_minutes=degree_minutes,
                trend_rate=trend_rate,
            )

        # PROACTIVE ZONE 2
        if zone3_threshold < degree_minutes <= zone2_threshold:
            offset = PROACTIVE_ZONE2_OFFSET

            # Skip boost during volatile cheap period (shared logic)
            if should_skip_volatile_boost(is_volatile, offset):
                offset = 0.0
                volatile_suffix = " [volatile: skipped]"
            else:
                volatile_suffix = ""

            # Apply warm house weight reduction
            weight = PROACTIVE_ZONE2_WEIGHT
            weight, warm_suffix = self._apply_warm_house_reduction(
                weight, indoor_temp, target_temp, "Z2"
            )

            # Apply weight reduction during volatile cheap periods
            if is_volatile:
                weight = weight * VOLATILE_WEIGHT_REDUCTION

            return ProactiveLayerDecision(
                name="Z2",
                offset=offset,
                weight=weight,
                reason=f"DM {degree_minutes:.0f}, boost recovery speed{warm_suffix}{volatile_suffix}",
                zone="Z2",
                degree_minutes=degree_minutes,
                trend_rate=trend_rate,
            )

        # PROACTIVE ZONE 3
        if zone4_threshold < degree_minutes <= zone3_threshold:
            deficit_severity = (zone3_threshold - degree_minutes) / (
                zone3_threshold - zone4_threshold
            )
            offset = PROACTIVE_ZONE3_OFFSET_MIN + (deficit_severity * PROACTIVE_ZONE3_OFFSET_RANGE)

            # Skip boost during volatile cheap period (shared logic)
            if should_skip_volatile_boost(is_volatile, offset):
                offset = 0.0
                volatile_suffix = " [volatile: skipped]"
            else:
                volatile_suffix = ""

            # Apply warm house weight reduction
            weight = PROACTIVE_ZONE3_WEIGHT
            weight, warm_suffix = self._apply_warm_house_reduction(
                weight, indoor_temp, target_temp, "Z3"
            )

            # Apply weight reduction during volatile cheap periods
            if is_volatile:
                weight = weight * VOLATILE_WEIGHT_REDUCTION

            return ProactiveLayerDecision(
                name="Z3",
                offset=offset,
                weight=weight,
                reason=f"DM {degree_minutes:.0f}, prevent deeper debt{warm_suffix}{volatile_suffix}",
                zone="Z3",
                degree_minutes=degree_minutes,
                trend_rate=trend_rate,
            )

        # PROACTIVE ZONE 4
        if zone5_threshold < degree_minutes <= zone4_threshold:
            offset = PROACTIVE_ZONE4_OFFSET

            # Skip boost during volatile cheap period (shared logic)
            if should_skip_volatile_boost(is_volatile, offset):
                offset = 0.0
                volatile_suffix = " [volatile: skipped]"
            else:
                volatile_suffix = ""

            # Apply warm house weight reduction
            weight = PROACTIVE_ZONE4_WEIGHT
            weight, warm_suffix = self._apply_warm_house_reduction(
                weight, indoor_temp, target_temp, "Z4"
            )

            # Apply weight reduction during volatile cheap periods
            if is_volatile:
                weight = weight * VOLATILE_WEIGHT_REDUCTION

            return ProactiveLayerDecision(
                name="Z4",
                offset=offset,
                weight=weight,
                reason=f"DM {degree_minutes:.0f}, strong prevention{warm_suffix}{volatile_suffix}",
                zone="Z4",
                degree_minutes=degree_minutes,
                trend_rate=trend_rate,
            )

        # PROACTIVE ZONE 5 (no warm house reduction - at warning boundary)
        if expected_dm["warning"] < degree_minutes <= zone5_threshold:
            offset = PROACTIVE_ZONE5_OFFSET

            # Skip boost during volatile cheap period (shared logic)
            if should_skip_volatile_boost(is_volatile, offset):
                offset = 0.0
                volatile_suffix = " [volatile: skipped]"
            else:
                volatile_suffix = ""

            # Apply weight reduction during volatile cheap periods
            weight = PROACTIVE_ZONE5_WEIGHT
            if is_volatile:
                weight = weight * VOLATILE_WEIGHT_REDUCTION

            return ProactiveLayerDecision(
                name="Z5",
                offset=offset,
                weight=weight,
                reason=f"DM {degree_minutes:.0f}, approaching warning{volatile_suffix}",
                zone="Z5",
                degree_minutes=degree_minutes,
                trend_rate=trend_rate,
            )

        # Outside proactive zones
        return ProactiveLayerDecision(
            name="Proactive",
            offset=0.0,
            weight=0.0,
            reason="Not needed",
            zone="NONE",
            degree_minutes=degree_minutes,
            trend_rate=trend_rate,
        )

    def _calculate_expected_dm_for_temperature(self, outdoor_temp: float) -> dict:
        """Calculate expected DM range for given outdoor temperature.

        Args:
            outdoor_temp: Current outdoor temperature (°C)

        Returns:
            Dictionary with normal and warning thresholds
        """
        dm_range = self.climate_detector.get_expected_dm_range(outdoor_temp)

        return {
            "normal": dm_range["normal_max"],
            "warning": dm_range["warning"],
        }

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
from typing import Callable, Optional

from homeassistant.util import dt as dt_util

from ..const import (
    THERMAL_MASS_CONCRETE_UFH_THRESHOLD,
    THERMAL_MASS_TIMBER_UFH_THRESHOLD,
    UFH_CONCRETE_PREDICTION_HORIZON,
    UFH_RADIATOR_PREDICTION_HORIZON,
    UFH_TIMBER_PREDICTION_HORIZON,
    ANTI_WINDUP_CAUSATION_WINDOW_MINUTES,
    ANTI_WINDUP_COOLDOWN_MINUTES,
    ANTI_WINDUP_DM_DROPPING_RATE,
    ANTI_WINDUP_MIN_POSITIVE_OFFSET,
    ANTI_WINDUP_MIN_SAMPLES,
    ANTI_WINDUP_MIN_SPAN_MINUTES,
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
    DM_RATE_WINDOW_MINUTES,
    DM_THERMAL_MASS_BUFFER_TIMBER,
    DM_THRESHOLD_AUX_LIMIT,
    LAYER_WEIGHT_EMERGENCY,
    LAYER_WEIGHT_PROACTIVE_MIN,
    MIN_OFFSET,
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
    RAPID_COOLING_BOOST_MAX,
    RAPID_COOLING_BOOST_MULTIPLIER,
    RAPID_COOLING_OUTDOOR_THRESHOLD,
    RAPID_COOLING_THRESHOLD,
    RAPID_COOLING_WEIGHT,
    THERMAL_CHANGE_MODERATE,
    THERMAL_CHANGE_MODERATE_COOLING,
    THERMAL_RECOVERY_DAMPING_FACTOR,
    THERMAL_RECOVERY_FORECAST_DROP_THRESHOLD,
    THERMAL_RECOVERY_FORECAST_HORIZON,
    THERMAL_RECOVERY_MIN_CONFIDENCE,
    THERMAL_RECOVERY_OUTDOOR_DROPPING_THRESHOLD,
    THERMAL_RECOVERY_RAPID_FACTOR,
    THERMAL_RECOVERY_RAPID_THRESHOLD,
    THERMAL_RECOVERY_T1_MIN_OFFSET,
    THERMAL_RECOVERY_T2_MIN_OFFSET,
    THERMAL_RECOVERY_T3_MIN_OFFSET,
    THERMAL_RECOVERY_WARMING_THRESHOLD,
    UPDATE_INTERVAL_MINUTES,
    VOLATILE_WEIGHT_REDUCTION,
    # DM recovery constants (shared with dhw_optimizer)
    DM_RECOVERY_MAX_HOURS,
    DM_RECOVERY_RATE_COLD,
    DM_RECOVERY_RATE_MILD,
    DM_RECOVERY_RATE_VERY_COLD,
)
from .climate_zones import ClimateZoneDetector, keep_triggers_clear_of_the_compressor_band
from ..utils.degree_minutes import dm_rate_per_hour
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
    # Authoritative discriminator for safety dispatch in the decision engine.
    # "EMERGENCY" (DM past the aux limit), "T3", "T2", "T1", "WARNING", "CAUTION",
    # "COOLDOWN", "ANTI_WINDUP", "OK". See const.DM_TIER_EMERGENCY / DM_RECOVERY_TIERS.
    tier: str = ""
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
        """How far ahead this house must look to act in time. See prediction_horizon_for."""
        return prediction_horizon_for(self.thermal_mass)


def prediction_horizon_for(thermal_mass: float) -> float:
    """How far ahead a house of this thermal mass must look to act in time - heavier fabric, longer lag.

    A concrete slab's slow time constant (~70 h) makes six hours its LAG, not its horizon: a
    flat 12 h window cannot see the slow, deep outdoor slide that actually drains it (only a
    sudden plunge, which the pump's own curve already catches), so the pre-heat never fires.
    Horizon is UFH-type-specific; see docs/research/03_concrete_slab_response.md and the
    UFH_*_PREDICTION_HORIZON constants.

    Shared by ThermalModel and AdaptiveThermalModel, so the weather pre-heat and the learned
    pre-heat read the same horizon from the same setting: the owner's thermal mass.

    Returns:
        Prediction horizon in hours.
    """
    if thermal_mass >= THERMAL_MASS_CONCRETE_UFH_THRESHOLD:
        return UFH_CONCRETE_PREDICTION_HORIZON
    if thermal_mass >= THERMAL_MASS_TIMBER_UFH_THRESHOLD:
        return UFH_TIMBER_PREDICTION_HORIZON
    return UFH_RADIATOR_PREDICTION_HORIZON


def apply_thermal_mass_buffer(base_thresholds: dict, heating_type: str) -> dict:
    """Move the degree-minute thresholds to suit how slowly this house responds.

    The slower the emitter, the SOONER it must start recovering. Degree minutes are NEGATIVE, so the
    buffer DIVIDES (multiplying would deepen the threshold and delay the response).

    Shared by EmergencyLayer and ProactiveLayer on purpose: a threshold is a property of the house,
    not the layer. When only one layer applied the buffer there was a band of DM in which neither
    responded.

    The divide can undo the warm-side ceiling, so keep_triggers_clear_of_the_compressor_band is
    re-applied HERE, on the value the layers actually read - clamping before the last thing that
    changes it is no clamp at all (e.g. -110 / 1.3 = -85, back inside the compressor's own cycling
    band, once flagged a warm-morning slab house as in thermal debt).

    Args:
        base_thresholds: Climate-aware thresholds from ClimateZoneDetector
        heating_type: "radiator", "concrete_ufh", "concrete_slab", "timber", "timber_ufh"

    Returns:
        The same thresholds, moved to suit the emitter's thermal lag.
    """
    if heating_type in ("concrete_ufh", "concrete_slab"):
        multiplier = DM_THERMAL_MASS_BUFFER_CONCRETE
    elif heating_type in ("timber", "timber_ufh"):
        multiplier = DM_THERMAL_MASS_BUFFER_TIMBER
    else:
        multiplier = DM_THERMAL_MASS_BUFFER_RADIATOR

    return keep_triggers_clear_of_the_compressor_band(
        {
            "normal_min": base_thresholds["normal_min"] / multiplier,
            "normal_max": base_thresholds["normal_max"] / multiplier,
            "warning": base_thresholds["warning"] / multiplier,
            # The auxiliary-heat limit is hardware. It is the same for every emitter.
            "critical": base_thresholds["critical"],
        }
    )


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

    Past DM -1500 the layer stops cuts (a floor at 0) unless the room is above its band; it never
    commands a raise there - a saturated compressor turns a raise into immersion heat (F-124).

    The recovery tiers raise only for a room below its floor (the target less the mode's
    room_floor_below_target). At or above it the room needs no heat, and degree minutes are no
    reason to buy any: a raise there deepens DM toward the pump's immersion heater and banks heat
    the room then overshoots with. The closed loop on the owner's F750 + slab (four plant fits,
    autumn, winter and spring 2025-26) bought most of its immersion heat with the room at or above
    its target. Without a room sensor the tiers act on degree minutes alone, as NIBE does.

    VOLATILE-PRICE SUPPRESSION (deliberate smoothness/recovery trade-off): when the current
    spot-price run is shorter than VOLATILE_MIN_DURATION_QUARTERS, should_skip_volatile_boost()
    zeroes the T1/T2/T3 recovery offset and cuts weight to VOLATILE_WEIGHT_REDUCTION, to avoid
    compressor cycling on brief windows. Cost: DM may keep falling during a volatile run. Bounded
    because the DM <= DM_THRESHOLD_AUX_LIMIT check at the TOP of evaluate_layer returns first, so the
    EMERGENCY floor is never suppressed.
    """

    def __init__(
        self,
        climate_detector: ClimateZoneDetector,
        heating_type: str = "radiator",
        get_thermal_trend: Optional[Callable[[], dict]] = None,
        get_outdoor_trend: Optional[Callable[[], dict]] = None,
    ):
        """Initialize emergency layer.

        Args:
            climate_detector: ClimateZoneDetector for context-aware thresholds
            heating_type: Heating system type ("radiator", "concrete_ufh", "timber")
            get_thermal_trend: Callable returning thermal trend dict
            get_outdoor_trend: Callable returning outdoor trend dict
        """
        self.climate_detector = climate_detector
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

        DM_RATE_WINDOW_MINUTES of readings, scaled by thermal mass buffer.
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

        # 30 min of 5-min readings: 6 samples x 1.3 = 8 for concrete, x 1.15 = 7 for timber
        return round(DM_RATE_WINDOW_MINUTES / UPDATE_INTERVAL_MINUTES * multiplier)

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
        rate = dm_rate_per_hour(
            ((at, dm, 0) for at, dm in self._dm_history),
            timedelta(minutes=ANTI_WINDUP_MIN_SPAN_MINUTES),
        )
        return (0.0, False) if rate is None else (rate, True)

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
        """Hold the offset at its current value while heat is in transit.

        Anti-windup fires when degree minutes are FALLING even though the offset is already
        positive: raising S1 widens the (BT25 - S1) gap that DM integrates, so asking for
        more heat makes the number worse for hours before the thermal mass responds. The
        response is to stop escalating and wait.

        The cap is `current_offset` - hold, do not climb. It used to be

            max(current_offset, current_offset * ANTI_WINDUP_OFFSET_CAP_MULTIPLIER)

        with the multiplier at 0.7, which is identically `current_offset` for every positive
        offset, and anti-windup only triggers at an offset of at least
        ANTI_WINDUP_MIN_POSITIVE_OFFSET. So the multiplier never changed an outcome at any
        input, while reading like a 30% safety reduction. Holding is the real behaviour, and
        it is now the stated one; ACTIVE reduction is a separate, deliberate path keyed on
        how bad the spiral is (ANTI_WINDUP_REDUCTION_THRESHOLD and the rate-proportional
        reduction below it), which is where a reduction belongs.

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
        # Hold at the offset the pump is already running. See the docstring: the former
        # max(current, current * 0.7) was `current` at every reachable input.
        max_offset = current_offset

        if calculated_offset > max_offset:
            _LOGGER.info(
                "%s anti-windup: holding offset at %.2f°C instead of %.2f°C (heat in transit)",
                tier_name,
                max_offset,
                calculated_offset,
            )
            return max_offset, f"held at {max_offset:.1f}°C instead of {calculated_offset:.1f}°C"

        return calculated_offset, ""

    def evaluate_layer(
        self,
        nibe_state,
        weather_data,
        target_temp: float,
        tolerance_range: float,
        room_floor: float,
        is_volatile: bool = False,
    ) -> EmergencyLayerDecision:
        """Emergency layer: Smart context-aware thermal debt response.

        A measured room at or above its floor gets no recovery raise: it needs no heat, whatever
        degree minutes say. Below the floor the tiers recover as before.

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data (for damping decisions)
            target_temp: Target indoor temperature
            tolerance_range: Temperature tolerance range
            room_floor: The lowest room temperature the mode accepts before it buys heat
            is_volatile: True if in volatile cheap period (brief duration)

        Returns:
            EmergencyLayerDecision with context-aware emergency response
        """
        degree_minutes = nibe_state.degree_minutes
        outdoor_temp = nibe_state.outdoor_temp
        indoor_temp = nibe_state.indoor_temp
        current_offset = getattr(nibe_state, "current_offset", 0.0)
        # dt_util.utcnow(), never a naive datetime.now(): this fallback feeds the anti-windup
        # causation window, and a naive datetime there raises TypeError inside the emergency layer -
        # the one path that must never fail.
        timestamp = getattr(nibe_state, "timestamp", None) or dt_util.utcnow()

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
        # Comfort reasoning needs a REAL indoor reading. Without a room sensor the adapter reports
        # DEFAULT_INDOOR_TEMP (== target), so a room test against it would permanently disable DM
        # protection on exactly the systems that depend on it most. Skip the room cases and use the
        # degree-minute tiers, as NIBE does sensorless.
        indoor_is_measured = getattr(nibe_state, "indoor_temp_valid", True)
        too_warm = indoor_is_measured and temp_deviation > tolerance_range

        # ========================================
        # AUX LIMIT: past DM -1500, no cuts - and no raise (F-124)
        # ========================================
        # MUST come before anti-windup and the cooldown, which return early. Past this threshold the
        # compressor is flat out and the immersion heater already running (from about -1450 on the
        # owner's F750). Raising the curve cannot add compressor heat: S1 climbs where the supply
        # cannot follow, DM falls faster and the heater steps up - in the owner's winter +10 latched
        # for hours and cooked the house. What protects the house is not CUTTING here, so this tier
        # is a floor at 0 that _aggregate_layers applies to the other layers' verdict. A room above
        # its band skips it: a deep DM there means our own target is too high, and lowering it is
        # what recovers DM.
        if degree_minutes <= DM_THRESHOLD_AUX_LIMIT and not too_warm:
            return EmergencyLayerDecision(
                name="Thermal Debt",
                offset=0.0,
                weight=1.0,
                reason=(
                    f"EMERGENCY: DM {degree_minutes:.0f} past aux limit {DM_THRESHOLD_AUX_LIMIT}"
                    " - no cuts"
                ),
                tier="EMERGENCY",
                degree_minutes=degree_minutes,
                threshold_used=DM_THRESHOLD_AUX_LIMIT,
                dm_rate=dm_rate,
            )

        # ========================================
        # ANTI-WINDUP: Prevent offset raises that make DM worse (Jan 2026 fix)
        # ========================================
        # Physics: Raising offset increases S1 (target), but BT25 (actual) can't catch up
        # → Gap (BT25 - S1) becomes MORE negative → DM drops FASTER
        # Solution: If anti-windup detects this pattern, don't calculate higher offsets at all

        # Check 1: Are we in cooldown from a previous anti-windup activation?
        if self._is_in_cooldown(timestamp):
            remaining_min = int((self._anti_windup_cooldown_until - timestamp).total_seconds() / 60)
            return EmergencyLayerDecision(
                name="Anti-windup Cooldown",
                offset=current_offset,
                weight=0.7,
                reason=f"In cooldown ({remaining_min} min remaining) - not raising offset",
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
                    new_offset = max(MIN_OFFSET, current_offset - reduction)
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
                    weight=0.7,
                    reason=reason,
                    tier="ANTI_WINDUP",
                    degree_minutes=degree_minutes,
                    threshold_used=0.0,
                    anti_windup_active=True,
                    dm_rate=dm_rate,
                )

        # A room at or above its floor needs no heat: no recovery raise, cheap hour or not. It used
        # to recover at the target in a cheap quarter - at night, onto a compressor that is flat out
        # by then, so the raise bought immersion heat for a room that did not need it.
        if indoor_is_measured and indoor_temp >= room_floor:
            return EmergencyLayerDecision(
                name="Thermal Debt",
                offset=0.0,
                weight=0.0,
                reason=(
                    f"DM {degree_minutes:.0f} but room {indoor_temp:.1f}°C at or above its floor "
                    f"{room_floor:.1f}°C - no raise"
                ),
                tier="OK",
                degree_minutes=degree_minutes,
                dm_rate=dm_rate,
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

        # The ladder ends here. Z1..Z5 (ProactiveLayer) cover the band from just-negative DM
        # down to the thermal-mass-adjusted warning threshold, and T1 picks up at exactly that
        # threshold (DM_CRITICAL_T1_MARGIN is 0, by design: T1 IS the warning response). The
        # coverage is contiguous and verified for every heating type by
        # tests/unit/optimization/test_the_degree_minute_ladder_has_no_gaps.py.
        #
        # Two further branches used to sit here, and NEITHER COULD EVER RUN:
        #
        #   WARNING  `degree_minutes < expected_dm["warning"]`, placed after T1's
        #            `degree_minutes <= warning - DM_CRITICAL_T1_MARGIN`. With the margin at 0
        #            the two tests are the same test, so T1 always returned first.
        #   CAUTION  `degree_minutes < expected_dm["normal"]`. Every climate zone defines
        #            dm_normal_range[1] == dm_warning_threshold - "warning" MEANS "deeper than
        #            the normal range" - so this is also T1's condition, one step further down.
        #
        # They were removed rather than resurrected: reviving either needs a NEW threshold
        # invented between the existing ones, and the Z-zones already occupy that space with
        # real, reachable, graduated responses. Deleting them changed no pump behaviour; it
        # removed two tiers the docs promised and the code could not deliver.

        # OK: within the normal range for this outdoor temperature.
        return EmergencyLayerDecision(
            name="Thermal Debt",
            offset=0.0,
            weight=0.0,
            reason=f"OK (DM: {degree_minutes:.0f})",
            tier="OK",
            degree_minutes=degree_minutes,
            dm_rate=dm_rate,
        )

    def _get_thermal_mass_adjusted_thresholds(self, base_thresholds: dict) -> dict:
        """Thresholds moved to suit this house's thermal lag. See apply_thermal_mass_buffer."""
        adjusted = apply_thermal_mass_buffer(base_thresholds, self.heating_type)

        _LOGGER.debug(
            "Thermal mass adjusted thresholds: heating type '%s' → warning %.0f (base: %.0f)",
            self.heating_type,
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
    ) -> tuple[float, str]:
        """Apply thermal recovery damping when the house is warming naturally.

        Prevents concrete slab thermal overshoot during recovery periods. A recovery tier is only
        reached by a room below its floor, so there is no overshoot left to damp against.

        Args:
            base_offset: Original recovery offset
            tier_name: Recovery tier name for logging
            min_damped_offset: Minimum allowed offset after damping
            weather_data: Weather forecast data
            current_outdoor_temp: Current outdoor temperature

        Returns:
            Tuple of (damped_offset, damping_reason_string)
        """
        thermal_trend = self._get_thermal_trend() or {}
        outdoor_trend = self._get_outdoor_trend() or {}

        # Check trend confidence
        if thermal_trend.get("confidence", 0.0) < THERMAL_RECOVERY_MIN_CONFIDENCE:
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
                damping_reason = f"rapid warming {warming_rate:.2f}°C/h"
            else:
                warming_factor = THERMAL_RECOVERY_DAMPING_FACTOR
                damping_reason = f"warming {warming_rate:.2f}°C/h"

            damped_offset = max(base_offset * warming_factor, min_damped_offset)

            _LOGGER.info(
                "%s thermal recovery damping: %s, offset %.2f°C → %.2f°C",
                tier_name,
                damping_reason,
                base_offset,
                damped_offset,
            )

            return damped_offset, damping_reason

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

    A measured room at or above its floor gets none of it: the zones raise to prevent a shortfall,
    and a room at its floor has none. Warmer than that, a negative DM is the pump catching up with
    the curve, and every raise deepens it toward the immersion heater. (A room more than 0.5 °C
    above the target once only cut the zones' weight to 30 %, Z5 excepted.)
    """

    def __init__(
        self,
        climate_detector: ClimateZoneDetector,
        get_thermal_trend: Optional[Callable[[], dict]] = None,
        heating_type: str = "radiator",
    ):
        """Initialize proactive layer.

        Args:
            climate_detector: ClimateZoneDetector for context-aware thresholds
            get_thermal_trend: Callable returning thermal trend dict
            heating_type: Heating system type. The proactive layer must read the SAME thresholds
                as the emergency layer, or a band of degree minutes falls between them in which
                neither responds.
        """
        self.climate_detector = climate_detector
        self.heating_type = heating_type
        self._get_thermal_trend = get_thermal_trend or (
            lambda: {"rate_per_hour": 0.0, "confidence": 0.0}
        )

    def evaluate_layer(
        self,
        nibe_state,
        weather_data,
        target_temp: float,
        room_floor: float,
        is_volatile: bool = False,
    ) -> ProactiveLayerDecision:
        """Proactive debt prevention with climate-aware thresholds and trend prediction.

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data (for forecast validation)
            target_temp: Target indoor temperature
            room_floor: The lowest room temperature the mode accepts before it buys heat
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

        # A measured room at or above its floor needs no heat. Without a room sensor the indoor
        # reading is a placeholder, and the zones act on degree minutes alone.
        if getattr(nibe_state, "indoor_temp_valid", True) and indoor_temp >= room_floor:
            return ProactiveLayerDecision(
                name="Proactive",
                offset=0.0,
                weight=0.0,
                reason=f"Room {indoor_temp:.1f}°C at or above its floor {room_floor:.1f}°C",
                zone="NONE",
                degree_minutes=degree_minutes,
                trend_rate=trend_rate,
            )

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

                weight = RAPID_COOLING_WEIGHT

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
                        f"{forecast_reason}{volatile_suffix}"
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

            weight = LAYER_WEIGHT_PROACTIVE_MIN

            # Apply weight reduction during volatile cheap periods
            if is_volatile:
                weight = weight * VOLATILE_WEIGHT_REDUCTION

            return ProactiveLayerDecision(
                name="Z1",
                offset=offset,
                weight=weight,
                reason=f"DM {degree_minutes:.0f}, gentle heating prevents debt{reason_suffix}{volatile_suffix}",
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

            weight = PROACTIVE_ZONE2_WEIGHT

            # Apply weight reduction during volatile cheap periods
            if is_volatile:
                weight = weight * VOLATILE_WEIGHT_REDUCTION

            return ProactiveLayerDecision(
                name="Z2",
                offset=offset,
                weight=weight,
                reason=f"DM {degree_minutes:.0f}, boost recovery speed{volatile_suffix}",
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

            weight = PROACTIVE_ZONE3_WEIGHT

            # Apply weight reduction during volatile cheap periods
            if is_volatile:
                weight = weight * VOLATILE_WEIGHT_REDUCTION

            return ProactiveLayerDecision(
                name="Z3",
                offset=offset,
                weight=weight,
                reason=f"DM {degree_minutes:.0f}, prevent deeper debt{volatile_suffix}",
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

            weight = PROACTIVE_ZONE4_WEIGHT

            # Apply weight reduction during volatile cheap periods
            if is_volatile:
                weight = weight * VOLATILE_WEIGHT_REDUCTION

            return ProactiveLayerDecision(
                name="Z4",
                offset=offset,
                weight=weight,
                reason=f"DM {degree_minutes:.0f}, strong prevention{volatile_suffix}",
                zone="Z4",
                degree_minutes=degree_minutes,
                trend_rate=trend_rate,
            )

        # PROACTIVE ZONE 5
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
        dm_range = apply_thermal_mass_buffer(
            self.climate_detector.get_expected_dm_range(outdoor_temp), self.heating_type
        )

        return {
            "normal": dm_range["normal_max"],
            "warning": dm_range["warning"],
        }

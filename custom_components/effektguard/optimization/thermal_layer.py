"""Thermal model for building thermal behavior prediction.

Models heat storage and loss characteristics for predictive control.
Enables pre-heating and thermal energy banking strategies.
Includes emergency thermal debt response layer (Phase 6 refactor).
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from ..const import (
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
    QuarterClassification,
    SAFETY_EMERGENCY_OFFSET,
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
    WARNING_CAUTION_OFFSET,
    WARNING_CAUTION_WEIGHT,
    WARNING_DEVIATION_DIVISOR_MODERATE,
    WARNING_DEVIATION_DIVISOR_SEVERE,
    WARNING_DEVIATION_THRESHOLD,
    WARNING_OFFSET_MAX_MODERATE,
    WARNING_OFFSET_MAX_SEVERE,
    WARNING_OFFSET_MIN_MODERATE,
    WARNING_OFFSET_MIN_SEVERE,
)
from .climate_zones import ClimateZoneDetector

_LOGGER = logging.getLogger(__name__)


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

    def evaluate_layer(
        self,
        nibe_state,
        weather_data,
        price_data,
        target_temp: float,
        tolerance_range: float,
        get_current_datetime: Optional[Callable] = None,
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

        Returns:
            EmergencyLayerDecision with context-aware emergency response
        """
        degree_minutes = nibe_state.degree_minutes
        outdoor_temp = nibe_state.outdoor_temp
        indoor_temp = nibe_state.indoor_temp

        temp_deviation = indoor_temp - target_temp

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

        # Case 2: At target + Expensive/Normal Price (and not at absolute limit)
        if temp_deviation >= 0 and degree_minutes > DM_THRESHOLD_AUX_LIMIT:
            is_cheap = self._is_price_cheap(price_data, get_current_datetime)

            if not is_cheap:
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

            reason_parts = [
                f"DM {degree_minutes:.0f} near absolute max "
                f"(threshold: {t3_threshold:.0f}, margin: {margin_to_limit:.0f})"
            ]
            if damping_reason:
                reason_parts.append(f"[{damping_reason}]")

            return EmergencyLayerDecision(
                name="Thermal Recovery T3",
                offset=damped_offset,
                weight=DM_CRITICAL_T3_WEIGHT,
                reason=" ".join(reason_parts),
                tier="T3",
                degree_minutes=degree_minutes,
                threshold_used=t3_threshold,
                damping_applied=bool(damping_reason),
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

            reason_parts = [
                f"DM {degree_minutes:.0f} approaching T3 "
                f"(threshold: {t2_threshold:.0f}, margin: {margin_to_limit:.0f})"
            ]
            if damping_reason:
                reason_parts.append(f"[{damping_reason}]")

            return EmergencyLayerDecision(
                name="Thermal Recovery T2",
                offset=damped_offset,
                weight=DM_CRITICAL_T2_WEIGHT,
                reason=" ".join(reason_parts),
                tier="T2",
                degree_minutes=degree_minutes,
                threshold_used=t2_threshold,
                damping_applied=bool(damping_reason),
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

            reason_parts = [
                f"DM {degree_minutes:.0f} beyond expected for {outdoor_temp:.1f}°C "
                f"(threshold: {t1_threshold:.0f})"
            ]
            if damping_reason:
                reason_parts.append(f"[{damping_reason}]")

            return EmergencyLayerDecision(
                name="Thermal Recovery T1",
                offset=damped_offset,
                weight=DM_CRITICAL_T1_WEIGHT,
                reason=" ".join(reason_parts),
                tier="T1",
                degree_minutes=degree_minutes,
                threshold_used=t1_threshold,
                damping_applied=bool(damping_reason),
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

            reason = (
                f"DM {degree_minutes:.0f} beyond expected for "
                f"{outdoor_temp:.1f}°C (expected: {expected_dm['normal']:.0f}, "
                f"warning: {expected_dm['warning']:.0f}, deviation: {deviation:.0f})"
            )

            return EmergencyLayerDecision(
                name="Thermal Debt Warning",
                offset=offset,
                weight=LAYER_WEIGHT_EMERGENCY,
                reason=reason,
                tier="WARNING",
                degree_minutes=degree_minutes,
                threshold_used=expected_dm["warning"],
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
            )

        # OK: Within normal range
        return EmergencyLayerDecision(
            name="Thermal Debt",
            offset=0.0,
            weight=0.0,
            reason=f"OK (DM: {degree_minutes:.0f})",
            tier="OK",
            degree_minutes=degree_minutes,
        )

    def _is_price_cheap(self, price_data, get_current_datetime: Optional[Callable] = None) -> bool:
        """Check if current price is classified as CHEAP."""
        if not price_data or not price_data.today or not self.price_analyzer:
            return False

        try:
            if get_current_datetime:
                now = get_current_datetime()
            else:
                from homeassistant.util import dt as dt_util

                now = dt_util.now()

            current_quarter = (now.hour * 4) + (now.minute // 15)
            if current_quarter < len(price_data.today):
                classification = self.price_analyzer.get_current_classification(current_quarter)
                return classification == QuarterClassification.CHEAP
        except (AttributeError, IndexError, TypeError):
            pass

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
        thermal_trend = self._get_thermal_trend()
        outdoor_trend = self._get_outdoor_trend()

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

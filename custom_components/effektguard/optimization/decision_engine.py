"""Multi-layer decision engine for optimization.

Implements layered decision-making architecture that integrates:
- Safety constraints (temperature limits, thermal debt prevention)
- Effect tariff protection (15-minute peak avoidance)
- Weather prediction (pre-heating before cold)
- Mathematical weather compensation with adaptive climate zones
- Spot price optimization (cost reduction)
- Comfort maintenance (temperature tolerance)
- Emergency recovery (degree minutes critical threshold)

Each layer votes on offset, final decision is weighted aggregation.
Globally applicable with latitude-based climate zone detection.
Automatically adapts from Arctic (-30°C) to Mild (5°C) climates without configuration.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from homeassistant.util import dt as dt_util

from ..const import (
    COMMON_SENSE_COLD_SNAP_THRESHOLD,
    COMMON_SENSE_FORECAST_HORIZON,
    COMMON_SENSE_TEMP_ABOVE_TARGET,
    DEFAULT_HEAT_LOSS_COEFFICIENT,
    DEFAULT_TARGET_TEMP,
    DEFAULT_TOLERANCE,
    DEFAULT_WEATHER_COMPENSATION_WEIGHT,
    DM_CRITICAL_T1_MARGIN,
    DM_CRITICAL_T1_OFFSET,
    DM_CRITICAL_T1_PEAK_AWARE_OFFSET,
    DM_CRITICAL_T1_WEIGHT,
    DM_CRITICAL_T2_MARGIN,
    DM_CRITICAL_T2_OFFSET,
    DM_CRITICAL_T2_PEAK_AWARE_OFFSET,
    DM_CRITICAL_T2_WEIGHT,
    DM_CRITICAL_T3_MARGIN,
    DM_CRITICAL_T3_MAX,
    DM_CRITICAL_T3_OFFSET,
    DM_CRITICAL_T3_PEAK_AWARE_OFFSET,
    DM_CRITICAL_T3_WEIGHT,
    DM_THRESHOLD_AUX_LIMIT,
    DM_THERMAL_MASS_BUFFER_CONCRETE,
    DM_THERMAL_MASS_BUFFER_RADIATOR,
    DM_THERMAL_MASS_BUFFER_TIMBER,
    EFFECT_MARGIN_PREDICTIVE,
    EFFECT_MARGIN_WARNING,
    EFFECT_OFFSET_CRITICAL,
    EFFECT_OFFSET_PREDICTIVE,
    EFFECT_OFFSET_WARNING_RISING,
    EFFECT_OFFSET_WARNING_STABLE,
    EFFECT_PREDICTIVE_MODERATE_COOLING_INCREASE,
    EFFECT_PREDICTIVE_RAPID_COOLING_INCREASE,
    EFFECT_PREDICTIVE_RAPID_COOLING_THRESHOLD,
    EFFECT_PREDICTIVE_WARMING_DECREASE,
    EFFECT_WEIGHT_CRITICAL,
    EFFECT_WEIGHT_PREDICTIVE,
    EFFECT_WEIGHT_WARNING_RISING,
    EFFECT_WEIGHT_WARNING_STABLE,
    LAYER_WEIGHT_COMFORT_MAX,
    LAYER_WEIGHT_COMFORT_MIN,
    LAYER_WEIGHT_COMFORT_HIGH,
    LAYER_WEIGHT_COMFORT_SEVERE,
    LAYER_WEIGHT_COMFORT_CRITICAL,
    COMFORT_CORRECTION_MILD,
    COMFORT_CORRECTION_STRONG,
    COMFORT_CORRECTION_CRITICAL,
    COMFORT_DM_COOLING_THRESHOLD,
    LAYER_WEIGHT_EMERGENCY,
    LAYER_WEIGHT_PRICE,
    LAYER_WEIGHT_PROACTIVE_MAX,
    LAYER_WEIGHT_PROACTIVE_MIN,
    LAYER_WEIGHT_PREDICTION,
    LAYER_WEIGHT_SAFETY,
    MIN_TEMP_LIMIT,
    PEAK_AWARE_EFFECT_THRESHOLD,
    PEAK_AWARE_EFFECT_WEIGHT_MIN,
    PROACTIVE_ZONE1_OFFSET,
    PROACTIVE_ZONE1_THRESHOLD_PERCENT,
    PROACTIVE_ZONE2_OFFSET,
    PROACTIVE_ZONE2_THRESHOLD_PERCENT,
    PROACTIVE_ZONE2_WEIGHT,
    PROACTIVE_ZONE3_OFFSET_MIN,
    PROACTIVE_ZONE3_OFFSET_RANGE,
    PROACTIVE_ZONE3_THRESHOLD_PERCENT,
    PROACTIVE_ZONE4_OFFSET,
    PROACTIVE_ZONE4_THRESHOLD_PERCENT,
    PROACTIVE_ZONE4_WEIGHT,
    PROACTIVE_ZONE5_OFFSET,
    PROACTIVE_ZONE5_THRESHOLD_PERCENT,
    PROACTIVE_ZONE5_WEIGHT,
    QuarterClassification,
    RAPID_COOLING_BOOST_MAX,
    RAPID_COOLING_BOOST_MULTIPLIER,
    RAPID_COOLING_OUTDOOR_THRESHOLD,
    RAPID_COOLING_THRESHOLD,
    RAPID_COOLING_WEIGHT,
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
    THERMAL_RECOVERY_T1_NAME,
    THERMAL_RECOVERY_T2_MIN_OFFSET,
    THERMAL_RECOVERY_T2_NAME,
    THERMAL_RECOVERY_T3_MIN_OFFSET,
    THERMAL_RECOVERY_T3_NAME,
    THERMAL_RECOVERY_WARMING_THRESHOLD,
    TOLERANCE_RANGE_MULTIPLIER,
    TREND_BOOST_OFFSET_LIMIT,
    TREND_DAMPING_COOLING_BOOST,
    TREND_DAMPING_NEUTRAL,
    TREND_DAMPING_WARMING,
    WARNING_CAUTION_OFFSET,
    WARNING_CAUTION_WEIGHT,
    WARNING_DEVIATION_DIVISOR_MODERATE,
    WARNING_DEVIATION_DIVISOR_SEVERE,
    WARNING_DEVIATION_THRESHOLD,
    WARNING_OFFSET_MAX_MODERATE,
    WARNING_OFFSET_MAX_SEVERE,
    WARNING_OFFSET_MIN_MODERATE,
    WARNING_OFFSET_MIN_SEVERE,
    WEATHER_BASE_INTENSITY_MULTIPLIER,
    WEATHER_BASE_OFFSET_MAX,
    WEATHER_COMP_DEFER_DM_CRITICAL,
    WEATHER_COMP_DEFER_DM_LIGHT,
    WEATHER_COMP_DEFER_DM_MODERATE,
    WEATHER_COMP_DEFER_DM_SIGNIFICANT,
    WEATHER_COMP_DEFER_WEIGHT_CRITICAL,
    WEATHER_COMP_DEFER_WEIGHT_LIGHT,
    WEATHER_COMP_DEFER_WEIGHT_MODERATE,
    WEATHER_COMP_DEFER_WEIGHT_SIGNIFICANT,
    WEATHER_FORECAST_DROP_THRESHOLD,
    WEATHER_FORECAST_HORIZON,
    WEATHER_GENTLE_OFFSET,
    WEATHER_INDOOR_COOLING_CONFIRMATION,
    WEATHER_PREDICTION_LEAD_TIME_FACTOR,
    WEATHER_TEMP_DROP_DIVISOR,
    WEATHER_INTENSITY_AGGRESSIVE_DEFICIT,
    WEATHER_INTENSITY_AGGRESSIVE_FACTOR,
    WEATHER_INTENSITY_GENTLE_FACTOR,
    WEATHER_INTENSITY_STABLE_RATE,
    WEATHER_INTENSITY_NORMAL_FACTOR,
    WEATHER_INTENSITY_MODERATE_FACTOR,
    LAYER_WEIGHT_WEATHER_PREDICTION,
    WEATHER_WEIGHT_CAP,
    PRICE_TOLERANCE_DIVISOR,
    COMFORT_DEAD_ZONE,
    COMFORT_CORRECTION_MULT,
    MULTIPLIER_BOOST_30_PERCENT,
    MULTIPLIER_REDUCTION_20_PERCENT,
    THERMAL_CHANGE_MODERATE,
    THERMAL_CHANGE_MODERATE_COOLING,
    DEFAULT_CURVE_SENSITIVITY,
    PRICE_FORECAST_BASE_HORIZON,
    PRICE_FORECAST_CHEAP_THRESHOLD,
    PRICE_FORECAST_EXPENSIVE_THRESHOLD,
    PRICE_FORECAST_MIN_DURATION,
    PRICE_FORECAST_PREHEAT_OFFSET,
    PRICE_FORECAST_REDUCTION_OFFSET,
    PRICE_VOLATILE_SCAN_QUARTERS_EACH_DIRECTION,
    PRICE_VOLATILE_MIN_THRESHOLD,
    PRICE_VOLATILE_MAX_THRESHOLD,
    PRICE_VOLATILE_WEIGHT_REDUCTION,
)
from .climate_zones import ClimateZoneDetector
from .weather_compensation import AdaptiveClimateSystem, WeatherCompensationCalculator

_LOGGER = logging.getLogger(__name__)


@dataclass
class LayerDecision:
    """Decision from a single optimization layer.

    Each layer proposes an offset and provides reasoning.
    """

    offset: float  # Proposed heating curve offset (°C)
    weight: float  # Layer weight/priority (0.0-1.0)
    reason: str  # Human-readable explanation


@dataclass
class OptimizationDecision:
    """Final optimization decision from decision engine.

    Aggregates all layer votes into single decision with explanation.
    """

    offset: float  # Final heating curve offset (°C)
    layers: list[LayerDecision] = field(default_factory=list)
    reasoning: str = ""


class DecisionEngine:
    """Multi-layer optimization decision engine.

    Layer priority (highest to lowest):
    1. Safety layer (always enforced)
    2. Emergency layer (degree minutes critical)
    3. Effect tariff protection
    4. Prediction layer (Phase 6 - learned pre-heating)
    5. Weather compensation layer (NEW - mathematical flow temp optimization)
    6. Weather prediction
    7. Spot price optimization
    8. Comfort maintenance
    """

    def __init__(
        self,
        price_analyzer,
        effect_manager,
        thermal_model,
        config: dict[str, Any],
        thermal_predictor=None,  # Phase 6 - Optional predictor
        weather_learner=None,  # Phase 6 - Optional weather pattern learner
        heat_pump_model=None,  # Heat pump model profile
    ):
        """Initialize decision engine with dependencies.

        Args:
            price_analyzer: PriceAnalyzer for spot price classification
            effect_manager: EffectManager for peak tracking
            thermal_model: ThermalModel for predictions
            config: Configuration options
            thermal_predictor: Optional ThermalStatePredictor for learned pre-heating (Phase 6)
            weather_learner: Optional WeatherPatternLearner for unusual weather detection (Phase 6)
            heat_pump_model: Optional heat pump model profile for validation and limits
        """
        self.price = price_analyzer
        self.effect = effect_manager
        self.thermal = thermal_model
        self.config = config
        self.predictor = thermal_predictor  # Phase 6
        self.weather_learner = weather_learner  # Phase 6
        self.heat_pump_model = heat_pump_model  # Model profile

        # Configuration with defaults
        self.target_temp = config.get("target_indoor_temp", DEFAULT_TARGET_TEMP)
        self.tolerance = config.get("tolerance", DEFAULT_TOLERANCE)
        self.tolerance_range = self.tolerance * TOLERANCE_RANGE_MULTIPLIER

        # Weather compensation enabled/disabled
        self.enable_weather_compensation = config.get("enable_weather_compensation", True)
        self.weather_comp_weight = config.get(
            "weather_compensation_weight", DEFAULT_WEATHER_COMPENSATION_WEIGHT
        )

        # Adaptive climate system - combines universal zones with learning
        # Automatically detects climate zone from latitude (no country-specific code needed!)
        latitude = config.get("latitude", 59.33)  # Default to Stockholm
        self.climate_system = AdaptiveClimateSystem(
            latitude=latitude, weather_learner=weather_learner
        )

        # Climate zone detector for DM threshold adaptation
        # Uses same latitude, provides context-aware degree minutes thresholds
        self.climate_detector = ClimateZoneDetector(latitude)

        _LOGGER.info(
            "Climate zone: %s (%s) - DM thresholds adapted for %s",
            self.climate_detector.zone_info.name,
            self.climate_detector.zone_key,
            self.climate_detector.zone_info.description,
        )

        # Weather compensation calculator
        # Heat loss coefficient from learned values or config, fallback to default
        heat_loss_coeff = config.get("heat_loss_coefficient", DEFAULT_HEAT_LOSS_COEFFICIENT)
        radiator_output = config.get("radiator_rated_output", None)
        heating_type = config.get("heating_type", "radiator")  # radiator, concrete_ufh, timber

        self.weather_comp = WeatherCompensationCalculator(
            heat_loss_coefficient=heat_loss_coeff,
            radiator_rated_output=radiator_output,
            heating_type=heating_type,
        )

        _LOGGER.info(
            "Weather compensation initialized: HC=%.1f W/°C, radiator=%s W, type=%s",
            heat_loss_coeff,
            radiator_output if radiator_output else "not configured",
            heating_type,
        )

        # Manual override state (Phase 5 service support)
        self._manual_override_offset: float | None = None
        self._manual_override_until: Any = None  # datetime or None

    def set_manual_override(self, offset: float, duration_minutes: int = 0) -> None:
        """Set manual override for heating curve offset.

        Used by force_offset and boost_heating services.

        Args:
            offset: Manual offset value (-10 to +10°C)
            duration_minutes: Duration in minutes (0 = until next cycle)
        """
        from datetime import timedelta

        self._manual_override_offset = offset

        if duration_minutes > 0:
            self._manual_override_until = dt_util.now() + timedelta(minutes=duration_minutes)
            _LOGGER.info(
                "Manual override set: %s°C until %s",
                offset,
                self._manual_override_until.strftime("%Y-%m-%d %H:%M"),
            )
        else:
            self._manual_override_until = None
            _LOGGER.info("Manual override set: %s°C until next cycle", offset)

    def clear_manual_override(self) -> None:
        """Clear manual override, return to automatic optimization."""
        self._manual_override_offset = None
        self._manual_override_until = None
        _LOGGER.info("Manual override cleared")

    def _check_manual_override(self) -> float | None:
        """Check if manual override is active and still valid.

        Returns:
            Override offset if active, None if expired or not set
        """
        if self._manual_override_offset is None:
            return None

        # Check if time-based override expired
        if self._manual_override_until:
            if dt_util.now() >= self._manual_override_until:
                _LOGGER.info("Manual override expired, returning to automatic")
                self.clear_manual_override()
                return None

        return self._manual_override_offset

    def _get_thermal_trend(self) -> dict:
        """Get current indoor temperature trend data.

        Returns:
            Dictionary with trend info, or empty dict if unavailable
        """
        if self.predictor and len(self.predictor.state_history) >= 8:
            return self.predictor.get_current_trend()
        return {
            "trend": "unknown",
            "rate_per_hour": 0.0,
            "confidence": 0.0,
            "samples": 0,
        }

    def _get_outdoor_trend(self) -> dict[str, Any]:
        """Get outdoor temperature trend (BT1 real-time).

        Returns:
            Outdoor trend data or empty if not available
        """
        if hasattr(self, "predictor") and self.predictor:
            return self.predictor.get_outdoor_trend()
        return {"trend": "unknown", "rate_per_hour": 0.0, "confidence": 0.0}

    def _apply_thermal_recovery_damping(
        self,
        base_offset: float,
        tier_name: str,
        min_damped_offset: float,
        weather_data=None,
        current_outdoor_temp: float | None = None,
        indoor_temp: float | None = None,
        target_temp: float | None = None,
    ) -> tuple[float, str]:
        """Apply thermal recovery damping when house warming naturally or overshooting.

        Prevents concrete slab thermal overshoot during recovery periods.
        When T1/T2/T3 active AND (house warming from solar OR already overshooting),
        reduce offset to prevent excessive heat storage in thermal mass.

        Args:
            base_offset: Original recovery offset (e.g., 2.5°C for T2)
            tier_name: Recovery tier name for logging (e.g., "STRONG RECOVERY")
            min_damped_offset: Minimum allowed offset after damping (safety floor)
            weather_data: Weather forecast data (optional, for cold weather check)
            current_outdoor_temp: Current outdoor temperature (°C)
            indoor_temp: Current indoor temperature (°C) for overshoot detection
            target_temp: Target indoor temperature (°C) for overshoot detection

        Returns:
            Tuple of (damped_offset, damping_reason_string)
            If no damping applied: (base_offset, "")

        Damping Conditions:
            Warming-based (original logic):
                1. Indoor warming >= 0.3°C/h (significant solar gain)
                2. Outdoor not dropping < -0.5°C/h (not fighting cold spell)
                3. Sufficient trend confidence >= 0.4 (~1 hour data)
                4. Weather forecast NOT showing significant cold within 6 hours

            Overshoot-based (NEW - Oct 23, 2025):
                If indoor_temp > target_temp:
                    Severe ≥1.5°C: 80% strength (0.8 multiplier)
                    Moderate ≥1.0°C: 90% strength (0.9 multiplier)
                    Mild ≥0.5°C: 95% strength (0.95 multiplier)
                Overshoot damping multiplies with warming damping for compound effect

        Example:
            T2 base offset 2.5°C + warming 0.4°C/h + stable forecast → damped to 1.5°C (60%)
            OR: T2 base 2.5°C + 2.0°C overshoot → damped to 2.0°C (80% strength)
            OR: Both warming + overshoot → compound: 2.5°C × 0.6 × 0.8 = 1.2°C
        """
        thermal_trend = self._get_thermal_trend()
        outdoor_trend = self._get_outdoor_trend()

        # Calculate overshoot damping factor (independent of warming)
        overshoot_factor = 1.0  # Default: no overshoot penalty
        overshoot_reason = ""

        if indoor_temp is not None and target_temp is not None:
            overshoot = indoor_temp - target_temp
            if overshoot >= THERMAL_RECOVERY_OVERSHOOT_SEVERE_THRESHOLD:  # ≥1.5°C
                overshoot_factor = THERMAL_RECOVERY_OVERSHOOT_SEVERE_DAMPING  # 0.8
                overshoot_reason = f"severe overshoot +{overshoot:.1f}°C"
            elif overshoot >= THERMAL_RECOVERY_OVERSHOOT_MODERATE_THRESHOLD:  # ≥1.0°C
                overshoot_factor = THERMAL_RECOVERY_OVERSHOOT_MODERATE_DAMPING  # 0.9
                overshoot_reason = f"moderate overshoot +{overshoot:.1f}°C"
            elif overshoot >= THERMAL_RECOVERY_OVERSHOOT_MILD_THRESHOLD:  # ≥0.5°C
                overshoot_factor = THERMAL_RECOVERY_OVERSHOOT_MILD_DAMPING  # 0.95
                overshoot_reason = f"mild overshoot +{overshoot:.1f}°C"

        # Check if we have sufficient trend confidence
        if thermal_trend.get("confidence", 0.0) < THERMAL_RECOVERY_MIN_CONFIDENCE:
            # Even without trend data, apply overshoot damping if present
            if overshoot_factor < 1.0:
                damped_offset = base_offset * overshoot_factor
                damped_offset = max(damped_offset, min_damped_offset)

                _LOGGER.info(
                    "%s overshoot damping: %s, " "offset %.2f°C → %.2f°C (factor %.2f, min %.1f°C)",
                    tier_name,
                    overshoot_reason,
                    base_offset,
                    damped_offset,
                    overshoot_factor,
                    min_damped_offset,
                )

                return damped_offset, overshoot_reason

            return base_offset, ""

        warming_rate = thermal_trend.get("rate_per_hour", 0.0)
        outdoor_rate = outdoor_trend.get("rate_per_hour", 0.0)

        # Check weather forecast for incoming cold
        # If significant cold is coming within 6 hours, don't damp - we need the heat!
        if weather_data and weather_data.forecast_hours and current_outdoor_temp is not None:
            # Check next 6 hours for significant temperature drop
            forecast_horizon = int(THERMAL_RECOVERY_FORECAST_HORIZON)
            forecast_temps = [h.temperature for h in weather_data.forecast_hours[:forecast_horizon]]

            if forecast_temps:
                min_forecast_temp = min(forecast_temps)
                forecast_drop = min_forecast_temp - current_outdoor_temp

                if forecast_drop < THERMAL_RECOVERY_FORECAST_DROP_THRESHOLD:
                    # Significant cold coming - don't damp, we need to prepare!
                    _LOGGER.info(
                        "%s: Cold weather forecast (%.1f°C drop to %.1f°C within %dh), "
                        "maintaining full offset %.2f°C",
                        tier_name,
                        abs(forecast_drop),
                        min_forecast_temp,
                        forecast_horizon,
                        base_offset,
                    )
                    return base_offset, ""

        # Apply damping if:
        # 1. Indoor warming significantly (>0.3°C/h = solar gain detected)
        # 2. Outdoor not rapidly dropping (not fighting cold spell)
        # 3. Forecast stable (no significant cold incoming)
        if (
            warming_rate >= THERMAL_RECOVERY_WARMING_THRESHOLD
            and outdoor_rate >= THERMAL_RECOVERY_OUTDOOR_DROPPING_THRESHOLD
        ):
            # Choose damping factor based on warming intensity
            if warming_rate >= THERMAL_RECOVERY_RAPID_THRESHOLD:  # Rapid warming >0.5°C/h
                warming_factor = THERMAL_RECOVERY_RAPID_FACTOR  # 0.4 = reduce to 40%
                warming_reason = f"rapid warming {warming_rate:.2f}°C/h"
            else:  # Moderate warming 0.3-0.5°C/h
                warming_factor = THERMAL_RECOVERY_DAMPING_FACTOR  # 0.6 = reduce to 60%
                warming_reason = f"warming {warming_rate:.2f}°C/h"

            # Combine warming and overshoot damping (multiply factors)
            combined_factor = warming_factor * overshoot_factor

            # Build comprehensive damping reason
            if overshoot_reason:
                damping_reason = f"{warming_reason} + {overshoot_reason}"
            else:
                damping_reason = warming_reason

            # Apply combined damping but maintain safety minimum
            damped_offset = base_offset * combined_factor
            damped_offset = max(damped_offset, min_damped_offset)

            _LOGGER.info(
                "%s thermal recovery damping: %s, "
                "offset %.2f°C → %.2f°C (warming %.2f × overshoot %.2f = %.2f, min %.1f°C)",
                tier_name,
                damping_reason,
                base_offset,
                damped_offset,
                warming_factor,
                overshoot_factor,
                combined_factor,
                min_damped_offset,
            )

            return damped_offset, damping_reason

        # No warming-based damping, but apply overshoot damping if present
        if overshoot_factor < 1.0:
            damped_offset = base_offset * overshoot_factor
            damped_offset = max(damped_offset, min_damped_offset)

            _LOGGER.info(
                "%s overshoot damping: %s, " "offset %.2f°C → %.2f°C (factor %.2f, min %.1f°C)",
                tier_name,
                overshoot_reason,
                base_offset,
                damped_offset,
                overshoot_factor,
                min_damped_offset,
            )

            return damped_offset, overshoot_reason

        return base_offset, ""

    def _is_cooling_rapidly(
        self, thermal_trend: dict, threshold: float = RAPID_COOLING_THRESHOLD
    ) -> bool:
        """Check if house is cooling rapidly.

        Args:
            thermal_trend: Trend data from _get_thermal_trend()
            threshold: Cooling rate threshold (°C/h, negative)

        Returns:
            True if cooling faster than threshold
        """
        return thermal_trend.get("rate_per_hour", 0.0) < threshold

    def _is_warming_rapidly(
        self, thermal_trend: dict, threshold: float = THERMAL_CHANGE_MODERATE
    ) -> bool:
        """Check if house is warming rapidly.

        Args:
            thermal_trend: Trend data from _get_thermal_trend()
            threshold: Warming rate threshold (°C/h, positive)

        Returns:
            True if warming faster than threshold
        """
        return thermal_trend.get("rate_per_hour", 0.0) > threshold

    def _get_preheat_lead_time(self) -> float:
        """Get required pre-heating lead time based on heating system type.

        Lead time is the hours needed to warm house before cold arrives.
        Based on thermal lag of heating system.

        Returns:
            Lead time in hours
        """
        # Get prediction horizon (based on UFH type from thermal model)
        horizon = self.thermal.get_prediction_horizon()

        # Lead time = 50% of prediction horizon
        # Concrete UFH: 12h horizon → 6h lead time
        # Timber UFH: 6h horizon → 3h lead time
        # Radiator: 2h horizon → 1h lead time
        lead_time = horizon * WEATHER_PREDICTION_LEAD_TIME_FACTOR

        return lead_time

    def _calculate_preheat_intensity(
        self,
        temp_drop: float,
        thermal_trend: dict,
        indoor_deficit: float,
    ) -> tuple[float, str]:
        """Calculate adaptive pre-heating intensity (Phase 3.3).

        Modulates pre-heating strength based on indoor state and trend to prevent
        over-heating when house already warm, or boost when house struggling.

        Args:
            temp_drop: Forecasted outdoor temperature drop (°C)
            thermal_trend: Indoor temperature trend data
            indoor_deficit: Current indoor temp below target (°C)

        Returns:
            Tuple of (offset, reasoning)

        References:
            MASTER_IMPLEMENTATION_PLAN.md: Phase 3.3 - Adaptive Pre-Heat Intensity
        """
        trend_rate = thermal_trend.get("rate_per_hour", 0.0)

        # Base intensity from forecast
        base_intensity = abs(temp_drop) / WEATHER_TEMP_DROP_DIVISOR
        base_offset = min(
            base_intensity * WEATHER_BASE_INTENSITY_MULTIPLIER, WEATHER_BASE_OFFSET_MAX
        )

        # Modulate based on indoor state and trend
        if (
            indoor_deficit > WEATHER_INTENSITY_AGGRESSIVE_DEFICIT
            and trend_rate < THERMAL_CHANGE_MODERATE_COOLING
        ):
            intensity_factor = WEATHER_INTENSITY_AGGRESSIVE_FACTOR
            reason = "aggressive: already cooling below target"
        elif indoor_deficit < RAPID_COOLING_THRESHOLD and trend_rate > COMFORT_DEAD_ZONE:
            intensity_factor = WEATHER_INTENSITY_GENTLE_FACTOR
            reason = "gentle: already warm and rising"
        elif (
            abs(indoor_deficit) < COMFORT_DEAD_ZONE
            and abs(trend_rate) < WEATHER_INTENSITY_STABLE_RATE
        ):
            intensity_factor = WEATHER_INTENSITY_NORMAL_FACTOR
            reason = "normal: stable at target"
        else:
            intensity_factor = WEATHER_INTENSITY_MODERATE_FACTOR
            reason = "moderate: mixed conditions"

        final_offset = base_offset * intensity_factor
        return final_offset, reason

    def calculate_decision(
        self,
        nibe_state,
        price_data,
        weather_data,
        current_peak: float,
        current_power: float,
    ) -> OptimizationDecision:
        """Calculate optimal heating offset using multi-layer approach.

        Decision layers (ordered by priority):
        1. Manual override (if active, Phase 5 services)
        2. Safety layer: Prevent extreme temperatures
        3. Emergency layer: Respond to critical degree minutes
        4. Effect tariff layer: Peak protection
        5. Weather prediction layer: Pre-heat before cold
        6. Spot price layer: Base optimization
        7. Comfort layer: Stay within tolerance

        Args:
            nibe_state: Current NIBE heat pump state
            price_data: GE-Spot price data (native 15-min intervals)
            weather_data: Weather forecast data
            current_peak: Current monthly peak threshold (kW) - from peak_this_month sensor
            current_power: Current whole-house power consumption (kW) - from peak_today sensor

        Returns:
            OptimizationDecision with offset, reasoning, and layer votes
        """
        _LOGGER.debug("Calculating optimization decision")

        # Check for manual override first (Phase 5 service support)
        manual_override = self._check_manual_override()
        if manual_override is not None:
            _LOGGER.info("Using manual override: %.2f°C", manual_override)
            return OptimizationDecision(
                offset=manual_override,
                layers=[
                    LayerDecision(
                        offset=manual_override,
                        weight=1.0,
                        reason=f"Manual override: {manual_override:.1f}°C",
                    )
                ],
                reasoning=f"Manual override active: {manual_override:.1f}°C",
            )

        # Update price analyzer with latest data
        if price_data:
            self.price.update_prices(price_data)

        # Validate power consumption if model available
        if self.heat_pump_model and hasattr(nibe_state, "power_kw"):
            validation = self._validate_power_consumption(
                nibe_state.power_kw, nibe_state.outdoor_temp
            )
            if validation["warning"]:
                _LOGGER.log(
                    (logging.WARNING if validation["severity"] == "warning" else logging.INFO),
                    validation["warning"],
                )

        # Calculate all layer decisions (ordered by priority)
        layers = [
            self._safety_layer(nibe_state),
            self._emergency_layer(
                nibe_state, weather_data, price_data
            ),  # Pass price for smart recovery
            self._proactive_debt_prevention_layer(
                nibe_state, weather_data
            ),  # Phase 3.2: Pass weather_data for forecast validation
            self._effect_layer(
                nibe_state, current_peak, current_power
            ),  # Pass actual whole-house power
            self._prediction_layer(nibe_state, weather_data),  # Phase 6 - Learned pre-heating
            self._weather_compensation_layer(
                nibe_state, weather_data
            ),  # Mathematical WC with Swedish adaptations
            self._weather_layer(nibe_state, weather_data),  # Simple pre-heating logic
            self._price_layer(nibe_state, price_data),
            self._comfort_layer(nibe_state),
        ]

        # Aggregate layers with priority weighting
        raw_offset = self._aggregate_layers(layers)

        # NEW: Trend-aware damping to prevent overshoot/undershoot
        thermal_trend = self._get_thermal_trend()
        trend_rate = thermal_trend.get("rate_per_hour", 0.0)

        if self._is_warming_rapidly(thermal_trend):
            # House warming rapidly
            if raw_offset > 0:
                # Heating + Warming -> Damp to prevent overshoot
                damping_factor = TREND_DAMPING_WARMING
                reason_suffix = f" (damped 25%: warming {trend_rate:.2f}°C/h)"
            else:
                # Cooling + Warming -> Maintain full cooling (don't damp)
                # If we are trying to cool and house is warming, we need full power!
                damping_factor = TREND_DAMPING_NEUTRAL
                reason_suffix = ""

        elif self._is_cooling_rapidly(thermal_trend):
            # House cooling rapidly
            if raw_offset > 0:
                # Heating + Cooling -> Boost to prevent undershoot
                # But only if not already at high offset (safety limit)
                if raw_offset < TREND_BOOST_OFFSET_LIMIT:
                    damping_factor = TREND_DAMPING_COOLING_BOOST
                    reason_suffix = f" (boosted 15%: cooling {trend_rate:.2f}°C/h)"
                else:
                    damping_factor = TREND_DAMPING_NEUTRAL
                    reason_suffix = " (at safety limit, no boost)"
            else:
                # Cooling + Cooling -> Damp to prevent undershoot (reduce cooling)
                # If we are trying to cool and house is cooling fast, back off
                damping_factor = TREND_DAMPING_WARMING  # Reuse 0.75 factor
                reason_suffix = f" (damped 25%: cooling {trend_rate:.2f}°C/h)"
        else:
            # Normal rate of change
            damping_factor = TREND_DAMPING_NEUTRAL
            reason_suffix = ""

        final_offset = raw_offset * damping_factor

        # Generate human-readable reasoning
        reasoning = self._generate_reasoning(layers, final_offset) + reason_suffix

        _LOGGER.info("Decision: offset %.2f°C - %s", final_offset, reasoning)

        return OptimizationDecision(
            offset=final_offset,
            layers=layers,
            reasoning=reasoning,
        )

    def _safety_layer(self, nibe_state) -> LayerDecision:
        """Safety layer: Enforce absolute minimum temperature only.

        Phase 2: Upper temperature limit removed - now handled dynamically by comfort layer
        based on user's target temperature + tolerance setting.

        This ensures temperature control adapts to user preferences rather
        than using a fixed maximum that may be inappropriate.

        Hard limit: 18°C minimum indoor temperature
        This layer always has maximum weight.

        Args:
            nibe_state: Current NIBE state

        Returns:
            LayerDecision with safety constraints
        """
        indoor_temp = nibe_state.indoor_temp

        if indoor_temp < MIN_TEMP_LIMIT:
            # Too cold - emergency heating
            offset = SAFETY_EMERGENCY_OFFSET
            return LayerDecision(
                offset=offset,
                weight=LAYER_WEIGHT_SAFETY,
                reason=f"SAFETY: Too cold ({indoor_temp:.1f}°C < {MIN_TEMP_LIMIT}°C)",
            )
        else:
            # Within safe limits (no fixed upper limit - comfort layer handles dynamically)
            return LayerDecision(
                offset=0.0,
                weight=0.0,
                reason="Safety OK",
            )

    def _emergency_layer(self, nibe_state, weather_data=None, price_data=None) -> LayerDecision:
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

        CRITICAL RULE (Nov 27, 2025):
        Never heat when indoor temperature exceeds target + tolerance.
        Thermal debt while overheating means "flow temp was correctly reduced".
        Let natural cooling bring temp down first, THEN recover DM during cheap periods.
        This prevents wasteful heating during expensive hours when house is already warm.

        Args:
            nibe_state: Current NIBE state

        Returns:
            LayerDecision with context-aware emergency response
        """
        degree_minutes = nibe_state.degree_minutes
        outdoor_temp = nibe_state.outdoor_temp
        indoor_temp = nibe_state.indoor_temp
        target_temp = self.target_temp

        # FIRST CHECK: Smart Debt Recovery (User Request Nov 29, 2025)
        # 1. If above tolerance: Definitely stop (existing logic)
        # 2. If at target (>= target) AND Price is NOT CHEAP: Ignore debt
        #    "Ignore DM if the indoor temp is already at target and price is normal or expensive"
        #    Exception: Absolute safety limit (-1500) always triggers

        temp_error = indoor_temp - target_temp
        tolerance = self.tolerance_range

        # Case 1: Too warm (above tolerance)
        if temp_error > tolerance:
            # Too warm - emergency heating would be counterproductive
            # DM will naturally improve as house cools and we maintain target temp later
            return LayerDecision(
                offset=0.0,
                weight=0.0,
                reason=f"DM {degree_minutes:.0f} BUT indoor {indoor_temp:.1f}°C is {temp_error:.1f}°C over target - let cool naturally first",
            )

        # Case 2: At target + Expensive/Normal Price (and not at absolute limit)
        if temp_error >= 0 and degree_minutes > DM_THRESHOLD_AUX_LIMIT:
            # Check if price is cheap
            is_cheap = False
            if price_data and price_data.today:
                now = dt_util.now()
                current_quarter = (now.hour * 4) + (now.minute // 15)
                # Safety check for index
                if current_quarter < len(price_data.today):
                    classification = self.price.get_current_classification(current_quarter)
                    if classification == QuarterClassification.CHEAP:
                        is_cheap = True

            if not is_cheap:
                return LayerDecision(
                    offset=0.0,
                    weight=0.0,
                    reason=f"Smart Recovery: At target ({indoor_temp:.1f}°C) & Price not cheap - ignoring DM {degree_minutes:.0f}",
                )

        # HARD LIMIT: DM -1500 absolute maximum (never exceed)
        if degree_minutes <= DM_THRESHOLD_AUX_LIMIT:
            # At absolute safety limit - maximum emergency response
            # This applies regardless of outdoor temperature or conditions
            offset = SAFETY_EMERGENCY_OFFSET
            return LayerDecision(
                offset=offset,
                weight=1.0,
                reason=f"AUX LIMIT: DM {degree_minutes:.0f} at aux limit -1500 - EMERGENCY",
            )

        # Calculate context-aware thresholds based on outdoor temperature
        # Colder weather = expect deeper normal DM, so thresholds adapt
        base_expected_dm = self._calculate_expected_dm_for_temperature(outdoor_temp)

        # Apply thermal mass adjustment (Oct 23, 2025)
        # High thermal mass systems need tighter thresholds to prevent v0.1.0 solar gain problem
        # Concrete slab: 30% tighter, Timber: 15% tighter, Radiator: standard
        heating_type = self.config.get("heating_type", "radiator")
        expected_dm_range = self.climate_detector.get_expected_dm_range(outdoor_temp)
        adjusted_dm_range = self._get_thermal_mass_adjusted_thresholds(
            expected_dm_range, heating_type
        )
        expected_dm = {
            "normal": adjusted_dm_range["normal_max"],  # Use deep end of normal range
            "warning": adjusted_dm_range["warning"],  # Thermal mass adjusted warning threshold
        }

        # Distance from absolute maximum (how much safety margin remains)
        margin_to_limit = degree_minutes - DM_THRESHOLD_AUX_LIMIT  # Positive value

        # MULTI-TIER CLIMATE-AWARE CRITICAL INTERVENTION (Oct 19, 2025 redesign)
        # THERMAL MASS ENHANCEMENT (Oct 23, 2025): Adjusted thresholds based on thermal lag
        # Previous fixed thresholds (-800, -1000, -1200) caused false positives in Arctic climates
        # and inadequate intervention in mild climates. New design calculates tiers dynamically
        # based on climate-aware WARNING threshold + margin + thermal mass buffer.
        #
        # Examples (without thermal mass adjustment):
        # - Paris (WARNING -350):     T1=-350,  T2=-550,  T3=-750
        # - Stockholm (WARNING -700): T1=-700,  T2=-900,  T3=-1100
        # - Kiruna (WARNING -1200):   T1=-1200, T2=-1400, T3=-1450 (capped)
        #
        # Examples (with thermal mass adjustment - concrete slab 1.3×):
        # - Paris concrete:     T1=-455,  T2=-655,  T3=-855   (30% tighter)
        # - Stockholm concrete: T1=-910,  T2=-1110, T3=-1310  (30% tighter)
        # - Kiruna concrete:    T1=-1450, T2=-1450, T3=-1450  (capped at T3 max)
        #
        # Philosophy: Use climate zone + thermal mass knowledge to define what's "critical"

        # Calculate climate-aware + thermal mass aware tier thresholds
        warning_threshold = expected_dm["warning"]
        t1_threshold = warning_threshold - DM_CRITICAL_T1_MARGIN  # At WARNING threshold
        t2_threshold = warning_threshold - DM_CRITICAL_T2_MARGIN  # WARNING + 200 DM
        t3_threshold = max(
            warning_threshold - DM_CRITICAL_T3_MARGIN,  # WARNING + 400 DM
            DM_CRITICAL_T3_MAX,  # Capped at -1450 (50 DM from absolute max)
        )

        # CRITICAL TIER 3: Most severe intervention (within 50-300 DM of absolute maximum)
        # Applies thermal recovery damping even at T3 to prevent overshoot
        # T3 minimum (2.0°C) ensures aggressive recovery even when damped
        if degree_minutes <= t3_threshold:
            base_offset = DM_CRITICAL_T3_OFFSET

            # Apply thermal recovery damping (even at emergency T3 level)
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
                f"{THERMAL_RECOVERY_T3_NAME}: DM {degree_minutes:.0f} near absolute max "
                f"(threshold: {t3_threshold:.0f}, margin: {margin_to_limit:.0f})"
            ]
            if damping_reason:
                reason_parts.append(f"[{damping_reason}]")

            return LayerDecision(
                offset=damped_offset,
                weight=DM_CRITICAL_T3_WEIGHT,
                reason=" ".join(reason_parts),
            )

        # STRONG RECOVERY TIER 2: Severe thermal debt - strong recovery before reaching T3
        # Thermal Recovery Damping (Oct 20, 2025): Prevent concrete slab overshoot
        # Uses general _apply_thermal_recovery_damping() helper
        if degree_minutes <= t2_threshold:
            base_offset = DM_CRITICAL_T2_OFFSET

            # Apply thermal recovery damping if house warming naturally
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
                f"{THERMAL_RECOVERY_T2_NAME}: DM {degree_minutes:.0f} approaching T3 "
                f"(threshold: {t2_threshold:.0f}, margin: {margin_to_limit:.0f})"
            ]
            if damping_reason:
                reason_parts.append(f"[{damping_reason}]")

            return LayerDecision(
                offset=damped_offset,
                weight=DM_CRITICAL_T2_WEIGHT,
                reason=" ".join(reason_parts),
            )

        # MODERATE RECOVERY TIER 1: Serious thermal debt - prevent escalation to T2
        # Triggers at climate-aware WARNING threshold (where thermal debt becomes abnormal)
        # Applies thermal recovery damping to prevent overshoot
        if degree_minutes <= t1_threshold:
            base_offset = DM_CRITICAL_T1_OFFSET

            # Apply thermal recovery damping
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
                f"{THERMAL_RECOVERY_T1_NAME}: DM {degree_minutes:.0f} beyond expected for {outdoor_temp:.1f}°C "
                f"(threshold: {t1_threshold:.0f})"
            ]
            if damping_reason:
                reason_parts.append(f"[{damping_reason}]")

            return LayerDecision(
                offset=damped_offset,
                weight=DM_CRITICAL_T1_WEIGHT,
                reason=" ".join(reason_parts),
            )

        # WARNING: DM is significantly beyond expected range for this temperature
        # Check if we're deeper than expected + safety margin
        # No thermal recovery damping - WARNING already moderate (0.8-1.8°C)
        if degree_minutes < expected_dm["warning"]:
            # Beyond expected range - recovery needed
            # Severity scales with how far beyond expected we are
            deviation = expected_dm["warning"] - degree_minutes

            # Strengthened offset calculation (Oct 19, 2025 fix)
            # Previous formula was too weak: +0.5-2.0°C allowed DM to worsen
            # New formula provides stronger intervention earlier
            if deviation > WARNING_DEVIATION_THRESHOLD:  # Severe deviation beyond expected
                offset = min(
                    WARNING_OFFSET_MAX_SEVERE,
                    WARNING_OFFSET_MIN_SEVERE + (deviation / WARNING_DEVIATION_DIVISOR_SEVERE),
                )
            else:  # Moderate deviation
                offset = min(
                    WARNING_OFFSET_MAX_MODERATE,
                    WARNING_OFFSET_MIN_MODERATE + (deviation / WARNING_DEVIATION_DIVISOR_MODERATE),
                )

            # WARNING: Beyond expected range - moderate to strong recovery
            # Philosophy: "If we don't need to heat, we shouldn't"
            # Let weather compensation and price optimization have their say
            # DHW blocking will fix root cause of thermal debt from DHW interference
            percent_beyond = (
                abs(deviation / expected_dm["warning"]) if expected_dm["warning"] else 0
            )

            reason = (
                f"WARNING: DM {degree_minutes:.0f} beyond expected for "
                f"{outdoor_temp:.1f}°C (expected: {expected_dm['normal']:.0f}, "
                f"warning: {expected_dm['warning']:.0f}, deviation: {deviation:.0f}, "
                f"{percent_beyond:.0%} beyond)"
            )

            return LayerDecision(
                offset=offset,
                weight=LAYER_WEIGHT_EMERGENCY,  # Strong suggestion, but not absolute override
                reason=reason,
            )

        # CAUTION: DM is approaching expected limits
        elif degree_minutes < expected_dm["normal"]:
            # Slightly beyond normal - gentle correction
            offset = WARNING_CAUTION_OFFSET
            return LayerDecision(
                offset=offset,
                weight=WARNING_CAUTION_WEIGHT,
                reason=f"CAUTION: DM {degree_minutes:.0f} at {outdoor_temp:.1f}°C - monitoring",
            )

        else:
            # DM within normal expected range for this temperature
            return LayerDecision(
                offset=0.0,
                weight=0.0,
                reason=f"Thermal debt OK (DM: {degree_minutes:.0f} at {outdoor_temp:.1f}°C)",
            )

    def _calculate_expected_dm_for_temperature(self, outdoor_temp: float) -> dict[str, float]:
        """Calculate expected DM range for given outdoor temperature.

        CLIMATE-AWARE ADAPTATION:
        Uses ClimateZoneDetector to get context-aware DM thresholds based on:
        1. Climate zone (Extreme Cold → Standard) provides base expectations
        2. Current outdoor temperature adjusts thresholds dynamically
        3. Automatically adapts from Arctic (-30°C) to mild climates (5°C)

        Examples:
        - Kiruna (Extreme Cold, -30°C): DM -800 to -1200 is normal
        - Stockholm (Cold, -10°C): DM -450 to -700 is normal
        - Copenhagen (Moderate Cold, 0°C): DM -300 to -500 is normal

        This replaces hardcoded temperature bands with universal climate-aware logic.

        Args:
            outdoor_temp: Current outdoor temperature (°C)

        Returns:
            Dictionary with:
            - normal: Expected normal operating DM
            - warning: Start warning/recovery at this DM

        References:
            - climate_zones.py: ClimateZoneDetector implementation
            - IMPLEMENTATION_PLAN/FUTURE/CLIMATE_ZONE_DM_INTEGRATION.md
        """
        # Use climate zone detector to get context-aware DM range
        dm_range = self.climate_detector.get_expected_dm_range(outdoor_temp)

        _LOGGER.debug(
            "Expected DM for %s zone at %.1f°C: normal %.0f to %.0f, warning %.0f",
            self.climate_detector.zone_info.name,
            outdoor_temp,
            dm_range["normal_min"],
            dm_range["normal_max"],
            dm_range["warning"],
        )

        return {
            "normal": dm_range["normal_max"],  # Use deep end of normal range
            "warning": dm_range["warning"],  # Warning threshold
        }

    def _get_thermal_mass_adjusted_thresholds(
        self,
        base_thresholds: dict,
        heating_type: str,
    ) -> dict:
        """Adjust DM thresholds based on thermal mass (prevents overshoot with lag).

        High thermal mass systems need tighter thresholds because:
        - Long thermal lag (6+ hours for concrete slab)
        - Current DM doesn't immediately affect indoor temperature
        - Solar gain can mask underlying thermal debt accumulation
        - Need larger buffer to handle sunset/weather changes

        Args:
            base_thresholds: Climate-aware thresholds from ClimateZoneDetector
                {"normal_min": -60, "normal_max": -276, "warning": -276, "critical": -1500}
            heating_type: Heating system type ("concrete_ufh", "timber", "radiator", etc.)

        Returns:
            Adjusted thresholds with thermal mass buffer applied

        Example:
            Stockholm at 10°C → base warning -276
            Concrete slab:      -276 × 1.3 = -359 (tighter)
            Timber:             -276 × 1.15 = -317 (moderate)
            Radiator:           -276 × 1.0 = -276 (standard)

        Notes:
            - Critical threshold (-1500) never adjusted (absolute maximum)
            - Prevents v0.1.0 failure mode (DM -700 during solar gain)
            - Maintains thermal buffer for forecast uncertainty

        Research:
            - v0.1.0 case: DM -700 allowed → 1.5°C drop at sunset
            - Concrete lag: 6-12 hours observed in production systems

        References:
            IMPLEMENTATION_PLAN/THERMAL_MASS_AND_CONTEXT_FIXES_OCT23.md
        """
        # Get multiplier based on heating type
        if heating_type in ("concrete_ufh", "concrete_slab"):
            multiplier = DM_THERMAL_MASS_BUFFER_CONCRETE  # 1.3
        elif heating_type in ("timber", "timber_ufh"):
            multiplier = DM_THERMAL_MASS_BUFFER_TIMBER  # 1.15
        else:  # radiator or unknown (conservative default)
            multiplier = DM_THERMAL_MASS_BUFFER_RADIATOR  # 1.0

        # Apply multiplier to thresholds (more negative = tighter)
        adjusted = {
            "normal_min": base_thresholds["normal_min"] * multiplier,
            "normal_max": base_thresholds["normal_max"] * multiplier,
            "warning": base_thresholds["warning"] * multiplier,
            "critical": base_thresholds["critical"],  # Never adjust absolute maximum
        }

        _LOGGER.debug(
            "Thermal mass adjusted thresholds: heating type '%s' (multiplier %.2f) "
            "→ warning %.0f (base: %.0f), critical %.0f",
            heating_type,
            multiplier,
            adjusted["warning"],
            base_thresholds["warning"],
            adjusted["critical"],
        )

        return adjusted

    def _proactive_debt_prevention_layer(self, nibe_state, weather_data=None) -> LayerDecision:
        """Proactive thermal debt prevention with climate-aware thresholds and trend prediction.

        PHILOSOPHY:
        - Continuous modulation beats forced cycling
        - Thermal debt from forced stops worse than running low power
        - Prevent peaks by maintaining gentle background heating

        NEW - PREDICTIVE TREND ANALYSIS:
        Uses indoor temperature trend to detect problems 30-60 minutes before they occur.
        Rapid cooling (-0.3°C/h or faster) combined with outdoor cold and temperature deficit
        indicates thermal debt will develop soon - intervene proactively.

        PHASE 3.2 - FORECAST VALIDATION:
        Validates rapid cooling trend against weather forecast to reduce false positives.
        When both signals agree (indoor cooling + outdoor cooling forecast), boost response.
        When they disagree (indoor cooling + stable forecast), reduce response (likely temporary).

        CLIMATE-AWARE DESIGN:
        Unlike hardcoded DM thresholds, this layer adapts to climate and outdoor temperature:
        - Arctic winter (-30°C): Zone 1 at DM -120, Zone 2 at DM -320, Zone 3 at DM -640
        - Cold winter (-10°C): Zone 1 at DM -68, Zone 2 at DM -180, Zone 3 at DM -360
        - Mild climate (0°C): Zone 1 at DM -45, Zone 2 at DM -120, Zone 3 at DM -240

        Thresholds calculated as percentages of climate-aware expected DM (normal_max):
        - Zone 1 (15% of normal_max): Early warning, gentle nudge before compressor starts
        - Zone 2 (40% of normal_max): Moderate action when compressor running
        - Zone 3 (80% of normal_max): Strong action when approaching warning threshold

        NOTE: Degree Minutes (DM) are negative values! Percentages make them LESS negative:
        - expected_dm["normal"] = -800 (Arctic example)
        - zone1_threshold = -800 * 0.15 = -120 (less negative = earlier intervention)
        - zone2_threshold = -800 * 0.40 = -320 (moderate)
        - zone3_threshold = -800 * 0.80 = -640 (more negative = closer to limit)        THERMAL LAG CONSIDERATION:
        UFH systems have significant thermal lag - changes take hours to manifest:
        - Concrete slab UFH: 6+ hours lag (UFH_CONCRETE_PREDICTION_HORIZON = 12h)
        - Timber UFH: 2-3 hours lag (UFH_TIMBER_PREDICTION_HORIZON = 6h)
        - Radiators: <1 hour lag (UFH_RADIATOR_PREDICTION_HORIZON = 2h)

        This layer provides GENTLE nudges that work with thermal inertia rather than
        fighting it. Small offsets (+0.5-1.5°C) accumulate effect over hours, preventing
        deep debt without causing overshoots.

        Think of it as "steering a ship" - small rudder adjustments early prevent
        large course corrections later.

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data (optional, for forecast validation)

        Returns:
            LayerDecision with climate-aware proactive gentle heating

        References:
            MASTER_IMPLEMENTATION_PLAN.md: Phase 3.2 - Forecast Validation of Trend
        """
        degree_minutes = nibe_state.degree_minutes
        outdoor_temp = nibe_state.outdoor_temp
        indoor_temp = nibe_state.indoor_temp

        # NEW: Get temperature trend for predictive intervention
        thermal_trend = self._get_thermal_trend()
        trend_rate = thermal_trend.get("rate_per_hour", 0.0)

        # Calculate current deficit
        deficit = self.target_temp - indoor_temp

        # Predict deficit in 1 hour if trend continues
        predicted_deficit_1h = deficit - trend_rate  # Negative rate increases deficit

        # ========================================
        # COMMON SENSE CHECK: Don't heat if well above target with no cold snap
        # ========================================
        # If indoor is significantly above target AND weather is stable/warming,
        # then thermal debt prevention is not needed - we have plenty of thermal margin.
        # This prevents unnecessary heating when conditions are actually comfortable.
        if deficit < -COMMON_SENSE_TEMP_ABOVE_TARGET:  # Above target threshold
            # Check if weather forecast shows NO significant cooling
            forecast_stable = True
            if weather_data and weather_data.forecast_hours:
                forecast_hours = weather_data.forecast_hours[:COMMON_SENSE_FORECAST_HORIZON]
                if forecast_hours:
                    min_forecast_temp = min(h.temperature for h in forecast_hours)
                    # Cold snap = forecast drops >threshold from current
                    forecast_stable = (
                        outdoor_temp - min_forecast_temp
                    ) < COMMON_SENSE_COLD_SNAP_THRESHOLD

            if forecast_stable:
                return LayerDecision(
                    offset=0.0,
                    weight=0.0,
                    reason=f"Common sense: Indoor {indoor_temp:.1f}°C is {abs(deficit):.1f}°C above target, weather stable - no heating needed (DM {degree_minutes:.0f})",
                )

        # ========================================
        # NEW: RAPID COOLING DETECTION (Predictive)
        # ========================================
        # Detect rapid cooling BEFORE thermal debt accumulates in degree minutes
        # This gives 30-60 minutes advance warning vs reactive DM-based approach
        if self._is_cooling_rapidly(thermal_trend):
            # House cooling faster than threshold (from RAPID_COOLING_THRESHOLD)
            # This often precedes thermal debt if outdoor temp is cold

            if outdoor_temp < RAPID_COOLING_OUTDOOR_THRESHOLD and deficit > THERMAL_CHANGE_MODERATE:
                # Cold outside + already below target + rapid cooling = trouble ahead
                boost = min(
                    abs(trend_rate) * RAPID_COOLING_BOOST_MULTIPLIER,
                    RAPID_COOLING_BOOST_MAX,
                )

                # PHASE 3.2: Validate against weather forecast to reduce false positives
                forecast_reason = ""
                if weather_data and weather_data.forecast_hours:
                    next_3h_temps = [h.temperature for h in weather_data.forecast_hours[:3]]
                    if next_3h_temps:
                        min_forecast_temp = min(next_3h_temps)
                        temp_will_drop = min_forecast_temp < outdoor_temp - 1.0

                        if temp_will_drop:
                            boost *= MULTIPLIER_BOOST_30_PERCENT
                            forecast_reason = " (forecast confirms cooling)"
                        else:
                            boost *= MULTIPLIER_REDUCTION_20_PERCENT
                            forecast_reason = " (forecast stable, likely temporary)"

                return LayerDecision(
                    offset=boost,
                    weight=RAPID_COOLING_WEIGHT,
                    reason=(
                        f"Proactive: Rapid cooling ({trend_rate:.2f}°C/h), "
                        f"deficit {deficit:.1f}°C → {predicted_deficit_1h:.1f}°C in 1h"
                        f"{forecast_reason}"
                    ),
                )

        # Get climate-aware expected DM range for current conditions
        expected_dm = self._calculate_expected_dm_for_temperature(outdoor_temp)

        # Climate-aware thresholds (adapt to outdoor temp and climate zone)
        # In Arctic winter (-30°C), these thresholds will be much deeper than mild climate (5°C)
        zone1_threshold = expected_dm["normal"] * PROACTIVE_ZONE1_THRESHOLD_PERCENT
        zone2_threshold = expected_dm["normal"] * PROACTIVE_ZONE2_THRESHOLD_PERCENT
        zone3_threshold = expected_dm["normal"] * PROACTIVE_ZONE3_THRESHOLD_PERCENT

        # PROACTIVE ZONE 1: Early warning (gentle nudge at any meaningful thermal debt)
        # 5% threshold adapts: Arctic -30°C → DM -60, Mild 10°C → DM -10 (climate-aware)
        if zone2_threshold < degree_minutes <= zone1_threshold:
            # Gentle nudge to prevent deeper deficit
            offset = PROACTIVE_ZONE1_OFFSET

            # NEW: Boost more if also cooling rapidly
            if trend_rate < THERMAL_CHANGE_MODERATE_COOLING:
                offset *= MULTIPLIER_BOOST_30_PERCENT
                reason_suffix = f" (trend: {trend_rate:.2f}°C/h)"
            else:
                reason_suffix = ""

            return LayerDecision(
                offset=offset,
                weight=LAYER_WEIGHT_PROACTIVE_MIN,
                reason=f"Proactive Z1: DM {degree_minutes:.0f} (threshold: {zone1_threshold:.0f}), gentle heating prevents debt{reason_suffix}",
            )

        # PROACTIVE ZONE 2: Compressor running, monitor trend
        # Example: Arctic -30°C → -320, Cold -10°C → -180, Mild 0°C → -120
        elif zone3_threshold < degree_minutes <= zone2_threshold:
            # Compressor running, check if deficit still growing
            # Slight boost to help it catch up faster
            offset = PROACTIVE_ZONE2_OFFSET
            return LayerDecision(
                offset=offset,
                weight=PROACTIVE_ZONE2_WEIGHT,
                reason=f"Proactive Z2: DM {degree_minutes:.0f} (threshold: {zone2_threshold:.0f}), boost recovery speed",
            )

        # PROACTIVE ZONE 3: Significant deficit (beyond typical)
        # Uses expected_dm["normal"] directly (already climate-aware)
        elif expected_dm["normal"] < degree_minutes <= zone3_threshold:
            # Deficit growing beyond typical - stronger action
            # Scale offset based on how far into warning zone
            deficit_severity = (zone3_threshold - degree_minutes) / (
                zone3_threshold - expected_dm["normal"]
            )  # 0-1 scale
            offset = PROACTIVE_ZONE3_OFFSET_MIN + (deficit_severity * PROACTIVE_ZONE3_OFFSET_RANGE)

            return LayerDecision(
                offset=offset,
                weight=LAYER_WEIGHT_PROACTIVE_MAX,
                reason=f"Proactive Z3: DM {degree_minutes:.0f} (threshold: {zone3_threshold:.0f}), prevent deeper debt (severity: {deficit_severity:.2f})",
            )

        # NEW: Extended proactive zones (Oct 19, 2025 fix)
        # Previous gap: Zone 3 stopped at 50% of normal_max, leaving huge gap to WARNING
        # New zones provide continuous escalation to prevent thermal debt worsening

        # Calculate extended zone thresholds
        zone4_threshold = expected_dm["normal"] * PROACTIVE_ZONE4_THRESHOLD_PERCENT
        zone5_threshold = expected_dm["warning"] * PROACTIVE_ZONE5_THRESHOLD_PERCENT

        # PROACTIVE ZONE 4: Strong prevention (between normal and warning)
        # At -4.3°C: DM -293 to -586 (was gap before, now has intervention)
        if zone4_threshold < degree_minutes <= expected_dm["normal"]:
            offset = PROACTIVE_ZONE4_OFFSET
            return LayerDecision(
                offset=offset,
                weight=PROACTIVE_ZONE4_WEIGHT,
                reason=f"Proactive Z4: DM {degree_minutes:.0f} (threshold: {zone4_threshold:.0f}), strong prevention",
            )

        # PROACTIVE ZONE 5: Very strong prevention (beyond warning threshold)
        # At -4.3°C: DM -586 to -733 (prevents escalation to CRITICAL tiers)
        elif zone5_threshold < degree_minutes <= zone4_threshold:
            offset = PROACTIVE_ZONE5_OFFSET
            return LayerDecision(
                offset=offset,
                weight=PROACTIVE_ZONE5_WEIGHT,
                reason=f"Proactive Z5: DM {degree_minutes:.0f} (threshold: {zone5_threshold:.0f}), very strong prevention",
            )

        # Outside proactive zones - let emergency/critical layers handle it
        return LayerDecision(
            offset=0.0,
            weight=0.0,
            reason="Proactive prevention not needed",
        )

    def _effect_layer(self, nibe_state, current_peak: float, current_power: float) -> LayerDecision:
        """Effect tariff protection with PREDICTIVE peak avoidance (Phase 4).

        Uses indoor temperature trend to predict heating demand in next 15 minutes.
        Acts BEFORE power spikes instead of reacting to them.

        PHILOSOPHY:
        Traditional reactive approach waits until power is high, then reduces.
        Predictive approach sees house cooling rapidly → knows compressor will ramp up soon
        → reduces offset NOW before spike occurs → smoother power profile.

        Args:
            nibe_state: Current NIBE state
            current_peak: Current monthly peak (kW) - from peak_this_month sensor
            current_power: Current whole-house power consumption (kW) - from peak_today sensor

        Returns:
            LayerDecision with predictive peak protection

        References:
            MASTER_IMPLEMENTATION_PLAN.md: Phase 4 - Predictive Peak Avoidance
        """
        # current_power is always provided from coordinator's peak tracking (actual measurements)

        # Get current quarter
        now = dt_util.now()
        current_quarter = (now.hour * 4) + (now.minute // 15)  # 0-95

        # Check if approaching monthly 15-minute peak
        limit_decision = self.effect.should_limit_power(current_power, current_quarter)

        # PHASE 4: Get thermal trend for predictive analysis
        thermal_trend = self._get_thermal_trend()
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
            return LayerDecision(
                offset=EFFECT_OFFSET_CRITICAL,
                weight=EFFECT_WEIGHT_CRITICAL,
                reason=f"Peak: CRITICAL ({current_power:.1f}/{current_peak:.1f} kW, Q{current_quarter})",
            )
        elif predicted_margin < EFFECT_MARGIN_PREDICTIVE and predicted_power_increase > 0:
            # PREDICTIVE: Will approach peak in next 15 min - act NOW
            # This is the key innovation: prevent spike before it happens
            return LayerDecision(
                offset=EFFECT_OFFSET_PREDICTIVE,
                weight=EFFECT_WEIGHT_PREDICTIVE,
                reason=(
                    f"Peak: PREDICTIVE avoidance "
                    f"(predicted {predicted_power:.1f} kW, {prediction_reason}, Q{current_quarter})"
                ),
            )
        elif limit_decision.severity == "WARNING":
            # Close to peak - check if trend shows increasing demand
            if predicted_power_increase > 0:
                # Warning + rising demand = moderate reduction
                return LayerDecision(
                    offset=EFFECT_OFFSET_WARNING_RISING,
                    weight=EFFECT_WEIGHT_WARNING_RISING,
                    reason=f"Peak: WARNING + demand rising ({prediction_reason}, Q{current_quarter})",
                )
            else:
                # Warning but demand stable/falling = gentle reduction
                return LayerDecision(
                    offset=EFFECT_OFFSET_WARNING_STABLE,
                    weight=EFFECT_WEIGHT_WARNING_STABLE,
                    reason=f"Peak: WARNING + demand {prediction_reason} (Q{current_quarter})",
                )
        else:
            # Safe margin - no action needed
            return LayerDecision(
                offset=0.0,
                weight=0.0,
                reason=f"Peak: Safe margin ({current_power:.1f}/{current_peak:.1f} kW, Q{current_quarter})",
            )

    def _prediction_layer(self, nibe_state, weather_data) -> LayerDecision:
        """Prediction layer: Learned pre-heating using thermal state predictor.

        Uses learned building thermal characteristics to make intelligent
        pre-heating decisions based on predicted temperature evolution.

        This layer uses actual learned thermal response rather than generic
        thermal mass assumptions, providing more accurate pre-heating.

        Phase 6 - Self-learning capability

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data

        Returns:
            LayerDecision with learned pre-heating recommendation
        """
        # Skip if predictor not available or not enough data
        if not self.predictor:
            return LayerDecision(offset=0.0, weight=0.0, reason="Predictor not initialized")

        if len(self.predictor.state_history) < 96:  # Less than 24 hours of data
            return LayerDecision(
                offset=0.0,
                weight=0.0,
                reason=f"Learning: {len(self.predictor.state_history)}/96 observations",
            )

        # Skip if no weather forecast available
        if not weather_data or not weather_data.forecast_hours:
            return LayerDecision(offset=0.0, weight=0.0, reason="No weather forecast")

        try:
            # Use UFH-type-specific forecast horizon for learned predictions
            # Concrete slab: 24h, Timber: 12h, Radiators: 6h
            prediction_horizon = int(self.thermal.get_prediction_horizon())
            forecast_temps = [
                hour.temperature for hour in weather_data.forecast_hours[:prediction_horizon]
            ]

            if not forecast_temps:
                return LayerDecision(offset=0.0, weight=0.0, reason="Empty weather forecast")

            # Check if pre-heating is recommended
            # Use half of prediction horizon as lookahead (balance between early and late)
            hours_ahead = prediction_horizon // 2
            preheat_decision = self.predictor.should_pre_heat(
                target_temp=self.target_temp,
                hours_ahead=hours_ahead,
                future_outdoor_temps=forecast_temps,
                current_outdoor_temp=nibe_state.outdoor_temp,
                current_indoor_temp=nibe_state.indoor_temp,
                thermal_mass=self.thermal.thermal_mass,
                insulation_quality=self.thermal.insulation_quality,
            )

            if preheat_decision.should_preheat:
                # Thermal predictor now accounts for current overshoot as stored thermal energy
                # The recommended_offset is already adjusted for overshoot in should_pre_heat()
                # Weight 0.65 - slightly higher than base price layer (0.6)
                # but lower than effect/weather layers (0.7-0.8)
                return LayerDecision(
                    offset=preheat_decision.recommended_offset,
                    weight=LAYER_WEIGHT_PREDICTION,
                    reason=f"Learned pre-heat: {preheat_decision.reason}",
                )
            else:
                return LayerDecision(
                    offset=0.0,
                    weight=0.0,
                    reason="Learned: No pre-heat needed",
                )

        except AttributeError as err:
            _LOGGER.error(
                "Thermal model API compatibility error: %s. "
                "This indicates the thermal model is missing required attributes. "
                "Check that AdaptiveThermalModel has insulation_quality property. "
                "Falling back to basic optimization without pre-heating.",
                err,
                exc_info=True,
            )
            return LayerDecision(
                offset=0.0,
                weight=0.0,
                reason="Thermal model API error - contact support",
            )
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as err:
            _LOGGER.warning("Prediction calculation failed: %s", err)
            return LayerDecision(
                offset=0.0,
                weight=0.0,
                reason=f"Prediction error: {err}",
            )

    def _weather_layer(self, nibe_state, weather_data) -> LayerDecision:
        """Simplified weather prediction layer - Pure forecast-based gentle pre-heating.

        Philosophy (Oct 20, 2025):
        "The heating we add NOW shows up in 6 hours - pre-heat BEFORE cold arrives"

        Problem: Concrete slab 6-hour thermal lag causes reactive heating to arrive
        too late, resulting in thermal debt spirals (DM -1000 @ 04:00) followed by
        massive overshoot (26°C @ 16:00 from heat added 6 hours earlier).

        Solution: Simple proactive pre-heating:
        1. PRIMARY: Forecast shows ≥5°C drop in next 12h → +0.5°C gentle pre-heat
        2. CONFIRMATION: Indoor cooling ≥0.5°C/h → Confirms forecast, maintains +0.5°C
        3. MODERATION: Let SAFETY, COMFORT, EFFECT layers handle naturally via weighted aggregation

        Weight scaling by thermal mass:
        - Concrete slab (1.5): 0.85 × 1.5 = 1.275 (very high priority, 6h lag)
        - Timber UFH (1.0): 0.85 × 1.0 = 0.85 (high priority, 2-3h lag)
        - Radiators (0.5): 0.85 × 0.5 = 0.425 (moderate priority, <1h lag)

        Real-world validation:
        - Prevents 20:00→04:00 emergency thermal debt cycles
        - Prevents 16:00 overshoot from overnight heating
        - No complex calculations - just gentle constant pre-heat

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data

        Returns:
            LayerDecision with gentle pre-heating recommendation
        """
        if not weather_data or not weather_data.forecast_hours:
            return LayerDecision(offset=0.0, weight=0.0, reason="No weather data")

        # Check forecast for significant temperature drop
        current_outdoor = nibe_state.outdoor_temp
        forecast_hours = weather_data.forecast_hours[: int(WEATHER_FORECAST_HORIZON)]

        if not forecast_hours:
            return LayerDecision(offset=0.0, weight=0.0, reason="No forecast data")

        # Find minimum temperature in forecast period
        min_temp = min(f.temperature for f in forecast_hours)
        temp_drop = min_temp - current_outdoor

        # PRIMARY TRIGGER: Forecast shows ≥5°C drop
        forecast_triggered = temp_drop <= WEATHER_FORECAST_DROP_THRESHOLD

        # CONFIRMATION TRIGGER: Indoor already cooling (confirms forecast)
        thermal_trend = self._get_thermal_trend()
        trend_rate = thermal_trend.get("rate_per_hour", 0.0)
        trend_confidence = thermal_trend.get("confidence", 0.0)

        indoor_cooling = (
            trend_rate <= WEATHER_INDOOR_COOLING_CONFIRMATION
            and trend_confidence > 0.4  # Sufficient data confidence
        )

        if forecast_triggered or indoor_cooling:
            # Calculate thermal mass-adjusted weight
            # Concrete slab: 0.85 × 1.5 = 1.275 (beats most layers except SAFETY/EFFECT_CRITICAL)
            # Timber: 0.85 × 1.0 = 0.85 (beats price, comfort, proactive)
            # Radiators: 0.85 × 0.5 = 0.425 (lower priority, fast response)
            weather_weight = min(
                LAYER_WEIGHT_WEATHER_PREDICTION * self.thermal.thermal_mass,
                WEATHER_WEIGHT_CAP,  # Cap below Safety (1.0)
            )

            # Determine trigger reason
            if forecast_triggered and indoor_cooling:
                trigger = f"Forecast {temp_drop:.1f}°C drop + Indoor cooling {trend_rate:.2f}°C/h (confirmed)"
            elif forecast_triggered:
                trigger = f"Forecast {temp_drop:.1f}°C drop in {WEATHER_FORECAST_HORIZON:.0f}h (proactive)"
            else:
                trigger = f"Indoor cooling {trend_rate:.2f}°C/h (reactive confirmation)"

            return LayerDecision(
                offset=WEATHER_GENTLE_OFFSET,  # Constant +0.5°C (simple, predictable)
                weight=weather_weight,
                reason=(
                    f"Weather pre-heat: {trigger} → "
                    f"+{WEATHER_GENTLE_OFFSET:.1f}°C @ weight {weather_weight:.2f} "
                    f"(base {LAYER_WEIGHT_WEATHER_PREDICTION:.2f} × thermal_mass {self.thermal.thermal_mass:.1f})"
                ),
            )

        return LayerDecision(
            offset=0.0,
            weight=0.0,
            reason="Weather: No pre-heating needed (forecast stable, indoor stable)",
        )

    def _weather_compensation_layer(self, nibe_state, weather_data) -> LayerDecision:
        """Mathematical weather compensation layer with adaptive climate system.

        Calculates optimal flow temperature using:
        - Universal flow temperature formula (validated across manufacturers)
        - Heat transfer method (if radiator specs available)
        - UFH-specific adjustments (concrete/timber)
        - Adaptive climate zones (latitude-based, globally applicable)
        - Weather learning (unusual pattern detection)

        Automatically adapts to global climates:
        - Kiruna, Sweden (-30°C) → Arctic zone → 2.5°C base margin
        - Stockholm, Sweden (-10°C) → Cold zone → 1.0°C base margin
        - London, UK (0°C) → Temperate zone → 0.5°C base margin
        - Paris, France (5°C) → Mild zone → 0.0°C base margin

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data

        Returns:
            LayerDecision with mathematically calculated offset
        """
        # Check if feature is enabled
        if not self.enable_weather_compensation:
            return LayerDecision(offset=0.0, weight=0.0, reason="Weather compensation disabled")

        if not weather_data or not weather_data.forecast_hours:
            return LayerDecision(offset=0.0, weight=0.0, reason="No weather data")

        current_outdoor = nibe_state.outdoor_temp
        current_flow = nibe_state.flow_temp

        # Calculate optimal flow temperature using physics-based formulas
        flow_calc = self.weather_comp.calculate_optimal_flow_temp(
            indoor_setpoint=self.target_temp,
            outdoor_temp=current_outdoor,
            prefer_method="auto",  # Combines universal formula + heat transfer if available
        )

        # Adaptive climate system safety adjustments
        # Replaces hardcoded Swedish thresholds with universal climate zones + learning
        unusual_weather = False
        unusual_severity = 0.0

        # Check for unusual weather patterns if weather learner available
        if self.weather_learner and weather_data.forecast_hours:
            try:
                # Extract forecast for unusual weather detection
                forecast_temps = [h.temperature for h in weather_data.forecast_hours[:24]]
                unusual = self.weather_learner.detect_unusual_weather(
                    current_date=dt_util.now(),
                    forecast=forecast_temps,
                )

                if unusual.is_unusual:
                    unusual_weather = True
                    # Map severity string to 0.0-1.0 scale
                    unusual_severity = 1.0 if unusual.severity == "extreme" else 0.5

                    _LOGGER.info(
                        "Unusual weather detected: %s (deviation: %.1f°C)",
                        unusual.recommendation,
                        unusual.deviation_from_typical,
                    )
            except (AttributeError, KeyError, ValueError, TypeError) as e:
                _LOGGER.warning("Weather learning check failed: %s", e)

        # Get adaptive safety margin from climate system
        safety_margin = self.climate_system.get_safety_margin(
            outdoor_temp=current_outdoor,
            unusual_weather_detected=unusual_weather,
            unusual_severity=unusual_severity,
        )

        # Apply safety margin to calculated flow temp
        adjusted_flow_temp = flow_calc.flow_temp + safety_margin

        # Calculate required offset from current flow temperature
        # NIBE curve sensitivity: ~1.5°C flow change per 1°C offset
        curve_sensitivity = DEFAULT_CURVE_SENSITIVITY
        required_offset = self.weather_comp.calculate_required_offset(
            optimal_flow_temp=adjusted_flow_temp,
            current_flow_temp=current_flow,
            curve_sensitivity=curve_sensitivity,
        )

        # Get dynamic weight from climate system
        dynamic_weight = self.climate_system.get_dynamic_weight(
            outdoor_temp=current_outdoor,
            unusual_weather_detected=unusual_weather,
        )

        # Apply user-configured weight adjustment
        final_weight = dynamic_weight * self.weather_comp_weight

        # Defer weather compensation when thermal debt exists (Conservative strategy)
        # Allow thermal reality (DM + comfort + proactive) to override outdoor temp optimization
        degree_minutes = nibe_state.degree_minutes
        if degree_minutes < WEATHER_COMP_DEFER_DM_CRITICAL:
            # Critical debt: 39% reduction (0.49 → 0.30)
            defer_factor = WEATHER_COMP_DEFER_WEIGHT_CRITICAL / DEFAULT_WEATHER_COMPENSATION_WEIGHT
            defer_reason = f"Critical debt (DM {degree_minutes:.0f})"
        elif degree_minutes < WEATHER_COMP_DEFER_DM_SIGNIFICANT:
            # Significant debt: 29% reduction (0.49 → 0.35)
            defer_factor = (
                WEATHER_COMP_DEFER_WEIGHT_SIGNIFICANT / DEFAULT_WEATHER_COMPENSATION_WEIGHT
            )
            defer_reason = f"Significant debt (DM {degree_minutes:.0f})"
        elif degree_minutes < WEATHER_COMP_DEFER_DM_MODERATE:
            # Moderate debt: 18% reduction (0.49 → 0.40)
            defer_factor = WEATHER_COMP_DEFER_WEIGHT_MODERATE / DEFAULT_WEATHER_COMPENSATION_WEIGHT
            defer_reason = f"Moderate debt (DM {degree_minutes:.0f})"
        elif degree_minutes < WEATHER_COMP_DEFER_DM_LIGHT:
            # Light debt: 8% reduction (0.49 → 0.45)
            defer_factor = WEATHER_COMP_DEFER_WEIGHT_LIGHT / DEFAULT_WEATHER_COMPENSATION_WEIGHT
            defer_reason = f"Light debt (DM {degree_minutes:.0f})"
        else:
            # No debt: full weather comp weight
            defer_factor = 1.0
            defer_reason = None

        final_weight = final_weight * defer_factor

        # Build comprehensive reasoning
        zone_info = self.climate_system.get_climate_info()
        reasoning_parts = [
            f"Math WC: {flow_calc.method}",
            f"Zone: {zone_info['name']}",
            f"Optimal: {flow_calc.flow_temp:.1f}°C",
        ]

        if safety_margin > 0:
            reasoning_parts.append(f"Safety: +{safety_margin:.1f}°C")
            reasoning_parts.append(f"Adjusted: {adjusted_flow_temp:.1f}°C")

        if unusual_weather:
            reasoning_parts.append(f"Unusual weather (severity={unusual_severity:.1f})")

        reasoning_parts.append(f"Current: {current_flow:.1f}°C → offset: {required_offset:+.1f}°C")
        reasoning_parts.append(f"Weight: {final_weight:.2f}")

        if defer_reason:
            reasoning_parts.append(f"Deferred: {defer_reason}")

        reasoning = "; ".join(reasoning_parts)

        _LOGGER.debug("Weather compensation layer: %s", reasoning)

        return LayerDecision(
            offset=required_offset,
            weight=final_weight,
            reason=reasoning,
        )

    def _price_layer(self, nibe_state, price_data) -> LayerDecision:
        """Spot price layer: Forward-looking optimization from native 15-minute GE-Spot data.

        Enhanced Nov 27, 2025: Added forward-looking price analysis
        - Looks ahead 4 hours for significant price changes
        - Reduces heating when much cheaper period coming soon
        - Pre-heats when much more expensive period approaching

        Args:
            nibe_state: Current NIBE state (for strategic overshoot context)
            price_data: GE-Spot price data with native 15-min intervals

        Returns:
            LayerDecision with price-based offset
        """
        if not price_data or not price_data.today:
            return LayerDecision(offset=0.0, weight=0.0, reason="No price data")

        now = dt_util.now()
        current_quarter = (now.hour * 4) + (now.minute // 15)  # 0-95

        # Bound check quarter index (safety)
        if current_quarter >= len(price_data.today):
            _LOGGER.warning(
                "Current quarter %d exceeds available periods (%d)",
                current_quarter,
                len(price_data.today),
            )
            current_quarter = min(current_quarter, len(price_data.today) - 1)

        # Get current period classification and price
        classification = self.price.get_current_classification(current_quarter)
        current_period = price_data.today[current_quarter]
        current_price = current_period.price

        # Initialize variables (moved outside if-block to avoid UnboundLocalError)
        current_cheap_too_brief = False
        remaining_cheap_quarters = 0
        is_volatile_period = False
        volatile_reason = ""

        # Forward-looking price analysis - horizon scales with thermal mass
        # Base 4h × thermal_mass (0.5-2.0) → 2.0-8.0 hour adaptive horizon
        # Calculate horizon based on thermal mass (higher mass = longer lookahead)
        thermal_mass = self.thermal.thermal_mass if self.thermal else 1.0
        forecast_hours = PRICE_FORECAST_BASE_HORIZON * thermal_mass

        forecast_quarters = int(forecast_hours * 4)  # Convert hours to 15-min quarters

        lookahead_end = min(current_quarter + forecast_quarters, 96)
        upcoming_periods = price_data.today[current_quarter + 1 : lookahead_end]

        # Also check tomorrow's first periods if we're near end of day
        if price_data.has_tomorrow and lookahead_end >= 96:
            remaining_quarters = forecast_quarters - (96 - current_quarter - 1)
            upcoming_periods.extend(price_data.tomorrow[:remaining_quarters])

        forecast_adjustment = 0.0
        forecast_reason = ""

        if upcoming_periods and current_price > 0:
            # Find index and count duration of min/max prices
            min_idx = next(
                i
                for i, p in enumerate(upcoming_periods)
                if p.price == min(p.price for p in upcoming_periods)
            )
            max_idx = next(
                i
                for i, p in enumerate(upcoming_periods)
                if p.price == max(p.price for p in upcoming_periods)
            )

            # Count consecutive quarters around min price meeting CHEAP threshold
            cheap_duration = 1
            for i in range(min_idx + 1, len(upcoming_periods)):
                if upcoming_periods[i].price / current_price < PRICE_FORECAST_CHEAP_THRESHOLD:
                    cheap_duration += 1
                else:
                    break
            for i in range(min_idx - 1, -1, -1):
                if upcoming_periods[i].price / current_price < PRICE_FORECAST_CHEAP_THRESHOLD:
                    cheap_duration += 1
                else:
                    break

            # Count consecutive quarters around max price meeting EXPENSIVE threshold
            expensive_duration = 1
            for i in range(max_idx + 1, len(upcoming_periods)):
                if upcoming_periods[i].price / current_price > PRICE_FORECAST_EXPENSIVE_THRESHOLD:
                    expensive_duration += 1
                else:
                    break
            for i in range(max_idx - 1, -1, -1):
                if upcoming_periods[i].price / current_price > PRICE_FORECAST_EXPENSIVE_THRESHOLD:
                    expensive_duration += 1
                else:
                    break

            # Calculate price ratios
            min_price_ratio = min(p.price for p in upcoming_periods) / current_price
            max_price_ratio = max(p.price for p in upcoming_periods) / current_price

            # Check if current CHEAP period is too brief for meaningful heating (Nov 29, 2025)
            # Compressor needs ~45min to efficiently use cheap electricity
            if classification == QuarterClassification.CHEAP:
                # Count remaining consecutive CHEAP quarters (including current)
                remaining_cheap_quarters = 1
                for q in range(current_quarter + 1, 96):
                    if self.price.get_current_classification(q) == QuarterClassification.CHEAP:
                        remaining_cheap_quarters += 1
                    else:
                        break

                # Check tomorrow if cheap continues to end of today
                if (
                    remaining_cheap_quarters < PRICE_FORECAST_MIN_DURATION
                    and current_quarter + remaining_cheap_quarters >= 96
                    and self.price.has_tomorrow_prices()
                ):
                    for q in range(96):
                        if self.price.get_tomorrow_classification(q) == QuarterClassification.CHEAP:
                            remaining_cheap_quarters += 1
                        else:
                            break

                current_cheap_too_brief = remaining_cheap_quarters < PRICE_FORECAST_MIN_DURATION

            # Volatile/jumpy price detection (Nov 30, 2025)
            # Bidirectional scan: 1h backward + 1h forward (total ~2h window)
            # Logic: If we're IN THE MIDDLE of volatile period (prices jumping around us),
            # we see many non-NORMAL periods in both directions → hold steady instead of chasing

            # Scan window: N quarters back + current + N forward (N = PRICE_VOLATILE_SCAN_QUARTERS_EACH_DIRECTION)
            scan_start = max(0, current_quarter - PRICE_VOLATILE_SCAN_QUARTERS_EACH_DIRECTION)
            scan_end = min(current_quarter + PRICE_VOLATILE_SCAN_QUARTERS_EACH_DIRECTION, 95)

            expensive_count = 0
            cheap_count = 0
            peak_count = 0
            normal_count = 0

            # Scan bidirectionally through today
            for quarter_idx in range(scan_start, scan_end + 1):
                quarter_classification = self.price.get_current_classification(quarter_idx)
                if quarter_classification == QuarterClassification.PEAK:
                    peak_count += 1
                elif quarter_classification == QuarterClassification.EXPENSIVE:
                    expensive_count += 1
                elif quarter_classification == QuarterClassification.CHEAP:
                    cheap_count += 1
                elif quarter_classification == QuarterClassification.NORMAL:
                    normal_count += 1

            # Check tomorrow if forward scan would extend past midnight
            if (
                current_quarter + PRICE_VOLATILE_SCAN_QUARTERS_EACH_DIRECTION > 95
                and price_data.has_tomorrow
            ):
                # How many quarters we need from tomorrow
                remaining_scan = (
                    current_quarter + PRICE_VOLATILE_SCAN_QUARTERS_EACH_DIRECTION
                ) - 95
                for quarter_idx in range(min(remaining_scan, 96)):
                    quarter_classification = self.price.get_tomorrow_classification(quarter_idx)
                    if quarter_classification == QuarterClassification.PEAK:
                        peak_count += 1
                    elif quarter_classification == QuarterClassification.EXPENSIVE:
                        expensive_count += 1
                    elif quarter_classification == QuarterClassification.CHEAP:
                        cheap_count += 1
                    elif quarter_classification == QuarterClassification.NORMAL:
                        normal_count += 1

            # Calculate non-NORMAL count (EXPENSIVE + CHEAP + PEAK)
            non_normal_count = expensive_count + cheap_count + peak_count
            actual_scan_periods = (scan_end + 1) - scan_start  # Quarters actually scanned today
            if (
                current_quarter + PRICE_VOLATILE_SCAN_QUARTERS_EACH_DIRECTION > 95
                and price_data.has_tomorrow
            ):
                actual_scan_periods += min(
                    (current_quarter + PRICE_VOLATILE_SCAN_QUARTERS_EACH_DIRECTION) - 95, 96
                )  # Add tomorrow quarters

            # Volatile if:
            # 1. >= 6 non-NORMAL periods (75% of 8 quarters = definitely jumpy)
            # 2. >= 3 non-NORMAL AND mix of different types (not just all CHEAP or all EXPENSIVE)
            has_mix = sum([expensive_count > 0, cheap_count > 0, peak_count > 0]) >= 2

            if non_normal_count >= PRICE_VOLATILE_MAX_THRESHOLD:
                # Definitely volatile - 6+ non-NORMAL in 2h window around current time
                is_volatile_period = True
                volatile_reason = f" | Volatile: {non_normal_count}/{actual_scan_periods} non-NORMAL in ±{actual_scan_periods * 15 // 2}min window - holding steady"
            elif non_normal_count >= PRICE_VOLATILE_MIN_THRESHOLD and has_mix:
                # Moderately volatile - 3+ non-NORMAL AND mixed types (jumpy)
                is_volatile_period = True
                types = []
                if peak_count > 0:
                    types.append(f"{peak_count}×PEAK")
                if expensive_count > 0:
                    types.append(f"{expensive_count}×EXP")
                if cheap_count > 0:
                    types.append(f"{cheap_count}×CHEAP")
                volatile_reason = f" | Volatile: {' + '.join(types)} mixed in ±{actual_scan_periods * 15 // 2}min - reducing confidence"

            # Apply normal forecast logic regardless of volatility
            # Volatility will reduce weight, not block smart decisions
            if classification == QuarterClassification.CHEAP:
                # Pre-heat before upcoming expensive periods (if sustained AND far enough away)
                # Only act if expensive period is at least 45min in future (same as duration filter)
                if (
                    max_price_ratio > PRICE_FORECAST_EXPENSIVE_THRESHOLD
                    and expensive_duration >= PRICE_FORECAST_MIN_DURATION
                    and max_idx >= PRICE_FORECAST_MIN_DURATION
                ):
                    increase_percent = int((max_price_ratio - 1) * 100)
                    forecast_adjustment = PRICE_FORECAST_PREHEAT_OFFSET
                    forecast_reason = f" | Forecast: {increase_percent}% more expensive in {max_idx//4}h - pre-heat now"
                elif max_price_ratio > PRICE_FORECAST_EXPENSIVE_THRESHOLD:
                    # Expensive period exists but doesn't meet criteria - explain why
                    if expensive_duration < PRICE_FORECAST_MIN_DURATION:
                        forecast_reason = f" | Forecast: Expensive period too brief ({expensive_duration * 15}min < {PRICE_FORECAST_MIN_DURATION * 15}min)"
                    elif max_idx < PRICE_FORECAST_MIN_DURATION:
                        forecast_reason = f" | Forecast: Expensive period too soon ({max_idx * 15}min < {PRICE_FORECAST_MIN_DURATION * 15}min lookahead)"

            elif classification in [
                QuarterClassification.EXPENSIVE,
                QuarterClassification.PEAK,
            ]:
                # Wait for upcoming cheap periods (if sustained AND far enough away)
                # Only reduce if cheap period is at least 45min in future (same as duration filter)
                if (
                    min_price_ratio < PRICE_FORECAST_CHEAP_THRESHOLD
                    and cheap_duration >= PRICE_FORECAST_MIN_DURATION
                    and min_idx >= PRICE_FORECAST_MIN_DURATION
                ):
                    savings_percent = int((1 - min_price_ratio) * 100)
                    forecast_adjustment = PRICE_FORECAST_REDUCTION_OFFSET
                    forecast_reason = (
                        f" | Forecast: {savings_percent}% cheaper in {min_idx//4}h - reduce heating"
                    )
                elif min_price_ratio < PRICE_FORECAST_CHEAP_THRESHOLD:
                    # Cheap period exists but doesn't meet criteria - explain why
                    if cheap_duration < PRICE_FORECAST_MIN_DURATION:
                        forecast_reason = f" | Forecast: Cheap period too brief ({cheap_duration * 15}min < {PRICE_FORECAST_MIN_DURATION * 15}min)"
                    elif min_idx < PRICE_FORECAST_MIN_DURATION:
                        forecast_reason = f" | Forecast: Cheap period too soon ({min_idx * 15}min < {PRICE_FORECAST_MIN_DURATION * 15}min lookahead)"

            else:  # NORMAL - check both directions, take most significant sustained change
                # Apply same lookahead requirement: price change must be ≥45min away to act on
                expensive_valid = (
                    max_price_ratio > PRICE_FORECAST_EXPENSIVE_THRESHOLD
                    and expensive_duration >= PRICE_FORECAST_MIN_DURATION
                    and max_idx >= PRICE_FORECAST_MIN_DURATION
                )
                cheap_valid = (
                    min_price_ratio < PRICE_FORECAST_CHEAP_THRESHOLD
                    and cheap_duration >= PRICE_FORECAST_MIN_DURATION
                    and min_idx >= PRICE_FORECAST_MIN_DURATION
                )

                if expensive_valid and cheap_valid:
                    # Both valid - choose larger magnitude
                    if (max_price_ratio - 1.0) > (1.0 - min_price_ratio):
                        increase_percent = int((max_price_ratio - 1) * 100)
                        forecast_adjustment = PRICE_FORECAST_PREHEAT_OFFSET
                        forecast_reason = f" | Forecast: {increase_percent}% more expensive in {max_idx//4}h - pre-heat now"
                    else:
                        savings_percent = int((1 - min_price_ratio) * 100)
                        forecast_adjustment = PRICE_FORECAST_REDUCTION_OFFSET
                        forecast_reason = f" | Forecast: {savings_percent}% cheaper in {min_idx//4}h - reduce heating"
                elif expensive_valid:
                    increase_percent = int((max_price_ratio - 1) * 100)
                    forecast_adjustment = PRICE_FORECAST_PREHEAT_OFFSET
                    forecast_reason = f" | Forecast: {increase_percent}% more expensive in {max_idx//4}h - pre-heat now"
                elif cheap_valid:
                    savings_percent = int((1 - min_price_ratio) * 100)
                    forecast_adjustment = PRICE_FORECAST_REDUCTION_OFFSET
                    forecast_reason = (
                        f" | Forecast: {savings_percent}% cheaper in {min_idx//4}h - reduce heating"
                    )
                else:
                    # Check if either direction had potential but didn't meet criteria
                    if max_price_ratio > PRICE_FORECAST_EXPENSIVE_THRESHOLD:
                        if expensive_duration < PRICE_FORECAST_MIN_DURATION:
                            forecast_reason = f" | Forecast: Expensive period too brief ({expensive_duration * 15}min)"
                        elif max_idx < PRICE_FORECAST_MIN_DURATION:
                            forecast_reason = (
                                f" | Forecast: Expensive period too soon ({max_idx * 15}min)"
                            )
                    elif min_price_ratio < PRICE_FORECAST_CHEAP_THRESHOLD:
                        if cheap_duration < PRICE_FORECAST_MIN_DURATION:
                            forecast_reason = (
                                f" | Forecast: Cheap period too brief ({cheap_duration * 15}min)"
                            )
                        elif min_idx < PRICE_FORECAST_MIN_DURATION:
                            forecast_reason = (
                                f" | Forecast: Cheap period too soon ({min_idx * 15}min)"
                            )

        # Get base offset for current classification
        base_offset = self.price.get_base_offset(
            current_quarter,
            classification,
            current_period.is_daytime,
        )

        # Skip heating boost for brief CHEAP periods (Nov 29, 2025)
        # Compressor needs ~45min to be efficient - don't boost for short dips
        brief_cheap_reason = ""
        if current_cheap_too_brief and base_offset > 0:
            brief_cheap_reason = f" | Current cheap period too brief ({remaining_cheap_quarters * 15}min < {PRICE_FORECAST_MIN_DURATION * 15}min) - skipping boost"
            base_offset = 0.0  # Treat as NORMAL instead of CHEAP

        # Adjust for tolerance setting (1-10 scale)
        tolerance_factor = self.tolerance / PRICE_TOLERANCE_DIVISOR  # 0.2-2.0
        adjusted_offset = base_offset * tolerance_factor

        # Apply forecast adjustment (additive to base classification)
        final_offset = adjusted_offset + forecast_adjustment

        # Check for strategic overshoot context when pre-heating
        strategic_context = ""
        if forecast_adjustment > 0 and nibe_state.indoor_temp > self.target_temp:
            overshoot = nibe_state.indoor_temp - self.target_temp
            # Calculate cost savings multiplier from price forecast
            if upcoming_periods and current_price > 0:
                max_upcoming = max(p.price for p in upcoming_periods)
                cost_multiplier = max_upcoming / current_price
                strategic_context = f" | Strategic storage: +{overshoot:.1f}°C overshoot acceptable for {cost_multiplier:.1f}x cost savings"

        # Apply very aggressive volatility weight reduction (Nov 30, 2025)
        # During volatile periods, drastically reduce price layer influence
        # Math: 0.8 × 0.1 = 0.08 (price layer reduced to 10% of normal strength)
        # Effect: Thermal/comfort/weather layers dominate, preventing chase behavior
        price_weight = LAYER_WEIGHT_PRICE
        if is_volatile_period:
            price_weight = LAYER_WEIGHT_PRICE * PRICE_VOLATILE_WEIGHT_REDUCTION  # 0.8 → 0.08

        # DEBUG: Log price analysis with thermal mass horizon calculation
        _LOGGER.debug(
            "Price Q%d (%02d:%02d): %.2f öre → %s | Horizon: %.1fh (%.1f base × %.1f thermal_mass) | Base: %.1f°C | Forecast adj: %.1f°C | Final: %.1f°C | Weight: %.2f%s%s%s",
            current_quarter,
            now.hour,
            now.minute,
            current_price,
            classification.name,
            forecast_hours,
            PRICE_FORECAST_BASE_HORIZON,
            thermal_mass,
            adjusted_offset,
            forecast_adjustment,
            final_offset,
            price_weight,
            volatile_reason if is_volatile_period else "",
            strategic_context,
            brief_cheap_reason,
        )

        return LayerDecision(
            offset=final_offset,
            weight=price_weight,
            reason=f"GE-Spot Q{current_quarter}: {classification.name} ({'day' if current_period.is_daytime else 'night'}) | Horizon: {forecast_hours:.1f}h ({PRICE_FORECAST_BASE_HORIZON:.1f} × {thermal_mass:.1f}){forecast_reason}{volatile_reason if is_volatile_period else ''}{strategic_context}{brief_cheap_reason}",
        )

    def _comfort_layer(self, nibe_state) -> LayerDecision:
        """Comfort layer: Reactive adjustment to maintain comfort.

        Provides gentle steering toward target even within tolerance zone.
        This ensures temperature doesn't drift unnecessarily during cheap periods.

        Args:
            nibe_state: Current NIBE state

        Returns:
            LayerDecision with comfort correction
        """
        temp_error = nibe_state.indoor_temp - self.target_temp

        # Temperature tolerance based on user setting
        tolerance = self.tolerance_range  # ±0.4-4.0°C

        # Fixed dead zone (very close to target, no action needed)
        dead_zone = COMFORT_DEAD_ZONE

        if abs(temp_error) < dead_zone:
            # Very close to target - we're right on target
            return LayerDecision(offset=0.0, weight=0.0, reason="Temp at target")
        elif abs(temp_error) < tolerance:
            # Within comfort zone but drifting from target
            # Gentle correction to maintain target during favorable conditions
            # Lower weight (0.2) so it doesn't override cost optimization,
            # but provides gentle steering when prices are similar
            correction = -temp_error * COMFORT_CORRECTION_MULT  # Gentle correction

            if temp_error > 0:
                reason = f"Slightly warm (+{temp_error:.1f}°C), gentle reduce"
            else:
                reason = f"Slightly cool ({temp_error:.1f}°C), gentle boost"

            return LayerDecision(
                offset=correction,
                weight=LAYER_WEIGHT_COMFORT_MIN,  # Low weight - advisory only
                reason=reason,
            )
        elif temp_error > tolerance:
            # Too warm - graduated response based on severity (Phase 2: Temperature Control Fixes)
            # This replaces the old fixed 0.5 weight with dynamic scaling
            overshoot = temp_error - tolerance

            # SAFETY CHECK: Don't apply cooling if thermal debt accumulating
            # If DM < -200, heat pump already struggling - cooling would worsen debt
            # This prevents sensor errors or outliers from creating thermal debt
            if nibe_state.degree_minutes < COMFORT_DM_COOLING_THRESHOLD:
                return LayerDecision(
                    offset=0.0,
                    weight=0.0,
                    reason=f"Overheat detected ({temp_error:.1f}°C) BUT DM {nibe_state.degree_minutes:.0f} - blocking cooling to prevent thermal debt",
                )

            # Graduated weight scaling based on overshoot severity:
            # - 0-1°C over tolerance: weight 0.7 (high priority, standard correction)
            # - 1-2°C over tolerance: weight 0.9 (very high priority, strong correction)
            # - 2°C+ over tolerance: weight 1.0 (CRITICAL - same as safety layer, emergency correction)
            if overshoot >= 2.0:
                weight = LAYER_WEIGHT_COMFORT_CRITICAL  # 1.0 - forces cooling, overrides all layers
                correction = -overshoot * COMFORT_CORRECTION_CRITICAL  # 1.5x multiplier
                reason = f"CRITICAL overheat: {temp_error:.1f}°C over target (emergency cooling)"
            elif overshoot >= 1.0:
                weight = LAYER_WEIGHT_COMFORT_SEVERE  # 0.9 - very high priority
                correction = -overshoot * COMFORT_CORRECTION_STRONG  # 1.2x multiplier
                reason = f"Severe overheat: {temp_error:.1f}°C over target"
            else:
                weight = LAYER_WEIGHT_COMFORT_HIGH  # 0.7 - high priority
                correction = -overshoot * COMFORT_CORRECTION_MILD  # 1.0x multiplier
                reason = f"Too warm: {temp_error:.1f}°C over target"

            return LayerDecision(offset=correction, weight=weight, reason=reason)
        else:
            # Too cold, increase heating strongly
            correction = -(temp_error + tolerance) * 0.5
            return LayerDecision(
                offset=correction,
                weight=LAYER_WEIGHT_COMFORT_MAX,
                reason=f"Too cold: {-temp_error:.1f}°C under",
            )

    def _aggregate_layers(self, layers: list[LayerDecision]) -> float:
        """Aggregate layer decisions into final offset.

        Uses weighted average with special handling for high-priority layers.
        Layer priority order (highest to lowest):
        1. Safety layer (absolute limits)
        2. Emergency layer (thermal debt) - ALWAYS overrides peak protection
        3. Effect layer (peak protection)
        4. Other layers

        Oct 19, 2025: Enhanced peak-aware emergency mode
        When emergency layer is critical AND effect/peak layers are strongly negative,
        apply minimal offset to prevent DM worsening without creating new peaks.

        Nov 29, 2025: Updated for weighted mixing (T3=0.95)
        Allows Emergency T3 to mix with Price/Weather in normal conditions,
        but protects it from being overridden by Critical Peak (1.0).

        Args:
            layers: List of layer decisions

        Returns:
            Final offset value
        """
        # 1. Safety Layer (Absolute Priority)
        # Always enforced if critical (weight >= 1.0)
        if len(layers) > 0 and layers[0].weight >= 1.0:
            return layers[0].offset

        # 2. Emergency vs Peak Conflict Resolution
        # If Emergency is strong (T2=0.85, T3=0.95) AND Peak is Critical (1.0),
        # we need a compromise. We don't want Peak to crush Emergency (unsafe),
        # nor Emergency to ignore Peak (expensive).
        emergency_layer = layers[1] if len(layers) > 1 else None
        effect_layer = layers[3] if len(layers) > 3 else None

        if (
            emergency_layer
            and emergency_layer.weight >= 0.85  # T2 or T3 active
            and effect_layer
            and effect_layer.weight >= 1.0  # Peak Critical active
        ):
            emergency_offset = emergency_layer.offset

            # Apply Peak-Aware Compromise Logic
            # Scale minimal offset based on emergency severity
            if emergency_offset >= DM_CRITICAL_T3_OFFSET:  # T3
                minimal_offset = DM_CRITICAL_T3_PEAK_AWARE_OFFSET
            elif emergency_offset >= DM_CRITICAL_T2_OFFSET:  # T2
                minimal_offset = DM_CRITICAL_T2_PEAK_AWARE_OFFSET
            else:  # T1
                minimal_offset = DM_CRITICAL_T1_PEAK_AWARE_OFFSET

            _LOGGER.info(
                "Peak-aware emergency mode: reducing offset from %.2f to %.2f (Critical Peak protection active)",
                emergency_offset,
                minimal_offset,
            )
            return minimal_offset

        # 3. Critical Overrides (Standard)
        # Any remaining layer with weight >= 1.0 overrides weighted average
        # (e.g., Critical Peak when Emergency is not strong)
        critical_layers = [layer for layer in layers if layer.weight >= 1.0]

        if critical_layers:
            # For critical layers, take the strongest vote
            max_offset = max(layer.offset for layer in critical_layers)
            min_offset = min(layer.offset for layer in critical_layers)

            # If conflicting critical votes, take the more conservative (lower magnitude? No, safer)
            # Actually, if we have multiple criticals (e.g. Peak vs Comfort Critical),
            # we should probably prioritize Safety/Peak.
            # But Safety is handled in step 1.
            # So this is likely Peak vs Comfort Critical.
            # Peak (-3.0) vs Comfort Critical (-3.0). Same.
            # Peak (-3.0) vs Comfort Critical (+3.0 - too cold).
            # If too cold (Comfort Critical) and Peak Critical (-3.0).
            # Comfort Critical is 1.0. Peak is 1.0.
            # We should probably respect Peak to avoid fees, unless Safety triggers.
            # Current logic: abs(max) > abs(min).
            # If max=+3, min=-3. Returns +3.
            # If max=+1, min=-3. Returns -3.
            # This logic favors the "stronger" intervention.
            if abs(max_offset) > abs(min_offset):
                return max_offset
            else:
                return min_offset

        # 4. Weighted Average
        # Mixes all layers (including T3=0.95, Price=0.8, Weather=0.85)
        total_weight = sum(layer.weight for layer in layers)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(layer.offset * layer.weight for layer in layers)
        return weighted_sum / total_weight

    def _generate_reasoning(
        self,
        layers: list[LayerDecision],
        final_offset: float,
    ) -> str:
        """Generate human-readable reasoning from layer decisions.

        Prioritizes critical layers (weight >= 1.0) in the output to make it clear
        which layer is driving the decision. Advisory layers shown separately.

        Args:
            layers: List of layer decisions
            final_offset: Final aggregated offset

        Returns:
            Reasoning string with critical layers highlighted
        """
        # Separate critical from advisory layers
        critical_layers = [layer for layer in layers if layer.weight >= 1.0]
        advisory_layers = [layer for layer in layers if 0 < layer.weight < 1.0]

        if not critical_layers and not advisory_layers:
            return "No optimization active"

        # Build reasoning string
        reasons = []

        if critical_layers:
            # Critical layers drive the decision - show them prominently
            critical_reasons = [layer.reason for layer in critical_layers]
            reasons.extend(critical_reasons)

            # Show advisory layers as supplementary info (if any)
            if advisory_layers and len(advisory_layers) <= 2:
                # Only show a few advisory layers to avoid clutter
                advisory_summary = ", ".join(f"{layer.reason}" for layer in advisory_layers[:2])
                reasons.append(f"[Advisory: {advisory_summary}]")
        else:
            # No critical layers - all are advisory, show normally
            reasons = [layer.reason for layer in advisory_layers]

        return " | ".join(reasons)

    def _validate_power_consumption(
        self,
        current_power_kw: float,
        outdoor_temp: float,
    ) -> dict[str, Any]:
        """Validate current power against model expectations.

        Args:
            current_power_kw: Current electrical consumption (kW)
            outdoor_temp: Outdoor temperature (°C)

        Returns:
            Dict with validation status and warnings
        """
        if not self.heat_pump_model:
            return {"valid": True, "warning": None}

        min_power, max_power = self.heat_pump_model.typical_electrical_range_kw

        # Allow 20% margin for startup/defrost
        max_with_margin = max_power * 1.2

        if current_power_kw > max_with_margin:
            return {
                "valid": False,
                "warning": f"Power {current_power_kw:.1f}kW exceeds {self.heat_pump_model.model_name} max {max_power:.1f}kW (auxiliary heating active?)",
                "severity": "warning",
            }

        # Check if unusually low (possible sensor issue)
        if current_power_kw < min_power * 0.5 and outdoor_temp < 0:
            return {
                "valid": True,
                "warning": f"Power {current_power_kw:.1f}kW below expected for {outdoor_temp:.1f}°C",
                "severity": "info",
            }

        return {"valid": True, "warning": None}

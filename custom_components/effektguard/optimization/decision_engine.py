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
from datetime import datetime, timedelta
from typing import Any, Optional, TypedDict

from homeassistant.util import dt as dt_util

from ..const import (
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
    LAYER_WEIGHT_COMFORT_MAX,
    LAYER_WEIGHT_COMFORT_MIN,
    LAYER_WEIGHT_COMFORT_HIGH,
    LAYER_WEIGHT_COMFORT_CRITICAL,
    COMFORT_CORRECTION_MILD,
    LAYER_WEIGHT_EMERGENCY,
    LAYER_WEIGHT_PRICE,
    LAYER_WEIGHT_PROACTIVE_MIN,
    LAYER_WEIGHT_PREDICTION,
    LAYER_WEIGHT_SAFETY,
    MIN_TEMP_LIMIT,
    OVERSHOOT_PROTECTION_COLD_SNAP_THRESHOLD,
    OVERSHOOT_PROTECTION_FORECAST_HORIZON,
    OVERSHOOT_PROTECTION_FULL,
    OVERSHOOT_PROTECTION_OFFSET_MAX,
    OVERSHOOT_PROTECTION_OFFSET_MIN,
    OVERSHOOT_PROTECTION_START,
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
    PRICE_TOLERANCE_MIN,
    PRICE_TOLERANCE_MAX,
    PRICE_TOLERANCE_FACTOR_MIN,
    PRICE_TOLERANCE_FACTOR_MAX,
    COMFORT_DEAD_ZONE,
    COMFORT_CORRECTION_MULT,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    MULTIPLIER_BOOST_30_PERCENT,
    MULTIPLIER_REDUCTION_20_PERCENT,
    THERMAL_CHANGE_MODERATE,
    THERMAL_CHANGE_MODERATE_COOLING,
    PRICE_FORECAST_BASE_HORIZON,
    PRICE_FORECAST_CHEAP_THRESHOLD,
    PRICE_FORECAST_EXPENSIVE_THRESHOLD,
    PRICE_FORECAST_MIN_DURATION,
    PRICE_FORECAST_PREHEAT_OFFSET,
    PRICE_FORECAST_REDUCTION_OFFSET,
    PRICE_OFFSET_PEAK,
    PRICE_PRE_PEAK_OFFSET,
    PRICE_VOLATILE_WEIGHT_REDUCTION,
)
from .climate_zones import ClimateZoneDetector
from .comfort_layer import ComfortLayer, ComfortLayerDecision
from .thermal_layer import (
    EmergencyLayer,
    EmergencyLayerDecision,
    ProactiveLayer,
    ProactiveLayerDecision,
)
from .weather_layer import (
    AdaptiveClimateSystem,
    WeatherCompensationCalculator,
    WeatherCompensationLayer,
    WeatherCompensationLayerDecision,
    WeatherLayerDecision,
    WeatherPredictionLayer,
)

_LOGGER = logging.getLogger(__name__)


class OutdoorTrendDict(TypedDict):
    """Outdoor temperature trend from BT1 sensor."""

    trend: str  # "warming", "cooling", "stable", "unknown"
    rate_per_hour: float
    confidence: float


class PowerValidationDict(TypedDict, total=False):
    """Power consumption validation result."""

    valid: bool
    warning: str | None
    severity: str


@dataclass
class LayerDecision:
    """Decision from a single optimization layer.

    Each layer proposes an offset and provides reasoning.
    """

    name: str  # Layer name for display (e.g., "Safety", "Spot Price", "Comfort")
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

        # Optimization mode configuration (comfort/balanced/savings)
        # This affects dead zones, layer weights, and price behavior
        self._update_mode_config()

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

        # Weather prediction layer for proactive pre-heating
        self.weather_prediction = WeatherPredictionLayer(
            thermal_mass=thermal_model.thermal_mass if thermal_model else 1.0
        )

        # Weather compensation layer for mathematical flow temp optimization
        self.weather_comp_layer = WeatherCompensationLayer(
            weather_comp=self.weather_comp,
            climate_system=self.climate_system,
            weather_learner=self.weather_learner,
            weather_comp_weight=self.weather_comp_weight,
        )

        # Emergency layer for thermal debt response
        self.emergency_layer = EmergencyLayer(
            climate_detector=self.climate_detector,
            price_analyzer=self.price,
            heating_type=config.get("heating_type", "radiator"),
            get_thermal_trend=self._get_thermal_trend,
            get_outdoor_trend=self._get_outdoor_trend,
        )

        # Proactive layer for thermal debt prevention
        self.proactive_layer = ProactiveLayer(
            climate_detector=self.climate_detector,
            get_thermal_trend=self._get_thermal_trend,
        )

        # Comfort layer for reactive temperature adjustments
        self.comfort_layer = ComfortLayer(
            get_thermal_trend=self._get_thermal_trend,
            thermal_model=self.thermal,
            mode_config=self.mode_config,
            tolerance_range=self.tolerance_range,
            target_temp=self.target_temp,
        )

        # Manual override state (Phase 5 service support)
        self._manual_override_offset: float | None = None
        self._manual_override_until: Optional[datetime] = None

    def _update_mode_config(self) -> None:
        """Update cached mode configuration from current optimization mode.

        Called on init and when mode changes. Caches the mode config to avoid
        repeated lookups during layer calculations.
        """
        mode = self.config.get("optimization_mode", OPTIMIZATION_MODE_BALANCED)
        self.mode_config = MODE_CONFIGS.get(mode, MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED])
        _LOGGER.debug(
            "Optimization mode: %s (dead_zone=%.1f, comfort_mult=%.1f, price_mult=%.1f)",
            mode,
            self.mode_config.dead_zone,
            self.mode_config.comfort_weight_multiplier,
            self.mode_config.price_tolerance_multiplier,
        )

    def set_manual_override(self, offset: float, duration_minutes: int = 0) -> None:
        """Set manual override for heating curve offset.

        Used by force_offset and boost_heating services.

        Args:
            offset: Manual offset value (-10 to +10°C)
            duration_minutes: Duration in minutes (0 = until next cycle)
        """
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

    def _get_outdoor_trend(self) -> OutdoorTrendDict:
        """Get outdoor temperature trend (BT1 real-time).

        Returns:
            Outdoor trend data or empty if not available
        """
        if hasattr(self, "predictor") and self.predictor:
            outdoor_trend = self.predictor.get_outdoor_trend()
            return {
                "trend": outdoor_trend.get("trend", "unknown"),
                "rate_per_hour": outdoor_trend.get("rate_per_hour", 0.0),
                "confidence": outdoor_trend.get("confidence", 0.0),
            }
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
            price_data: Spot price data (native 15-min intervals)
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
                        name="Manual Override",
                        offset=manual_override,
                        weight=1.0,
                        reason=f"User-set offset: {manual_override:.1f}°C",
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
            self._comfort_layer(nibe_state, weather_data, price_data),
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
                name="Safety",
                offset=offset,
                weight=LAYER_WEIGHT_SAFETY,
                reason=f"Too cold ({indoor_temp:.1f}°C < {MIN_TEMP_LIMIT}°C)",
            )
        else:
            # Within safe limits (no fixed upper limit - comfort layer handles dynamically)
            return LayerDecision(
                name="Safety",
                offset=0.0,
                weight=0.0,
                reason="OK",
            )

    def _emergency_layer(self, nibe_state, weather_data=None, price_data=None) -> LayerDecision:
        """Emergency layer: Smart context-aware thermal debt response.

        Delegates to EmergencyLayer.evaluate_layer() for the actual logic.

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data
            price_data: Price data

        Returns:
            LayerDecision with context-aware emergency response
        """
        # Delegate to EmergencyLayer for the actual logic
        emergency_decision = self.emergency_layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            price_data=price_data,
            target_temp=self.target_temp,
            tolerance_range=self.tolerance_range,
        )

        # Convert EmergencyLayerDecision to LayerDecision
        return LayerDecision(
            name=emergency_decision.name,
            offset=emergency_decision.offset,
            weight=emergency_decision.weight,
            reason=emergency_decision.reason,
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

        Delegates to ProactiveLayer.evaluate_layer() for the actual logic.

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data (optional, for forecast validation)

        Returns:
            LayerDecision with climate-aware proactive gentle heating

        References:
            thermal_layer.py: ProactiveLayer class for full implementation
        """
        result = self.proactive_layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=self.target_temp,
        )
        return LayerDecision(
            name=result.name,
            offset=result.offset,
            weight=result.weight,
            reason=result.reason,
        )

    def _effect_layer(self, nibe_state, current_peak: float, current_power: float) -> LayerDecision:
        """Effect tariff protection with PREDICTIVE peak avoidance (Phase 4).

        Delegates to EffectManager.evaluate_layer() for the actual logic.

        Args:
            nibe_state: Current NIBE state
            current_peak: Current monthly peak (kW) - from peak_this_month sensor
            current_power: Current whole-house power consumption (kW) - from peak_today sensor

        Returns:
            LayerDecision with predictive peak protection

        References:
            MASTER_IMPLEMENTATION_PLAN.md: Phase 4 - Predictive Peak Avoidance
        """
        # Get thermal trend for predictive analysis
        thermal_trend = self._get_thermal_trend()

        # Delegate to EffectManager for the actual logic
        effect_decision = self.effect.evaluate_layer(
            current_peak=current_peak,
            current_power=current_power,
            thermal_trend=thermal_trend,
            enable_peak_protection=self.config.get("enable_peak_protection", True),
        )

        # Convert EffectLayerDecision to LayerDecision
        return LayerDecision(
            name=effect_decision.name,
            offset=effect_decision.offset,
            weight=effect_decision.weight,
            reason=effect_decision.reason,
        )

    def _prediction_layer(self, nibe_state, weather_data) -> LayerDecision:
        """Prediction layer: Learned pre-heating using thermal state predictor.

        Delegates to ThermalStatePredictor.evaluate_layer() for the actual logic.

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data

        Returns:
            LayerDecision with learned pre-heating recommendation
        """
        # Skip if predictor not available
        if not self.predictor:
            return LayerDecision(
                name="Learned Pre-heat", offset=0.0, weight=0.0, reason="Predictor not initialized"
            )

        # Delegate to ThermalStatePredictor for the actual logic
        prediction_decision = self.predictor.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=self.target_temp,
            thermal_model=self.thermal,
        )

        # Convert PredictionLayerDecision to LayerDecision
        return LayerDecision(
            name=prediction_decision.name,
            offset=prediction_decision.offset,
            weight=prediction_decision.weight,
            reason=prediction_decision.reason,
        )

    def _weather_layer(self, nibe_state, weather_data) -> LayerDecision:
        """Weather prediction layer: Proactive pre-heating based on forecast.

        Delegates to WeatherPredictionLayer.evaluate_layer() for the actual logic.

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data

        Returns:
            LayerDecision with gentle pre-heating recommendation
        """
        # Get thermal trend for the layer
        thermal_trend = self._get_thermal_trend()

        # Delegate to WeatherPredictionLayer for the actual logic
        weather_decision = self.weather_prediction.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            thermal_trend=thermal_trend,
            enable_weather_prediction=self.config.get("enable_weather_prediction", True),
        )

        # Convert WeatherLayerDecision to LayerDecision
        return LayerDecision(
            name=weather_decision.name,
            offset=weather_decision.offset,
            weight=weather_decision.weight,
            reason=weather_decision.reason,
        )

    def _weather_compensation_layer(self, nibe_state, weather_data) -> LayerDecision:
        """Mathematical weather compensation layer with adaptive climate system.

        Delegates to WeatherCompensationLayer.evaluate_layer() for the actual logic.

        Calculates optimal flow temperature using:
        - Universal flow temperature formula (validated across manufacturers)
        - Heat transfer method (if radiator specs available)
        - UFH-specific adjustments (concrete/timber)
        - Adaptive climate zones (latitude-based, globally applicable)
        - Weather learning (unusual pattern detection)

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data

        Returns:
            LayerDecision with mathematically calculated offset
        """
        # Delegate to WeatherCompensationLayer for the actual logic
        comp_decision = self.weather_comp_layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=self.target_temp,
            enable_weather_compensation=self.enable_weather_compensation,
        )

        # Convert WeatherCompensationLayerDecision to LayerDecision
        return LayerDecision(
            name=comp_decision.name,
            offset=comp_decision.offset,
            weight=comp_decision.weight,
            reason=comp_decision.reason,
        )

    def _price_layer(self, nibe_state, price_data) -> LayerDecision:
        """Spot price layer: Forward-looking optimization from native 15-minute spot price data.

        Delegates to PriceAnalyzer.evaluate_layer() for the actual logic.

        Args:
            nibe_state: Current NIBE state (for strategic overshoot context)
            price_data: Spot price data with native 15-min intervals

        Returns:
            LayerDecision with price-based offset
        """
        # Get thermal mass from thermal model
        thermal_mass = self.thermal.thermal_mass if self.thermal else 1.0

        # Delegate to PriceAnalyzer for the actual logic
        price_decision = self.price.evaluate_layer(
            nibe_state=nibe_state,
            price_data=price_data,
            thermal_mass=thermal_mass,
            target_temp=self.target_temp,
            tolerance=self.tolerance,
            mode_config=self.mode_config,
            gespot_entity=self.config.get("gespot_entity", "unknown"),
            enable_price_optimization=self.config.get("enable_price_optimization", True),
        )

        # Convert PriceLayerDecision to LayerDecision
        return LayerDecision(
            name=price_decision.name,
            offset=price_decision.offset,
            weight=price_decision.weight,
            reason=price_decision.reason,
        )

    def _comfort_layer(self, nibe_state, weather_data=None, price_data=None) -> LayerDecision:
        """Comfort layer: Reactive adjustment to maintain comfort.

        Delegates to ComfortLayer.evaluate_layer() for the actual logic.

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast (for cold snap → higher heat loss)
            price_data: Spot price data (for expensive period timing)

        Returns:
            LayerDecision with comfort correction

        References:
            comfort_layer.py: ComfortLayer class for full implementation
        """
        # Update comfort layer with current state before evaluation
        self.comfort_layer.mode_config = self.mode_config
        self.comfort_layer.tolerance_range = self.tolerance_range
        self.comfort_layer.target_temp = self.target_temp

        result = self.comfort_layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            price_data=price_data,
        )
        return LayerDecision(
            name=result.name,
            offset=result.offset,
            weight=result.weight,
            reason=result.reason,
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
    ) -> PowerValidationDict:
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
                "severity": "info",
            }

        # Check if unusually low (possible sensor issue)
        if current_power_kw < min_power * 0.5 and outdoor_temp < 0:
            return {
                "valid": True,
                "warning": f"Power {current_power_kw:.1f}kW below expected for {outdoor_temp:.1f}°C",
                "severity": "info",
            }

        return {"valid": True, "warning": None}

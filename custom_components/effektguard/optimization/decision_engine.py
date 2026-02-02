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
from typing import TYPE_CHECKING, Optional, TypedDict

from homeassistant.util import dt as dt_util

from ..const import (
    DEFAULT_HEAT_LOSS_COEFFICIENT,
    DEFAULT_TARGET_TEMP,
    DEFAULT_THERMAL_MASS,
    DEFAULT_TOLERANCE,
    DEFAULT_WEATHER_COMPENSATION_WEIGHT,
    DM_CRITICAL_T1_PEAK_AWARE_OFFSET,
    DM_CRITICAL_T2_OFFSET,
    DM_CRITICAL_T2_PEAK_AWARE_OFFSET,
    DM_CRITICAL_T3_OFFSET,
    DM_CRITICAL_T3_PEAK_AWARE_OFFSET,
    THERMAL_MASS_CONCRETE_UFH_THRESHOLD,
    THERMAL_MASS_TIMBER_UFH_THRESHOLD,
    LAYER_WEIGHT_SAFETY,
    MIN_TEMP_LIMIT,
    SAFETY_EMERGENCY_OFFSET,
    TOLERANCE_RANGE_MULTIPLIER,
    TREND_BOOST_OFFSET_LIMIT,
    TREND_DAMPING_COOLING_BOOST,
    TREND_DAMPING_NEUTRAL,
    TREND_DAMPING_WARMING,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
)
from ..utils.volatile_helpers import get_volatile_info
from .climate_zones import ClimateZoneDetector
from .comfort_layer import ComfortLayer
from .thermal_layer import (
    EmergencyLayer,
    ProactiveLayer,
    is_cooling_rapidly,
    is_warming_rapidly,
)
from .weather_layer import (
    AdaptiveClimateSystem,
    WeatherCompensationCalculator,
    WeatherCompensationLayer,
    WeatherPredictionLayer,
)

if TYPE_CHECKING:
    from ..models.types import EffektGuardConfigDict

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
    anti_windup_active: bool = False  # True when anti-windup is driving the decision


def get_safe_default_decision() -> OptimizationDecision:
    """Get safe default decision when optimization fails.

    Returns zero offset to maintain current operation without changes.
    Used as fallback when optimization engine encounters errors.

    Moved from coordinator._get_safe_default_decision for shared reuse.

    Returns:
        OptimizationDecision with 0.0 offset and safe mode reasoning
    """
    return OptimizationDecision(
        offset=0.0,
        layers=[
            LayerDecision(
                name="Safe Mode",
                offset=0.0,
                weight=1.0,
                reason="Safe mode: optimization unavailable",
            )
        ],
        reasoning="Safe mode active - maintaining current operation",
    )


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
        config: "EffektGuardConfigDict",
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
        self.update_mode_config()

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
        # Infer heating_type from thermal_mass if not explicitly set
        # Uses THERMAL_MASS_*_THRESHOLD constants from const.py
        thermal_mass_value = DEFAULT_THERMAL_MASS  # Fallback to default (1.0)
        if thermal_model is not None:
            try:
                tm_attr = getattr(thermal_model, "thermal_mass", None)
                if isinstance(tm_attr, (int, float)):
                    thermal_mass_value = float(tm_attr)
            except (TypeError, AttributeError):
                pass  # Use default if thermal_model is mocked or has no thermal_mass

        if "heating_type" in config:
            heating_type = config["heating_type"]
        elif thermal_mass_value >= THERMAL_MASS_CONCRETE_UFH_THRESHOLD:
            heating_type = "concrete_ufh"
        elif thermal_mass_value >= THERMAL_MASS_TIMBER_UFH_THRESHOLD:
            heating_type = "timber"
        else:
            heating_type = "radiator"

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
            heating_type=heating_type,  # Use inferred heating_type (lines 335-342)
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

    def update_mode_config(self) -> None:
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

    def calculate_decision(
        self,
        nibe_state,
        price_data,
        weather_data,
        current_peak: float,
        current_power: float,
        temp_lux_active: bool = False,
        dhw_heating_end: datetime | None = None,
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

        # Update comfort layer with current state before evaluation
        self.comfort_layer.mode_config = self.mode_config
        self.comfort_layer.tolerance_range = self.tolerance_range
        self.comfort_layer.target_temp = self.target_temp

        # Calculate all layer decisions (ordered by priority)
        # Note: We call evaluate_layer() directly on layer objects instead of using wrappers

        # 1. Safety Layer (Inline)
        safety_decision = self._safety_layer(nibe_state)

        # Check volatility once for reuse across all layers
        is_volatile = get_volatile_info(self.price, price_data).is_volatile

        # 2. Emergency Layer
        emergency_decision = self.emergency_layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            price_data=price_data,
            target_temp=self.target_temp,
            tolerance_range=self.tolerance_range,
            is_volatile=is_volatile,
        )

        # 3. Proactive Layer
        proactive_decision = self.proactive_layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=self.target_temp,
            is_volatile=is_volatile,
        )

        # 4. Effect Layer
        effect_result = self.effect.evaluate_layer(
            current_peak=current_peak,
            current_power=current_power,
            thermal_trend=self._get_thermal_trend(),
            enable_peak_protection=self.config.get("enable_peak_protection", True),
        )
        effect_decision = LayerDecision(
            name=effect_result.name,
            offset=effect_result.offset,
            weight=effect_result.weight,
            reason=effect_result.reason,
        )

        # 5. Prediction Layer
        if self.predictor:
            pred_result = self.predictor.evaluate_layer(
                nibe_state=nibe_state,
                weather_data=weather_data,
                target_temp=self.target_temp,
                thermal_model=self.thermal,
            )
            prediction_decision = LayerDecision(
                name=pred_result.name,
                offset=pred_result.offset,
                weight=pred_result.weight,
                reason=pred_result.reason,
            )
        else:
            prediction_decision = LayerDecision(
                name="Learned Pre-heat", offset=0.0, weight=0.0, reason="Predictor not initialized"
            )

        # 6. Weather Compensation Layer
        comp_result = self.weather_comp_layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=self.target_temp,
            enable_weather_compensation=self.enable_weather_compensation,
            temp_lux_active=temp_lux_active,
            dhw_heating_end=dhw_heating_end,
        )
        comp_decision = LayerDecision(
            name=comp_result.name,
            offset=comp_result.offset,
            weight=comp_result.weight,
            reason=comp_result.reason,
        )

        # 7. Weather Prediction Layer
        weather_result = self.weather_prediction.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            thermal_trend=self._get_thermal_trend(),
            enable_weather_prediction=self.config.get("enable_weather_prediction", True),
        )
        weather_decision = LayerDecision(
            name=weather_result.name,
            offset=weather_result.offset,
            weight=weather_result.weight,
            reason=weather_result.reason,
        )

        # 8. Price Layer
        price_result = self.price.evaluate_layer(
            nibe_state=nibe_state,
            price_data=price_data,
            thermal_mass=self.thermal.thermal_mass if self.thermal else 1.0,
            target_temp=self.target_temp,
            tolerance=self.tolerance,
            mode_config=self.mode_config,
            gespot_entity=self.config.get("gespot_entity", "unknown"),
            enable_price_optimization=self.config.get("enable_price_optimization", True),
        )
        price_decision = LayerDecision(
            name=price_result.name,
            offset=price_result.offset,
            weight=price_result.weight,
            reason=price_result.reason,
        )

        # 9. Comfort Layer
        comfort_result = self.comfort_layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            price_data=price_data,
        )
        comfort_decision = LayerDecision(
            name=comfort_result.name,
            offset=comfort_result.offset,
            weight=comfort_result.weight,
            reason=comfort_result.reason,
        )

        layers = [
            safety_decision,
            emergency_decision,
            proactive_decision,
            effect_decision,
            prediction_decision,
            comp_decision,
            weather_decision,
            price_decision,
            comfort_decision,
        ]

        # Aggregate layers with priority weighting
        raw_offset = self._aggregate_layers(layers)

        # NEW: Trend-aware damping to prevent overshoot/undershoot
        thermal_trend = self._get_thermal_trend()
        trend_rate = thermal_trend.get("rate_per_hour", 0.0)

        if is_warming_rapidly(thermal_trend):
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

        elif is_cooling_rapidly(thermal_trend):
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
        reasoning = self._generate_reasoning(layers) + reason_suffix

        _LOGGER.info("Decision: offset %.2f°C - %s", final_offset, reasoning)

        # Propagate anti-windup flag from emergency layer to final decision (Feb 2026)
        # When anti-windup is active, the coordinator must bypass the volatile blocker
        # to allow the offset reduction to take effect immediately.
        # Physics: Anti-windup detects that raising S1 is making DM worse (BT25-S1 gap grows).
        # The volatile blocker must not block this safety-critical reduction.
        anti_windup = getattr(emergency_decision, "anti_windup_active", False)

        return OptimizationDecision(
            offset=final_offset,
            layers=layers,
            reasoning=reasoning,
            anti_windup_active=anti_windup,
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
    ) -> str:
        """Generate human-readable reasoning from layer decisions.

        Prioritizes critical layers (weight >= 1.0) in the output to make it clear
        which layer is driving the decision. Advisory layers shown separately.

        Args:
            layers: List of layer decisions

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
            critical_reasons = [f"[{layer.name}] {layer.reason}" for layer in critical_layers]
            reasons.extend(critical_reasons)

            # Show advisory layers as supplementary info (if any)
            if advisory_layers and len(advisory_layers) <= 2:
                # Only show a few advisory layers to avoid clutter
                advisory_summary = ", ".join(
                    f"[{layer.name}] {layer.reason}" for layer in advisory_layers[:2]
                )
                reasons.append(f"[Advisory: {advisory_summary}]")
        else:
            # No critical layers - all are advisory, show normally
            reasons = [f"[{layer.name}] {layer.reason}" for layer in advisory_layers]

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

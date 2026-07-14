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
from typing import TYPE_CHECKING, Final, Optional, TypedDict

from homeassistant.util import dt as dt_util

from ..const import (
    WEATHER_FORECAST_HORIZON,
    COMPRESSOR_RISK_HIGH,
    DEFAULT_HEAT_LOSS_COEFFICIENT,
    DEFAULT_TARGET_TEMP,
    DEFAULT_THERMAL_MASS,
    DEFAULT_TOLERANCE,
    DEFAULT_WEATHER_COMPENSATION_WEIGHT,
    DM_CRITICAL_PEAK_AWARE_OFFSETS,
    DM_RECOVERY_TIERS,
    DM_THRESHOLD_AUX_LIMIT,
    DM_TIER_EMERGENCY,
    THERMAL_MASS_CONCRETE_UFH_THRESHOLD,
    THERMAL_MASS_TIMBER_UFH_THRESHOLD,
    LAYER_WEIGHT_SAFETY,
    MAX_OFFSET,
    MIN_OFFSET,
    MIN_TARGET_TEMP,
    MIN_TEMP_LIMIT,
    POWER_VALIDATION_MARGIN,
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
    EmergencyLayerDecision,
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


# Display name of the safety layer. _aggregate_layers looks the layer up by name rather
# than by list position, so reordering the layers cannot silently re-target safety logic.
SAFETY_LAYER_NAME: Final = "Safety"

# Display name of the comfort layer. It is the ONLY layer that can see under-heating caused by a
# negative curve offset: degree minutes are DM = integral(BT25 - S1), so lowering the curve lowers
# S1 and DM improves as the house gets colder. _aggregate_layers looks it up by name to floor a
# cost layer that is coasting the house out of its comfort band.
COMFORT_LAYER_NAME: Final = "Comfort"


@dataclass
class LayerDecision:
    """Decision from a single optimization layer.

    Each layer proposes an offset and provides reasoning.
    """

    name: str  # Layer name for display (e.g., "Safety", "Spot Price", "Comfort")
    offset: float  # Proposed heating curve offset (°C)
    weight: float  # Layer weight/priority (0.0-1.0)
    reason: str  # Human-readable explanation
    # True for layers that optimize for COST (spot price, effect tariff) rather than for
    # comfort, safety, or physics. Cost layers are barred from reducing heat while the
    # thermal-debt layer is recovering - see DecisionEngine._aggregate_layers.
    is_cost_layer: bool = False


@dataclass
class OptimizationDecision:
    """Final optimization decision from decision engine.

    Aggregates all layer votes into single decision with explanation.
    """

    offset: float  # Final heating curve offset (°C)
    layers: list[LayerDecision] = field(default_factory=list)
    reasoning: str = ""
    anti_windup_active: bool = False  # True when anti-windup is driving the decision
    is_manual_override: bool = False  # True for user-commanded offsets (force_offset/boost)
    # True when an ABSOLUTE safety path produced this offset: indoor below MIN_TEMP_LIMIT,
    # or degree minutes past DM_THRESHOLD_AUX_LIMIT. The coordinator's offset-volatility
    # blocker must never defer such a decision - it exists to damp price-driven
    # flip-flopping, and deferring an aux-limit recovery for 45 minutes lets DM plunge
    # further while the immersion heater runs.
    is_emergency: bool = False


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
            thermal_mass=thermal_model.thermal_mass if thermal_model else 1.0,
            forecast_horizon=self._forecast_horizon_for(thermal_model),
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
            heating_type=heating_type,  # Must match EmergencyLayer: one ladder, not two
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
        self._manual_override_one_shot = False

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
            self._manual_override_one_shot = False
            self._manual_override_until = dt_util.now() + timedelta(minutes=duration_minutes)
            _LOGGER.info(
                "Manual override set: %s°C until %s",
                offset,
                self._manual_override_until.strftime("%Y-%m-%d %H:%M"),
            )
        else:
            self._manual_override_one_shot = True
            self._manual_override_until = None
            _LOGGER.info("Manual override set: %s°C until next cycle", offset)

    def clear_manual_override(self) -> None:
        """Clear manual override, return to automatic optimization."""
        self._manual_override_offset = None
        self._manual_override_until = None
        self._manual_override_one_shot = False
        _LOGGER.info("Manual override cleared")

    def consume_manual_override(self) -> None:
        """Consume an override whose public duration was zero.

        A zero-duration override means one applied control cycle, not forever. Reads and startup
        observation do not consume it because neither reaches the pump.
        """
        if self._manual_override_one_shot:
            self.clear_manual_override()

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

    @staticmethod
    def _absolute_safety_floor(nibe_state) -> float | None:
        """Lowest offset the system may apply regardless of user intent or cost.

        These are the two conditions the project treats as non-negotiable:
          - indoor below MIN_TEMP_LIMIT: the house is getting dangerously cold.
          - degree minutes at or past DM_THRESHOLD_AUX_LIMIT: NIBE engages the auxiliary
            immersion heater here. Declining to recover does not avoid that - it
            guarantees it, while the debt keeps deepening.

        Applied as a FLOOR, not a replacement: a user asking for MORE heat than safety
        requires still gets what they asked for. Only a command that would leave the
        system below the safety floor is raised to it.

        The indoor check is skipped when the reading is not a measurement (no room
        sensor): DEFAULT_INDOOR_TEMP sits above MIN_TEMP_LIMIT, so trusting it would
        mean the floor could never engage on such a system. DM still protects it.

        Returns:
            The minimum permitted offset (°C), or None when neither condition applies.
        """
        if getattr(nibe_state, "indoor_temp_valid", True) and (
            nibe_state.indoor_temp < MIN_TEMP_LIMIT
        ):
            return SAFETY_EMERGENCY_OFFSET
        if nibe_state.degree_minutes <= DM_THRESHOLD_AUX_LIMIT:
            return SAFETY_EMERGENCY_OFFSET
        return None

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

    @property
    def target_temp(self) -> float:
        """The indoor temperature being aimed for."""
        return self._target_temp

    @target_temp.setter
    def target_temp(self, target: float) -> None:
        """Refuse a target the safety layer would answer with an emergency.

        Below MIN_TEMP_LIMIT the safety layer commands MAX_OFFSET. The comfort layer drives TOWARD
        the target - so a target below the floor means comfort pulls the house down into the
        emergency zone and safety hauls it back out, MIN_OFFSET to MAX_OFFSET, on a real compressor,
        for as long as the setpoint stands. And the safety boost carries is_emergency=True, so it
        bypasses the volatility blocker that exists to stop precisely that thrashing. Measured, with
        a target of 15 °C: -10.00 at 19.0 °C, +10.00 at 17.9 °C. (Audit F-085.)

        Even AT the floor it is wrong: the house would sit at its target with the emergency trigger
        0.0 °C below it, and ordinary control noise would fire a full boost. So the lowest target
        this system can hold is MIN_TARGET_TEMP, one default tolerance clear of the floor.

        Enforced in the setter, not in the callers. The climate entity no longer OFFERS a lower
        target - and that does nothing for the owner who set one before this landed, because Home
        Assistant keeps the stored value across the upgrade. Stored options, a hot reload, a
        migration, a hand-edited entry: they all assign this attribute, and they are all refused.
        """
        if target < MIN_TARGET_TEMP:
            _LOGGER.warning(
                "Target %.1f°C cannot be held - the safety layer treats anything below %.1f°C as an "
                "emergency, so this target could only be met by fighting it. Holding %.1f°C.",
                target,
                MIN_TEMP_LIMIT,
                MIN_TARGET_TEMP,
            )
            target = MIN_TARGET_TEMP

        self._target_temp = target

    def calculate_decision(
        self,
        nibe_state,
        price_data,
        weather_data,
        current_peak: float,
        current_power: float,
        temp_lux_active: bool = False,
        dhw_heating_end: datetime | None = None,
        compressor_risk: str | None = None,
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

        # Check for manual override first (Phase 5 service support).
        #
        # A user command is authoritative, but it is NOT permitted to hold the system
        # below the absolute safety floor. force_offset(-10) held for hours while the
        # house drops below MIN_TEMP_LIMIT, or while DM sits past the aux limit, is not a
        # preference - it is a fault. The floor only ever raises the offset, so a user
        # asking for MORE heat (e.g. boost_heating) is passed through untouched.
        manual_override = self._check_manual_override()
        if manual_override is not None:
            safety_floor = self._absolute_safety_floor(nibe_state)

            if safety_floor is not None and manual_override < safety_floor:
                _LOGGER.warning(
                    "Manual override %.1f°C raised to %.1f°C: absolute safety floor active "
                    "(indoor %.1f°C, DM %.0f)",
                    manual_override,
                    safety_floor,
                    nibe_state.indoor_temp,
                    nibe_state.degree_minutes,
                )
                return OptimizationDecision(
                    offset=self._clamp_offset(safety_floor),
                    layers=[
                        LayerDecision(
                            name=SAFETY_LAYER_NAME,
                            offset=safety_floor,
                            weight=LAYER_WEIGHT_SAFETY,
                            reason=(
                                f"Safety floor overrides manual {manual_override:.1f}°C "
                                f"(indoor {nibe_state.indoor_temp:.1f}°C, "
                                f"DM {nibe_state.degree_minutes:.0f})"
                            ),
                        )
                    ],
                    reasoning=(
                        f"Manual override {manual_override:.1f}°C raised to "
                        f"{safety_floor:.1f}°C by absolute safety floor"
                    ),
                    is_manual_override=True,
                    is_emergency=True,
                )

            _LOGGER.info("Using manual override: %.2f°C", manual_override)
            return OptimizationDecision(
                offset=self._clamp_offset(manual_override),
                layers=[
                    LayerDecision(
                        name="Manual Override",
                        offset=manual_override,
                        weight=1.0,
                        reason=f"User-set offset: {manual_override:.1f}°C",
                    )
                ],
                reasoning=f"Manual override active: {manual_override:.1f}°C",
                is_manual_override=True,
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
            is_cost_layer=True,  # Effect tariff optimizes cost, not comfort or safety
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
            is_cost_layer=True,  # Spot price optimizes cost, not comfort or safety
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

        # HOW FAR OUT OF ITS COMFORT BAND IS THE HOUSE? A cost layer is allowed to coast it around
        # inside that band - that is the thermal battery - but not out of it, and degree minutes
        # cannot tell the difference (see _aggregate_layers step 4).
        starvation = self._starvation_fraction(nibe_state)

        # Aggregate layers with priority weighting
        raw_offset = self._aggregate_layers(layers, starvation=starvation)

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

        # An ABSOLUTE safety path outranks the wear guard: a house below MIN_TEMP_LIMIT, or degree
        # minutes past the pump's aux-start, gets everything the machine has. Wear is a cost;
        # a cold house is a failure.
        is_emergency = (
            safety_decision.weight >= LAYER_WEIGHT_SAFETY
            or getattr(emergency_decision, "tier", "") == DM_TIER_EMERGENCY
        )

        final_offset, wear_note = self._limit_for_compressor_wear(
            final_offset, nibe_state, compressor_risk, is_emergency
        )

        # Generate human-readable reasoning
        reasoning = self._generate_reasoning(layers) + reason_suffix + wear_note

        _LOGGER.info("Decision: offset %.2f°C - %s", final_offset, reasoning)

        # Propagate anti-windup flag from emergency layer to final decision (Feb 2026)
        # When anti-windup is active, the coordinator must bypass the volatile blocker
        # to allow the offset reduction to take effect immediately.
        # Physics: Anti-windup detects that raising S1 is making DM worse (BT25-S1 gap grows).
        # The volatile blocker must not block this safety-critical reduction.
        anti_windup = getattr(emergency_decision, "anti_windup_active", False)

        # `is_emergency` was computed above, before the wear guard, so that the guard could stand
        # aside for it. It also tells the coordinator's offset-volatility blocker not to defer this
        # decision: that blocker damps price-driven flip-flopping, and deferring an aux-limit
        # recovery for 45 minutes lets DM keep falling while the immersion heater runs.

        return OptimizationDecision(
            offset=final_offset,
            layers=layers,
            reasoning=reasoning,
            anti_windup_active=anti_windup,
            is_emergency=is_emergency,
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

        # Abstain when the indoor reading is a placeholder rather than a measurement.
        # DEFAULT_INDOOR_TEMP (21.0) is above MIN_TEMP_LIMIT (18.0), so a system with no
        # room sensor would otherwise report "OK" forever and this layer could never fire.
        # Such systems are protected by the degree-minute path instead.
        if not getattr(nibe_state, "indoor_temp_valid", True):
            return LayerDecision(
                name=SAFETY_LAYER_NAME,
                offset=0.0,
                weight=0.0,
                reason="No indoor sensor - abstaining (degree minutes protect this system)",
            )

        if indoor_temp < MIN_TEMP_LIMIT:
            # Too cold - emergency heating
            offset = SAFETY_EMERGENCY_OFFSET
            return LayerDecision(
                name=SAFETY_LAYER_NAME,
                offset=offset,
                weight=LAYER_WEIGHT_SAFETY,
                reason=f"Too cold ({indoor_temp:.1f}°C < {MIN_TEMP_LIMIT}°C)",
            )
        else:
            # Within safe limits (no fixed upper limit - comfort layer handles dynamically)
            return LayerDecision(
                name=SAFETY_LAYER_NAME,
                offset=0.0,
                weight=0.0,
                reason="OK",
            )

    @staticmethod
    def _limit_for_compressor_wear(
        offset: float,
        nibe_state,
        compressor_risk: str | None,
        is_emergency: bool,
    ) -> tuple[float, str]:
        """Decline to ask a saturated compressor for more heat. Never ask it for less.

        The offset works by raising the pump's calculated supply setpoint, S1. That only produces
        heat while the compressor has frequency left to give. Above 100 Hz for a quarter of an hour
        it has none: the setpoint goes up, the compressor cannot follow, and all that is bought is

          * wear, from holding the machine at full frequency longer than it needs to be, and
          * a DEEPER degree-minute deficit, because DM = integral(BT25 - S1) and S1 just rose while
            BT25 could not (audit F-124).

        The auxiliary heater exists for precisely this moment - NIBE sets "start addition" where it
        does (menu 4.9.3) so the compressor need not grind at maximum for hours. Refusing its help
        by demanding more from a saturated compressor trades cheap kWh for expensive compressor.

        This costs no comfort, and that is forced rather than argued: the extra offset was not
        producing heat, so declining to ask for it cannot take any away. It HOLDS the offset at what
        the pump is already being asked for; it never reduces it, and the absolute safety paths
        (indoor below MIN_TEMP_LIMIT, the aux-limit emergency) return before this is ever reached.
        """
        if is_emergency or compressor_risk != COMPRESSOR_RISK_HIGH:
            return offset, ""

        held = min(offset, nibe_state.current_offset)
        if held >= offset:
            return offset, ""

        _LOGGER.info(
            "Compressor saturated (%s): holding offset at %.2f°C instead of %.2f°C. A higher "
            "setpoint buys no heat from a compressor already at maximum - only wear and a deeper "
            "DM deficit. The auxiliary heater is what adds heat here.",
            compressor_risk,
            held,
            offset,
        )
        return held, f" | Compressor at maximum: holding {held:+.1f}°C (asked {offset:+.1f}°C)"

    @staticmethod
    def _forecast_horizon_for(thermal_model) -> float:
        """How far ahead the pre-heat layer must scan for this house.

        WEATHER_FORECAST_HORIZON is the FLOOR that every house gets. A thermal model may only
        EXTEND it, never shrink it: seeing further ahead costs a little early pre-heat, while
        seeing less far can cost the cold snap entirely.

        The layer scanned that fixed floor whatever the house was built of, and a concrete slab
        cannot see a slow slide inside it. A 15 C fall spread over two days shows only 3.8 C in any
        twelve hours - under WEATHER_FORECAST_DROP_THRESHOLD - so the pre-heat never fired, while
        the sudden plunge that DID trigger it is the case the pump's own curve already handles.
        UFH_CONCRETE_PREDICTION_HORIZON has said 24 hours all along; nothing could reach it.

        A model that cannot state a horizon gets the floor.
        """
        if thermal_model is None:
            return WEATHER_FORECAST_HORIZON
        try:
            horizon = float(thermal_model.get_prediction_horizon())
        except (AttributeError, TypeError, ValueError):
            return WEATHER_FORECAST_HORIZON
        return max(WEATHER_FORECAST_HORIZON, horizon)

    @staticmethod
    def _clamp_offset(offset: float) -> float:
        """Clamp an offset to the pump's valid range.

        This is the engine's single, unconditional bound. The adapter clamps again at
        write time as defence in depth, but it only does so inside its fractional
        accumulator branch - so before this existed, the unclamped float still reached
        the coordinator, the sensors, and the learning recorder.
        """
        return max(MIN_OFFSET, min(offset, MAX_OFFSET))

    def _aggregate_layers(self, layers: list[LayerDecision], starvation: float = 0.0) -> float:
        """Aggregate layer decisions into the final offset.

        SAFETY CONTRACT - the invariant this method exists to enforce:

            A cost layer (spot price, effect tariff) must NEVER reduce heating while
            the thermal-debt layer is actively recovering.

        Priority order:
            1. Safety layer            - indoor below MIN_TEMP_LIMIT. Absolute.
            2. EMERGENCY tier          - DM past DM_THRESHOLD_AUX_LIMIT. Absolute.
            3. Recovery tiers T1/T2/T3 - cost layers may MODERATE the response down to
                                         the tier's peak-aware offset, never reverse it.
            4. Remaining critical      - weight >= LAYER_WEIGHT_SAFETY, safety-biased tie-break.
            5. Weighted average        - everything else.

        Tiers are read from `EmergencyLayerDecision.tier`, never inferred from weights or
        offset magnitudes: damping mutates the offset, and a weight is a tuning knob, so
        inferring from either lets a retuned or damped tier fall through into the cost-layer
        override path.

        Args:
            layers: Layer decisions, as built by calculate_decision

        Returns:
            Final offset (°C), always within [MIN_OFFSET, MAX_OFFSET]
        """
        safety_layer = next((layer for layer in layers if layer.name == SAFETY_LAYER_NAME), None)
        emergency_layer = next(
            (layer for layer in layers if isinstance(layer, EmergencyLayerDecision)), None
        )

        # 1. Safety layer: indoor temperature below the absolute floor.
        if safety_layer is not None and safety_layer.weight >= LAYER_WEIGHT_SAFETY:
            return self._clamp_offset(safety_layer.offset)

        # 2. EMERGENCY tier: DM past the auxiliary-heat limit.
        # Nothing may throttle this. Past DM_THRESHOLD_AUX_LIMIT the immersion heater
        # engages; suppressing recovery to protect the effect tariff does not avoid the
        # peak, it guarantees a bigger one from the aux heater while the debt deepens.
        if emergency_layer is not None and emergency_layer.tier == DM_TIER_EMERGENCY:
            _LOGGER.warning(
                "Aux-limit emergency: DM %.0f - applying %.1f°C, overriding all cost layers",
                emergency_layer.degree_minutes,
                emergency_layer.offset,
            )
            return self._clamp_offset(emergency_layer.offset)

        # 3. Recovery tiers (T1/T2/T3): thermal debt beyond the climate-aware warning
        # threshold. The emergency layer only reaches a recovery tier when the house is
        # NOT above tolerance (its "too warm" case returns tier OK first), so removing
        # heat here always deepens the debt.
        if emergency_layer is not None and emergency_layer.tier in DM_RECOVERY_TIERS:
            peak_aware_floor = DM_CRITICAL_PEAK_AWARE_OFFSETS[emergency_layer.tier]

            if self._has_critical_cost_layer(layers):
                # Peak-aware compromise: enough to stop DM worsening, small enough not to
                # grow the monthly peak. Selected by TIER, so a damped T3 still gets T3's
                # compromise rather than T1's.
                _LOGGER.info(
                    "Peak-aware %s recovery: %.2f°C (critical cost layer active, DM %.0f)",
                    emergency_layer.tier,
                    peak_aware_floor,
                    emergency_layer.degree_minutes,
                )
                return self._clamp_offset(peak_aware_floor)

            # No critical cost layer: let the tier mix with the other layers, but never
            # below the tier's minimum recovery offset.
            weighted = self._weighted_average(layers)
            return self._clamp_offset(max(weighted, peak_aware_floor))

        # 4. Remaining critical layers (no thermal-debt recovery in progress).
        critical_layers = [layer for layer in layers if layer.weight >= LAYER_WEIGHT_SAFETY]
        if critical_layers:
            max_offset = max(layer.offset for layer in critical_layers)
            min_offset = min(layer.offset for layer in critical_layers)
            # Safety-biased tie-break: on equal magnitude prefer the HEATING vote.
            # (`>` here returned the negative vote on an exact tie, and
            # SAFETY_EMERGENCY_OFFSET/+10 vs PRICE_OFFSET_PEAK/-10 tie by construction.)
            chosen = max_offset if abs(max_offset) >= abs(min_offset) else min_offset

            # A COST LAYER MAY COAST THE HOUSE WITHIN ITS COMFORT BAND. IT MAY NOT COAST IT OUT.
            # This step takes the critical layer's vote ALONE, so with a price layer at PEAK the
            # comfort layer never enters the sum and cost kept cutting heat into an already-cold
            # house until the hard floor fired three degrees later. Nothing else catches it: DM is
            # blind by construction (DM = integral(BT25 - S1), so lowering the curve lowers S1 and
            # DM *improves* as the house cools).
            #
            # So floor a cost reduction at the comfort layer's own demand, which is graduated by how
            # far out the house is. The floor is RAMPED in via `starvation`, not switched at a
            # threshold - a boolean there is bang-bang and chatters a Modbus write on every dither
            # of the indoor sensor; see _starvation_fraction.
            if chosen < 0 and starvation > 0.0 and self._all_critical_are_cost(critical_layers):
                comfort = next(
                    (layer for layer in layers if layer.name == COMFORT_LAYER_NAME), None
                )
                if comfort is not None and comfort.offset > chosen:
                    # `min` states the invariant AND keeps it exact: the floor stops the cut, it
                    # never becomes a heat source of its own. Without it, the blend at starvation
                    # 1.0 lands on 0.9000000000000004 rather than comfort's 0.9.
                    floored = min(comfort.offset, chosen + (comfort.offset - chosen) * starvation)
                    _LOGGER.debug(
                        "Cost layer asked for %.2f°C with the house %.0f%% of the way out of its "
                        "comfort band; floored at %.2f°C (comfort wants %.2f°C)",
                        chosen,
                        starvation * 100.0,
                        floored,
                        comfort.offset,
                    )
                    chosen = floored

            return self._clamp_offset(chosen)

        # 5. Weighted average of everything else.
        return self._clamp_offset(self._weighted_average(layers))

    def _starvation_fraction(self, nibe_state) -> float:
        """How far the house has been coasted out of its comfort band, from 0.0 to 1.0.

        THERE ARE TWO BANDS HERE, AND THE DIFFERENCE BETWEEN THEM IS THE RAMP.

            inner = target - tolerance_range     the cost layers' playground. Above this the
                                                 thermal battery runs free, which is the entire
                                                 point of the integration.
            outer = target - tolerance           the band the OWNER actually asked for. At this
                                                 point a cost layer has spent everything it was
                                                 lent and the comfort layer's demand is the floor.

        Between them the floor is blended, so the control law is continuous. It used to be a
        boolean at `inner`, and a boolean on a temperature threshold is a bang-bang controller:
        indoor 20.80 C gave -10.00 and indoor 20.79 C gave +0.01. A real indoor sensor dithers by
        more than a hundredth of a degree, so the house sat on that boundary flipping the curve
        between its extremes, and every flip is a write to the pump.

        Abstains (0.0) when there is no valid indoor reading: without one this cannot be measured,
        and degree minutes are structurally blind to it - DM = integral(BT25 - S1), so lowering the
        curve lowers S1 and DM *improves* as the house gets colder.
        """
        if not getattr(nibe_state, "indoor_temp_valid", True):
            return 0.0

        inner = self.target_temp - self.tolerance_range
        outer = self.target_temp - self.tolerance

        if nibe_state.indoor_temp >= inner:
            return 0.0
        if nibe_state.indoor_temp <= outer or inner <= outer:
            return 1.0

        return (inner - nibe_state.indoor_temp) / (inner - outer)

    @staticmethod
    def _all_critical_are_cost(critical_layers: list[LayerDecision]) -> bool:
        """True when EVERY layer voting at critical weight is a cost layer.

        The floor in step 4 must not weaken a critical SAFETY or physics vote - only a vote that
        exists to save money. If anything other than cost is also critical, the tie-break already
        had a non-cost opinion to weigh and there is nothing to protect the house from.
        """
        return bool(critical_layers) and all(
            getattr(layer, "is_cost_layer", False) for layer in critical_layers
        )

    @staticmethod
    def _has_critical_cost_layer(layers: list[LayerDecision]) -> bool:
        """True if a cost layer (spot price or effect tariff) is voting at critical weight.

        Both the price layer (PEAK quarters) and the effect layer (at the monthly peak)
        promote themselves to LAYER_WEIGHT_SAFETY. Non-cost layers never set the flag, so
        `getattr` defaults them to False - the emergency and proactive layers use their own
        decision dataclasses and do not carry this field.
        """
        return any(
            getattr(layer, "is_cost_layer", False) and layer.weight >= LAYER_WEIGHT_SAFETY
            for layer in layers
        )

    @staticmethod
    def _weighted_average(layers: list[LayerDecision]) -> float:
        """Weighted average of all layer votes (0.0 when no layer is voting)."""
        total_weight = sum(layer.weight for layer in layers)
        if total_weight == 0:
            return 0.0
        return sum(layer.offset * layer.weight for layer in layers) / total_weight

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

        # The compressor range plus the machine's own immersion heater is the machine's
        # plausible ceiling. Winter draw above the compressor range alone is the elpatron
        # doing its job - flagging that fired every five minutes all January, which teaches
        # the owner to ignore this channel. Only a reading the HARDWARE cannot produce is
        # worth a line.
        immersion_kw = float(getattr(self.heat_pump_model, "immersion_heater_kw", 0.0) or 0.0)
        machine_ceiling = (max_power + immersion_kw) * POWER_VALIDATION_MARGIN

        if current_power_kw > machine_ceiling:
            return {
                "valid": False,
                "warning": (
                    f"Power {current_power_kw:.1f}kW exceeds what a "
                    f"{self.heat_pump_model.model_name} can draw "
                    f"(compressor max {max_power:.1f}kW + immersion {immersion_kw:.1f}kW). "
                    f"Check the sensor's unit and scaling."
                ),
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

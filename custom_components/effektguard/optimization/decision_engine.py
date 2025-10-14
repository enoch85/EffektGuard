"""Multi-layer decision engine for optimization.

Implements layered decision-making architecture that integrates:
- Safety constraints (temperature limits, thermal debt prevention)
- Effect tariff protection (15-minute peak avoidance)
- Weather prediction (pre-heating before cold)
- Spot price optimization (cost reduction)
- Comfort maintenance (temperature tolerance)
- Emergency recovery (degree minutes critical threshold)

Each layer votes on offset, final decision is weighted aggregation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from homeassistant.util import dt as dt_util

from ..const import (
    DEFAULT_TARGET_TEMP,
    DEFAULT_TOLERANCE,
    DM_THRESHOLD_CRITICAL,
    DM_THRESHOLD_WARNING,
    MAX_TEMP_LIMIT,
    MIN_TEMP_LIMIT,
    QuarterClassification,
)

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
    4. Weather prediction
    5. Spot price optimization
    6. Comfort maintenance
    """

    def __init__(
        self,
        price_analyzer,
        effect_manager,
        thermal_model,
        config: dict[str, Any],
    ):
        """Initialize decision engine with dependencies.

        Args:
            price_analyzer: PriceAnalyzer for spot price classification
            effect_manager: EffectManager for peak tracking
            thermal_model: ThermalModel for predictions
            config: Configuration options
        """
        self.price = price_analyzer
        self.effect = effect_manager
        self.thermal = thermal_model
        self.config = config

        # Configuration with defaults
        self.target_temp = config.get("target_temperature", DEFAULT_TARGET_TEMP)
        self.tolerance = config.get("tolerance", DEFAULT_TOLERANCE)
        self.tolerance_range = self.tolerance * 0.4  # Scale: 1-10 -> 0.4-4.0°C

    def calculate_decision(
        self,
        nibe_state,
        price_data,
        weather_data,
        current_peak: float,
    ) -> OptimizationDecision:
        """Calculate optimal heating offset using multi-layer approach.

        Decision layers (ordered by priority):
        1. Safety layer: Prevent extreme temperatures
        2. Emergency layer: Respond to critical degree minutes
        3. Effect tariff layer: Peak protection
        4. Weather prediction layer: Pre-heat before cold
        5. Spot price layer: Base optimization
        6. Comfort layer: Stay within tolerance

        Args:
            nibe_state: Current NIBE heat pump state
            price_data: GE-Spot price data (native 15-min intervals)
            weather_data: Weather forecast data
            current_peak: Current monthly peak (kW)

        Returns:
            OptimizationDecision with offset, reasoning, and layer votes
        """
        _LOGGER.debug("Calculating optimization decision")

        # Update price analyzer with latest data
        if price_data:
            self.price.update_prices(price_data)

        # Calculate all layer decisions
        layers = [
            self._safety_layer(nibe_state),
            self._emergency_layer(nibe_state),
            self._effect_layer(nibe_state, current_peak),
            self._weather_layer(nibe_state, weather_data),
            self._price_layer(price_data),
            self._comfort_layer(nibe_state),
        ]

        # Aggregate layers with priority weighting
        final_offset = self._aggregate_layers(layers)

        # Generate human-readable reasoning
        reasoning = self._generate_reasoning(layers, final_offset)

        _LOGGER.info("Decision: offset %.2f°C - %s", final_offset, reasoning)

        return OptimizationDecision(
            offset=final_offset,
            layers=layers,
            reasoning=reasoning,
        )

    def _safety_layer(self, nibe_state) -> LayerDecision:
        """Safety layer: Enforce absolute temperature limits.

        Hard limits: 18-24°C indoor temperature
        This layer always has maximum weight.

        Args:
            nibe_state: Current NIBE state

        Returns:
            LayerDecision with safety constraints
        """
        indoor_temp = nibe_state.indoor_temp

        if indoor_temp < MIN_TEMP_LIMIT:
            # Too cold - emergency heating
            offset = 5.0
            return LayerDecision(
                offset=offset,
                weight=1.0,
                reason=f"SAFETY: Too cold ({indoor_temp:.1f}°C < {MIN_TEMP_LIMIT}°C)",
            )
        elif indoor_temp > MAX_TEMP_LIMIT:
            # Too hot - reduce heating
            offset = -5.0
            return LayerDecision(
                offset=offset,
                weight=1.0,
                reason=f"SAFETY: Too hot ({indoor_temp:.1f}°C > {MAX_TEMP_LIMIT}°C)",
            )
        else:
            # Within safe limits
            return LayerDecision(
                offset=0.0,
                weight=0.0,
                reason="Safety OK",
            )

    def _emergency_layer(self, nibe_state) -> LayerDecision:
        """Emergency layer: Respond to critical degree minutes (thermal debt).

        Based on research:
        - DM -400: Warning threshold, stop cost optimization
        - DM -500: Critical threshold, emergency recovery
        - Source: Forum_Summary.md (stevedvo case: DM -500 = 15kW spikes)

        Args:
            nibe_state: Current NIBE state

        Returns:
            LayerDecision with emergency response
        """
        degree_minutes = nibe_state.degree_minutes

        if degree_minutes <= DM_THRESHOLD_CRITICAL:
            # Critical thermal debt - emergency recovery
            offset = 3.0
            return LayerDecision(
                offset=offset,
                weight=1.0,
                reason=f"EMERGENCY: Critical DM {degree_minutes:.0f} (≤{DM_THRESHOLD_CRITICAL})",
            )
        elif degree_minutes <= DM_THRESHOLD_WARNING:
            # Warning - gentle recovery, stop further reductions
            offset = 1.5
            return LayerDecision(
                offset=offset,
                weight=0.9,
                reason=f"WARNING: DM {degree_minutes:.0f} approaching danger (≤{DM_THRESHOLD_WARNING})",
            )
        else:
            # Thermal debt OK
            return LayerDecision(
                offset=0.0,
                weight=0.0,
                reason=f"Thermal debt OK (DM: {degree_minutes:.0f})",
            )

    def _effect_layer(self, nibe_state, current_peak: float) -> LayerDecision:
        """Effect tariff protection layer: Avoid creating new 15-minute peak.

        Args:
            nibe_state: Current NIBE state
            current_peak: Current monthly peak (kW)

        Returns:
            LayerDecision with peak protection
        """
        # Estimate current power
        current_power = self._estimate_heat_pump_power(nibe_state)

        # Get current quarter
        now = dt_util.now()
        current_quarter = (now.hour * 4) + (now.minute // 15)  # 0-95

        # Check if approaching monthly 15-minute peak
        limit_decision = self.effect.should_limit_power(current_power, current_quarter)

        if limit_decision.severity == "CRITICAL":
            # Within 0.5 kW of peak, aggressively reduce
            return LayerDecision(
                offset=-3.0,
                weight=1.0,
                reason=f"CRITICAL: Approaching 15-min peak (Q{current_quarter})",
            )
        elif limit_decision.severity == "WARNING":
            # Within 1.0 kW, moderate reduction
            return LayerDecision(
                offset=-1.5,
                weight=0.8,
                reason=f"WARNING: Near 15-min peak (Q{current_quarter})",
            )
        else:
            return LayerDecision(
                offset=0.0,
                weight=0.0,
                reason=f"Peak OK (Q{current_quarter})",
            )

    def _weather_layer(self, nibe_state, weather_data) -> LayerDecision:
        """Weather prediction layer: Pre-heat before predicted cold periods.

        Uses dynamic thresholds based on building thermal mass and
        rate-of-change analysis over 6-hour forecast window.

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data

        Returns:
            LayerDecision with pre-heating recommendation
        """
        if not weather_data or not weather_data.forecast_hours:
            return LayerDecision(offset=0.0, weight=0.0, reason="No weather data")

        forecast_6h = weather_data.forecast_hours[:6]  # 6-hour window
        current_outdoor = nibe_state.outdoor_temp

        # Calculate minimum temperature and rate of change
        min_temp = min(f.temperature for f in forecast_6h)
        temp_drop = min_temp - current_outdoor
        hourly_rate = temp_drop / len(forecast_6h) if forecast_6h else 0.0

        # Dynamic threshold based on building thermal mass
        # Higher thermal mass = can handle bigger drops without pre-heat
        # Range: -1.0°C to -4.0°C depending on thermal mass (0.5-2.0)
        threshold = -2.0 * self.thermal.thermal_mass

        # Pre-heat if both conditions met:
        # 1. Temperature drop exceeds dynamic threshold
        # 2. Rate of cooling is significant (> 0.5°C/hour)
        if temp_drop < threshold and hourly_rate < -0.5:
            # Calculate pre-heat target accounting for thermal decay
            preheat_target = self.thermal.calculate_preheating_target(
                current_temp=nibe_state.indoor_temp,
                desired_temp=self.target_temp,
                hours_until_peak=len(forecast_6h),
                outdoor_temp=current_outdoor,
                forecast_min_temp=min_temp,
            )
            offset = preheat_target - self.target_temp
            return LayerDecision(
                offset=offset,
                weight=0.7,
                reason=f"Pre-heat: {temp_drop:.1f}°C drop, {hourly_rate:.1f}°C/h rate",
            )
        else:
            return LayerDecision(
                offset=0.0,
                weight=0.0,
                reason=f"Weather OK: {temp_drop:.1f}°C drop < {threshold:.1f}°C threshold",
            )

    def _price_layer(self, price_data) -> LayerDecision:
        """Spot price layer: Base optimization from native 15-minute GE-Spot data.

        Args:
            price_data: GE-Spot price data with native 15-min intervals

        Returns:
            LayerDecision with price-based offset
        """
        if not price_data or not price_data.today:
            return LayerDecision(offset=0.0, weight=0.0, reason="No price data")

        now = dt_util.now()
        current_quarter = (now.hour * 4) + (now.minute // 15)  # 0-95

        # Get current period classification
        classification = self.price.get_current_classification(current_quarter)
        current_period = price_data.today[current_quarter]

        # Get base offset for classification
        base_offset = self.price.get_base_offset(
            current_quarter,
            classification,
            current_period.is_daytime,
        )

        # Adjust for tolerance setting (1-10 scale)
        # Higher tolerance = more aggressive optimization
        tolerance_factor = self.tolerance / 5.0  # 0.2-2.0
        adjusted_offset = base_offset * tolerance_factor

        return LayerDecision(
            offset=adjusted_offset,
            weight=0.6,
            reason=f"GE-Spot Q{current_quarter}: {classification.name} ({'day' if current_period.is_daytime else 'night'})",
        )

    def _comfort_layer(self, nibe_state) -> LayerDecision:
        """Comfort layer: Reactive adjustment to maintain comfort.

        Args:
            nibe_state: Current NIBE state

        Returns:
            LayerDecision with comfort correction
        """
        temp_error = nibe_state.indoor_temp - self.target_temp

        # Temperature tolerance based on user setting
        tolerance = self.tolerance_range  # ±0.4-4.0°C

        if abs(temp_error) < tolerance:
            # Within comfort zone
            return LayerDecision(offset=0.0, weight=0.0, reason="Temp OK")
        elif temp_error > tolerance:
            # Too warm, reduce heating
            correction = -(temp_error - tolerance) * 0.5
            return LayerDecision(
                offset=correction,
                weight=0.5,
                reason=f"Too warm: {temp_error:.1f}°C over",
            )
        else:
            # Too cold, increase heating
            correction = -(temp_error + tolerance) * 0.5
            return LayerDecision(
                offset=correction,
                weight=0.5,
                reason=f"Too cold: {-temp_error:.1f}°C under",
            )

    def _aggregate_layers(self, layers: list[LayerDecision]) -> float:
        """Aggregate layer decisions into final offset.

        Uses weighted average with special handling for high-priority layers.

        Args:
            layers: List of layer decisions

        Returns:
            Final offset value
        """
        # Separate high-priority layers (weight = 1.0)
        critical_layers = [layer for layer in layers if layer.weight >= 1.0]

        if critical_layers:
            # If any critical layer votes, take the strongest vote
            # (safety and emergency layers)
            max_offset = max(layer.offset for layer in critical_layers)
            min_offset = min(layer.offset for layer in critical_layers)

            # If conflicting critical votes, take the more conservative
            if abs(max_offset) > abs(min_offset):
                return max_offset
            else:
                return min_offset

        # Otherwise, weighted average of all layers
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

        Args:
            layers: List of layer decisions
            final_offset: Final aggregated offset

        Returns:
            Reasoning string
        """
        # Find active layers (non-zero weight)
        active_layers = [layer for layer in layers if layer.weight > 0]

        if not active_layers:
            return "No optimization active"

        # Build reasoning string
        reasons = [layer.reason for layer in active_layers]
        return " | ".join(reasons)

    def _estimate_heat_pump_power(self, nibe_state) -> float:
        """Estimate heat pump power consumption from state.

        Args:
            nibe_state: Current NIBE state

        Returns:
            Estimated power consumption in kW
        """
        # Basic estimation based on compressor status and outdoor temp
        if not nibe_state.is_heating:
            return 0.1  # Standby power

        # Rough estimation: colder outdoor = higher power
        outdoor_temp = nibe_state.outdoor_temp
        base_power = 4.0  # kW baseline

        # Adjust for outdoor temperature
        if outdoor_temp < -10:
            return base_power * 1.3
        elif outdoor_temp < 0:
            return base_power * 1.1
        else:
            return base_power

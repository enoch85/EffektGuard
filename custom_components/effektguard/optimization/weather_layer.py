"""Weather compensation calculations for optimal flow temperature.

Implements scientifically-validated mathematical formulas from OpenEnergyMonitor research:
1. André Kühne's Universal Formula (validated across Vaillant, Daikin, Mitsubishi, NIBE)
2. Timbones' Heat Transfer Method (radiator output approach)
3. UFH-specific flow temperature adjustments
4. Adaptive Climate System (combines universal zones with weather learning)

References:
    - Mathematical_Enhancement_Summary.md
    - OpenEnergyMonitor.org community research
    - Timbones' calculation spreadsheet
    - HeatpumpMonitor.org performance data
    - POST_PHASE_5_ROADMAP.md Phase 6 - Adaptive learning
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol

from ..const import (
    DEFAULT_CURVE_SENSITIVITY,
    DEFAULT_HEAT_LOSS_COEFFICIENT,
    DEFAULT_WEATHER_COMPENSATION_WEIGHT,
    KUEHNE_COEFFICIENT,
    KUEHNE_POWER,
    LAYER_WEIGHT_WEATHER_PREDICTION,
    RADIATOR_POWER_COEFFICIENT,
    RADIATOR_RATED_DT,
    UFH_FLOW_REDUCTION_CONCRETE,
    UFH_FLOW_REDUCTION_TIMBER,
    UFH_MIN_FLOW_TEMP_CONCRETE,
    UFH_MIN_FLOW_TEMP_TIMBER,
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
    WEATHER_WEIGHT_CAP,
)
from .climate_zones import ClimateZoneDetector

_LOGGER = logging.getLogger(__name__)


class WeatherLearnerProtocol(Protocol):
    """Protocol for weather learning interface."""

    def detect_unusual_weather(
        self, current_date: datetime, forecast: list[float]
    ) -> "UnusualWeatherResult":
        """Detect unusual weather patterns from forecast."""
        ...


class UnusualWeatherResult(Protocol):
    """Protocol for unusual weather detection result."""

    is_unusual: bool
    severity: str
    deviation_from_typical: float
    recommendation: str


@dataclass
class WeatherLayerDecision:
    """Decision from the weather prediction layer.

    Encapsulates the pre-heating recommendation based on weather forecast
    and thermal trend analysis.
    """

    name: str
    offset: float
    weight: float
    reason: str


@dataclass
class WeatherCompensationLayerDecision:
    """Decision from the mathematical weather compensation layer.

    Encapsulates the flow temperature optimization based on:
    - Universal flow temperature formula (André Kühne)
    - Heat transfer method (Timbones, if radiator specs available)
    - UFH-specific adjustments
    - Adaptive climate zones
    - Weather learning (unusual pattern detection)
    """

    name: str
    offset: float
    weight: float
    reason: str
    # Additional diagnostic fields
    optimal_flow_temp: float = 0.0
    adjusted_flow_temp: float = 0.0
    safety_margin: float = 0.0
    unusual_weather: bool = False
    defer_factor: float = 1.0


@dataclass
class FlowTempCalculation:
    """Result of flow temperature calculation with reasoning."""

    flow_temp: float  # Calculated optimal flow temperature (°C)
    method: str  # Calculation method used
    heating_type: str  # "radiator", "concrete_ufh", "timber_ufh", etc.
    confidence: float  # 0-1 confidence in calculation
    reasoning: str  # Explanation of calculation
    raw_kuehne: Optional[float] = None  # Raw Kühne result before adjustments
    raw_timbones: Optional[float] = None  # Raw Timbones result before adjustments


class WeatherCompensationCalculator:
    """Calculate optimal flow temperatures using validated mathematical formulas.

    Implements three complementary methods:
    1. André Kühne's formula - Universal physics-based calculation
    2. Timbones' method - Radiator-specific heat transfer approach
    3. UFH adjustments - Specialized underfloor heating optimization
    """

    def __init__(
        self,
        heat_loss_coefficient: float = DEFAULT_HEAT_LOSS_COEFFICIENT,
        radiator_rated_output: Optional[float] = None,
        heating_type: str = "radiator",
    ):
        """Initialize weather compensation calculator.

        Args:
            heat_loss_coefficient: Building heat loss in W/°C (typical 100-300)
            radiator_rated_output: Total rated radiator output at DT50 in Watts
            heating_type: "radiator", "concrete_ufh", "timber_ufh", "mixed"
        """
        self.heat_loss_coefficient = heat_loss_coefficient
        self.radiator_rated_output = radiator_rated_output
        self.heating_type = heating_type

        _LOGGER.debug(
            "WeatherCompensationCalculator initialized: HC=%.1f W/°C, "
            "radiator_output=%s W, type=%s",
            heat_loss_coefficient,
            radiator_rated_output,
            heating_type,
        )

    def calculate_kuehne_flow_temp(
        self,
        indoor_setpoint: float,
        outdoor_temp: float,
    ) -> float:
        """Calculate optimal flow temperature using André Kühne's universal formula.

        Formula: TFlow = 2.55 × (HC × (Tset - Tout))^0.78 + Tset

        Note: HC must be in kW/K for correct results!

        Validated across manufacturers: Vaillant, Daikin, Mitsubishi, NIBE.
        Based on heat transfer physics, not manufacturer-specific curves.

        Args:
            indoor_setpoint: Target indoor temperature (°C)
            outdoor_temp: Current outdoor temperature (°C)

        Returns:
            Optimal flow temperature (°C)

        References:
            Mathematical_Enhancement_Summary.md: André Kühne's formula
            OpenEnergyMonitor community validation data
        """
        # Calculate temperature differential
        temp_diff = indoor_setpoint - outdoor_temp

        # Ensure positive differential (can't heat when outdoor > indoor setpoint)
        if temp_diff <= 0:
            return indoor_setpoint

        # André Kühne's formula
        # TFlow = 2.55 × (HC × (Tset - Tout))^0.78 + Tset
        # Convert heat loss coefficient from W/°C to kW/K
        heat_loss_kw = self.heat_loss_coefficient / 1000.0
        heat_term = heat_loss_kw * temp_diff
        flow_temp = KUEHNE_COEFFICIENT * (heat_term**KUEHNE_POWER) + indoor_setpoint

        _LOGGER.debug(
            "Kühne formula: outdoor=%.1f°C, indoor_target=%.1f°C, "
            "temp_diff=%.1f°C, HC=%.3f kW/K -> flow=%.1f°C",
            outdoor_temp,
            indoor_setpoint,
            temp_diff,
            heat_loss_kw,
            flow_temp,
        )

        return flow_temp

    def calculate_timbones_flow_temp(
        self,
        indoor_setpoint: float,
        outdoor_temp: float,
        flow_return_dt: float = 5.0,
    ) -> Optional[float]:
        """Calculate optimal flow temperature using Timbones' heat transfer method.

        Based on radiator output calculations and heat loss coefficient.
        Requires radiator_rated_output to be configured.

        Formula:
        1. Heat demand = heat_loss_coefficient × (indoor - outdoor)
        2. Required DT = 50K × (heat_demand / radiator_output)^(1/1.3)
        3. Flow temp = indoor + required_DT + (flow_return_dt / 2)

        Args:
            indoor_setpoint: Target indoor temperature (°C)
            outdoor_temp: Current outdoor temperature (°C)
            flow_return_dt: Design flow-return temperature differential (°C)

        Returns:
            Optimal flow temperature (°C), or None if radiator output not configured

        References:
            Timbones' calculation spreadsheet (OpenEnergyMonitor community)
            Radiator output formula: Heat = Rated × (DT/50K)^1.3
        """
        if self.radiator_rated_output is None:
            _LOGGER.debug("Timbones method requires radiator_rated_output configuration")
            return None

        # Calculate heat demand
        temp_diff = indoor_setpoint - outdoor_temp
        if temp_diff <= 0:
            return indoor_setpoint

        heat_demand = self.heat_loss_coefficient * temp_diff

        # Calculate required radiator temperature differential
        # From radiator equation: Output = Rated × (ΔT/50K)^1.3
        # Inverted: ΔT = 50K × (Output/Rated)^(1/1.3)
        output_ratio = heat_demand / self.radiator_rated_output
        required_dt = RADIATOR_RATED_DT * (output_ratio ** (1 / RADIATOR_POWER_COEFFICIENT))

        # Mean water temperature = room temp + required DT
        mean_water_temp = indoor_setpoint + required_dt

        # Flow temperature = MWT + half of flow-return differential
        flow_temp = mean_water_temp + (flow_return_dt / 2)

        _LOGGER.debug(
            "Timbones method: heat_demand=%.0f W, radiator_output=%.0f W, "
            "required_DT=%.1f K, MWT=%.1f°C -> flow=%.1f°C",
            heat_demand,
            self.radiator_rated_output,
            required_dt,
            mean_water_temp,
            flow_temp,
        )

        return flow_temp

    def apply_ufh_adjustment(
        self,
        radiator_flow_temp: float,
        ufh_type: str,
    ) -> float:
        """Apply underfloor heating adjustments to radiator-calculated flow temp.

        UFH systems require lower flow temperatures due to larger heat exchange surface.
        Adjustments based on real-world UFH installations and thermal properties.

        Args:
            radiator_flow_temp: Flow temperature calculated for radiators (°C)
            ufh_type: "concrete_slab", "timber", or "radiator" (no adjustment)

        Returns:
            Adjusted flow temperature for UFH system (°C)

        References:
            Mathematical_Enhancement_Summary.md: UFH-specific optimizations
            Floor_Heating_Enhancements.md: Thermal lag and mass modeling
        """
        if ufh_type == "concrete_slab":
            # Concrete slab UFH: 8°C reduction, minimum 25°C
            ufh_flow_temp = radiator_flow_temp - UFH_FLOW_REDUCTION_CONCRETE
            ufh_flow_temp = max(ufh_flow_temp, UFH_MIN_FLOW_TEMP_CONCRETE)

            _LOGGER.debug(
                "UFH concrete adjustment: radiator=%.1f°C -> UFH=%.1f°C "
                "(reduction=%.1f°C, min=%.1f°C)",
                radiator_flow_temp,
                ufh_flow_temp,
                UFH_FLOW_REDUCTION_CONCRETE,
                UFH_MIN_FLOW_TEMP_CONCRETE,
            )

        elif ufh_type == "timber":
            # Timber UFH: 5°C reduction, minimum 22°C
            ufh_flow_temp = radiator_flow_temp - UFH_FLOW_REDUCTION_TIMBER
            ufh_flow_temp = max(ufh_flow_temp, UFH_MIN_FLOW_TEMP_TIMBER)

            _LOGGER.debug(
                "UFH timber adjustment: radiator=%.1f°C -> UFH=%.1f°C "
                "(reduction=%.1f°C, min=%.1f°C)",
                radiator_flow_temp,
                ufh_flow_temp,
                UFH_FLOW_REDUCTION_TIMBER,
                UFH_MIN_FLOW_TEMP_TIMBER,
            )

        else:
            # Radiator system - no adjustment
            ufh_flow_temp = radiator_flow_temp

        return ufh_flow_temp

    def calculate_optimal_flow_temp(
        self,
        indoor_setpoint: float,
        outdoor_temp: float,
        prefer_method: str = "kuehne",
        flow_return_dt: float = 5.0,
    ) -> FlowTempCalculation:
        """Calculate optimal flow temperature using best available method.

        Prioritizes André Kühne's formula by default (universal, physics-based).
        Falls back to Timbones' method if configured and requested.
        Applies UFH adjustments automatically based on heating_type.

        Args:
            indoor_setpoint: Target indoor temperature (°C)
            outdoor_temp: Current outdoor temperature (°C)
            prefer_method: "kuehne" (default), "timbones", or "auto"
            flow_return_dt: Design flow-return differential (°C)

        Returns:
            FlowTempCalculation with optimal temperature and reasoning
        """
        raw_kuehne = None
        raw_timbones = None
        method_used = "kuehne"
        confidence = 0.9  # High confidence in physics-based formula

        # Calculate using André Kühne's formula (always available)
        raw_kuehne = self.calculate_kuehne_flow_temp(indoor_setpoint, outdoor_temp)
        flow_temp = raw_kuehne

        # Try Timbones' method if configured and requested
        if prefer_method in ("timbones", "auto") and self.radiator_rated_output is not None:
            raw_timbones = self.calculate_timbones_flow_temp(
                indoor_setpoint, outdoor_temp, flow_return_dt
            )

            if raw_timbones is not None:
                if prefer_method == "timbones":
                    # User explicitly prefers Timbones
                    flow_temp = raw_timbones
                    method_used = "timbones"
                    confidence = 0.85  # Slightly lower (requires radiator spec)
                elif prefer_method == "auto":
                    # Average both methods for robustness
                    flow_temp = (raw_kuehne + raw_timbones) / 2
                    method_used = "kuehne+timbones"
                    confidence = 0.95  # Higher confidence with multiple methods

        # Apply UFH adjustments if applicable
        if self.heating_type in ("concrete_ufh", "timber"):
            ufh_type = "concrete_slab" if self.heating_type == "concrete_ufh" else "timber"
            flow_temp = self.apply_ufh_adjustment(flow_temp, ufh_type)

        # Build reasoning string
        reasoning_parts = [f"Outdoor: {outdoor_temp:.1f}°C, Indoor target: {indoor_setpoint:.1f}°C"]

        if method_used == "kuehne":
            reasoning_parts.append(
                f"André Kühne formula: {raw_kuehne:.1f}°C "
                f"(HC={self.heat_loss_coefficient:.0f} W/°C)"
            )
        elif method_used == "timbones":
            reasoning_parts.append(
                f"Timbones method: {raw_timbones:.1f}°C "
                f"(radiator={self.radiator_rated_output:.0f}W)"
            )
        elif method_used == "kuehne+timbones":
            reasoning_parts.append(
                f"Combined: Kühne={raw_kuehne:.1f}°C, "
                f"Timbones={raw_timbones:.1f}°C, avg={flow_temp:.1f}°C"
            )

        if self.heating_type in ("concrete_ufh", "timber"):
            reasoning_parts.append(f"UFH adjustment applied ({self.heating_type})")

        reasoning = "; ".join(reasoning_parts)

        return FlowTempCalculation(
            flow_temp=flow_temp,
            method=method_used,
            heating_type=self.heating_type,
            confidence=confidence,
            reasoning=reasoning,
            raw_kuehne=raw_kuehne,
            raw_timbones=raw_timbones,
        )

    def calculate_required_offset(
        self,
        optimal_flow_temp: float,
        current_flow_temp: float,
        curve_sensitivity: float = 1.5,
    ) -> float:
        """Calculate heating curve offset needed to achieve optimal flow temperature.

        Args:
            optimal_flow_temp: Target flow temperature from weather compensation (°C)
            current_flow_temp: Current actual flow temperature (°C)
            curve_sensitivity: Flow temp change per offset unit (°C/offset)
                              Typical NIBE: 1.5°C per offset unit

        Returns:
            Recommended heating curve offset adjustment (°C)
        """
        temp_deviation = optimal_flow_temp - current_flow_temp
        offset_adjustment = temp_deviation / curve_sensitivity

        _LOGGER.debug(
            "Offset calculation: optimal=%.1f°C, current=%.1f°C, "
            "error=%.1f°C, sensitivity=%.1f -> offset=%.1f",
            optimal_flow_temp,
            current_flow_temp,
            temp_deviation,
            curve_sensitivity,
            offset_adjustment,
        )

        return offset_adjustment


class AdaptiveClimateSystem:
    """Combine universal climate zones with adaptive weather learning.

    DESIGN PHILOSOPHY:
    - Universal math (André Kühne, Timbones) works globally
    - Climate zones provide baseline safety margins
    - Weather learning adapts to local unusual patterns
    - No country-specific hardcoding needed

    This system automatically adapts to:
    - Kiruna, Sweden (-30°C winters) → Extreme Cold zone
    - Stockholm, Sweden (-10°C winters) → Cold zone
    - Copenhagen, Denmark (0°C winters) → Moderate Cold zone
    - Paris, France (5°C winters) → Standard zone

    All without configuration changes or country-specific code!

    Note: Now uses dedicated ClimateZoneDetector from climate_zones.py module.
    """

    def __init__(self, latitude: float, weather_learner: Optional = None):
        """Initialize adaptive climate system.

        Args:
            latitude: Home location latitude in degrees (positive = North)
            weather_learner: Optional WeatherPatternLearner for adaptive margins
        """
        self.latitude = abs(latitude)  # Use absolute value for hemisphere independence
        self.weather_learner = weather_learner

        # Use dedicated ClimateZoneDetector module
        self.detector = ClimateZoneDetector(latitude)
        self.climate_zone = self.detector.zone_key  # Compatibility property

        _LOGGER.info(
            "AdaptiveClimateSystem initialized: latitude=%.2f° -> %s zone "
            "(winter avg low: %.1f°C, base safety margin: %.1f°C)",
            latitude,
            self.detector.zone_info.name,
            self.detector.zone_info.winter_avg_low,
            self.detector.zone_info.safety_margin_base,
        )

    def get_safety_margin(
        self,
        outdoor_temp: float,
        unusual_weather_detected: bool = False,
        unusual_severity: float = 0.0,
    ) -> float:
        """Calculate safety margin for current conditions.

        Combines:
        1. Climate zone baseline (latitude-based)
        2. Current outdoor temperature severity
        3. Weather learning (unusual pattern detection)

        This replaces hardcoded Swedish thresholds with universal adaptive system.

        Args:
            outdoor_temp: Current outdoor temperature (°C)
            unusual_weather_detected: Whether weather learner detected unusual pattern
            unusual_severity: Severity of unusual weather (0.0-1.0)

        Returns:
            Safety margin in °C to add to calculated flow temperature
        """
        base_margin = self.detector.zone_info.safety_margin_base

        # Temperature-based adjustment (colder = more margin)
        # Scale linearly from zone's winter avg low
        winter_avg_low = self.detector.zone_info.winter_avg_low
        temp_margin = 0.0

        if outdoor_temp < winter_avg_low:
            # Colder than typical winter low - add extra margin
            # 0.1°C margin per degree below winter average
            temp_margin = (winter_avg_low - outdoor_temp) * 0.1

        # Weather learning adjustment
        learning_margin = 0.0
        if unusual_weather_detected:
            # Unusual weather gets extra margin scaled by severity
            # Range: 0.5°C to 1.5°C for severity 0.0 to 1.0
            # Works with or without weather_learner (can be triggered externally)
            learning_margin = 0.5 + (unusual_severity * 1.0)

            _LOGGER.debug(
                "Unusual weather detected (severity=%.2f): +%.1f°C margin",
                unusual_severity,
                learning_margin,
            )

        total_margin = base_margin + temp_margin + learning_margin

        _LOGGER.debug(
            "Safety margin calculation: zone_base=%.1f°C, temp_adj=%.1f°C, "
            "learning=%.1f°C -> total=%.1f°C",
            base_margin,
            temp_margin,
            learning_margin,
            total_margin,
        )

        return total_margin

    def get_dynamic_weight(
        self,
        outdoor_temp: float,
        unusual_weather_detected: bool = False,
    ) -> float:
        """Calculate dynamic weight for weather compensation layer.

        Adjusts influence based on:
        - Temperature severity (colder = more important)
        - Unusual weather patterns (higher weight when unusual)

        Args:
            outdoor_temp: Current outdoor temperature (°C)
            unusual_weather_detected: Whether weather learner detected unusual pattern

        Returns:
            Weight for decision engine layer (0.0-1.0)
        """
        winter_avg_low = self.detector.zone_info.winter_avg_low

        # Base weight varies by temperature severity
        if outdoor_temp < winter_avg_low:
            # Very cold - weather compensation critical
            base_weight = 0.85
        elif outdoor_temp < (winter_avg_low + 5):
            # Cold - weather compensation important
            base_weight = 0.75
        elif outdoor_temp < 5:
            # Mild cold - weather compensation useful
            base_weight = 0.65
        else:
            # Warm - weather compensation less critical
            base_weight = 0.50

        # Increase weight during unusual weather
        if unusual_weather_detected:
            base_weight = min(0.95, base_weight + 0.15)

        _LOGGER.debug(
            "Dynamic weight: outdoor=%.1f°C (zone_low=%.1f°C), " "unusual=%s -> weight=%.2f",
            outdoor_temp,
            winter_avg_low,
            unusual_weather_detected,
            base_weight,
        )

        return base_weight

    def get_climate_info(self) -> dict:
        """Get climate zone information for diagnostics.

        Returns:
            Dictionary with zone name, latitude range, winter avg, examples, etc.
        """
        return {
            "detected_zone": self.detector.zone_key,
            "name": self.detector.zone_info.name,
            "description": self.detector.zone_info.description,
            "winter_avg_low": self.detector.zone_info.winter_avg_low,
            "safety_margin_base": self.detector.zone_info.safety_margin_base,
            "examples": self.detector.zone_info.examples,
            "user_latitude": self.latitude,
        }


class WeatherPredictionLayer:
    """Weather prediction layer for proactive pre-heating.

    Philosophy (Oct 20, 2025):
    "The heating we add NOW shows up in 6 hours - pre-heat BEFORE cold arrives"

    Problem: Concrete slab 6-hour thermal lag causes reactive heating to arrive
    too late, resulting in thermal debt spirals (DM -1000 @ 04:00) followed by
    massive overshoot (26°C @ 16:00 from heat added 6 hours earlier).

    Solution: Simple proactive pre-heating:
    1. PRIMARY: Forecast shows ≥5°C drop in next 12h → +0.5°C gentle pre-heat
    2. CONFIRMATION: Indoor cooling ≥0.5°C/h → Confirms forecast, maintains +0.5°C
    3. MODERATION: Let SAFETY, COMFORT, EFFECT layers handle naturally via weighted aggregation
    """

    def __init__(self, thermal_mass: float = 1.0):
        """Initialize weather prediction layer.

        Args:
            thermal_mass: Building thermal mass (0.5=light, 1.0=medium, 1.5=heavy)
        """
        self.thermal_mass = thermal_mass

    def evaluate_layer(
        self,
        nibe_state,
        weather_data,
        thermal_trend: dict,
        enable_weather_prediction: bool = True,
    ) -> WeatherLayerDecision:
        """Weather prediction layer: Proactive pre-heating based on forecast.

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
            thermal_trend: Trend data from predictor (rate_per_hour, confidence)
            enable_weather_prediction: Whether layer is enabled by user

        Returns:
            WeatherLayerDecision with gentle pre-heating recommendation
        """
        # Check if weather prediction is enabled
        if not enable_weather_prediction:
            return WeatherLayerDecision(
                name="Weather Pre-heat",
                offset=0.0,
                weight=0.0,
                reason="Disabled by user",
            )

        if not weather_data or not weather_data.forecast_hours:
            return WeatherLayerDecision(
                name="Weather Pre-heat",
                offset=0.0,
                weight=0.0,
                reason="No weather data",
            )

        # Check forecast for significant temperature drop
        current_outdoor = nibe_state.outdoor_temp
        forecast_hours = weather_data.forecast_hours[: int(WEATHER_FORECAST_HORIZON)]

        if not forecast_hours:
            return WeatherLayerDecision(
                name="Weather Pre-heat",
                offset=0.0,
                weight=0.0,
                reason="No forecast data",
            )

        # Find minimum temperature in forecast period
        min_temp = min(f.temperature for f in forecast_hours)
        temp_drop = min_temp - current_outdoor

        # PRIMARY TRIGGER: Forecast shows ≥5°C drop
        forecast_triggered = temp_drop <= WEATHER_FORECAST_DROP_THRESHOLD

        # CONFIRMATION TRIGGER: Indoor already cooling (confirms forecast)
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
                LAYER_WEIGHT_WEATHER_PREDICTION * self.thermal_mass,
                WEATHER_WEIGHT_CAP,  # Cap below Safety (1.0)
            )

            # Determine trigger reason
            if forecast_triggered and indoor_cooling:
                trigger = f"Forecast {temp_drop:.1f}°C drop + Indoor cooling {trend_rate:.2f}°C/h (confirmed)"
            elif forecast_triggered:
                trigger = f"Forecast {temp_drop:.1f}°C drop in {WEATHER_FORECAST_HORIZON:.0f}h (proactive)"
            else:
                trigger = f"Indoor cooling {trend_rate:.2f}°C/h (reactive confirmation)"

            return WeatherLayerDecision(
                name="Weather Pre-heat",
                offset=WEATHER_GENTLE_OFFSET,  # Constant +0.5°C (simple, predictable)
                weight=weather_weight,
                reason=trigger,
            )

        return WeatherLayerDecision(
            name="Weather Pre-heat",
            offset=0.0,
            weight=0.0,
            reason="Forecast stable, indoor stable",
        )


class WeatherCompensationLayer:
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
    """

    def __init__(
        self,
        weather_comp: WeatherCompensationCalculator,
        climate_system: AdaptiveClimateSystem,
        weather_learner: Optional[WeatherLearnerProtocol] = None,
        weather_comp_weight: float = 1.0,
    ):
        """Initialize weather compensation layer.

        Args:
            weather_comp: WeatherCompensationCalculator instance for flow temp calculation
            climate_system: AdaptiveClimateSystem for zone-based adjustments
            weather_learner: Optional weather pattern learner for unusual detection
            weather_comp_weight: User-configured weight adjustment (default 1.0)
        """
        self.weather_comp = weather_comp
        self.climate_system = climate_system
        self.weather_learner = weather_learner
        self.weather_comp_weight = weather_comp_weight

    def evaluate_layer(
        self,
        nibe_state,
        weather_data,
        target_temp: float,
        enable_weather_compensation: bool = True,
        get_current_datetime: Optional[callable] = None,
        temp_lux_active: bool = False,
    ) -> WeatherCompensationLayerDecision:
        """Evaluate the weather compensation layer.

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data
            target_temp: Target indoor temperature (°C)
            enable_weather_compensation: Whether feature is enabled
            get_current_datetime: Callable returning current datetime (for testing)

        Returns:
            WeatherCompensationLayerDecision with mathematically calculated offset
        """
        # Check if feature is enabled
        if not enable_weather_compensation:
            return WeatherCompensationLayerDecision(
                name="Math WC", offset=0.0, weight=0.0, reason="Disabled"
            )

        # Skip weather compensation during DHW/lux heating
        # When NIBE heats DHW, flow temp reads DHW charging temp (45-60°C),
        # not space heating flow. This would cause incorrect negative offsets.
        # Check temp_lux_active from coordinator (checks temp_lux switch state)
        if temp_lux_active:
            _LOGGER.debug(
                "Weather comp skipped: DHW/lux heating active (flow=%.1f°C)",
                nibe_state.flow_temp,
            )
            return WeatherCompensationLayerDecision(
                name="Math WC",
                offset=0.0,
                weight=0.0,
                reason="DHW/lux active - flow temp not valid for space heating",
            )

        if not weather_data or not weather_data.forecast_hours:
            return WeatherCompensationLayerDecision(
                name="Math WC", offset=0.0, weight=0.0, reason="No weather data"
            )

        current_outdoor = nibe_state.outdoor_temp
        current_flow = nibe_state.flow_temp

        # Calculate optimal flow temperature using physics-based formulas
        flow_calc = self.weather_comp.calculate_optimal_flow_temp(
            indoor_setpoint=target_temp,
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
                # Get current datetime (allow injection for testing)
                if get_current_datetime is not None:
                    current_date = get_current_datetime()
                else:
                    # Import here to avoid circular imports and allow mocking
                    from homeassistant.util import dt as dt_util

                    current_date = dt_util.now()

                # Extract forecast for unusual weather detection
                forecast_temps = [h.temperature for h in weather_data.forecast_hours[:24]]
                unusual = self.weather_learner.detect_unusual_weather(
                    current_date=current_date,
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
        required_offset = self.weather_comp.calculate_required_offset(
            optimal_flow_temp=adjusted_flow_temp,
            current_flow_temp=current_flow,
            curve_sensitivity=DEFAULT_CURVE_SENSITIVITY,
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
        defer_factor = 1.0
        defer_reason = None

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

        return WeatherCompensationLayerDecision(
            name="Math WC",
            offset=required_offset,
            weight=final_weight,
            reason=reasoning,
            optimal_flow_temp=flow_calc.flow_temp,
            adjusted_flow_temp=adjusted_flow_temp,
            safety_margin=safety_margin,
            unusual_weather=unusual_weather,
            defer_factor=defer_factor,
        )

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
from typing import Optional

from ..const import (
    DEFAULT_HEAT_LOSS_COEFFICIENT,
    KUEHNE_COEFFICIENT,
    KUEHNE_POWER,
    RADIATOR_POWER_COEFFICIENT,
    RADIATOR_RATED_DT,
    UFH_FLOW_REDUCTION_CONCRETE,
    UFH_FLOW_REDUCTION_TIMBER,
    UFH_MIN_FLOW_TEMP_CONCRETE,
    UFH_MIN_FLOW_TEMP_TIMBER,
)
from .climate_zones import ClimateZoneDetector

_LOGGER = logging.getLogger(__name__)


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
        temp_error = optimal_flow_temp - current_flow_temp
        offset_adjustment = temp_error / curve_sensitivity

        _LOGGER.debug(
            "Offset calculation: optimal=%.1f°C, current=%.1f°C, "
            "error=%.1f°C, sensitivity=%.1f -> offset=%.1f",
            optimal_flow_temp,
            current_flow_temp,
            temp_error,
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

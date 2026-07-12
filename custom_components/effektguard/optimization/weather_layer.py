"""Weather compensation: the flow temperature the emitters need for the current weather.

TODO: Rename this module to flow_temp_layer.py and classes to FlowTemp* for clarity.
      "Weather compensation" is confusing - this layer optimizes FLOW TEMPERATURE
      based on outdoor conditions, not weather forecasting. (Dec 19, 2025)

The flow temperature comes from the EN 442 emitter law (see utils/emitter.py), anchored either
on the emitters' rated output or on the system's design point. This layer then:

1. adds a climate-zone safety margin (latitude-derived, globally applicable),
2. converts the result into a curve OFFSET relative to what the pump is currently delivering,
3. defers to thermal reality by shedding weight when degree minutes show real thermal debt.

The offset is a TRIM on the pump's own heating curve, not a replacement for it: if the curve is
correctly tuned the correction is near zero, and it is bounded either way.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol

from homeassistant.util import dt as dt_util

from ..const import (
    DEFAULT_CURVE_SENSITIVITY,
    DEFAULT_DESIGN_FLOW_TEMP_RADIATOR,
    DEFAULT_DESIGN_FLOW_TEMP_UFH,
    DEFAULT_DESIGN_OUTDOOR_TEMP,
    DEFAULT_DESIGN_SPREAD,
    DEFAULT_HEAT_LOSS_COEFFICIENT,
    DEFAULT_WEATHER_COMPENSATION_WEIGHT,
    DHW_WEATHER_COOLDOWN_MINUTES,
    LAYER_WEIGHT_WEATHER_PREDICTION,
    RADIATOR_POWER_COEFFICIENT,
    RADIATOR_RATED_DT,
    UFH_POWER_COEFFICIENT,
    WEATHER_COMP_DEFER_DM_CRITICAL,
    WEATHER_COMP_DEFER_DM_LIGHT,
    WEATHER_COMP_DEFER_DM_MODERATE,
    WEATHER_COMP_DEFER_DM_SIGNIFICANT,
    WEATHER_COMP_DEFER_WEIGHT_CRITICAL,
    WEATHER_COMP_DEFER_WEIGHT_LIGHT,
    WEATHER_COMP_DEFER_WEIGHT_MODERATE,
    WEATHER_COMP_DEFER_WEIGHT_SIGNIFICANT,
    WEATHER_COMP_MAX_OFFSET,
    WEATHER_FORECAST_DROP_THRESHOLD,
    WEATHER_FORECAST_HORIZON,
    WEATHER_PREHEAT_OFFSET,
    WEATHER_INDOOR_COOLING_CONFIRMATION,
    WEATHER_WEIGHT_CAP,
)
from ..utils.emitter import en442_flow_temp
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
    - the EN 442 emitter law (rated-output anchor, or the system design point)
    - adaptive climate zones
    - weather learning (unusual pattern detection)
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
    method: str  # "en442_rated_output" or "en442_design_point"
    heating_type: str  # "radiator", "concrete_ufh", "timber_ufh", etc.
    confidence: float  # 0-1 confidence in calculation
    reasoning: str  # Explanation of calculation
    raw_design_point: Optional[float] = None  # EN 442 anchored on the system design point
    raw_rated_output: Optional[float] = None  # EN 442 anchored on the emitters' rated output


class WeatherCompensationCalculator:
    """Calculate optimal flow temperatures using validated mathematical formulas.

    One law - the EN 442 emitter characteristic equation - with two anchors:
    1. the emitters' rated output at DT50, when the installer has supplied it (preferred);
    2. otherwise the system's design point (design flow temperature at the design outdoor temp).

    Underfloor heating is handled by its own exponent (EN 1264, n ~ 1.1) and its own design flow
    temperature, not by subtracting a fixed amount from a radiator curve.
    """

    def __init__(
        self,
        heat_loss_coefficient: float = DEFAULT_HEAT_LOSS_COEFFICIENT,
        radiator_rated_output: Optional[float] = None,
        heating_type: str = "radiator",
        design_outdoor_temp: float = DEFAULT_DESIGN_OUTDOOR_TEMP,
        design_flow_temp: Optional[float] = None,
        design_spread: float = DEFAULT_DESIGN_SPREAD,
    ):
        """Initialize weather compensation calculator.

        Args:
            heat_loss_coefficient: Building heat loss in W/°C (typical 100-300)
            radiator_rated_output: Total rated emitter output at DT50 in Watts
            heating_type: "radiator", "concrete_ufh", "timber_ufh", "mixed"
            design_outdoor_temp: Dimensioning outdoor temperature (DUT/DVUT, °C)
            design_flow_temp: Supply temperature needed at the design outdoor temperature (°C).
                Defaults by emitter type, since an underfloor system is dimensioned far cooler
                than radiators.
            design_spread: Flow-return spread at the design load (°C)
        """
        self.heat_loss_coefficient = heat_loss_coefficient
        self.radiator_rated_output = radiator_rated_output
        self.heating_type = heating_type
        self.design_outdoor_temp = design_outdoor_temp
        self.design_spread = design_spread

        # Underfloor heating has its own emitter exponent (EN 1264, n ~ 1.1) and a far lower
        # design flow temperature. Its lower temperatures belong HERE, in the curve itself - not
        # as a fixed subtraction applied to a radiator curve afterwards.
        self.is_underfloor = heating_type in ("concrete_ufh", "timber_ufh", "timber")
        self.emitter_exponent = (
            UFH_POWER_COEFFICIENT if self.is_underfloor else RADIATOR_POWER_COEFFICIENT
        )
        if design_flow_temp is not None:
            self.design_flow_temp = design_flow_temp
        elif self.is_underfloor:
            self.design_flow_temp = DEFAULT_DESIGN_FLOW_TEMP_UFH
        else:
            self.design_flow_temp = DEFAULT_DESIGN_FLOW_TEMP_RADIATOR

        _LOGGER.debug(
            "WeatherCompensationCalculator initialized: type=%s, design %.1f°C @ %.1f°C outdoor, "
            "spread=%.1f°C, emitter exponent n=%.2f, HC=%.1f W/°C, rated_output=%s W",
            heating_type,
            self.design_flow_temp,
            self.design_outdoor_temp,
            self.design_spread,
            self.emitter_exponent,
            heat_loss_coefficient,
            radiator_rated_output,
        )

    def calculate_design_point_flow_temp(
        self,
        indoor_setpoint: float,
        outdoor_temp: float,
    ) -> float:
        """Flow temperature from the EN 442 emitter law, anchored on the system design point.

        The default path: it needs only quantities an installer knows (the dimensioning outdoor
        temperature, the supply temperature the system needs at it, the flow-return spread, and
        the emitter type), and by construction it reproduces the design flow temperature at the
        design outdoor temperature - so a correctly tuned pump curve gets a near-zero correction.

        Args:
            indoor_setpoint: Target indoor temperature (°C)
            outdoor_temp: Current outdoor temperature (°C)

        Returns:
            Required flow temperature (°C)
        """
        flow_temp = en442_flow_temp(
            indoor_setpoint=indoor_setpoint,
            outdoor_temp=outdoor_temp,
            design_outdoor_temp=self.design_outdoor_temp,
            design_flow_temp=self.design_flow_temp,
            design_spread=self.design_spread,
            emitter_exponent=self.emitter_exponent,
        )

        _LOGGER.debug(
            "EN 442 (design point): outdoor=%.1f°C, target=%.1f°C, design=%.1f°C@%.1f°C, "
            "spread=%.1f°C, n=%.2f -> flow=%.1f°C",
            outdoor_temp,
            indoor_setpoint,
            self.design_flow_temp,
            self.design_outdoor_temp,
            self.design_spread,
            self.emitter_exponent,
            flow_temp,
        )

        return flow_temp

    def calculate_rated_output_flow_temp(
        self,
        indoor_setpoint: float,
        outdoor_temp: float,
        flow_return_dt: float,
    ) -> Optional[float]:
        """The same EN 442 law, parameterised by the emitters' rated output instead.

        When the installer knows the total rated emitter output at ΔT50, the law can be anchored
        on the EN 442 rating point directly:

            ΔT = ΔT_N × (Φ / Φ_N) ** (1 / n)

        Preferred over the design-point form, because Φ_N is a measured nameplate figure. Same
        physics either way - both invert `Φ / Φ_N = (ΔT / ΔT_N) ** n`; only the anchor differs.

        Args:
            indoor_setpoint: Target indoor temperature (°C)
            outdoor_temp: Current outdoor temperature (°C)
            flow_return_dt: Flow-return spread at the design load (°C)

        Returns:
            Required flow temperature (°C), or None if the rated output is not configured.
        """
        if self.radiator_rated_output is None or self.radiator_rated_output <= 0:
            return None

        load = indoor_setpoint - outdoor_temp
        if load <= 0:
            return indoor_setpoint

        heat_demand = self.heat_loss_coefficient * load
        output_ratio = heat_demand / self.radiator_rated_output

        # The exponent is the EMITTER's: underfloor uses EN 1264's n ~ 1.1, not a radiator's 1.3.
        required_dt = RADIATOR_RATED_DT * (output_ratio ** (1.0 / self.emitter_exponent))

        flow_temp = indoor_setpoint + required_dt + (flow_return_dt / 2.0)

        _LOGGER.debug(
            "EN 442 (rated output): demand=%.0f W, rated=%.0f W, ratio=%.3f, n=%.2f, "
            "required ΔT=%.1f K -> flow=%.1f°C",
            heat_demand,
            self.radiator_rated_output,
            output_ratio,
            self.emitter_exponent,
            required_dt,
            flow_temp,
        )

        return flow_temp

    def calculate_optimal_flow_temp(
        self,
        indoor_setpoint: float,
        outdoor_temp: float,
        flow_return_dt: Optional[float] = None,
    ) -> FlowTempCalculation:
        """Flow temperature the emitters need, by the EN 442 emitter law.

        One law, two anchors: the emitters' rated output when the installer has supplied it,
        otherwise the system's design point.

        Args:
            indoor_setpoint: Target indoor temperature (°C)
            outdoor_temp: Current outdoor temperature (°C)
            flow_return_dt: Flow-return spread (°C). Defaults to the configured design spread.

        Returns:
            FlowTempCalculation with the required temperature and its reasoning.
        """
        spread = flow_return_dt if flow_return_dt is not None else self.design_spread

        raw_rated_output = self.calculate_rated_output_flow_temp(
            indoor_setpoint, outdoor_temp, spread
        )
        raw_design_point = self.calculate_design_point_flow_temp(indoor_setpoint, outdoor_temp)

        if raw_rated_output is not None:
            flow_temp = raw_rated_output
            method_used = "en442_rated_output"
            confidence = 0.95  # anchored on a measured nameplate figure
            detail = (
                f"EN 442 via rated output: {raw_rated_output:.1f}°C "
                f"(emitters {self.radiator_rated_output:.0f} W @ ΔT50, n={self.emitter_exponent})"
            )
        else:
            flow_temp = raw_design_point
            method_used = "en442_design_point"
            confidence = 0.9
            detail = (
                f"EN 442 via design point: {raw_design_point:.1f}°C "
                f"({self.design_flow_temp:.0f}°C @ {self.design_outdoor_temp:.0f}°C outdoor, "
                f"n={self.emitter_exponent})"
            )

        reasoning = "; ".join(
            [
                f"Outdoor: {outdoor_temp:.1f}°C, Indoor target: {indoor_setpoint:.1f}°C",
                detail,
            ]
        )

        return FlowTempCalculation(
            flow_temp=flow_temp,
            method=method_used,
            heating_type=self.heating_type,
            confidence=confidence,
            reasoning=reasoning,
            raw_design_point=raw_design_point,
            raw_rated_output=raw_rated_output,
        )

    def calculate_required_offset(
        self,
        optimal_flow_temp: float,
        current_flow_temp: float,
        curve_sensitivity: float = DEFAULT_CURVE_SENSITIVITY,
    ) -> float:
        """Curve offset needed to move the flow temperature to the calculated optimum.

        Weather compensation TRIMS the pump's own heating curve; it does not replace it. If the
        curve is correctly tuned this correction is near zero. Bounded so that a mis-configured
        design point can never command a large swing.

        Args:
            optimal_flow_temp: Target flow temperature from weather compensation (°C)
            current_flow_temp: Current actual flow temperature (°C)
            curve_sensitivity: Flow temp change per offset unit (°C/offset)

        Returns:
            Recommended heating curve offset adjustment (°C), bounded by WEATHER_COMP_MAX_OFFSET.
        """
        temp_deviation = optimal_flow_temp - current_flow_temp
        offset_adjustment = temp_deviation / curve_sensitivity

        bounded = max(-WEATHER_COMP_MAX_OFFSET, min(WEATHER_COMP_MAX_OFFSET, offset_adjustment))

        if bounded != offset_adjustment:
            _LOGGER.debug(
                "Weather compensation offset %.1f°C bounded to %.1f°C (optimal=%.1f°C, "
                "current=%.1f°C) - a correction this large means the pump's curve or the "
                "configured design point disagrees with reality",
                offset_adjustment,
                bounded,
                optimal_flow_temp,
                current_flow_temp,
            )
        else:
            _LOGGER.debug(
                "Offset calculation: optimal=%.1f°C, current=%.1f°C, "
                "error=%.1f°C, sensitivity=%.1f -> offset=%.1f",
                optimal_flow_temp,
                current_flow_temp,
                temp_deviation,
                curve_sensitivity,
                bounded,
            )

        return bounded


class AdaptiveClimateSystem:
    """Combine universal climate zones with adaptive weather learning.

    DESIGN PHILOSOPHY:
    - The emitter law (EN 442) works globally
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
    1. PRIMARY: Forecast shows ≥4°C drop in next 12h → +0.83°C gentle pre-heat
    2. CONFIRMATION: Indoor cooling ≥0.5°C/h → Confirms forecast, maintains offset
    3. MODERATION: Let SAFETY, COMFORT, EFFECT layers handle naturally via weighted aggregation
    """

    def __init__(
        self, thermal_mass: float = 1.0, forecast_horizon: float = WEATHER_FORECAST_HORIZON
    ):
        """Initialize weather prediction layer.

        Args:
            thermal_mass: Building thermal mass (0.5=light, 1.0=medium, 1.5=heavy)
            forecast_horizon: How far ahead to scan, in hours. From the thermal model, because it
                depends on what the house is built of. This layer took thermal_mass already and
                used it ONLY to scale its weight - it scanned a fixed twelve hours whatever the
                house was. A concrete slab reaches 63% of its response in about fourteen hours, and
                a 15 C fall spread over two days shows less than 4 C inside any twelve-hour window,
                so the drop never crossed the trigger and the pre-heat never fired at all.
        """
        self.thermal_mass = thermal_mass
        self.forecast_horizon = forecast_horizon

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
        forecast_hours = weather_data.forecast_hours[: int(self.forecast_horizon)]

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

        # PRIMARY TRIGGER: Forecast shows ≥3°C drop
        forecast_triggered = temp_drop <= WEATHER_FORECAST_DROP_THRESHOLD

        # CONFIRMATION TRIGGER: Indoor already cooling (confirms forecast)
        trend_rate = thermal_trend.get("rate_per_hour", 0.0)
        trend_confidence = thermal_trend.get("confidence", 0.0)

        indoor_cooling = (
            trend_rate <= WEATHER_INDOOR_COOLING_CONFIRMATION
            and trend_confidence > 0.4  # Sufficient data confidence
        )

        # Debug logging for weather pre-heat decisions
        _LOGGER.debug(
            "Weather Pre-heat check: outdoor=%.1f°C, forecast_min=%.1f°C, "
            "temp_drop=%.1f°C (threshold=%.1f°C), forecast_triggered=%s, "
            "trend_rate=%.2f°C/h (threshold=%.1f°C/h), trend_confidence=%.2f, "
            "indoor_cooling=%s",
            current_outdoor,
            min_temp,
            temp_drop,
            WEATHER_FORECAST_DROP_THRESHOLD,
            forecast_triggered,
            trend_rate,
            WEATHER_INDOOR_COOLING_CONFIRMATION,
            trend_confidence,
            indoor_cooling,
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
                trigger = (
                    f"Forecast {temp_drop:.1f}°C drop in {self.forecast_horizon:.0f}h (proactive)"
                )
            else:
                trigger = f"Indoor cooling {trend_rate:.2f}°C/h (reactive confirmation)"

            return WeatherLayerDecision(
                name="Weather Pre-heat",
                offset=WEATHER_PREHEAT_OFFSET,  # Constant +0.5°C (simple, predictable)
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

    Calculates the required flow temperature using:
    - the EN 442 emitter law (rated-output anchor, or the system design point)
    - adaptive climate zones (latitude-based, globally applicable)
    - weather learning (unusual pattern detection)

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
        dhw_heating_end: Optional[datetime] = None,
    ) -> WeatherCompensationLayerDecision:
        """Evaluate the weather compensation layer.

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data
            target_temp: Target indoor temperature (°C)
            enable_weather_compensation: Whether feature is enabled
            get_current_datetime: Callable returning current datetime (for testing)
            temp_lux_active: Whether DHW/lux heating is currently active
            dhw_heating_end: When DHW heating last stopped (for cooldown check)

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

        # Skip weather compensation during DHW cooldown period
        # After DHW stops, flow temperature remains elevated (45-55°C).
        # Wait for it to normalize before using weather compensation.
        if dhw_heating_end is not None:
            now = get_current_datetime() if get_current_datetime else dt_util.now()
            minutes_since_dhw = (now - dhw_heating_end).total_seconds() / 60
            if minutes_since_dhw < DHW_WEATHER_COOLDOWN_MINUTES:
                _LOGGER.debug(
                    "Weather comp skipped: DHW cooldown (%.1f min < %d min, flow=%.1f°C)",
                    minutes_since_dhw,
                    DHW_WEATHER_COOLDOWN_MINUTES,
                    nibe_state.flow_temp,
                )
                return WeatherCompensationLayerDecision(
                    name="Math WC",
                    offset=0.0,
                    weight=0.0,
                    reason=f"DHW cooldown ({minutes_since_dhw:.0f}/{DHW_WEATHER_COOLDOWN_MINUTES}min)",
                )

        if not weather_data or not weather_data.forecast_hours:
            return WeatherCompensationLayerDecision(
                name="Math WC", offset=0.0, weight=0.0, reason="No weather data"
            )

        current_outdoor = nibe_state.outdoor_temp
        current_flow = nibe_state.flow_temp

        # Flow temperature the emitters actually need, by the EN 442 emitter law
        flow_calc = self.weather_comp.calculate_optimal_flow_temp(
            indoor_setpoint=target_temp,
            outdoor_temp=current_outdoor,
        )

        # Adaptive climate system safety adjustments
        # Replaces hardcoded Swedish thresholds with universal climate zones + learning
        unusual_weather = False
        unusual_severity = 0.0

        # Check for unusual weather patterns if weather learner available
        if self.weather_learner and weather_data.forecast_hours:
            try:
                current_date = get_current_datetime() if get_current_datetime else dt_util.now()

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

        # The safety margin is an ASYMMETRIC TOLERANCE, not an addition to the setpoint.
        #
        # [required, required + margin] is acceptable: inside it the curve is left alone. Below
        # it the curve is running cold and is pulled up to what the emitter law demands. Above it
        # the curve is pulled back down to the top of the band, never below.
        #
        # Adding the margin to the setpoint instead makes the correction strictly positive at
        # every outdoor temperature, so a perfectly tuned curve is permanently told to add heat -
        # a DC bias, which is the same defect as a permanent setback with the sign flipped. The
        # margin means the curve MAY run warm in a hard winter, not that it must.
        required_flow = flow_calc.flow_temp
        if current_flow < required_flow:
            adjusted_flow_temp = required_flow
        elif current_flow > required_flow + safety_margin:
            adjusted_flow_temp = required_flow + safety_margin
        else:
            adjusted_flow_temp = current_flow

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

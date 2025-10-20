#!/usr/bin/env python3
"""EffektGuard Decision Engine Scenario Tester

A standalone utility for testing and optimizing multi-layer decision engine behavior
before deploying changes to production code.

WHAT IT DOES:
-------------
Simulates the decision engine's multi-layer voting system with different scenarios:
- Safety layer: Absolute temperature limits
- Emergency layer: Thermal debt (degree minutes) recovery
- Proactive layer: Prevent thermal debt accumulation
- Effect layer: Peak protection (15-minute effect tariff)
- Weather compensation: Mathematical flow temperature optimization
- Price layer: Spot price optimization (CHEAP/NORMAL/EXPENSIVE/PEAK)
- Comfort layer: Temperature error correction

Each layer votes with an offset (°C) and weight (0.0-1.0). Final decision is weighted
aggregation. Critical layers (weight 1.0) override all others.

USE CASES:
----------
1. Test new layer weights before code changes
2. Verify behavior under extreme conditions
3. Optimize offset values for different classifications
4. Understand layer interactions and priorities
5. Validate safety mechanisms

USAGE EXAMPLES:
---------------
# Run all built-in scenarios with current code defaults:
    python3 scripts/test_decision_scenarios.py

# Test specific scenario:
    python3 scripts/test_decision_scenarios.py --scenario negative_price

# Override specific conditions:
    python3 scripts/test_decision_scenarios.py --scenario custom \\
        --indoor 22.0 --outdoor 7.5 --dm -161 --flow-temp 35.0 --price -10.0

# Test with different layer weights:
    python3 scripts/test_decision_scenarios.py --scenario custom \\
        --indoor 21.0 --outdoor 7.5 --dm -300 --flow-temp 30.0 --price 45.0 \\
        --price-weight 0.85 --weather-comp-weight 0.40

# Tune proactive Z1 weight:
    python3 scripts/test_decision_scenarios.py --scenario negative_price \\
        --proactive-z1-weight 0.5

CURRENT PRODUCTION VALUES (from const.py, as of 2025-10-18):
------------------------------------------------------------
Layer Weights:
  - LAYER_WEIGHT_SAFETY: 1.0 (absolute priority)
  - LAYER_WEIGHT_EMERGENCY: 0.8 (high priority, DM beyond expected)
  - LAYER_WEIGHT_PRICE: 0.75 (strong influence)
  - LAYER_WEIGHT_WEATHER_PREDICTION: 0.49 (moderate influence)
  - LAYER_WEIGHT_PROACTIVE_MAX: 0.6 (Zone 3)
  - LAYER_WEIGHT_PROACTIVE_MIN: 0.3 (Zone 1)
  - LAYER_WEIGHT_COMFORT_MAX: 0.5
  - LAYER_WEIGHT_COMFORT_MIN: 0.2

Price Offsets (from price_analyzer.py):
  - CHEAP: +3.0°C (charge thermal battery!)
  - NORMAL: 0.0°C (maintain)
  - EXPENSIVE: -1.0°C (conserve, x1.5 during daytime)
  - PEAK: -2.0°C (minimize, x1.5 during daytime)

Effect/Peak Offsets (from decision_engine.py):
  - CRITICAL (at peak): -3.0°C @ weight 1.0
  - PREDICTIVE (approaching): -1.5°C @ weight 0.8
  - WARNING (rising demand): -1.0°C @ weight 0.7
  - WARNING (stable/falling): -0.5°C @ weight 0.6

Temperature Targets:
  - Default indoor: 21.0°C
  - Tolerance: 5 (1-10 scale)
  - Safe range: 18-24°C

AUTHORS:
--------
EffektGuard Development Team
https://github.com/enoch85/EffektGuard

LICENSE:
--------
MIT License - See LICENSE file for details
"""

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum

# Import constants from production code - single source of truth
sys.path.insert(0, "/workspaces/EffektGuard/custom_components/effektguard")
from const import (
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
    DM_THRESHOLD_ABSOLUTE_MAX,
    DEFAULT_TARGET_TEMP,
    EFFECT_MARGIN_PREDICTIVE,
    EFFECT_MARGIN_WARNING,
    EFFECT_MARGIN_WATCH,
    EFFECT_OFFSET_CRITICAL,
    EFFECT_OFFSET_PREDICTIVE,
    EFFECT_OFFSET_WARNING_RISING,
    EFFECT_OFFSET_WARNING_STABLE,
    EFFECT_WEIGHT_CRITICAL,
    EFFECT_WEIGHT_PREDICTIVE,
    EFFECT_WEIGHT_WARNING_RISING,
    EFFECT_WEIGHT_WARNING_STABLE,
    LAYER_WEIGHT_COMFORT_MAX,
    LAYER_WEIGHT_COMFORT_MIN,
    LAYER_WEIGHT_EMERGENCY,
    LAYER_WEIGHT_PRICE,
    LAYER_WEIGHT_PROACTIVE_MAX,
    LAYER_WEIGHT_PROACTIVE_MIN,
    LAYER_WEIGHT_SAFETY,
    LAYER_WEIGHT_WEATHER_PREDICTION,
    MAX_TEMP_LIMIT,
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
    SAFETY_EMERGENCY_OFFSET,
    TOLERANCE_RANGE_MULTIPLIER,
    WARNING_CAUTION_OFFSET,
    WARNING_CAUTION_WEIGHT,
    WARNING_DEVIATION_DIVISOR_MODERATE,
    WARNING_DEVIATION_DIVISOR_SEVERE,
    WARNING_DEVIATION_THRESHOLD,
    WARNING_OFFSET_MAX_MODERATE,
    WARNING_OFFSET_MAX_SEVERE,
    WARNING_OFFSET_MIN_MODERATE,
    WARNING_OFFSET_MIN_SEVERE,
    WEATHER_COMP_DEFER_DM_CRITICAL,
    WEATHER_COMP_DEFER_DM_LIGHT,
    WEATHER_COMP_DEFER_DM_MODERATE,
    WEATHER_COMP_DEFER_DM_SIGNIFICANT,
    WEATHER_COMP_DEFER_WEIGHT_CRITICAL,
    WEATHER_COMP_DEFER_WEIGHT_LIGHT,
    WEATHER_COMP_DEFER_WEIGHT_MODERATE,
    WEATHER_COMP_DEFER_WEIGHT_SIGNIFICANT,
    PRICE_TOLERANCE_DIVISOR,
    COMFORT_DEAD_ZONE,
    COMFORT_CORRECTION_MULT,
)

# Load climate zone constants that climate_zones.py needs
from const import (
    CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG,
    CLIMATE_ZONE_VERY_COLD_WINTER_AVG,
    CLIMATE_ZONE_COLD_WINTER_AVG,
    CLIMATE_ZONE_MODERATE_COLD_WINTER_AVG,
    CLIMATE_ZONE_STANDARD_WINTER_AVG,
)

# Import ClimateZoneDetector directly using importlib to bypass optimization/__init__.py
# Inject the const values into the module's globals before execution
spec = importlib.util.spec_from_file_location(
    "climate_zones",
    "/workspaces/EffektGuard/custom_components/effektguard/optimization/climate_zones.py",
)
climate_zones_module = importlib.util.module_from_spec(spec)

# Create a fake package structure so relative imports in climate_zones.py work
import types

fake_effektguard = types.ModuleType("effektguard")
fake_effektguard.__path__ = []
sys.modules["effektguard"] = fake_effektguard
sys.modules["effektguard.const"] = sys.modules["const"]

fake_optimization = types.ModuleType("effektguard.optimization")
fake_optimization.__path__ = []
sys.modules["effektguard.optimization"] = fake_optimization

climate_zones_module.__package__ = "effektguard.optimization"
sys.modules["effektguard.optimization.climate_zones"] = climate_zones_module

spec.loader.exec_module(climate_zones_module)
ClimateZoneDetector = climate_zones_module.ClimateZoneDetector


class QuarterClassification(Enum):
    """Price classification for 15-minute quarters."""

    CHEAP = "CHEAP"
    NORMAL = "NORMAL"
    EXPENSIVE = "EXPENSIVE"
    PEAK = "PEAK"


@dataclass
class MockNibeState:
    """Mock NIBE state for testing."""

    indoor_temp: float
    outdoor_temp: float
    degree_minutes: float
    flow_temp: float
    power_kw: float = 3.5


@dataclass
class MockPriceData:
    """Mock price data for testing."""

    current_price: float
    classification: QuarterClassification
    is_daytime: bool = True


@dataclass
class MockPeakData:
    """Mock effect tariff peak data for testing."""

    current_peak: float
    current_power: float
    current_quarter: int = 50  # Mid-day quarter


@dataclass
class MockWeatherData:
    """Mock weather data for testing."""

    current_temp: float
    forecast_min: float
    forecast_hours: list = None

    def __post_init__(self):
        if self.forecast_hours is None:
            self.forecast_hours = []


@dataclass
class LayerVote:
    """Individual layer vote."""

    name: str
    offset: float
    weight: float
    reason: str

    @property
    def weighted_value(self) -> float:
        return self.offset * self.weight


class ScenarioTester:
    """Test decision engine scenarios."""

    def __init__(
        self,
        price_weight: float = LAYER_WEIGHT_PRICE,
        weather_comp_weight: float = LAYER_WEIGHT_WEATHER_PREDICTION,
        emergency_weight: float = LAYER_WEIGHT_EMERGENCY,
        proactive_weight_z1: float = LAYER_WEIGHT_PROACTIVE_MIN,
        proactive_weight_z2: float = PROACTIVE_ZONE2_WEIGHT,
        proactive_weight_z3: float = LAYER_WEIGHT_PROACTIVE_MAX,
        comfort_weight_min: float = LAYER_WEIGHT_COMFORT_MIN,
        comfort_weight_max: float = LAYER_WEIGHT_COMFORT_MAX,
        cheap_offset: float = 3.0,  # From price_analyzer.py
        normal_offset: float = 0.0,
        expensive_offset: float = -1.0,
        peak_offset: float = -2.0,
    ):
        """Initialize tester with configurable parameters.

        Args:
            price_weight: Price layer weight (default: 0.75)
            weather_comp_weight: Weather compensation layer weight (default: 0.49)
            emergency_weight: Emergency layer weight (default: 0.8)
            proactive_weight_z1: Proactive Zone 1 weight (default: 0.3)
            proactive_weight_z2: Proactive Zone 2 weight (default: 0.4)
            proactive_weight_z3: Proactive Zone 3 weight (default: 0.6)
            comfort_weight_min: Comfort layer minimum weight (default: 0.2)
            comfort_weight_max: Comfort layer maximum weight (default: 0.5)
            cheap_offset: CHEAP price base offset °C (default: +3.0)
            normal_offset: NORMAL price base offset °C (default: 0.0)
            expensive_offset: EXPENSIVE price base offset °C (default: -1.0)
            peak_offset: PEAK price base offset °C (default: -2.0)
        """
        self.tolerance = 5  # User-facing 1-10 scale (5 = default balanced)
        self.config = {
            "target_indoor_temp": 22.0,  # Default to 22°C for testing
            "tolerance": self.tolerance,  # Use proper 1-10 scale
            "enable_weather_compensation": True,
            "weather_compensation_weight": weather_comp_weight,
            "latitude": 55.60,  # Malmö (Southern Sweden)
        }
        self.tolerance_range = self.tolerance * TOLERANCE_RANGE_MULTIPLIER

        # Initialize climate zone detector for Malmö/Southern Sweden
        self.climate_detector = ClimateZoneDetector(latitude=55.60)

        # Configurable layer weights
        self.price_weight = price_weight
        self.weather_comp_weight = weather_comp_weight
        self.emergency_weight = emergency_weight
        self.proactive_weight_z1 = proactive_weight_z1
        self.proactive_weight_z2 = proactive_weight_z2
        self.proactive_weight_z3 = proactive_weight_z3
        self.comfort_weight_min = comfort_weight_min
        self.comfort_weight_max = comfort_weight_max

        # Configurable price offsets
        self.price_offsets = {
            QuarterClassification.CHEAP: cheap_offset,
            QuarterClassification.NORMAL: normal_offset,
            QuarterClassification.EXPENSIVE: expensive_offset,
            QuarterClassification.PEAK: peak_offset,
        }

    def calculate_safety_layer(self, nibe_state: MockNibeState) -> LayerVote:
        """Calculate safety layer vote."""
        if nibe_state.indoor_temp < MIN_TEMP_LIMIT:
            return LayerVote(
                "Safety",
                offset=5.0,
                weight=1.0,
                reason=f"SAFETY: Too cold ({nibe_state.indoor_temp:.1f}°C)",
            )
        elif nibe_state.indoor_temp > MAX_TEMP_LIMIT:
            return LayerVote(
                "Safety",
                offset=-5.0,
                weight=1.0,
                reason=f"SAFETY: Too hot ({nibe_state.indoor_temp:.1f}°C)",
            )
        return LayerVote("Safety", offset=0.0, weight=0.0, reason="Within safe limits")

    def calculate_emergency_layer(self, nibe_state: MockNibeState) -> LayerVote:
        """Calculate emergency layer vote (climate-aware).

        Mirrors production code from decision_engine.py._emergency_layer()
        """
        dm = nibe_state.degree_minutes
        outdoor = nibe_state.outdoor_temp

        # Absolute maximum (hardcoded in production)
        if dm <= DM_THRESHOLD_ABSOLUTE_MAX:  # -1500
            return LayerVote(
                "Emergency",
                offset=5.0,
                weight=1.0,
                reason=f"ABSOLUTE MAX: DM {dm:.0f} at safety limit -1500 - EMERGENCY",
            )

        # Get climate-aware expected DM ranges for current outdoor temperature
        dm_range = self.climate_detector.get_expected_dm_range(outdoor)
        expected_normal = dm_range["normal_min"]  # Expected minimum for normal operation
        expected_warning = dm_range["warning"]  # Warning threshold (climate-aware)

        # MULTI-TIER CLIMATE-AWARE CRITICAL INTERVENTION (Oct 19, 2025 redesign)
        # Calculate tier thresholds dynamically based on climate-aware WARNING threshold
        # All values from const.py - single source of truth
        margin_to_limit = dm - DM_THRESHOLD_ABSOLUTE_MAX

        # Calculate climate-aware tier thresholds
        warning_threshold = expected_warning
        t1_threshold = warning_threshold - DM_CRITICAL_T1_MARGIN  # At WARNING threshold
        t2_threshold = warning_threshold - DM_CRITICAL_T2_MARGIN  # WARNING + 200 DM
        t3_threshold = max(
            warning_threshold - DM_CRITICAL_T3_MARGIN,  # WARNING + 400 DM
            DM_CRITICAL_T3_MAX,  # Capped at -1450 (50 DM from absolute max)
        )

        # CRITICAL TIER 3: Most severe intervention (within 50-300 DM of absolute maximum)
        if dm <= t3_threshold:
            return LayerVote(
                "Emergency",
                offset=DM_CRITICAL_T3_OFFSET,
                weight=DM_CRITICAL_T3_WEIGHT,
                reason=f"CRITICAL T3: DM {dm:.0f} near absolute max (threshold: {t3_threshold:.0f}, margin: {margin_to_limit:.0f})",
            )

        # CRITICAL TIER 2: Severe thermal debt - strong recovery before reaching T3
        if dm <= t2_threshold:
            return LayerVote(
                "Emergency",
                offset=DM_CRITICAL_T2_OFFSET,
                weight=DM_CRITICAL_T2_WEIGHT,
                reason=f"CRITICAL T2: DM {dm:.0f} approaching T3 (threshold: {t2_threshold:.0f}, margin: {margin_to_limit:.0f})",
            )

        # CRITICAL TIER 1: Serious thermal debt - prevent escalation to T2
        # Triggers at climate-aware WARNING threshold (where thermal debt becomes abnormal)
        if dm <= t1_threshold:
            return LayerVote(
                "Emergency",
                offset=DM_CRITICAL_T1_OFFSET,
                weight=DM_CRITICAL_T1_WEIGHT,
                reason=f"CRITICAL T1: DM {dm:.0f} beyond expected for {outdoor:.1f}°C (threshold: {t1_threshold:.0f})",
            )

        # WARNING: DM beyond expected range (strengthened Oct 19, 2025)
        if dm < expected_warning:
            deviation = expected_warning - dm

            # Strengthened offset calculation
            if deviation > WARNING_DEVIATION_THRESHOLD:  # Severe deviation
                offset = min(
                    WARNING_OFFSET_MAX_SEVERE,
                    WARNING_OFFSET_MIN_SEVERE + (deviation / WARNING_DEVIATION_DIVISOR_SEVERE),
                )
            else:  # Moderate deviation
                offset = min(
                    WARNING_OFFSET_MAX_MODERATE,
                    WARNING_OFFSET_MIN_MODERATE + (deviation / WARNING_DEVIATION_DIVISOR_MODERATE),
                )

            percent_beyond = abs(deviation / expected_warning) if expected_warning else 0

            return LayerVote(
                "Emergency",
                offset=offset,
                weight=self.emergency_weight,
                reason=f"WARNING: DM {dm:.0f} beyond expected for {outdoor:.1f}°C (expected: {expected_normal:.0f}, {percent_beyond:.0%} over)",
            )

        # CAUTION: Approaching expected limits
        elif dm < expected_normal:
            return LayerVote(
                "Emergency",
                offset=WARNING_CAUTION_OFFSET,
                weight=WARNING_CAUTION_WEIGHT,
                reason=f"CAUTION: DM {dm:.0f} at {outdoor:.1f}°C - monitoring",
            )

        return LayerVote(
            "Emergency",
            offset=0.0,
            weight=0.0,
            reason=f"Thermal debt OK (DM: {dm:.0f} at {outdoor:.1f}°C)",
        )

    def calculate_proactive_layer(self, nibe_state: MockNibeState) -> LayerVote:
        """Calculate proactive debt prevention layer (updated Oct 19, 2025)."""
        dm = nibe_state.degree_minutes
        outdoor = nibe_state.outdoor_temp

        # Get climate-aware expected DM ranges (same as emergency layer)
        dm_range = self.climate_detector.get_expected_dm_range(outdoor)
        expected_normal = dm_range["normal_min"]  # Expected minimum for normal operation
        expected_warning = dm_range["warning"]  # Warning threshold (climate-aware)

        # Zone thresholds (percentages of expected_normal)
        zone1 = expected_normal * PROACTIVE_ZONE1_THRESHOLD_PERCENT
        zone2 = expected_normal * PROACTIVE_ZONE2_THRESHOLD_PERCENT
        zone3 = expected_normal * PROACTIVE_ZONE3_THRESHOLD_PERCENT

        # NEW ZONES (Oct 19, 2025): Fill gap between Proactive and WARNING/CRITICAL
        zone4_start = expected_normal * PROACTIVE_ZONE4_THRESHOLD_PERCENT
        zone4_end = expected_warning
        zone5_end = expected_warning * PROACTIVE_ZONE5_THRESHOLD_PERCENT

        if zone2 < dm <= zone1:
            return LayerVote(
                "Proactive Z1",
                offset=PROACTIVE_ZONE1_OFFSET,
                weight=self.proactive_weight_z1,
                reason=f"DM {dm:.0f} (threshold: {zone1:.0f}), gentle heating prevents debt",
            )
        elif zone3 < dm <= zone2:
            return LayerVote(
                "Proactive Z2",
                offset=PROACTIVE_ZONE2_OFFSET,
                weight=self.proactive_weight_z2,
                reason=f"DM {dm:.0f} (threshold: {zone2:.0f}), boost recovery speed",
            )
        elif zone4_start < dm <= zone3:
            deficit_severity = (zone3 - dm) / (zone3 - expected_normal)
            offset = PROACTIVE_ZONE3_OFFSET_MIN + (
                min(deficit_severity, 1.0) * PROACTIVE_ZONE3_OFFSET_RANGE
            )
            return LayerVote(
                "Proactive Z3",
                offset=offset,
                weight=self.proactive_weight_z3,
                reason=f"DM {dm:.0f} (threshold: {zone3:.0f}), prevent deeper debt (severity: {deficit_severity:.2f})",
            )
        elif zone4_end < dm <= zone4_start:
            # NEW ZONE 4: Bridge gap to WARNING (Oct 19, 2025)
            return LayerVote(
                "Proactive Z4",
                offset=PROACTIVE_ZONE4_OFFSET,
                weight=PROACTIVE_ZONE4_WEIGHT,
                reason=f"DM {dm:.0f} extended debt prevention (bridge to WARNING)",
            )
        elif zone5_end < dm <= zone4_end:
            # NEW ZONE 5: Strong intervention before WARNING/CRITICAL (Oct 19, 2025)
            return LayerVote(
                "Proactive Z5",
                offset=PROACTIVE_ZONE5_OFFSET,
                weight=PROACTIVE_ZONE5_WEIGHT,
                reason=f"DM {dm:.0f} strong preventative recovery (pre-WARNING)",
            )

        return LayerVote("Proactive", offset=0.0, weight=0.0, reason="Not needed")

    def calculate_effect_layer(
        self, nibe_state: MockNibeState, peak_data: MockPeakData
    ) -> LayerVote:
        """Calculate effect tariff protection layer.

        Uses constants from const.py - single source of truth.
        """
        current_power = peak_data.current_power
        current_peak = peak_data.current_peak
        margin = current_peak - current_power

        # CRITICAL: Already at or exceeding peak
        if current_power >= current_peak:
            return LayerVote(
                "Effect/Peak",
                offset=EFFECT_OFFSET_CRITICAL,
                weight=EFFECT_WEIGHT_CRITICAL,
                reason=f"CRITICAL ({current_power:.1f}/{current_peak:.1f} kW, Q{peak_data.current_quarter})",
            )

        # PREDICTIVE: Will approach peak (margin < EFFECT_MARGIN_PREDICTIVE)
        elif margin < EFFECT_MARGIN_PREDICTIVE:
            return LayerVote(
                "Effect/Peak",
                offset=EFFECT_OFFSET_PREDICTIVE,
                weight=EFFECT_WEIGHT_PREDICTIVE,
                reason=f"PREDICTIVE avoidance (predicted {current_power:.1f} kW, Q{peak_data.current_quarter})",
            )

        # WARNING: Close to peak (margin < EFFECT_MARGIN_WARNING)
        elif margin < EFFECT_MARGIN_WARNING:
            return LayerVote(
                "Effect/Peak",
                offset=EFFECT_OFFSET_WARNING_RISING,
                weight=EFFECT_WEIGHT_WARNING_RISING,
                reason=f"WARNING + demand rising (Q{peak_data.current_quarter})",
            )

        # WATCH: Margin exists but monitoring (margin < EFFECT_MARGIN_WATCH)
        elif margin < EFFECT_MARGIN_WATCH:
            return LayerVote(
                "Effect/Peak",
                offset=EFFECT_OFFSET_WARNING_STABLE,
                weight=EFFECT_WEIGHT_WARNING_STABLE,
                reason=f"WATCH (margin {margin:.1f} kW, Q{peak_data.current_quarter})",
            )

        # Safe margin
        return LayerVote(
            "Effect/Peak",
            offset=0.0,
            weight=0.0,
            reason=f"Safe margin ({current_power:.1f}/{current_peak:.1f} kW)",
        )

    def calculate_prediction_layer(self, nibe_state: MockNibeState) -> LayerVote:
        """Calculate prediction layer (Phase 6 - learned pre-heating).

        Note: This is a placeholder. Real implementation requires thermal predictor
        with historical data and learned building characteristics.
        """
        # Simplified: Not implemented in standalone tester
        return LayerVote(
            "Prediction", offset=0.0, weight=0.0, reason="Not available (requires learning)"
        )

    def calculate_weather_layer(
        self, nibe_state: MockNibeState, weather_data: MockWeatherData
    ) -> LayerVote:
        """Calculate weather prediction layer (simple pre-heating)."""
        if not weather_data or not weather_data.forecast_hours:
            return LayerVote("Weather", offset=0.0, weight=0.0, reason="No forecast")

        # Simple logic: check for significant temperature drop
        current_outdoor = nibe_state.outdoor_temp
        min_forecast = (
            min(weather_data.forecast_hours) if weather_data.forecast_hours else current_outdoor
        )
        temp_drop = min_forecast - current_outdoor

        if temp_drop < -3.0:
            # Significant cold snap coming
            # Scale offset based on drop magnitude
            offset = min(abs(temp_drop) / 5.0 * 2.0, 2.5)
            return LayerVote(
                "Weather",
                offset=offset,
                weight=0.7,
                reason=f"Pre-heat: {temp_drop:.1f}°C drop forecast",
            )

        return LayerVote("Weather", offset=0.0, weight=0.0, reason="No pre-heating needed")

    def calculate_weather_comp_layer(self, nibe_state: MockNibeState) -> LayerVote:
        """Calculate weather compensation layer (simplified)."""
        outdoor = nibe_state.outdoor_temp
        indoor = self.config["target_indoor_temp"]
        current_flow = nibe_state.flow_temp

        # André Kühne's formula (simplified)
        # TFlow = 2.55 × (HC × (Tset - Tout))^0.78 + Tset
        heat_loss_coeff = 180.0  # W/°C for typical house
        temp_diff = indoor - outdoor

        if temp_diff <= 0:
            optimal_flow = indoor
        else:
            heat_loss_kw = heat_loss_coeff / 1000.0  # Convert W/°C to kW/K
            heat_term = heat_loss_kw * temp_diff
            optimal_flow = 2.55 * (heat_term**0.78) + indoor

        # Calculate required offset
        curve_sensitivity = 1.5
        required_offset = (optimal_flow - current_flow) / curve_sensitivity

        # Apply weather comp deferral based on thermal debt (Conservative strategy)
        # Defer weather compensation when thermal debt exists, allowing thermal reality
        # (DM + comfort + proactive) to override outdoor temperature optimization
        degree_minutes = nibe_state.degree_minutes
        base_weight = self.weather_comp_weight

        if degree_minutes < WEATHER_COMP_DEFER_DM_CRITICAL:
            # Critical debt: 39% reduction (0.49 → 0.30)
            defer_factor = WEATHER_COMP_DEFER_WEIGHT_CRITICAL / LAYER_WEIGHT_WEATHER_PREDICTION
            defer_note = f"; Deferred: Critical debt (DM {degree_minutes:.0f})"
        elif degree_minutes < WEATHER_COMP_DEFER_DM_SIGNIFICANT:
            # Significant debt: 29% reduction (0.49 → 0.35)
            defer_factor = WEATHER_COMP_DEFER_WEIGHT_SIGNIFICANT / LAYER_WEIGHT_WEATHER_PREDICTION
            defer_note = f"; Deferred: Significant debt (DM {degree_minutes:.0f})"
        elif degree_minutes < WEATHER_COMP_DEFER_DM_MODERATE:
            # Moderate debt: 18% reduction (0.49 → 0.40)
            defer_factor = WEATHER_COMP_DEFER_WEIGHT_MODERATE / LAYER_WEIGHT_WEATHER_PREDICTION
            defer_note = f"; Deferred: Moderate debt (DM {degree_minutes:.0f})"
        elif degree_minutes < WEATHER_COMP_DEFER_DM_LIGHT:
            # Light debt: 8% reduction (0.49 → 0.45)
            defer_factor = WEATHER_COMP_DEFER_WEIGHT_LIGHT / LAYER_WEIGHT_WEATHER_PREDICTION
            defer_note = f"; Deferred: Light debt (DM {degree_minutes:.0f})"
        else:
            # No debt: full weather comp weight
            defer_factor = 1.0
            defer_note = ""

        final_weight = base_weight * defer_factor

        return LayerVote(
            "Weather Comp",
            offset=required_offset,
            weight=final_weight,
            reason=f"Optimal {optimal_flow:.1f}°C vs {current_flow:.1f}°C{defer_note}",
        )

    def calculate_price_layer(self, price_data: MockPriceData) -> LayerVote:
        """Calculate price layer vote."""
        # Use configurable price offsets
        offset = self.price_offsets[price_data.classification]

        # Adjust for tolerance (1-10 scale, 5 = default = factor 1.0)
        # Higher tolerance = more aggressive optimization
        tolerance_factor = self.tolerance / PRICE_TOLERANCE_DIVISOR  # 0.2-2.0 range
        adjusted_offset = offset * tolerance_factor

        # Extra boost for negative prices
        price_note = ""
        if price_data.current_price < 0:
            price_note = f" (NEGATIVE: {price_data.current_price:.1f} öre!)"

        return LayerVote(
            "Price",
            offset=adjusted_offset,
            weight=self.price_weight,
            reason=f"{price_data.classification.name}{price_note}",
        )

    def calculate_comfort_layer(self, nibe_state: MockNibeState) -> LayerVote:
        """Calculate comfort layer vote."""
        temp_error = nibe_state.indoor_temp - self.config["target_indoor_temp"]
        dead_zone = COMFORT_DEAD_ZONE
        tolerance = self.tolerance_range

        if abs(temp_error) < dead_zone:
            return LayerVote("Comfort", offset=0.0, weight=0.0, reason="At target")
        elif abs(temp_error) < tolerance:
            # Gentle steering
            offset = -temp_error * COMFORT_CORRECTION_MULT
            return LayerVote(
                "Comfort",
                offset=offset,
                weight=self.comfort_weight_min,
                reason=f"Gentle steer ({temp_error:+.1f}°C from target)",
            )
        elif temp_error > tolerance:
            return LayerVote(
                "Comfort",
                offset=-1.5,
                weight=self.comfort_weight_max,
                reason=f"Too hot ({temp_error:+.1f}°C over)",
            )
        else:
            return LayerVote(
                "Comfort",
                offset=+1.5,
                weight=self.comfort_weight_max,
                reason=f"Too cold ({temp_error:+.1f}°C under)",
            )

    def aggregate_layers(self, layers: list[LayerVote]) -> tuple[float, list[LayerVote]]:
        """Aggregate layer votes into final offset.

        Oct 19, 2025: Peak-aware emergency mode
        When emergency layer is critical AND peak protection is strongly active,
        use minimal offset to prevent DM worsening without creating new peaks.
        """
        # Filter active layers
        active = [l for l in layers if l.weight > 0]

        if not active:
            return 0.0, active

        # Check for critical overrides (weight = 1.0)
        critical = [l for l in active if l.weight >= 1.0]
        if critical:
            # Find safety layer (index 0) and emergency layer (index 1)
            safety_layer = layers[0] if len(layers) > 0 else None
            emergency_layer = layers[1] if len(layers) > 1 else None
            effect_layer = layers[3] if len(layers) > 3 else None  # Effect is index 3

            # Safety always wins
            if safety_layer and safety_layer.weight >= 1.0:
                return safety_layer.offset, active

            # Emergency layer with peak-aware logic
            if emergency_layer and emergency_layer.weight >= 1.0:
                emergency_offset = emergency_layer.offset

                # Check if effect/peak layer is strongly negative
                if (
                    effect_layer
                    and effect_layer.offset < PEAK_AWARE_EFFECT_THRESHOLD
                    and effect_layer.weight > PEAK_AWARE_EFFECT_WEIGHT_MIN
                ):
                    # Peak protection is strongly active - use tier-appropriate minimal offset
                    # Scale minimal offset based on emergency severity
                    if emergency_offset >= DM_CRITICAL_T3_OFFSET:  # T3 (-1200)
                        minimal_offset = DM_CRITICAL_T3_PEAK_AWARE_OFFSET
                    elif emergency_offset >= DM_CRITICAL_T2_OFFSET:  # T2 (-1000)
                        minimal_offset = DM_CRITICAL_T2_PEAK_AWARE_OFFSET
                    else:  # T1 (-800)
                        minimal_offset = DM_CRITICAL_T1_PEAK_AWARE_OFFSET

                    print(
                        f"  ℹ️  Peak-aware emergency: reducing offset from {emergency_offset:.2f} to {minimal_offset:.2f}"
                    )
                    return minimal_offset, active
                else:
                    # No strong peak protection - apply full emergency offset
                    return emergency_offset, active

            # Take first critical layer as fallback
            return critical[0].offset, active

        # Weighted average
        total_weight = sum(l.weight for l in active)
        weighted_sum = sum(l.weighted_value for l in active)

        return weighted_sum / total_weight, active

    def test_scenario(
        self,
        name: str,
        nibe_state: MockNibeState,
        price_data: MockPriceData,
        peak_data: MockPeakData = None,
        weather_data: MockWeatherData = None,
    ) -> dict[str, Any]:
        """Test a specific scenario."""
        print(f"\n{'=' * 80}")
        print(f"SCENARIO: {name}")
        print(f"{'=' * 80}")

        # Print conditions
        print(f"\n📊 CONDITIONS:")
        print(
            f"  Indoor: {nibe_state.indoor_temp:.1f}°C (target: {self.config['target_indoor_temp']:.1f}°C)"
        )
        print(f"  Outdoor: {nibe_state.outdoor_temp:.1f}°C")
        print(f"  Degree Minutes: {nibe_state.degree_minutes:.0f}")
        print(f"  Flow Temp: {nibe_state.flow_temp:.1f}°C")
        print(f"  Price: {price_data.current_price:.1f} öre/kWh ({price_data.classification.name})")
        if price_data.current_price < 0:
            print(f"  💰 NEGATIVE PRICE: They're paying YOU to use electricity!")

        if peak_data:
            print(
                f"  Power: {peak_data.current_power:.1f} kW (peak: {peak_data.current_peak:.1f} kW)"
            )
            margin = peak_data.current_peak - peak_data.current_power
            print(f"  Peak Margin: {margin:.1f} kW")

        # Calculate all layers (in priority order)
        layers = [
            self.calculate_safety_layer(nibe_state),
            self.calculate_emergency_layer(nibe_state),
            self.calculate_proactive_layer(nibe_state),
        ]

        # Add effect layer if peak data provided
        if peak_data:
            layers.append(self.calculate_effect_layer(nibe_state, peak_data))

        # Continue with remaining layers
        layers.extend(
            [
                self.calculate_prediction_layer(nibe_state),
                self.calculate_weather_comp_layer(nibe_state),
                self.calculate_weather_layer(nibe_state, weather_data),
                self.calculate_price_layer(price_data),
                self.calculate_comfort_layer(nibe_state),
            ]
        )

        # Aggregate
        final_offset, active_layers = self.aggregate_layers(layers)

        # Print results
        print(f"\n🎯 LAYER VOTES:")
        print(f"{'Layer':<20} {'Offset':>8} {'Weight':>8} {'Weighted':>10} {'Reason':<40}")
        print(f"{'-' * 90}")

        for layer in active_layers:
            print(
                f"{layer.name:<20} {layer.offset:>+7.2f}°C {layer.weight:>7.2f} "
                f"{layer.weighted_value:>+9.3f} {layer.reason:<40}"
            )

        print(f"{'-' * 90}")
        total_weight = sum(l.weight for l in active_layers)
        weighted_sum = sum(l.weighted_value for l in active_layers)
        print(f"{'TOTAL':<20} {'':>8} {total_weight:>7.2f} {weighted_sum:>+9.3f}")

        # Winner analysis
        print(f"\n🏆 RESULT: {final_offset:+.2f}°C")

        if abs(final_offset) < 0.1:
            print(f"   Status: Maintain current heating (balanced)")
        elif final_offset > 0:
            print(f"   Status: Increase heating (charge thermal battery)")
        else:
            print(f"   Status: Reduce heating (conserve energy)")

        # Identify dominant layers
        if active_layers:
            sorted_layers = sorted(active_layers, key=lambda l: abs(l.weighted_value), reverse=True)
            dominant = sorted_layers[0]
            print(f"   Dominant: {dominant.name} ({dominant.weighted_value:+.3f})")

        return {
            "scenario": name,
            "final_offset": final_offset,
            "active_layers": active_layers,
            "conditions": {
                "indoor": nibe_state.indoor_temp,
                "outdoor": nibe_state.outdoor_temp,
                "dm": nibe_state.degree_minutes,
                "price": price_data.current_price,
            },
        }


def main():
    """Run scenario tests."""

    # Always using production constants - single source of truth
    print("✅ Using PRODUCTION constants from custom_components/effektguard/const.py")
    print()

    parser = argparse.ArgumentParser(
        description="Test EffektGuard decision engine scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all built-in scenarios
  python3 scripts/test_decision_scenarios.py

  # Test single scenario
  python3 scripts/test_decision_scenarios.py --scenario negative_price

  # Custom scenario
  python3 scripts/test_decision_scenarios.py --scenario custom \\
      --indoor 22 --outdoor 5 --dm -800 --flow-temp 38 \\
      --price -50 --price-class CHEAP

  # Override layer weights
  python3 scripts/test_decision_scenarios.py --scenario negative_price \\
      --price-weight 0.85 --weather-comp-weight 0.40

  # Override price offsets
  python3 scripts/test_decision_scenarios.py --scenario all \\
      --cheap-offset 4.0 --peak-offset -2.5

  # Show current defaults
  python3 scripts/test_decision_scenarios.py --show-defaults
        """,
    )

    # Scenario selection
    parser.add_argument(
        "--scenario",
        choices=[
            "all",
            "negative_price",
            "thermal_debt",
            "mild_weather",
            "expensive_peak",
            "custom",
        ],
        default="all",
        help="Scenario to test (default: all)",
    )

    # Custom scenario parameters
    custom_group = parser.add_argument_group("Custom Scenario Parameters")
    custom_group.add_argument("--indoor", type=float, help="Indoor temperature (°C)")
    custom_group.add_argument("--outdoor", type=float, help="Outdoor temperature (°C)")
    custom_group.add_argument("--dm", type=float, help="Degree minutes (thermal debt)")
    custom_group.add_argument("--flow-temp", type=float, help="Current flow temperature (°C)")
    custom_group.add_argument("--price", type=float, help="Electricity price (öre/kWh)")
    custom_group.add_argument(
        "--price-class",
        choices=["CHEAP", "NORMAL", "EXPENSIVE", "PEAK"],
        help="Price classification override",
    )
    custom_group.add_argument(
        "--daytime", action="store_true", help="Set time to daytime (default: nighttime)"
    )
    custom_group.add_argument("--current-power", type=float, help="Current power consumption (kW)")
    custom_group.add_argument("--current-peak", type=float, help="Current monthly peak (kW)")

    # Layer weight overrides
    weight_group = parser.add_argument_group(
        "Layer Weight Overrides", "Current production values shown in defaults"
    )
    weight_group.add_argument(
        "--price-weight",
        type=float,
        default=LAYER_WEIGHT_PRICE,
        help=f"Price layer weight (default: {LAYER_WEIGHT_PRICE})",
    )
    weight_group.add_argument(
        "--weather-comp-weight",
        type=float,
        default=LAYER_WEIGHT_WEATHER_PREDICTION,
        help=f"Weather compensation layer weight (default: {LAYER_WEIGHT_WEATHER_PREDICTION})",
    )
    weight_group.add_argument(
        "--emergency-weight",
        type=float,
        default=LAYER_WEIGHT_EMERGENCY,
        help=f"Emergency layer weight (default: {LAYER_WEIGHT_EMERGENCY})",
    )
    weight_group.add_argument(
        "--proactive-z1-weight",
        type=float,
        default=LAYER_WEIGHT_PROACTIVE_MIN,
        help=f"Proactive Zone 1 weight (default: {LAYER_WEIGHT_PROACTIVE_MIN})",
    )
    weight_group.add_argument(
        "--proactive-z2-weight",
        type=float,
        default=0.4,
        help="Proactive Zone 2 weight (default: 0.4)",
    )
    weight_group.add_argument(
        "--proactive-z3-weight",
        type=float,
        default=LAYER_WEIGHT_PROACTIVE_MAX,
        help=f"Proactive Zone 3 weight (default: {LAYER_WEIGHT_PROACTIVE_MAX})",
    )
    weight_group.add_argument(
        "--comfort-weight-min",
        type=float,
        default=LAYER_WEIGHT_COMFORT_MIN,
        help=f"Comfort layer min weight (default: {LAYER_WEIGHT_COMFORT_MIN})",
    )
    weight_group.add_argument(
        "--comfort-weight-max",
        type=float,
        default=LAYER_WEIGHT_COMFORT_MAX,
        help=f"Comfort layer max weight (default: {LAYER_WEIGHT_COMFORT_MAX})",
    )

    # Price offset overrides
    offset_group = parser.add_argument_group(
        "Price Offset Overrides", "Current production values shown in defaults"
    )
    offset_group.add_argument(
        "--cheap-offset", type=float, default=3.0, help="CHEAP price offset °C (default: +3.0)"
    )
    offset_group.add_argument(
        "--normal-offset", type=float, default=0.0, help="NORMAL price offset °C (default: 0.0)"
    )
    offset_group.add_argument(
        "--expensive-offset",
        type=float,
        default=-1.0,
        help="EXPENSIVE price offset °C (default: -1.0)",
    )
    offset_group.add_argument(
        "--peak-offset", type=float, default=-2.0, help="PEAK price offset °C (default: -2.0)"
    )

    # Other options
    parser.add_argument(
        "--show-defaults", action="store_true", help="Show current production defaults and exit"
    )

    args = parser.parse_args()

    # Show defaults and exit
    if args.show_defaults:
        print("=" * 80)
        print("CURRENT PRODUCTION DEFAULTS (from const.py & decision_engine.py)")
        print("=" * 80)
        print("\nLayer Weights (from const.py):")
        print(f"  LAYER_WEIGHT_SAFETY:        {LAYER_WEIGHT_SAFETY:.2f} (absolute priority)")
        print(f"  LAYER_WEIGHT_EMERGENCY:     {LAYER_WEIGHT_EMERGENCY:.2f} (DM beyond expected)")
        print(f"  LAYER_WEIGHT_PRICE:         {LAYER_WEIGHT_PRICE:.2f} (strong influence)")
        print(
            f"  LAYER_WEIGHT_WEATHER_PREDICTION:  {LAYER_WEIGHT_WEATHER_PREDICTION:.2f} (moderate influence)"
        )
        print(f"  LAYER_WEIGHT_PROACTIVE_MAX: {LAYER_WEIGHT_PROACTIVE_MAX:.2f} (Zone 3)")
        print(f"  LAYER_WEIGHT_PROACTIVE_MIN: {LAYER_WEIGHT_PROACTIVE_MIN:.2f} (Zone 1)")
        print(f"  Proactive Zone 2:           0.40 (hardcoded in decision_engine.py)")
        print(f"  LAYER_WEIGHT_COMFORT_MAX:   {LAYER_WEIGHT_COMFORT_MAX:.2f}")
        print(f"  LAYER_WEIGHT_COMFORT_MIN:   {LAYER_WEIGHT_COMFORT_MIN:.2f}")
        print("\nCritical Emergency Weights (hardcoded at weight 1.0):")
        print(f"  DM <= {DM_THRESHOLD_ABSOLUTE_MAX} (absolute max):  offset=+5.0°C, weight=1.0")
        print(f"  DM margin < 300 (critical):         offset=+3.0°C, weight=1.0")
        print("\nEffect/Peak Protection (from decision_engine._effect_layer):")
        print(f"  CRITICAL (at peak):      -3.0°C @ weight 1.0")
        print(f"  PREDICTIVE (margin<1kW): -1.5°C @ weight 0.8")
        print(f"  WARNING (rising):        -1.0°C @ weight 0.7")
        print(f"  WARNING (stable):        -0.5°C @ weight 0.6")
        print("\nPrice Offsets (from price_analyzer.get_base_offset):")
        print(f"  CHEAP:      {args.cheap_offset:+.1f}°C (charge thermal battery!)")
        print(f"  NORMAL:     {args.normal_offset:+.1f}°C (maintain)")
        print(f"  EXPENSIVE:  {args.expensive_offset:+.1f}°C (conserve, x1.5 if daytime)")
        print(f"  PEAK:       {args.peak_offset:+.1f}°C (minimize, x1.5 if daytime)")
        print("\nNote: Price offsets adjusted by tolerance factor (tolerance/5.0)")
        print(f"      Default tolerance=5 → factor=1.0 (no adjustment)")
        print("\nPhilosophy: 'Charge heat when cheap, without peaking the peak'")
        print("=" * 80)
        return

    # Auto-detect custom scenario if any custom parameters provided
    custom_params = ["indoor", "outdoor", "dm", "flow_temp", "price"]
    if args.scenario == "all" and any(getattr(args, p) is not None for p in custom_params):
        # User provided custom values, switch to custom scenario mode
        args.scenario = "custom"

    # Validate custom scenario
    if args.scenario == "custom":
        required = ["indoor", "outdoor", "dm", "flow_temp", "price"]
        missing = [f"--{r.replace('_', '-')}" for r in required if getattr(args, r) is None]
        if missing:
            parser.error(f"custom scenario requires: {', '.join(missing)}")

    tester = ScenarioTester(
        price_weight=args.price_weight,
        weather_comp_weight=args.weather_comp_weight,
        emergency_weight=args.emergency_weight,
        proactive_weight_z1=args.proactive_z1_weight,
        proactive_weight_z2=args.proactive_z2_weight,
        proactive_weight_z3=args.proactive_z3_weight,
        comfort_weight_min=args.comfort_weight_min,
        comfort_weight_max=args.comfort_weight_max,
        cheap_offset=args.cheap_offset,
        normal_offset=args.normal_offset,
        expensive_offset=args.expensive_offset,
        peak_offset=args.peak_offset,
    )

    scenarios = {}

    # Scenario 1: Negative Price with Thermal Debt (user's original question)
    if args.scenario in ["all", "negative_price"]:
        scenarios["negative_price"] = tester.test_scenario(
            name="Negative Price + Thermal Debt",
            nibe_state=MockNibeState(
                indoor_temp=22.0,
                outdoor_temp=7.5,
                degree_minutes=-161,  # From user's example - proactive Z1
                flow_temp=35.0,
            ),
            price_data=MockPriceData(
                current_price=-10.0, classification=QuarterClassification.CHEAP, is_daytime=False
            ),
            peak_data=MockPeakData(
                current_peak=6.5,
                current_power=3.2,  # Safe margin
            ),
        )

    # Scenario 2: Normal Mild Weather
    if args.scenario in ["all", "mild_weather"]:
        scenarios["mild_weather"] = tester.test_scenario(
            name="Mild Weather, Normal Operation",
            nibe_state=MockNibeState(
                indoor_temp=21.5,
                outdoor_temp=7.5,
                degree_minutes=-300,
                flow_temp=28.0,
            ),
            price_data=MockPriceData(
                current_price=45.0, classification=QuarterClassification.NORMAL
            ),
            peak_data=MockPeakData(
                current_peak=6.5,
                current_power=2.8,
            ),
        )

    # Scenario 3: Expensive Peak Hour
    if args.scenario in ["all", "expensive_peak"]:
        scenarios["expensive_peak"] = tester.test_scenario(
            name="Expensive Peak Hour",
            nibe_state=MockNibeState(
                indoor_temp=21.0,
                outdoor_temp=5.0,
                degree_minutes=-200,
                flow_temp=30.0,
            ),
            price_data=MockPriceData(
                current_price=180.0, classification=QuarterClassification.PEAK, is_daytime=True
            ),
            peak_data=MockPeakData(
                current_peak=6.5,
                current_power=6.3,  # Close to peak!
            ),
        )

    # Scenario 4: Severe Thermal Debt, Cold Weather
    if args.scenario in ["all", "thermal_debt"]:
        scenarios["thermal_debt"] = tester.test_scenario(
            name="Severe Thermal Debt in Cold",
            nibe_state=MockNibeState(
                indoor_temp=20.0,
                outdoor_temp=-5.0,
                degree_minutes=-700,
                flow_temp=40.0,
            ),
            price_data=MockPriceData(
                current_price=95.0, classification=QuarterClassification.EXPENSIVE
            ),
            peak_data=MockPeakData(
                current_peak=7.2,
                current_power=5.8,
            ),
        )

    # Custom scenario
    if args.scenario == "custom":
        # Auto-classify price if not specified
        if args.price_class:
            classification = QuarterClassification[args.price_class]
        else:
            # Auto-classify based on price value (simplified)
            if args.price < 20:
                classification = QuarterClassification.CHEAP
            elif args.price < 80:
                classification = QuarterClassification.NORMAL
            elif args.price < 150:
                classification = QuarterClassification.EXPENSIVE
            else:
                classification = QuarterClassification.PEAK

        # Create peak data if power/peak provided
        peak_data = None
        if args.current_power is not None and args.current_peak is not None:
            peak_data = MockPeakData(
                current_peak=args.current_peak,
                current_power=args.current_power,
            )

        scenarios["custom"] = tester.test_scenario(
            name="Custom Scenario",
            nibe_state=MockNibeState(
                indoor_temp=args.indoor,
                outdoor_temp=args.outdoor,
                degree_minutes=args.dm,
                flow_temp=args.flow_temp,
            ),
            price_data=MockPriceData(
                current_price=args.price,
                classification=classification,
                is_daytime=args.daytime,
            ),
            peak_data=peak_data,
        )

    # Summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n{'Scenario':<35} {'Offset':>10} {'Dominant Layer':<30}")
    print(f"{'-' * 80}")

    for name, result in scenarios.items():
        if result["active_layers"]:
            sorted_layers = sorted(
                result["active_layers"], key=lambda l: abs(l.weighted_value), reverse=True
            )
            dominant = sorted_layers[0].name
        else:
            dominant = "None"

        print(f"{result['scenario']:<35} {result['final_offset']:>+9.2f}°C {dominant:<30}")

    print(f"\n✅ Tested {len(scenarios)} scenario(s)")
    print(f"\n💡 TIP: Use --scenario to test individual scenarios")
    print(f"   Example: python3 scripts/test_decision_scenarios.py --scenario negative_price")


if __name__ == "__main__":
    main()

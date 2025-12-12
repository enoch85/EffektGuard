"""Constants for EffektGuard integration."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

# Domain
DOMAIN: Final = "effektguard"

# Configuration keys
CONF_NIBE_ENTITY: Final = "nibe_entity"
CONF_GESPOT_ENTITY: Final = "gespot_entity"
CONF_WEATHER_ENTITY: Final = "weather_entity"
CONF_DEGREE_MINUTES_ENTITY: Final = "degree_minutes_entity"  # Optional: NIBE degree minutes
CONF_POWER_SENSOR_ENTITY: Final = "power_sensor_entity"  # Optional: Power meter
CONF_DHW_TEMP_ENTITY: Final = "dhw_temp_entity"  # Optional: DHW temperature sensor (BT7)
CONF_NIBE_TEMP_LUX_ENTITY: Final = "nibe_temp_lux_entity"  # Optional: switch.temporary_lux_50004
CONF_ENABLE_DHW_OPTIMIZATION: Final = "enable_dhw_optimization"  # Enable intelligent DHW scheduling
CONF_DHW_DEMAND_PERIODS: Final = "dhw_demand_periods"  # High DHW demand periods (JSON list)
CONF_DHW_TARGET_TEMP: Final = "dhw_target_temp"  # User-configurable DHW target temperature (°C)
CONF_ADDITIONAL_INDOOR_SENSORS: Final = (
    "additional_indoor_sensors"  # Optional: List of extra temp sensors
)
CONF_INDOOR_TEMP_METHOD: Final = "indoor_temp_method"  # average, median, weighted
CONF_ENABLE_PRICE_OPTIMIZATION: Final = "enable_price_optimization"
CONF_ENABLE_PEAK_PROTECTION: Final = "enable_peak_protection"
CONF_ENABLE_WEATHER_PREDICTION: Final = "enable_weather_prediction"
CONF_ENABLE_HOT_WATER_OPTIMIZATION: Final = "enable_hot_water_optimization"
CONF_ENABLE_WEATHER_COMPENSATION: Final = "enable_weather_compensation"  # Universal formulas
CONF_ENABLE_OPTIMIZATION: Final = "enable_optimization"  # Master enable switch
CONF_TARGET_TEMPERATURE: Final = "target_temperature"
CONF_TOLERANCE: Final = "tolerance"
CONF_OPTIMIZATION_MODE: Final = "optimization_mode"
CONF_THERMAL_MASS: Final = "thermal_mass"
CONF_INSULATION_QUALITY: Final = "insulation_quality"
CONF_HEAT_PUMP_MODEL: Final = "heat_pump_model"
CONF_WEATHER_COMPENSATION_WEIGHT: Final = "weather_compensation_weight"  # 0.0-1.0

# Defaults
DEFAULT_TOLERANCE: Final = 0.5
DEFAULT_TARGET_TEMP: Final = 21.0
DEFAULT_INDOOR_TEMP: Final = 21.0  # Fallback when sensor unavailable
DEFAULT_THERMAL_MASS: Final = 1.0
DEFAULT_INSULATION_QUALITY: Final = 1.0
DEFAULT_HEAT_PUMP_MODEL: Final = "nibe_f750"  # Most common model
DEFAULT_OPTIMIZATION_MODE: Final = "balanced"
DEFAULT_WEATHER_COMPENSATION_WEIGHT: Final = 0.49  # User-configurable weight (matches layer weight)
DEFAULT_INDOOR_TEMP_METHOD: Final = "median"  # median more robust to outliers than average
DEFAULT_DHW_TARGET_TEMP: Final = 50.0  # °C - Default DHW target temperature

# Climate entity temperature limits (displayed in UI)
MIN_INDOOR_TEMP: Final = 15.0  # °C - minimum settable temperature
MAX_INDOOR_TEMP: Final = 25.0  # °C - maximum settable temperature
TEMP_STEP: Final = 0.5  # °C - temperature adjustment step

# Daytime/Nighttime boundaries (Swedish effect tariff periods)
# Swedish Effektavgift uses different weights for day vs night peaks
# Daytime: Full effect tariff weight, higher prices typical
# Nighttime: 50% effect tariff weight, lower prices typical
DAYTIME_START_HOUR: Final = 6  # 06:00 - daytime starts
DAYTIME_END_HOUR: Final = 22  # 22:00 - daytime ends (nighttime 22:00-06:00)
# Quarter equivalents (each hour = 4 quarters, so 6*4=24 and 22*4=88)
DAYTIME_START_QUARTER: Final = 24  # Quarter 24 = 06:00
DAYTIME_END_QUARTER: Final = 87  # Quarter 87 = 21:45 (last daytime quarter)

# Optimization modes for climate entity presets
OPTIMIZATION_MODE_COMFORT: Final = "comfort"  # Minimize deviation, accept higher costs
OPTIMIZATION_MODE_BALANCED: Final = "balanced"  # Balance comfort and savings
OPTIMIZATION_MODE_SAVINGS: Final = "savings"  # Maximize savings, wider tolerance


@dataclass(frozen=True)
class OptimizationModeConfig:
    """Configuration for each optimization mode.

    Defines how each mode affects the optimization behavior:
    - dead_zone: Temperature band around target where no action is taken
    - comfort_weight_multiplier: Scales comfort layer influence (higher = comfort wins more)
    - price_tolerance_multiplier: Scales price layer effect (higher = more aggressive)
    - peak_bypass_tolerance: If True, PEAK ignores tolerance scaling (always full reduction)
    - preheat_overshoot_allowed: How much overshoot to accept during pre-heating (°C)
    """

    dead_zone: float  # °C - no action within this range
    comfort_weight_multiplier: float  # Scales comfort layer weight
    price_tolerance_multiplier: float  # Scales price layer tolerance factor
    peak_bypass_tolerance: bool  # PEAK always uses full offset
    preheat_overshoot_allowed: float  # °C - acceptable overshoot during cheap pre-heat


# Mode configurations - these define the behavior philosophy
# Comfort: Rock-solid temperature, price is secondary
# Balanced: Smart trade-offs between comfort and savings
# Savings: Maximum savings, temperature can drift more
MODE_CONFIGS: Final[dict[str, OptimizationModeConfig]] = {
    OPTIMIZATION_MODE_COMFORT: OptimizationModeConfig(
        dead_zone=0.1,  # Tighter: react faster to any deviation
        comfort_weight_multiplier=1.3,  # Comfort layer wins more often
        price_tolerance_multiplier=0.7,  # Reduce price layer effect
        peak_bypass_tolerance=False,  # Respect tolerance even during PEAK
        preheat_overshoot_allowed=0.3,  # Minimal overshoot accepted
    ),
    OPTIMIZATION_MODE_BALANCED: OptimizationModeConfig(
        dead_zone=0.2,  # Standard dead zone
        comfort_weight_multiplier=1.0,  # Normal comfort influence
        price_tolerance_multiplier=1.0,  # Normal price effect
        peak_bypass_tolerance=False,  # Respect tolerance setting
        preheat_overshoot_allowed=0.5,  # Moderate overshoot OK
    ),
    OPTIMIZATION_MODE_SAVINGS: OptimizationModeConfig(
        dead_zone=0.3,  # Wider: ignore small deviations
        comfort_weight_multiplier=0.7,  # Price wins more often
        price_tolerance_multiplier=1.3,  # Amplify price effect
        peak_bypass_tolerance=True,  # PEAK always full reduction
        preheat_overshoot_allowed=1.0,  # Accept more overshoot for savings
    ),
}


# Config keys for internal use
CONF_TARGET_INDOOR_TEMP: Final = "target_indoor_temp"

# Limits
MIN_OFFSET: Final = -10.0
MAX_OFFSET: Final = 10.0
MIN_TEMP_LIMIT: Final = 18.0

# Service call rate limiting (boost, DHW, general)
HEATING_BOOST_COOLDOWN_MINUTES: Final = 45  # Space heating boost cooldown
DHW_BOOST_COOLDOWN_MINUTES: Final = 60  # DHW boost cooldown
DHW_CONTROL_MIN_INTERVAL_MINUTES: Final = 60  # Automatic DHW control rate limit (1 hour)
SERVICE_RATE_LIMIT_MINUTES: Final = 5  # General service call cooldown

# Decision engine layer weights
# Source: User feedback and optimization tuning (Oct 2025)
# Philosophy: "Charge heat when cheap, without peaking the peak"
LAYER_WEIGHT_SAFETY: Final = 1.0  # Absolute priority (temp limits)
LAYER_WEIGHT_EMERGENCY: Final = 0.8  # High priority (DM beyond expected)
LAYER_WEIGHT_PRICE: Final = 0.8  # Strong influence - balanced with other layers (Nov 27, 2025)
LAYER_WEIGHT_PROACTIVE_MIN: Final = 0.3  # Minimum proactive weight
LAYER_WEIGHT_PREDICTION: Final = 0.65  # Prediction layer weight (Phase 6)
LAYER_WEIGHT_COMFORT_MIN: Final = 0.2  # Minimum comfort weight
LAYER_WEIGHT_COMFORT_MAX: Final = 0.5  # Maximum comfort weight (legacy - unused after Phase 2)

# Graduated comfort layer weights (Phase 2: Temperature Control Fixes)
# Provides dynamic response to temperature overshoot severity
# Dec 2, 2025: Lowered thresholds - comfort layer triggers at 0.5°C and 1.0°C
COMFORT_OVERSHOOT_SEVERE: Final = 0.5  # °C over tolerance for severe response
COMFORT_OVERSHOOT_CRITICAL: Final = 1.0  # °C over tolerance for critical response
LAYER_WEIGHT_COMFORT_HIGH: Final = 0.7  # High priority: 0-0.5°C over tolerance
LAYER_WEIGHT_COMFORT_SEVERE: Final = 0.9  # Very high priority: 0.5-1°C over tolerance
LAYER_WEIGHT_COMFORT_CRITICAL: Final = (
    1.0  # Critical priority: 1°C+ over tolerance (same as safety)
)

# Graduated comfort layer correction multipliers (Phase 2)
COMFORT_CORRECTION_MILD: Final = 1.0  # 0-1°C over tolerance: standard correction
COMFORT_CORRECTION_STRONG: Final = 1.2  # 1-2°C over tolerance: strong correction
COMFORT_CORRECTION_CRITICAL: Final = 1.5  # 2°C+ over tolerance: emergency correction

# Effect tariff / Peak protection layer weights and offsets
# Oct 19, 2025: Increased weights to make peak protection more decisive
# Peak costs (effect tariff) are monthly charges that accumulate - worth protecting
EFFECT_WEIGHT_CRITICAL: Final = 1.0  # Already at peak - immediate action
EFFECT_WEIGHT_PREDICTIVE: Final = 0.85  # Will approach peak (<1kW margin) - act NOW
EFFECT_WEIGHT_WARNING_RISING: Final = 0.75  # Close to peak + demand rising
EFFECT_WEIGHT_WARNING_STABLE: Final = 0.65  # Close to peak + demand stable/falling
EFFECT_OFFSET_CRITICAL: Final = -3.0  # Aggressive reduction at power peak
EFFECT_OFFSET_PREDICTIVE: Final = -1.5  # Moderate reduction before peak
EFFECT_OFFSET_WARNING_RISING: Final = -1.0  # Gentle reduction near peak
EFFECT_OFFSET_WARNING_STABLE: Final = -0.5  # Light reduction near peak
EFFECT_MARGIN_PREDICTIVE: Final = 1.0  # kW margin threshold for predictive action
EFFECT_MARGIN_WARNING: Final = 1.5  # kW margin threshold for warning
EFFECT_MARGIN_WATCH: Final = 2.5  # kW margin threshold for watch

# Effect layer predictive power thresholds (Oct 19, 2025)
# Predicts future power demand based on thermal trend rate
# Adds predicted power increase to current demand for peak protection decisions
EFFECT_PREDICTIVE_RAPID_COOLING_THRESHOLD: Final = -0.4  # °C/h rapid cooling
EFFECT_PREDICTIVE_RAPID_COOLING_INCREASE: Final = 1.5  # kW predicted increase
EFFECT_PREDICTIVE_MODERATE_COOLING_INCREASE: Final = 0.5  # kW predicted increase
EFFECT_PREDICTIVE_WARMING_DECREASE: Final = -0.5  # kW predicted decrease

# Weather compensation deferral (Conservative strategy)
# Reduce weather comp weight when thermal debt exists, allowing thermal reality
# (DM + comfort + proactive) to override outdoor temperature optimization
WEATHER_COMP_DEFER_DM_LIGHT: Final = -150  # Start reducing influence (8% reduction)
WEATHER_COMP_DEFER_DM_MODERATE: Final = -200  # Clear thermal priority (18% reduction)
WEATHER_COMP_DEFER_DM_SIGNIFICANT: Final = -300  # Strong reduction (29% reduction)
WEATHER_COMP_DEFER_DM_CRITICAL: Final = -400  # Minimal weather comp influence (39% reduction)

WEATHER_COMP_DEFER_WEIGHT_LIGHT: Final = 0.45  # 8% reduction from 0.49
WEATHER_COMP_DEFER_WEIGHT_MODERATE: Final = 0.41  # 16% reduction from 0.49
WEATHER_COMP_DEFER_WEIGHT_SIGNIFICANT: Final = 0.36  # 27% reduction from 0.49
WEATHER_COMP_DEFER_WEIGHT_CRITICAL: Final = 0.30  # 39% reduction from 0.49

# Climate Zone Winter Baselines (Oct 19, 2025 - SMHI Climate Normals 1991-2020)
# These represent Jan-Feb average temperatures for each climate zone
# Used as baseline for seasonal adjustment calculations in weather_learning.py
# Source: SMHI Climate Normals 1991-2020 from representative Swedish stations
#
# Formula: typical_temp = winter_avg_low + seasonal_adjustment
#
# IMPORTANT: These are NOT "coldest possible temperatures" but JAN-FEB AVERAGES
# that serve as the baseline for seasonal temperature predictions.
#
CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG: Final = -20.0  # Conservative (SMHI Kiruna: -13.1°C)
CLIMATE_ZONE_VERY_COLD_WINTER_AVG: Final = -12.0  # SMHI Arvidsjaur Jan-Feb: -12.3°C
CLIMATE_ZONE_COLD_WINTER_AVG: Final = -8.0  # SMHI Östersund Jan-Feb: -7.8°C
CLIMATE_ZONE_MODERATE_COLD_WINTER_AVG: Final = -1.0  # SMHI Stockholm Jan-Feb: -1.1°C
CLIMATE_ZONE_STANDARD_WINTER_AVG: Final = 0.0  # SMHI Southern Sweden Jan-Feb: -0.2°C

# Safety thresholds - Degree Minutes (DM) / Gradminuter (GM)
# Based on real-world validation in Nordic conditions
#
# DM -1500 is safe operating limit across all Nordic climate zones.
# Validated in real-world Swedish conditions: -30°C to +5°C temperature range.
#
# CLIMATE-AWARE DESIGN:
# Climate zone module (climate_zones.py) provides context-aware DM thresholds that
# automatically adapt from Arctic (-30°C) to Mild (5°C) climates without configuration.
# No hardcoded temperature bands - uses latitude-based climate zones with outdoor temp adjustment.
#
DM_THRESHOLD_START: Final = -60  # Normal compressor start (NIBE standard)
DM_THRESHOLD_AUX_LIMIT: Final = -1500  # Auxiliary heat threshold (prevent expensive elpatron)

# Multi-tier CRITICAL thermal debt intervention (Oct 19, 2025 - Climate-Aware)
# Philosophy: Progressive escalation based on climate-aware WARNING threshold
# Tiers calculated dynamically as: WARNING + margin (adapts to climate zone)
# User preference: "DM max -800" → T1 triggers at climate-appropriate threshold
#
# Example climate-aware tier activation:
# - Paris (WARNING -350):     T1=-350,  T2=-550,  T3=-750
# - Stockholm (WARNING -700): T1=-700,  T2=-900,  T3=-1100
# - Kiruna (WARNING -1200):   T1=-1200, T2=-1400, T3=-1450 (capped)
#
# Prevents false positives in Arctic climates while maintaining safety in mild climates.
DM_CRITICAL_T1_MARGIN: Final = (
    0  # Tier 1: At WARNING threshold (climate-aware) - early aggressive intervention
)
DM_CRITICAL_T1_OFFSET: Final = 4.0  # Strong early boost (prevent DM spiral, avoid hours of high Hz)
DM_CRITICAL_T1_WEIGHT: Final = 0.65  # Below price (0.8) - allows price/comfort override
DM_CRITICAL_T2_MARGIN: Final = 200  # Tier 2: WARNING + 200 DM beyond
DM_CRITICAL_T2_OFFSET: Final = 7.0  # Very strong boost (decisive recovery before T3)
DM_CRITICAL_T2_WEIGHT: Final = 0.81  # Slightly above Price (0.8) - ensures recovery if cold
DM_CRITICAL_T3_MARGIN: Final = 400  # Tier 3: WARNING + 400 DM beyond (capped at -1450)
DM_CRITICAL_T3_OFFSET: Final = 8.5  # Maximum emergency boost (prevent hours of full Hz operation)
DM_CRITICAL_T3_WEIGHT: Final = 0.91  # Strong dominance over Price (0.8)
DM_CRITICAL_T3_MAX: Final = -1450  # Safety cap: 50 DM margin from absolute max (-1500)

# Peak-aware emergency mode minimal offsets (Oct 19, 2025)
# When CRITICAL tiers are active during peak hours, use reduced offsets to prevent
# DM worsening while minimizing peak creation. Escalates with thermal debt severity.
DM_CRITICAL_T1_PEAK_AWARE_OFFSET: Final = 0.5  # T1 minimal: Just enough to stabilize
DM_CRITICAL_T2_PEAK_AWARE_OFFSET: Final = 0.75  # T2 minimal: Moderate escalation
DM_CRITICAL_T3_PEAK_AWARE_OFFSET: Final = 1.0  # T3 minimal: Aggressive recovery needed
PEAK_AWARE_EFFECT_THRESHOLD: Final = -1.0  # Effect offset threshold for peak detection
PEAK_AWARE_EFFECT_WEIGHT_MIN: Final = 0.5  # Minimum effect weight for peak detection

# Thermal Recovery Damping - General (Oct 20, 2025)
# Prevent concrete slab thermal overshoot when solar gain naturally warms house during recovery
# Applies to ALL recovery tiers (T1, T2, T3, WARNING) when warming detected
#
# Scenario: Emergency recovery active at night → morning sun → rapid indoor warming
# Without damping: High offset continues → concrete slab stores excess heat → +2-3°C overshoot
# With damping: Detect warming trend → reduce offset → concrete slab doesn't overheat
#
# Philosophy: Let solar gain do the work during recovery, reduce artificial heating boost
THERMAL_RECOVERY_WARMING_THRESHOLD: Final = (
    0.3  # Indoor warming rate (°C/h) to trigger damping (significant solar gain)
)
THERMAL_RECOVERY_DAMPING_FACTOR: Final = (
    0.6  # Standard damping multiplier (0.6 = reduce to 60% when warming)
)
THERMAL_RECOVERY_RAPID_FACTOR: Final = (
    0.4  # Stronger damping for rapid warming >0.5°C/h (0.4 = reduce to 40%)
)
THERMAL_RECOVERY_RAPID_THRESHOLD: Final = 0.5  # Rapid warming threshold (°C/h)
THERMAL_RECOVERY_OUTDOOR_DROPPING_THRESHOLD: Final = (
    -0.5  # Outdoor cooling rate (°C/h) that blocks damping (safety: fighting cold spell)
)
THERMAL_RECOVERY_MIN_CONFIDENCE: Final = (
    0.4  # Minimum trend confidence to apply damping (0.4 = ~1 hour of data)
)
THERMAL_RECOVERY_FORECAST_HORIZON: Final = (
    6.0  # Hours to check forecast for cold weather (blocks damping if cold coming)
)
THERMAL_RECOVERY_FORECAST_DROP_THRESHOLD: Final = (
    -2.0  # Temperature drop (°C) that blocks damping (significant cold incoming)
)

# Overshoot-Aware Damping (Oct 23, 2025)
# ============================================================================
# Context: v0.1.0 showed DM recovery caused indoor 23.6°C vs target 21.5°C
#          (2.1°C overshoot during STRONG RECOVERY tier)
#
# Problem: Traditional damping only considers warming rate, not absolute error
#          Large overshoot = need stronger damping regardless of warming rate
#
# Solution: Overshoot severity multiplies with warming damping
#          Example: 2.0°C overshoot (severe) + rapid warming (0.4) = 0.8 × 0.4 = 0.32
#                   Final offset = base_offset × 0.32 (68% reduction for safety)
#
# Thresholds:
#   Severe ≥1.5°C: 80% strength (0.8 multiplier, strong reduction)
#   Moderate ≥1.0°C: 90% strength (0.9 multiplier, moderate reduction)
#   Mild ≥0.5°C: 95% strength (0.95 multiplier, gentle reduction)
#   <0.5°C: 100% strength (no overshoot penalty)
THERMAL_RECOVERY_OVERSHOOT_SEVERE_THRESHOLD: Final = 1.5  # °C over target (severe)
THERMAL_RECOVERY_OVERSHOOT_MODERATE_THRESHOLD: Final = 1.0  # °C over target (moderate)
THERMAL_RECOVERY_OVERSHOOT_MILD_THRESHOLD: Final = 0.5  # °C over target (mild)
THERMAL_RECOVERY_OVERSHOOT_SEVERE_DAMPING: Final = 0.8  # Multiplier for severe overshoot
THERMAL_RECOVERY_OVERSHOOT_MODERATE_DAMPING: Final = 0.9  # Multiplier for moderate overshoot
THERMAL_RECOVERY_OVERSHOOT_MILD_DAMPING: Final = 0.95  # Multiplier for mild overshoot

# Tier-specific minimum offsets after damping (safety floors)
# Each tier has appropriate minimum to maintain recovery progress
THERMAL_RECOVERY_T1_MIN_OFFSET: Final = 1.0  # T1 damped minimum (moderate recovery)
THERMAL_RECOVERY_T2_MIN_OFFSET: Final = 1.5  # T2 damped minimum (strong recovery)
THERMAL_RECOVERY_T3_MIN_OFFSET: Final = 2.0  # T3 damped minimum (critical - needs more boost)

# ============================================================================
# Anti-Windup Protection (Dec 9, 2025)
# ============================================================================
# Description:
#   Prevents DM oscillation caused by "chasing" thermal debt in UFH systems.
#   When offset is positive (adding heat), DM goes MORE negative initially
#   because S1 target rises but BT25 (actual) takes hours to catch up.
#
# Problem (Classic Integral Windup):
#   1. DM at -300 → System applies +2°C offset
#   2. S1 target jumps up immediately
#   3. BT25 takes 3-6 hours to catch up (concrete slab thermal lag)
#   4. DM drops to -500 (gap is now bigger!) → "Not working, add more!"
#   5. Offset raised to +3°C → DM drops to -700 (even worse!)
#   6. Eventually heat reaches slab, BT25 overshoots S1
#   7. DM swings to +100 (positive overshoot)
#   8. System backs off → cycle repeats
#
# Solution (Anti-Windup Logic):
#   If DM is dropping WHILE current_offset is already positive:
#   → Heat is "in transit" through the thermal mass
#   → DON'T escalate offset further - wait for heat to arrive
#   → Cap offset at current level until DM stabilizes or improves
#
# Key Insight:
#   Raising offset doesn't fix DM immediately - it makes DM WORSE temporarily!
#   Correct behavior: Set reasonable offset and WAIT for thermal lag.
#
# All parameters below are DERIVED from existing constants, not static values.
# This ensures consistency across the codebase and adapts to configuration.
#
# Reference:
#   - Wikipedia: Integral Windup
#   - Thermal mass buffers: DM_THERMAL_MASS_BUFFER_* (1.0 to 1.3)
#   - Update interval: UPDATE_INTERVAL_MINUTES (5 min default)
# ============================================================================

# DM drop rate threshold to detect "dropping while heating"
#
# Math derivation:
#   DM = ∫(BT25 - S1) dt, accumulated in °C·minutes
#   When offset is raised by X°C, S1 target rises immediately
#   But BT25 (actual flow) takes hours to catch up (thermal lag)
#   During lag: gap ≈ X°C → DM drops at X × 60 = X·60 DM/hour
#
#   Example: +2°C offset raise → gap ~2°C → DM drops ~120 DM/hour
#   Example: +1°C offset raise → gap ~1°C → DM drops ~60 DM/hour
#
# Threshold: -60 DM/hour = 1°C sustained gap = clear "heat in transit" signal
# This avoids false positives from normal ±0.3°C fluctuations (~18 DM/h)
#
# Reference: NIBE DM formula from F750 Service Manual Menu 4.9.3
ANTI_WINDUP_DM_DROPPING_RATE: Final = -60.0  # DM/hour (≈1°C gap between S1 and BT25)

# Base history size for anti-windup DM tracking (30 minutes = 6 samples at 5-min intervals)
# This is scaled by DM_THERMAL_MASS_BUFFER_* in EmergencyLayer.__init__
# Concrete (×1.3): 6 × 1.3 ≈ 8 samples = 40 min history
# Timber (×1.15): 6 × 1.15 ≈ 7 samples = 35 min history
# Radiator (×1.0): 6 samples = 30 min history
ANTI_WINDUP_DM_HISTORY_BASE_SIZE: Final = 6  # Base samples, scaled by thermal mass buffer

# Minimum samples needed to calculate reliable DM trend
# At least 15 minutes of data (3 samples at 5-min intervals)
ANTI_WINDUP_MIN_SAMPLES: Final = 3

# Minimum positive offset to trigger anti-windup check
# Below this, we're not actively pushing recovery hard enough to cause windup
ANTI_WINDUP_MIN_POSITIVE_OFFSET: Final = 0.5  # Offset must be at least +0.5°C

# When anti-windup is active, cap offset at this fraction of current offset
# This allows some recovery but prevents escalation
ANTI_WINDUP_OFFSET_CAP_MULTIPLIER: Final = 0.7  # Cap at 70% of normal recovery offset

# ============================================================================
# Thermal Mass Buffer Multipliers (DM Threshold Adjustment)
# ============================================================================
# Description:
#   High thermal mass systems need tighter DM thresholds to maintain thermal
#   buffer for forecast uncertainty. Long thermal lag (6+ hours for concrete)
#   means current DM doesn't immediately affect indoor temp.
#
# Research Context:
#   - v0.1.0 failure: DM -700 during solar gain → 1.5°C drop at sunset
#   - Concrete slab thermal lag: 6-12 hours
#   - Need buffer to handle: sunset, weather changes, forecast errors
#
# Multiplier Logic:
#   Applied to base climate-aware thresholds from ClimateZoneDetector
#   Example: Stockholm at 10°C → base warning -276
#            Concrete: -276 × 1.3 = -359 (tighter threshold)
#            Radiator: -276 × 1.0 = -276 (standard)
#
# Example Impact (Stockholm at 10°C):
#   Base thresholds: normal_max=-276, warning=-276, critical=-1500
#
#   Concrete slab (6-12h lag):
#     normal_max: -276 × 1.3 = -359
#     warning:    -276 × 1.3 = -359
#     critical:   -1500 (unchanged, absolute maximum)
#     → Activates T1 recovery earlier, maintains thermal buffer
#
#   Timber UFH (2-4h lag):
#     normal_max: -276 × 1.15 = -317
#     warning:    -276 × 1.15 = -317
#     critical:   -1500 (unchanged)
#     → Moderate tightening for medium-lag systems
#
#   Radiators (<1h lag):
#     normal_max: -276 × 1.0 = -276
#     warning:    -276 × 1.0 = -276
#     critical:   -1500 (unchanged)
#     → Standard thresholds, fast response allows more latitude
#
# Reference:
#   - COMPLETED/CLIMATE_AWARE_VALIDATION_OCT19.md
#   - IMPLEMENTATION_PLAN/THERMAL_MASS_AND_CONTEXT_FIXES_OCT23.md
# ============================================================================

DM_THERMAL_MASS_BUFFER_CONCRETE: Final = 1.3  # 30% tighter threshold (6-12h lag)
DM_THERMAL_MASS_BUFFER_TIMBER: Final = 1.15  # 15% tighter threshold (2-4h lag)
DM_THERMAL_MASS_BUFFER_RADIATOR: Final = 1.0  # Standard threshold (<1h lag)

# Heating type inference from thermal_mass setting
# Used when heating_type is not explicitly configured
# Maps user's thermal_mass slider value to appropriate heating system type
THERMAL_MASS_CONCRETE_UFH_THRESHOLD: Final = 1.5  # >= 1.5 = concrete underfloor heating
THERMAL_MASS_TIMBER_UFH_THRESHOLD: Final = 1.2  # >= 1.2 = timber underfloor heating
# Below 1.2 defaults to radiator heating

# Tolerance range multiplier (Oct 19, 2025)
# Scales user tolerance setting (1-10) to actual temperature range
TOLERANCE_RANGE_MULTIPLIER: Final = 0.4  # Scale: 1-10 -> 0.4-4.0°C

# Safety layer emergency offsets (Oct 19, 2025)
# Used for extreme temperature deviations and absolute DM maximum
SAFETY_EMERGENCY_OFFSET: Final = MAX_OFFSET  # Emergency temperature correction (too cold/hot)

# WARNING layer dynamic offsets (Oct 19, 2025)
# Progressive offset calculation based on DM deviation severity
# Severe deviation: DM deviation > 200 from expected warning threshold
# Moderate deviation: DM deviation <= 200 from expected warning threshold
WARNING_DEVIATION_THRESHOLD: Final = 200  # DM deviation threshold for severe vs moderate
WARNING_OFFSET_MAX_SEVERE: Final = 1.8  # Maximum offset for severe deviation
WARNING_OFFSET_MIN_SEVERE: Final = 1.0  # Minimum offset for severe deviation
WARNING_DEVIATION_DIVISOR_SEVERE: Final = 250  # Ramp steepness for severe (1.0 + dev/250)
WARNING_OFFSET_MAX_MODERATE: Final = 1.5  # Maximum offset for moderate deviation
WARNING_OFFSET_MIN_MODERATE: Final = 0.8  # Minimum offset for moderate deviation
WARNING_DEVIATION_DIVISOR_MODERATE: Final = 300  # Ramp steepness for moderate (0.8 + dev/300)

# WARNING layer caution zone (Oct 19, 2025)
# Gentle correction when slightly beyond normal operating range
WARNING_CAUTION_OFFSET: Final = 0.5  # Gentle correction for caution zone
WARNING_CAUTION_WEIGHT: Final = 0.5  # Caution layer weight

# Proactive layer zone thresholds (Oct 19, 2025)
# Five-zone system based on percentage of climate-aware expected DM thresholds
# Prevents thermal debt accumulation before reaching WARNING thresholds
#
# DESIGN: All proactive zones trigger BEFORE warning threshold!
# Z1-Z5 are PREVENTION layers. T1-T3 are RECOVERY layers (after warning).
#
PROACTIVE_ZONE1_THRESHOLD_PERCENT: Final = 0.10  # 10% of normal max (early warning)
PROACTIVE_ZONE2_THRESHOLD_PERCENT: Final = 0.30  # 30% of normal max (moderate)
PROACTIVE_ZONE3_THRESHOLD_PERCENT: Final = 0.50  # 50% of normal max (significant)
PROACTIVE_ZONE4_THRESHOLD_PERCENT: Final = 0.75  # 75% of normal max (strong)
PROACTIVE_ZONE5_THRESHOLD_PERCENT: Final = 1.00  # 100% of normal max (at warning boundary)

# Proactive layer zone offsets and weights (Oct 19, 2025)
# Progressive escalation as DM approaches warning threshold
# Zone offsets must be >= 1.0 to have immediate effect on NIBE (integer only)
# Lower offsets accumulate via fractional accumulator but take multiple cycles to apply
#
# ESCALATION HIERARCHY (must be strictly increasing):
# Z1 → Z2 → Z3 → Z4 → Z5 → WARNING → T1 → T2 → T3
#
# Effective contribution = offset × weight
# Z1: 1.0 × 0.30 = 0.30 (gentle nudge)
# Z2: 1.5 × 0.40 = 0.60 (moderate boost)
# Z3: 2.0 × 0.50 = 1.00 to 2.5 × 0.50 = 1.25 (significant)
# Z4: 2.5 × 0.55 = 1.38 (strong prevention)
# Z5: 3.0 × 0.60 = 1.80 (very strong, approaching WARNING)
# WARNING: 0.8-1.8 × 0.5-0.7 = 0.4-1.26 (at warning threshold)
# T1: 4.0 × 0.65 = 2.60 (recovery mode)
# T2: 7.0 × 0.81 = 5.67 (strong recovery)
# T3: 8.5 × 0.91 = 7.74 (emergency recovery)
#
PROACTIVE_ZONE1_OFFSET: Final = 1.0  # Light boost (immediate NIBE effect)
# Zone 1 weight uses LAYER_WEIGHT_PROACTIVE_MIN (0.3)
PROACTIVE_ZONE2_OFFSET: Final = 1.5  # Moderate boost
PROACTIVE_ZONE2_WEIGHT: Final = 0.4  # Mid-range weight
PROACTIVE_ZONE3_OFFSET_MIN: Final = 2.0  # Base offset for zone 3
PROACTIVE_ZONE3_OFFSET_RANGE: Final = 0.5  # Additional offset based on severity (0.0-0.5)
PROACTIVE_ZONE3_WEIGHT: Final = 0.50  # Zone 3 weight
PROACTIVE_ZONE4_OFFSET: Final = 2.5  # Strong prevention
PROACTIVE_ZONE4_WEIGHT: Final = 0.55  # Strong weight (below WARNING)
PROACTIVE_ZONE5_OFFSET: Final = 3.0  # Very strong prevention (bridging to WARNING)
PROACTIVE_ZONE5_WEIGHT: Final = 0.60  # Below T1 (0.65)

# Warm house weight reduction (Dec 10, 2025)
# When house is warm (above target + threshold), reduce proactive layer weight
# to prevent fighting with COAST (overshoot protection).
# Root cause: DM can be negative even when house is warm due to flow temp reduction.
# The correct response is to let COAST reduce heat, not boost DM recovery.
# Z5 exemption: At warning boundary, prioritize DM recovery over temperature
PROACTIVE_WARM_HOUSE_THRESHOLD: Final = 0.5  # °C above target to consider "warm"
PROACTIVE_WARM_HOUSE_WEIGHT_REDUCTION: Final = 0.3  # Apply 30% of normal weight when warm

# Overshoot Protection
# Graduated coasting response when indoor temp is above target with stable weather
# Based on Dec 1-2, 2025 production analysis: overshoot was ignored, causing DM spiral
# Key insight: DM recovers when we STOP heating, not by boosting
OVERSHOOT_PROTECTION_START: Final = (
    0.6  # °C above target to start responding (catches Dec 2 crisis)
)
OVERSHOOT_PROTECTION_FULL: Final = 1.5  # °C above target for full response
OVERSHOOT_PROTECTION_OFFSET_MIN: Final = -7.0  # Offset at start threshold (coast gently)
OVERSHOOT_PROTECTION_OFFSET_MAX: Final = MIN_OFFSET  # Offset at full threshold (full coast)
OVERSHOOT_PROTECTION_WEIGHT_MIN: Final = 0.5  # Weight at start threshold
OVERSHOOT_PROTECTION_WEIGHT_MAX: Final = 1.0  # Weight at full threshold (full override)
OVERSHOOT_PROTECTION_FORECAST_HORIZON: Final = 12  # Hours to check forecast stability
OVERSHOOT_PROTECTION_COLD_SNAP_THRESHOLD: Final = 3.0  # °C drop that qualifies as cold snap

# Price-aware overshoot protection (Dec 4, 2025)
# Rapid cooling detection (Oct 19, 2025)
# Detects rapid indoor temperature changes and boosts heating proactively
RAPID_COOLING_THRESHOLD: Final = -0.3  # °C/hour for rapid cooling detection
RAPID_COOLING_BOOST_MULTIPLIER: Final = 2.0  # Boost proportional to rate (rate * multiplier)
RAPID_COOLING_BOOST_MAX: Final = 3.0  # Maximum boost for rapid cooling (°C offset)
RAPID_COOLING_WEIGHT: Final = 0.8  # High priority weight (not safety-level)
RAPID_COOLING_OUTDOOR_THRESHOLD: Final = 0.0  # Cold weather threshold for boost (°C)

# Consolidated thermal change thresholds (Oct 19, 2025 - Deduplication)
# These constants consolidate multiple similar thresholds with the same semantic meaning
# Source: DUPLICATE_CONSTANTS_ANALYSIS.md - 78% duplicate reduction
THERMAL_CHANGE_MODERATE: Final = 0.3  # °C or °C/h - moderate temperature change threshold
THERMAL_CHANGE_MODERATE_COOLING: Final = -0.2  # °C/h - moderate cooling rate threshold

# Consolidated multipliers (Oct 19, 2025 - Deduplication)
# Standard multipliers used across multiple contexts for consistency
MULTIPLIER_REDUCTION_20_PERCENT: Final = 0.8  # 20% reduction (80% of original)
MULTIPLIER_BOOST_30_PERCENT: Final = 1.3  # 30% increase (130% of original)

# Trend-aware damping (Oct 19, 2025)
# Prevents overshoot/undershoot by damping offset based on indoor temperature trend
TREND_DAMPING_WARMING: Final = 0.75  # 25% reduction when warming rapidly
TREND_DAMPING_COOLING_BOOST: Final = 1.15  # 15% boost when cooling rapidly
TREND_BOOST_OFFSET_LIMIT: Final = 3.0  # Don't boost if offset already high (safety)
TREND_DAMPING_NEUTRAL: Final = 1.0  # No damping when trend stable

# Weather prediction layer - Simplified proactive pre-heating (Oct 20, 2025)
# Philosophy: "The heating we add NOW shows up in 6 hours - pre-heat BEFORE cold arrives"
#
# Problem: Concrete slab 6-hour thermal lag causes reactive heating to arrive too late,
#          resulting in thermal debt spirals (DM -1000) followed by 26°C overshoot.
#
# Solution: Simple forecast-based pre-heating scaled by thermal mass:
#   - Forecast ≥5°C drop in next 12h → +0.6°C gentle pre-heat
#   - Indoor cooling ≥0.5°C/h → confirms forecast, maintains +0.6°C
#   - Weight scaled by thermal mass: Concrete 1.28x, Timber 0.85x, Radiator 0.43x
#   - Let SAFETY, COMFORT, EFFECT layers moderate naturally via weighted aggregation
#
# Real-world validation: Prevents 20:00→04:00 emergency cycles and 16:00 overshoot
WEATHER_FORECAST_DROP_THRESHOLD: Final = -5.0  # °C drop in forecast (increased sensitivity)
WEATHER_FORECAST_HORIZON: Final = 12.0  # Hours to scan forecast (matches thermal lag)
WEATHER_GENTLE_OFFSET: Final = 0.83  # °C - gentle pre-heat (tuned Oct 20, was 0.5→0.6→0.7→0.77)
WEATHER_INDOOR_COOLING_CONFIRMATION: Final = -0.5  # °C/h - confirms forecast accuracy
LAYER_WEIGHT_WEATHER_PREDICTION: Final = 0.85  # Base weight (scaled by thermal mass)
WEATHER_WEIGHT_CAP: Final = 0.99  # Cap for weather weight (below Safety 1.0)

# Price layer constants (Oct 19, 2025)
# Tolerance scaling: maps UI range (0.5-3.0) to factor range (0.2-1.0)
# Formula: factor = 0.2 + (tolerance - 0.5) * 0.32
# At tolerance 0.5: factor = 0.2 (20% of offset - most conservative)
# At tolerance 3.0: factor = 1.0 (100% of offset - full effect)
PRICE_TOLERANCE_MIN: Final = 0.5  # UI slider minimum
PRICE_TOLERANCE_MAX: Final = 3.0  # UI slider maximum
PRICE_TOLERANCE_FACTOR_MIN: Final = 0.2  # Minimum scaling factor (20%)
PRICE_TOLERANCE_FACTOR_MAX: Final = 1.0  # Maximum scaling factor (100%)

# Heat pump compressor dynamics (Nov 28, 2025)
# Based on typical NIBE F-series compressor behavior
COMPRESSOR_RAMP_UP_MINUTES: Final = 30  # Minutes to reach full speed from idle
COMPRESSOR_COOL_DOWN_MINUTES: Final = 15  # Minutes for thermal stabilization after reduction
COMPRESSOR_MIN_CYCLE_MINUTES: Final = (
    COMPRESSOR_RAMP_UP_MINUTES + COMPRESSOR_COOL_DOWN_MINUTES
)  # 45min total

# Price forecast lookahead (Nov 27, 2025)
# Forward-looking price optimization: reduce heating when cheaper period coming soon
# Updated Nov 28, 2025: Horizon scales with thermal_mass (configurable 0.5-2.0)
# Updated Dec 2, 2025: Increased from 4h to 5h for better PEAK visibility
PRICE_FORECAST_BASE_HORIZON: Final = 5.0  # hours - base value multiplied by thermal_mass
# Examples of dynamic scaling:
#   thermal_mass 2.0 (heavy concrete slab): 4.0 × 2.0 = 8.0 hours lookahead
#   thermal_mass 1.2 (medium timber):       4.0 × 1.2 = 4.8 hours lookahead
#   thermal_mass 0.8 (light radiator):      4.0 × 0.8 = 3.2 hours lookahead
PRICE_FORECAST_CHEAP_THRESHOLD: Final = (
    0.5  # Price ratio - upcoming < 50% of current = "much cheaper"
)
PRICE_FORECAST_EXPENSIVE_THRESHOLD: Final = (
    1.5  # Price ratio - upcoming > 150% of current = "much more expensive"
)
PRICE_FORECAST_MIN_DURATION: Final = (
    COMPRESSOR_MIN_CYCLE_MINUTES // 15
)  # quarters - derived from compressor dynamics (45min / 15min = 3)
PRICE_FORECAST_REDUCTION_OFFSET: Final = (
    -1.5
)  # °C - reduce heating when cheap period coming (Dec 5, 2025: strengthened from -1.0)
PRICE_FORECAST_PREHEAT_OFFSET: Final = 2.0  # °C - pre-heat when expensive period coming

# Volatile price detection - Current run-length approach (Dec 2, 2025)
# A quarter is volatile if its own run is brief (< 3 quarters / 45 min).
# Compressor needs ~45 min to ramp up - brief periods cause wear.
#
# EXAMPLES:
# - EXPENSIVE(4Q), NORMAL(1Q), EXPENSIVE(4Q) → at NORMAL: run=1 → VOLATILE
# - NORMAL(3Q), CHEAP(1Q), NORMAL(3Q) → at CHEAP: run=1 → VOLATILE
# - NORMAL(2Q), CHEAP(4Q), NORMAL(2Q) → at CHEAP: run=4 → NOT VOLATILE
#
# PEAK EXCLUDED: Always gets full response (never suppressed)
# WEIGHT REDUCTION: 0.8 → 0.24 (70% reduction during volatility)
PRICE_VOLATILE_WEIGHT_REDUCTION: Final = 0.3  # Retain 30% weight during volatility (0.8 → 0.24)

# Price classification percentile thresholds (Dec 8, 2025)
# Define the boundaries between price classifications
PRICE_PERCENTILE_VERY_CHEAP: Final = 10  # Bottom 10% = VERY_CHEAP
PRICE_PERCENTILE_CHEAP: Final = 25  # 10-25% = CHEAP
PRICE_PERCENTILE_NORMAL: Final = 75  # 25-75% = NORMAL
PRICE_PERCENTILE_EXPENSIVE: Final = 90  # 75-90% = EXPENSIVE
# Above 90% = PEAK

# Price classification base offsets (Dec 3, 2025, updated Dec 8, 2025)
# Used by price_analyzer.py get_base_offset() for spot price optimization
# These are base values before daytime multiplier (1.5x for EXPENSIVE during 06:00-22:00)
PRICE_OFFSET_VERY_CHEAP: Final = 4.0  # °C - exceptional prices, aggressive pre-heating!
PRICE_OFFSET_CHEAP: Final = 1.5  # °C - good prices, moderate pre-heating (reduced from 3.0)
PRICE_OFFSET_NORMAL: Final = 0.0  # °C - maintain current heating
PRICE_OFFSET_EXPENSIVE: Final = -1.0  # °C - conserve, reduce heating
PRICE_OFFSET_PEAK: Final = MIN_OFFSET  # Maximum reduction (coast through expensive period)
PRICE_DAYTIME_MULTIPLIER: Final = 1.5  # Multiplier for EXPENSIVE during daytime (06:00-22:00)

# Pre-PEAK offset (Dec 2, 2025, Updated Dec 3, 2025)
# Start reducing heating 1 quarter BEFORE peak to allow pump slowdown
# Pump needs time to reduce - acting at PEAK start is too late
# Dec 3, 2025: Set to -6.0 for gradual decrease before PEAK (-10.0 was too aggressive)
PRICE_PRE_PEAK_OFFSET: Final = -6.0  # °C - gradual reduction before PEAK arrives

# Comfort layer constants (Oct 19, 2025)
COMFORT_DEAD_ZONE: Final = 0.2  # ±0.2°C dead zone (no action)
COMFORT_CORRECTION_MULT: Final = 0.3  # Gentle correction multiplier
COMFORT_DM_COOLING_THRESHOLD: Final = (
    -200  # Block cooling corrections when DM < -200 (thermal debt accumulating)
)

# Comfort layer thermal-aware calculations (Dec 8, 2025)
# Heat loss rate calculation: base_heat_loss = temp_diff / (insulation * HEAT_LOSS_DIVISOR)
# Well-insulated Swedish house loses ~0.15°C/hour at 20°C indoor/outdoor diff
# Formula: heat_loss = (temp_diff / REF_DIFF) * RATE / insulation
#          = temp_diff / (insulation * REF_DIFF / RATE)
# Updated Dec 12, 2025: Was hardcoded 10.0 which gave 13x too high heat loss rates
# Now dynamically calculated from reference constants (defined in prediction layer section below)
COMFORT_HEAT_LOSS_FLOOR: Final = 0.02  # Minimum effective heat loss rate (°C/h)
COMFORT_TOO_COLD_CORRECTION_MULT: Final = 0.5  # Multiplier for "too cold" correction

# Effect layer peak protection margins and offsets (Dec 8, 2025)
# Power margin thresholds for peak protection decisions (kW)
EFFECT_PEAK_MARGIN_CRITICAL: Final = 0.5  # kW - within 0.5 kW = critical
EFFECT_PEAK_MARGIN_WARNING: Final = 1.0  # kW - within 1.0 kW = warning
# Recommended offsets for peak protection
EFFECT_PEAK_OFFSET_EXCEEDING: Final = -3.0  # Aggressive reduction when exceeding peak
EFFECT_PEAK_OFFSET_CRITICAL: Final = -2.0  # Strong reduction within 0.5 kW
EFFECT_PEAK_OFFSET_WARNING: Final = -1.0  # Moderate reduction within 1.0 kW

# Prediction layer thresholds (Dec 8, 2025)
# Temperature trend detection thresholds
PREDICTION_TREND_RISING_THRESHOLD: Final = 0.3  # °C delta to classify as "rising"
PREDICTION_TREND_FALLING_THRESHOLD: Final = -0.3  # °C delta to classify as "falling"
# Default thermal responsiveness (learned or fallback)
PREDICTION_THERMAL_RESPONSIVENESS_DEFAULT: Final = 0.5  # Default if not learned
PREDICTION_THERMAL_RESPONSIVENESS_MIN: Final = 0.0  # Minimum valid responsiveness
PREDICTION_THERMAL_RESPONSIVENESS_MAX: Final = 1.0  # Maximum valid responsiveness
PREDICTION_MIN_HEATING_OFFSET: Final = 0.5  # Minimum offset for responsiveness learning
# Climate-aware pre-heating comfort thresholds (°C deficit before pre-heating)
PREDICTION_COMFORT_THRESHOLD_EXTREME_COLD: Final = 1.0  # Allow 1.0°C deficit at <-15°C
PREDICTION_COMFORT_THRESHOLD_COLD: Final = 0.7  # Allow 0.7°C deficit at -15°C to -5°C
PREDICTION_COMFORT_THRESHOLD_MILD: Final = 0.5  # Allow 0.5°C deficit at >-5°C
# Outdoor temperature boundaries for comfort threshold selection
PREDICTION_OUTDOOR_EXTREME_COLD: Final = -15.0  # °C threshold for extreme cold
PREDICTION_OUTDOOR_COLD: Final = -5.0  # °C threshold for cold
# Pre-heating offset multipliers by climate
PREDICTION_PREHEAT_MULT_EXTREME_COLD: Final = 1.2  # Conservative in extreme cold
PREDICTION_PREHEAT_MULT_COLD: Final = 1.5  # Moderate in cold
PREDICTION_PREHEAT_MULT_MILD: Final = 2.0  # Normal in mild
PREDICTION_PREHEAT_MAX_EXTREME_COLD: Final = 2.0  # Max offset in extreme cold
PREDICTION_PREHEAT_MAX_COLD: Final = 2.5  # Max offset in cold
PREDICTION_PREHEAT_MAX_MILD: Final = 3.0  # Max offset in mild
# Fallback heat loss model (when learned params unavailable)
# Reference: Well-insulated Swedish house loses ~0.1-0.2°C/hour at 20°C indoor/outdoor diff
PREDICTION_FALLBACK_HEAT_LOSS_RATE: Final = 0.15  # °C/hour at reference temp diff
PREDICTION_FALLBACK_HEAT_LOSS_REF_DIFF: Final = 20.0  # Reference indoor/outdoor temp diff
# Derived divisor for heat loss formula: divisor = REF_DIFF / RATE
# Used by both comfort_layer and prediction_layer for consistent heat loss calculations
HEAT_LOSS_DIVISOR: Final = (
    PREDICTION_FALLBACK_HEAT_LOSS_REF_DIFF / PREDICTION_FALLBACK_HEAT_LOSS_RATE
)
# Current overshoot threshold for logging thermal storage strategy
PREDICTION_OVERSHOOT_LOG_THRESHOLD: Final = 0.1  # °C overshoot to log storage strategy

# DHW optimizer space heating emergency thresholds (Dec 8, 2025)
DHW_SPACE_HEATING_DEFICIT_THRESHOLD: Final = 0.5  # °C indoor deficit to block DHW
DHW_SPACE_HEATING_OUTDOOR_THRESHOLD: Final = 0.0  # °C outdoor temp for emergency check
# DHW trend-based blocking thresholds
DHW_TREND_DEFICIT_THRESHOLD: Final = 0.3  # °C indoor deficit for trend check
DHW_TREND_RATE_THRESHOLD: Final = -0.3  # °C/h cooling rate for trend check

# UFH prediction horizons based on thermal lag research
# Prediction horizons for different UFH types
UFH_CONCRETE_PREDICTION_HORIZON: Final = (
    24.0  # hours - 6+ hour lag, needs 24h for extreme cold (20°C drops)
)
UFH_TIMBER_PREDICTION_HORIZON: Final = 12.0  # hours - 2-3 hour lag (increased from 6h)
UFH_RADIATOR_PREDICTION_HORIZON: Final = 6.0  # hours - <1 hour lag (increased from 2h)

DEFAULT_CURVE_SENSITIVITY: Final = 1.5  # NIBE curve sensitivity (~1.5°C flow change per 1°C offset)

# Weather compensation mathematical constants
KUEHNE_COEFFICIENT: Final = 2.55  # Universal coefficient for flow temperature calculation
KUEHNE_POWER: Final = 0.78  # Power coefficient for heat transfer physics
RADIATOR_POWER_COEFFICIENT: Final = 1.3  # BS EN442 standard radiator output
RADIATOR_RATED_DT: Final = 50.0  # Standard test DT (75°C flow, 65°C return, 20°C room)

# UFH flow temperature adjustments
UFH_FLOW_REDUCTION_CONCRETE: Final = 8.0  # °C reduction for concrete slab UFH
UFH_FLOW_REDUCTION_TIMBER: Final = 5.0  # °C reduction for timber/lightweight UFH
UFH_MIN_FLOW_TEMP_CONCRETE: Final = 25.0  # Minimum effective concrete slab temp
UFH_MIN_FLOW_TEMP_TIMBER: Final = 22.0  # Minimum effective timber UFH temp

# Heat loss coefficient defaults (W/°C)
DEFAULT_HEAT_LOSS_COEFFICIENT: Final = 180.0  # W/°C typical value

# Power estimation defaults (kW)
# Used when actual power sensor unavailable - fallback values
DEFAULT_BASE_POWER: Final = 3.0  # kW typical NIBE heat pump average
TEMP_FACTOR_MIN: Final = 0.5  # Minimum temperature scaling factor
TEMP_FACTOR_MAX: Final = 3.0  # Maximum temperature scaling factor

# Effect tariff peak tracking thresholds (kW)
# Source: Real-world NIBE heat pump consumption patterns
# Prevents recording standby/startup noise as legitimate peaks
# Start with lower threshold for initial learning phase
# System learns from actual usage patterns even during idle periods
PEAK_RECORDING_MINIMUM: Final = 0.5  # kW - lowered from 1.0 for better learning
# Typical NIBE consumption: standby 0.05-0.1 kW, heating 2.5-6.0 kW

# Update intervals
UPDATE_INTERVAL_MINUTES: Final = (
    5  # Coordinator update frequency + thermal predictor save throttle interval
)
QUARTER_INTERVAL_MINUTES: Final = 15  # Swedish Effektavgift measurement period

# Thermal predictor history constants (derived from UPDATE_INTERVAL_MINUTES)
SAMPLES_PER_HOUR: Final = 60 // UPDATE_INTERVAL_MINUTES  # 12 samples/hour with 5-min intervals

# Adaptive learning parameters
# Source: POST_PHASE_5_ROADMAP.md Phase 6 - Self-Learning Capability
LEARNING_OBSERVATION_WINDOW: Final = 672  # 1 week of 15-minute observations
LEARNING_MIN_OBSERVATIONS: Final = 96  # 24 hours minimum for basic learning
LEARNING_CONFIDENCE_THRESHOLD: Final = 0.7  # 70% confidence to use learned params

# Swedish climate regions - SMHI historical data (1961-1990)
# Source: Swedish_Climate_Adaptations.md
CLIMATE_SOUTHERN_SWEDEN: Final = "southern_sweden"  # Malmö/Gothenburg (0°C Jan avg)
CLIMATE_CENTRAL_SWEDEN: Final = "central_sweden"  # Stockholm (-4°C Jan avg)
CLIMATE_MID_NORTHERN_SWEDEN: Final = "mid_northern_sweden"  # Umeå/Östersund (-8°C Jan avg)
CLIMATE_NORTHERN_SWEDEN: Final = "northern_sweden"  # Luleå (-11°C Jan avg)
CLIMATE_NORTHERN_LAPLAND: Final = "northern_lapland"  # Kiruna (-13°C Jan avg)

# Climate zones - Import from dedicated module
# Source: optimization/climate_zones.py
# See IMPLEMENTATION_PLAN/FUTURE/CLIMATE_ZONE_DM_INTEGRATION.md
#
# DESIGN PHILOSOPHY:
# - Heating-focused zones (extreme_cold → standard)
# - Latitude-based automatic detection
# - Context-aware DM thresholds per zone
# - Used by decision engine for emergency layer
# - Used by weather compensation for safety margins
#
# Import is done at runtime to avoid circular dependencies - use the climate_zones module directly.

# Storage
STORAGE_VERSION: Final = 1
STORAGE_KEY: Final = f"{DOMAIN}_state"
STORAGE_KEY_LEARNING: Final = f"{DOMAIN}_learned_data"

# Services
SERVICE_SET_OPTIMIZATION_MODE: Final = "set_optimization_mode"
SERVICE_FORCE_UPDATE: Final = "force_update"
SERVICE_RESET_PEAKS: Final = "reset_peaks"
SERVICE_FORCE_OFFSET: Final = "force_offset"
SERVICE_RESET_PEAK_TRACKING: Final = "reset_peak_tracking"
SERVICE_BOOST_HEATING: Final = "boost_heating"
SERVICE_BOOST_DHW: Final = "boost_dhw"
SERVICE_CALCULATE_OPTIMAL_SCHEDULE: Final = "calculate_optimal_schedule"

# Service parameters
ATTR_OFFSET: Final = "offset"
ATTR_DURATION: Final = "duration"
ATTR_TARGET_TEMP: Final = "target_temp"

# Attributes
ATTR_CURRENT_OFFSET: Final = "current_offset"
ATTR_DECISION_REASONING: Final = "decision_reasoning"
ATTR_LAYER_VOTES: Final = "layer_votes"
ATTR_PEAK_TODAY: Final = "peak_today"
ATTR_PEAK_THIS_MONTH: Final = "peak_this_month"
ATTR_THERMAL_DEBT: Final = "thermal_debt"
ATTR_QUARTER_OF_DAY: Final = "quarter_of_day"
ATTR_OPTIONAL_FEATURES: Final = "optional_features_status"

# DHW (Domestic Hot Water) Optimization Constants
# Based on DHW_RESEARCH_FINDINGS.md and DHW_IMPLEMENTATION_CORRECTIONS.md
#
# Temperature hierarchy:
# - 20°C (DHW_SAFETY_CRITICAL): Hard floor, always heat (emergency)
# - 30°C (DHW_SAFETY_MIN): Price optimization minimum (allows tank to cool for better price-based heating)
# - 40°C (DHW_MIN_TEMP): User-configurable minimum (validation)
# - 45°C (MIN_DHW_TARGET_TEMP): Minimum user target / NIBE start threshold
# - 50°C (NIBE_DHW_NORMAL_TARGET): Normal comfort target
# - 60°C (DHW_MAX_TEMP): Maximum comfort temperature
#
DHW_MIN_TEMP: Final = 40.0  # °C - Minimum safe DHW temperature (user validation)
DHW_MAX_TEMP: Final = 60.0  # °C - Maximum normal DHW temperature (comfort)
DHW_MAX_TEMP_VALIDATION: Final = 65.0  # °C - Absolute maximum for validation (near Legionella temp)
DHW_PREHEAT_TARGET_OFFSET: Final = 5.0  # °C - Extra heating above target for optimal windows
MIN_DHW_TARGET_TEMP: Final = (
    45.0  # °C - Minimum user-configurable DHW target (safety + comfort threshold)
)
DHW_READY_THRESHOLD: Final = (
    52.0  # °C - DHW at normal target (50°C + 2°C buffer for "ready" status)
)

# Legionella Prevention (requires DHW tank immersion heater, Swedish: elpatron)
# Boverket.se official guidelines:
# - Legionella bacteria grow at 20-45°C (our new low-temp optimization range!)
# - Legionella dormant below 20°C
# - Killed at high temperatures (≥60°C)
# - Water heaters should maintain ≥60°C to prevent bacterial growth
#
# Heat pump limitation:
# - Compressor can only reach ~50-55°C max (COP/efficiency limits)
# - NIBE automatically engages DHW tank immersion heater to reach 60°C
# - This is standard operation for all NIBE Legionella protection
# - Our hygiene boost schedules this during cheapest electricity periods
#
# NOTE: DHW immersion heater (elpatron) is separate from space heating auxiliary heater.
# They are different electrical heating systems with different purposes.
DHW_LEGIONELLA_DETECT: Final = (
    55.0  # °C - BT7 temp indicating Legionella boost complete (observed 55.3°C in production)
)
DHW_LEGIONELLA_PREVENT_TEMP: Final = (
    56.0  # °C - Target temp for hygiene boost (kills bacteria, requires immersion heater)
)
DHW_LEGIONELLA_MAX_DAYS: Final = 14.0  # Days - Max time without high-temp cycle (hygiene)
DHW_HEATING_TIME_HOURS: Final = 1.5  # Hours to heat DHW tank (typically 1-2h)
DHW_SCHEDULING_WINDOW_MAX: Final = 24  # Max hours ahead for DHW scheduling
DHW_SCHEDULING_WINDOW_MIN: Final = 0.25  # Min hours ahead (15 min minimum for meaningful pre-heat)
DHW_MAX_WAIT_HOURS: Final = 36.0  # Max hours between DHW heating (hygiene/comfort)

# DHW thermal debt fallback thresholds (used only if climate detector unavailable)
# These are balanced fixed values for rare fallback scenarios
DM_DHW_BLOCK_FALLBACK: Final = -340.0  # Fallback: Never start DHW below this DM
DM_DHW_ABORT_FALLBACK: Final = -500.0  # Fallback: Abort DHW if reached during run

# DHW runtime safeguards (monitoring only - NIBE controls actual completion)
DHW_SAFETY_RUNTIME_MINUTES: Final = 30  # Safety minimum heating (emergency)
DHW_NORMAL_RUNTIME_MINUTES: Final = 45  # Normal DHW heating window
DHW_EXTENDED_RUNTIME_MINUTES: Final = 60  # High demand period heating
DHW_URGENT_RUNTIME_MINUTES: Final = 90  # Urgent pre-demand heating

# DHW demand period thresholds
DHW_URGENT_DEMAND_HOURS: Final = 0.5  # Start urgent heating 30 min before demand period

# NIBE Power Calculation Constants (Swedish 3-phase standard)
# All NIBE heat pumps in Sweden are 3-phase systems
NIBE_VOLTAGE_PER_PHASE: Final = (
    240.0  # V - Swedish 3-phase: 400V between phases, 240V phase-to-neutral
)
NIBE_POWER_FACTOR: Final = 0.95  # Conservative for inverter compressor (real likely 0.96-0.98)

# ============================================================================
# Airflow Optimizer Constants (Exhaust Air Heat Pump)
# ============================================================================
# Description:
#   Original thermodynamic calculations for exhaust air heat pump airflow optimization.
#   Enhanced airflow extracts more heat from exhaust air, improving COP when conditions
#   are favorable. Based on first-principles energy balance.
#
# Physics:
#   Net Benefit = (Extra heat extracted) + (COP improvement) - (Ventilation penalty)
#   Q_extract = ṁ × cp × ΔT = (flow_m³h / 3600) × 1.2 kg/m³ × 1.005 kJ/kg·K × 12°C
#   COP_enhanced ≈ 1.20 × COP_standard (warmer evaporator from enhanced airflow)
#   Q_penalty = ṁ × cp × (T_indoor - T_outdoor)
#
# When Enhanced Airflow Helps:
#   Outdoor °C | Min Compressor % | Expected Gain
#   +10        | 50%              | +1.3 kW
#   0          | 50%              | +0.9 kW
#   -5         | 62%              | +0.7 kW
#   -10        | 75%              | +0.4 kW
#   < -15      | Don't enhance    | Negative
#
# Reference:
#   - Original thermodynamic first-principles analysis
#   - NIBE F750 ventilation specifications
# ============================================================================

# Physical constants for airflow calculations
AIRFLOW_AIR_DENSITY: Final = 1.2  # kg/m³ at ~20°C
AIRFLOW_SPECIFIC_HEAT: Final = 1.005  # kJ/kg·K for air
AIRFLOW_EVAPORATOR_TEMP_DROP: Final = 12.0  # °C typical temperature drop through evaporator

# Default flow rates for NIBE F750 (m³/h)
# Standard: Normal ventilation speed (speed 2)
# Enhanced: Maximum ventilation speed (speed 4)
AIRFLOW_DEFAULT_STANDARD: Final = 150.0  # m³/h - Normal ventilation
AIRFLOW_DEFAULT_ENHANCED: Final = 252.0  # m³/h - Maximum ventilation

# COP improvement from enhanced airflow
# More air → warmer evaporator → better COP
# Empirically ~20% improvement at enhanced flow
AIRFLOW_COP_IMPROVEMENT_FACTOR: Final = 1.20  # 20% COP improvement
AIRFLOW_BASE_COP: Final = 3.3  # Typical NIBE F750 COP

# Compressor input power for benefit calculations
AIRFLOW_COMPRESSOR_INPUT_KW: Final = 2.0  # kW typical compressor electrical input

# Temperature thresholds
AIRFLOW_OUTDOOR_TEMP_MIN: Final = -15.0  # °C - Never enhance below this (penalty exceeds gains)
AIRFLOW_INDOOR_DEFICIT_MIN: Final = 0.2  # °C - Minimum deficit to trigger enhancement

# Compressor threshold calculation
# Minimum compressor % needed for enhanced flow to be beneficial
# Colder outside → need higher compressor output to justify extra ventilation
# Formula: threshold = 50 + slope * outdoor_temp (for outdoor_temp < 0)
AIRFLOW_COMPRESSOR_BASE_THRESHOLD: Final = 50.0  # % base threshold at 0°C
AIRFLOW_COMPRESSOR_SLOPE: Final = -2.5  # Increase by 2.5% per degree below 0°C

# Temperature deficit thresholds for duration calculation (°C)
AIRFLOW_DEFICIT_SMALL_THRESHOLD: Final = 0.3  # Small deficit boundary
AIRFLOW_DEFICIT_MODERATE_THRESHOLD: Final = 0.5  # Moderate deficit boundary
AIRFLOW_DEFICIT_LARGE_THRESHOLD: Final = 1.0  # Large deficit boundary

# Outdoor temperature thresholds for duration caps (°C)
AIRFLOW_TEMP_COLD_THRESHOLD: Final = -10.0  # Very cold - apply duration cap
AIRFLOW_TEMP_COOL_THRESHOLD: Final = -5.0  # Cool - apply moderate duration cap

# Duration calculation for enhanced airflow (minutes)
AIRFLOW_DURATION_SMALL_DEFICIT: Final = 15  # minutes for deficit < 0.3°C
AIRFLOW_DURATION_MODERATE_DEFICIT: Final = 20  # minutes for deficit 0.3-0.5°C
AIRFLOW_DURATION_LARGE_DEFICIT: Final = 45  # minutes for deficit 0.5-1.0°C
AIRFLOW_DURATION_EXTREME_DEFICIT: Final = 60  # minutes for deficit > 1.0°C
AIRFLOW_DURATION_COLD_CAP: Final = 20  # minutes cap when outdoor < -10°C
AIRFLOW_DURATION_COOL_CAP: Final = 30  # minutes cap when outdoor < -5°C

# Indoor temperature trend threshold for enhancement decision
AIRFLOW_TREND_WARMING_THRESHOLD: Final = 0.1  # °C/h - already warming, let stabilize

# Configuration keys
CONF_ENABLE_AIRFLOW_OPTIMIZATION: Final = "enable_airflow_optimization"
CONF_AIRFLOW_STANDARD_RATE: Final = "airflow_standard_rate"  # m³/h
CONF_AIRFLOW_ENHANCED_RATE: Final = "airflow_enhanced_rate"  # m³/h

# NIBE Enhanced Ventilation (F750/F730)
# Controls the "Increased Ventilation" switch on exhaust air heat pumps
# Entity pattern: switch.{device}_increased_ventilation
NIBE_VENTILATION_MIN_ENHANCED_DURATION: Final = 5  # Minimum minutes to run enhanced

DHW_SAFETY_CRITICAL: Final = 20.0  # °C - Hard floor, always heat below this (emergency)
DHW_SAFETY_MIN: Final = 30.0  # °C - Safety minimum (can defer if 20-30°C during expensive periods)
DHW_COOLING_RATE: Final = 0.5  # °C/hour - Conservative DHW tank cooling estimate

# NIBE Adapter Constants
NIBE_DEFAULT_SUPPLY_TEMP: Final = 35.0  # °C - Default supply/flow temp when sensor unavailable
NIBE_FRACTIONAL_ACCUMULATOR_THRESHOLD: Final = (
    1.0  # °C - Write to NIBE when accumulator crosses ±1.0
)

# DHW Recovery Time Estimation
# DM recovery rates based on heat pump efficiency at different outdoor temperatures
DM_RECOVERY_RATE_MILD: Final = 40.0  # DM/hour - Mild weather (>5°C), efficient heat pump
DM_RECOVERY_RATE_COLD: Final = 30.0  # DM/hour - Cold weather (0-5°C), moderate efficiency
DM_RECOVERY_RATE_VERY_COLD: Final = 20.0  # DM/hour - Very cold (<0°C), reduced efficiency
DM_RECOVERY_MAX_HOURS: Final = 12.0  # Maximum recovery time estimate (if longer, error condition)
DM_RECOVERY_SAFETY_BUFFER: Final = 20.0  # DM - Safety buffer above warning threshold

# Indoor temperature recovery estimation (used for test validation bounds)
INDOOR_TEMP_RECOVERY_MAX_HOURS: Final = 6.0  # Maximum recovery time for validation

# Space heating demand estimation (Nov 30, 2025)
# Used for DHW blocking decisions and heating demand display
# Uses actual power sensor reading (sensor.effektguard_nibe_power)
SPACE_HEATING_DEMAND_HIGH_THRESHOLD: Final = 6.0  # kW - DHW blocking threshold
SPACE_HEATING_DEMAND_MODERATE_THRESHOLD: Final = 2.0  # kW - Display threshold
SPACE_HEATING_DEMAND_LOW_THRESHOLD: Final = 0.5  # kW - Display threshold
SPACE_HEATING_DEMAND_DROP_HOURS: Final = 2.0  # Conservative estimate for demand to drop

# Savings Calculation Constants (Swedish electricity market)
# Swedish effect tariff - typical cost per kW of monthly peak
# Based on common Swedish grid operators (Ellevio ~55, Vattenfall/E.ON ~50 SEK/kW/month)
SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH: Final = 50.0  # Conservative average

# Baseline peak estimation - assumes optimization reduces peak by ~15%
# If no baseline observed, estimate unoptimized peak from current optimized peak
BASELINE_PEAK_MULTIPLIER: Final = 1.176  # Inverse of 0.85 (15% reduction)

# Monthly calculation constants
DAYS_PER_MONTH: Final = 30.0  # Average days for monthly savings calculation
ORE_TO_SEK_CONVERSION: Final = 100.0  # Convert öre to SEK (1 SEK = 100 öre)

# Baseline tracking - exponential moving average weights
BASELINE_EMA_WEIGHT_OLD: Final = 0.8  # 80% weight on existing baseline
BASELINE_EMA_WEIGHT_NEW: Final = 0.2  # 20% weight on new observation

# Default heat pump power for savings estimation (system-specific, will vary)
# F750: typically 2-7 kW depending on outdoor temp and compressor frequency
DEFAULT_HEAT_PUMP_POWER_KW: Final = 4.0  # Mid-range estimate for average conditions

# Power estimation constants (moved from coordinator for shared reuse)
# Standby power when compressor not running
POWER_STANDBY_KW: Final = 0.1

# Temperature-based power multipliers for basic estimation
POWER_TEMP_VERY_COLD_THRESHOLD: Final = -10.0  # °C
POWER_TEMP_COLD_THRESHOLD: Final = 0.0  # °C
POWER_MULTIPLIER_VERY_COLD: Final = 1.3  # <-10°C
POWER_MULTIPLIER_COLD: Final = 1.1  # -10 to 0°C
POWER_MULTIPLIER_MILD: Final = 1.0  # >0°C

# Compressor frequency-based power estimation
# Based on typical NIBE F750 performance curves:
# - 20 Hz (minimum): ~1.5-2.0 kW
# - 50 Hz (mid): ~3.5-4.5 kW
# - 80 Hz (maximum): ~6.0-7.0 kW
COMPRESSOR_HZ_MIN: Final = 20
COMPRESSOR_HZ_RANGE: Final = 60  # 80-20
COMPRESSOR_POWER_MIN_KW: Final = 1.5
COMPRESSOR_POWER_RANGE_KW: Final = 5.0  # 6.5-1.5
COMPRESSOR_POWER_MAX_KW: Final = 6.5

# Temperature factor for compressor power estimation
COMPRESSOR_TEMP_EXTREME_COLD_THRESHOLD: Final = -15.0
COMPRESSOR_TEMP_COLD_THRESHOLD: Final = -5.0
COMPRESSOR_TEMP_COOL_THRESHOLD: Final = 0.0
COMPRESSOR_TEMP_FACTOR_EXTREME_COLD: Final = 1.3
COMPRESSOR_TEMP_FACTOR_COLD: Final = 1.2
COMPRESSOR_TEMP_FACTOR_COOL: Final = 1.1
COMPRESSOR_TEMP_FACTOR_MILD: Final = 1.0


class OptimizationMode(StrEnum):
    """Optimization mode options."""

    BASIC = "basic"
    PRICE = "price"
    ADVANCED = "advanced"
    EXPERT = "expert"


class QuarterClassification(StrEnum):
    """Price period classification based on spot prices.

    Can represent hourly or 15-minute (quarterly) periods depending on price data granularity.

    Classifications (Dec 8, 2025):
        VERY_CHEAP: Bottom 10% (P10) - Exceptional prices, aggressive pre-heating
        CHEAP: 10-25% (P10-P25) - Good prices, moderate pre-heating
        NORMAL: 25-75% (P25-P75) - Average prices, maintain
        EXPENSIVE: 75-90% (P75-P90) - High prices, reduce heating
        PEAK: Top 10% (P90+) - Critical prices, maximum reduction
    """

    VERY_CHEAP = "very_cheap"  # Bottom 10% - exceptional prices
    CHEAP = "cheap"  # 10-25% - good prices
    NORMAL = "normal"
    EXPENSIVE = "expensive"
    PEAK = "peak"


class UFHType(StrEnum):
    """Heating system thermal response type (detected from lag time, not material)."""

    SLOW_RESPONSE = "slow_response"  # 6+ hours lag (typically concrete slab UFH)
    MEDIUM_RESPONSE = "medium_response"  # 2-3 hours lag (typically timber UFH)
    FAST_RESPONSE = "fast_response"  # <1 hour lag (typically radiators)
    UNKNOWN = "unknown"


class SystemType(StrEnum):
    """Heat pump system configuration type."""

    OPEN_LOOP = "open_loop"  # Direct UFH, requires Auto pump mode
    BUFFERED = "buffered"  # Buffer tank, Intermittent mode OK
    MIXED = "mixed"  # Both direct and buffered zones
    UNKNOWN = "unknown"


class PumpMode(StrEnum):
    """NIBE pump operation modes."""

    AUTO = "auto"  # Required for open-loop UFH
    INTERMITTENT = "intermittent"  # OK for buffered only
    CONTINUOUS = "continuous"
    UNKNOWN = "unknown"

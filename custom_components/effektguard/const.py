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
CONF_DHW_TEMP_ENTITY: Final = "dhw_temp_entity"  # Optional: DHW top temperature sensor (BT7)
# Manual sensor overrides for setups where name-pattern discovery cannot work
# (generic Modbus YAML, renamed entities, nibe_heatpump with re-added entries).
# Issue #18: local Modbus/nibe_heatpump users need these; discovery is fallback only.
CONF_OUTDOOR_TEMP_ENTITY: Final = "outdoor_temp_entity"  # Optional: BT1 override
CONF_INDOOR_TEMP_ENTITY: Final = "indoor_temp_entity"  # Optional: BT50/room sensor override
CONF_SUPPLY_TEMP_ENTITY: Final = "supply_temp_entity"  # Optional: BT2/BT25/BT63 override
CONF_RETURN_TEMP_ENTITY: Final = "return_temp_entity"  # Optional: BT3 override
CONF_DHW_CHARGING_TEMP_ENTITY: Final = "dhw_charging_temp_entity"  # Optional: BT6 override
CONF_NIBE_TEMP_LUX_ENTITY: Final = "nibe_temp_lux_entity"  # Optional: switch.temporary_lux_50004
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
CONF_TOLERANCE: Final = "tolerance"
CONF_OPTIMIZATION_MODE: Final = "optimization_mode"
CONF_THERMAL_MASS: Final = "thermal_mass"
CONF_INSULATION_QUALITY: Final = "insulation_quality"
CONF_HEAT_PUMP_MODEL: Final = "heat_pump_model"

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
DEFAULT_DHW_MORNING_HOUR: Final = 7  # Default morning DHW availability hour (07:00)
DEFAULT_DHW_EVENING_HOUR: Final = 18  # Default evening DHW availability hour (18:00)

# Climate entity temperature limits (displayed in UI).
# The MINIMUM is MIN_TEMP_LIMIT - the safety floor - not a separate number: a setpoint the
# safety layer answers with an emergency is not a setpoint (audit F-085).
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

# Swedish effect tariffs weight night quarters at half. This was a bare 0.5 in two places in
# effect_layer, and the fact that it is a WEIGHTING rather than a power was easy to lose sight of:
# the coordinator fed the savings baseline an unweighted peak and compared it against a weighted
# one, so a single night quarter reported 150 SEK/month of "savings" that were nothing but this
# number. Anything compared against a monthly peak has to be put through effective_tariff_power_kw.
NIGHT_TARIFF_WEIGHT: Final = 0.5

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

# The lowest target the system can actually HOLD, as opposed to the lowest it will not treat as an
# emergency. The safety layer fires below MIN_TEMP_LIMIT, and the comfort band is target ± tolerance
# - so a target sitting AT the floor puts the lower half of its own comfort band inside the
# emergency zone, and ordinary control noise trips a full MAX_OFFSET boost that carries
# is_emergency=True and bypasses the volatility blocker. A target must sit at least one tolerance
# above the floor (audit F-085).
#
# This is the slider's minimum, computed at the DEFAULT tolerance. The engine enforces the real
# bound - MIN_TEMP_LIMIT + the tolerance actually configured - and says so if it has to.
MIN_TARGET_TEMP: Final = MIN_TEMP_LIMIT + DEFAULT_TOLERANCE  # 18.5 °C at the default ±0.5

# Service call rate limiting (boost, DHW, general)
HEATING_BOOST_COOLDOWN_MINUTES: Final = 45  # Space heating boost cooldown
DHW_BOOST_COOLDOWN_MINUTES: Final = 60  # DHW boost cooldown
DHW_CONTROL_MIN_INTERVAL_MINUTES: Final = 60  # Automatic DHW control rate limit (1 hour)
SERVICE_RATE_LIMIT_MINUTES: Final = 5  # General service call cooldown

# DHW/Weather layer cooldown (Dec 19, 2025)
# After DHW heating stops, flow temperature remains elevated for a LONG time:
# - 20:25 (DHW stop): 55.4°C
# - 20:30 (+5 min): 48.3°C
# - 20:55 (+30 min): 46.3°C
# - 21:30 (+65 min): 44.3°C
# - 22:30 (+125 min): 41.8°C (still elevated!)
#
# The weather layer optimal flow at 8°C outdoor is ~26.6°C. Flow temp never
# reaches optimal even after 2 hours! When flow is much higher, weather layer
# sees huge "error" and recommends large negative offset.
#
# Skip weather layer for 30 minutes after DHW stops. This prevents the
# immediate volatility spike we saw at 20:30 (-3.75 → -5.24°C) and gives
# the system time to stabilize in space heating mode.
#
# TODO: Consider temperature-based cooldown instead (skip if flow > optimal + X°C).
# More adaptive but tricky to tune - with 10°C threshold it skips for 2+ hours.
# Time-based is simpler and catches the immediate spike which is the main issue.
DHW_WEATHER_COOLDOWN_MINUTES: Final = 30

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
LAYER_WEIGHT_COMFORT_HIGH: Final = 0.7  # High priority: 0-0.5°C over tolerance
LAYER_WEIGHT_COMFORT_CRITICAL: Final = (
    1.0  # Critical priority: 1°C+ over tolerance (same as safety)
)

# Graduated comfort layer correction multipliers (Phase 2)
COMFORT_CORRECTION_MILD: Final = 1.0  # 0-1°C over tolerance: standard correction

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

# How far the EXPECTED degree-minute band must stay clear of the absolute floor above.
# A house whose "normal" range reached the emergency trigger would be normal and in danger at the
# same time. These used to be bare `+ 100` / `+ 50` literals inside climate_zones.get_expected_dm_range
# - magic numbers clamping the safety floor itself (audit F-076).
DM_NORMAL_MIN_BUFFER: Final = 100  # DM - the shallow end of "normal" stays this clear of the floor
DM_WARNING_BUFFER: Final = 50  # DM - normal_max and warning stay this clear of it

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

# Emergency tier identifiers (see thermal_layer.EmergencyLayerDecision.tier)
#
# SAFETY DISPATCH MUST KEY ON THESE NAMES - never on a layer's weight or on the
# magnitude of its offset. Both are unreliable discriminators:
#   - The offset returned by a tier has already passed through thermal-recovery
#     damping, the anti-windup cap, and the volatile-boost skip, so a damped T3 can
#     emerge smaller than an undamped T1.
#   - A weight is a tuning parameter. Retuning DM_CRITICAL_T2_WEIGHT (0.85 -> 0.81)
#     silently moved T2 below a hardcoded `weight >= 0.85` gate, which dropped T2
#     recovery into the cost-layer override path and produced a -3.0 C heat REDUCTION
#     at deep thermal debt.
DM_TIER_EMERGENCY: Final = "EMERGENCY"  # DM <= DM_THRESHOLD_AUX_LIMIT (absolute priority)
DM_RECOVERY_TIERS: Final[frozenset[str]] = frozenset({"T1", "T2", "T3"})

# Minimal offsets applied when a recovery tier coincides with a CRITICAL cost layer
# (effect tariff at the monthly peak, or a PEAK spot-price quarter). Large enough to
# stop DM worsening, small enough not to grow the monthly peak. Keyed by tier name so
# that damping cannot change which tier's compromise is selected.
DM_CRITICAL_PEAK_AWARE_OFFSETS: Final[dict[str, float]] = {
    "T1": DM_CRITICAL_T1_PEAK_AWARE_OFFSET,
    "T2": DM_CRITICAL_T2_PEAK_AWARE_OFFSET,
    "T3": DM_CRITICAL_T3_PEAK_AWARE_OFFSET,
}

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
# Threshold: -50 DM/hour = ~0.8°C sustained gap = clear "heat in transit" signal
# Lowered from -60 to -50 (Dec 18, 2025) to catch borderline cases where
# heat is clearly in transit but rate is just above threshold.
# This avoids false positives from normal ±0.3°C fluctuations (~18 DM/h)
#
# Reference: NIBE DM formula from F750 Service Manual Menu 4.9.3
ANTI_WINDUP_DM_DROPPING_RATE: Final = -50.0  # DM/hour (≈0.8°C gap between S1 and BT25)

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

# Anti-windup cooldown period (Jan 2026 DM spiral analysis)
# After anti-windup prevents an offset raise, wait this long before trying again.
# Gives pump time to stabilize at achievable targets.
# Problem: Raising offset when DM is dropping makes DM drop FASTER (S1 increases but BT25 can't catch up).
# Solution: Prevent the raise AND wait before retrying to avoid oscillation.
ANTI_WINDUP_COOLDOWN_MINUTES: Final = 30

# Anti-windup active offset REDUCTION (Jan 2026 enhancement)
# When DM is dropping severely despite positive offset, actively REDUCE offset
# to give the pump achievable targets. Reduction is proportional to spiral severity.
#
# Based on debug.log analysis showing spirals from -53/h (mild) to -456/h (severe):
#   -50 to -100/h: Mild spiral - just prevent raise (existing behavior)
#   -100/h and worse: Active reduction needed
#
# Reduction formula: reduction = 1.0°C × (|dm_rate| / 100)
#   -100/h: reduce by 1.0°C
#   -200/h: reduce by 2.0°C
#   -300/h: reduce by 3.0°C
#   -400/h: reduce by 4.0°C
#   -456/h (observed max): reduce by 4.6°C
ANTI_WINDUP_REDUCTION_THRESHOLD: Final = -100.0  # DM/h - start reducing when spiral this bad
ANTI_WINDUP_REDUCTION_RATE_DIVISOR: Final = 100.0  # DM/h - divisor for proportional reduction
ANTI_WINDUP_REDUCTION_MULTIPLIER: Final = 1.0  # °C per unit of (|dm_rate| / divisor)

# Anti-windup causation window (Jan 2026 enhancement)
# Only trigger anti-windup if we raised offset recently.
# If offset has been stable for longer than this, DM drop is likely environmental
# (e.g., forecasted cold snap arrived), not a self-induced spiral.
# 90 min = long enough to see effect of raise, short enough to allow weather response.
ANTI_WINDUP_CAUSATION_WINDOW_MINUTES: Final = 90

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

# THE INNER BAND: how much of the owner's tolerance a cost layer may spend freely.
#
# The comment here used to read "Scales user tolerance setting (1-10) to actual temperature range;
# Scale: 1-10 -> 0.4-4.0°C". There is no 1-10 setting. `tolerance` is DEGREES and always has been -
# DEFAULT_TOLERANCE is 0.5, and MIN_TARGET_TEMP below is computed as MIN_TEMP_LIMIT + tolerance,
# which only makes sense in degrees. The comment described a design that never shipped, and it hid
# what the number actually does.
#
# What it does: an owner who asks for +/-0.5 C gets an INNER band of +/-0.2 C inside which the spot
# and effect layers may coast the house around for free - the thermal battery - and a ramp from
# there out to the +/-0.5 C they actually asked for, across which the comfort layer progressively
# takes the floor back. See DecisionEngine._starvation_fraction.
TOLERANCE_RANGE_MULTIPLIER: Final = 0.4  # of the owner's tolerance: 0.5 C -> a 0.2 C free band

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
# WHY ZONE 5 WAS UNREACHABLE (kept as a comment: it was a module-level string bound to a
# name nothing read, which is dead code however useful the words are).
# Zone 5 was unreachable for the whole life of this project, in every release and every climate zone.
#
# Its band is `warning < DM <= zone5_threshold`, and the percent below was 1.00 - so zone5_threshold
# came out at exactly `normal_max`. Every climate zone ALSO sets `dm_warning_threshold` to exactly the
# deep end of `dm_normal_range`, so `warning == normal_max == zone5_threshold` and the band read
# `-740 < DM <= -740`: the empty set. Both ends were the same number, and the same temperature
# adjustment is added to both, so they could never separate.
#
# 1.00 is the one value that gives this rung no step to stand on, and it contradicts the DESIGN note
# directly above: Z1-Z5 trigger BEFORE the warning threshold, so Z5's boundary must sit strictly before
# it. 0.875 splits the old Z4 band in half and hands the deeper half to Z5 - the ladder regains the
# +3.0 rung it was built with, and no threshold that governs when RECOVERY starts moves at all.

PROACTIVE_ZONE1_THRESHOLD_PERCENT: Final = 0.02  # 2% of normal max (ultra-early warning, Jan 2026)
PROACTIVE_ZONE2_THRESHOLD_PERCENT: Final = 0.30  # 30% of normal max (moderate)
PROACTIVE_ZONE3_THRESHOLD_PERCENT: Final = 0.50  # 50% of normal max (significant)
PROACTIVE_ZONE4_THRESHOLD_PERCENT: Final = 0.75  # 75% of normal max (strong)
PROACTIVE_ZONE5_THRESHOLD_PERCENT: Final = 0.875  # 87.5% - strictly BEFORE warning (see note above)

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
#   - Forecast ≥4°C drop in next 12h → +0.6°C gentle pre-heat
#   - Indoor cooling ≥0.5°C/h → confirms forecast, maintains +0.6°C
#   - Weight scaled by thermal mass: Concrete 1.28x, Timber 0.85x, Radiator 0.43x
#   - Let SAFETY, COMFORT, EFFECT layers moderate naturally via weighted aggregation
#
# Real-world validation: Prevents 20:00→04:00 emergency cycles and 16:00 overshoot
# The unit shown on the price sensor when the spot-price integration has not (yet) reported one.
# The sensor prefers the user's own integration's unit; this is only the gap-filler during startup
# or a brief outage. It is öre because that is what GE-Spot reports for SE4 (audit F-070).
# Home Assistant repair-issue id raised when there is no electricity price source at all.
# Without prices the price layer abstains entirely - which is correct - but the user has
# `enable_price_optimization` switched on and believes it is trading. A log line does not tell
# them; a repair issue does. (Audit F-123: the old code invented 96 quarters at 1.0 öre instead.)
PRICE_SOURCE_ISSUE_ID: Final = "no_price_source"

# EffektGuard drives hot water by toggling NIBE's temporary-lux switch. Home Assistant's own NIBE
# integration only maps that register (50004) for the F-series, so an S-series pump exposes no such
# entity at all - and the DHW half of EffektGuard then does nothing whatsoever. It said so in a
# _LOGGER.debug, while the UI carried on showing a hot-water status, a recommendation and a
# scheduled start time that could never fire. Same rule as the price source: a debug line is not
# telling anyone.
DHW_CONTROL_ISSUE_ID: Final = "no_dhw_control_entity"

PRICE_UNIT_FALLBACK: Final = "öre/kWh"

WEATHER_FORECAST_DROP_THRESHOLD: Final = -4.0  # °C drop in forecast (was -5.0, lowered Jan 2026)
WEATHER_FORECAST_HORIZON: Final = 12.0  # Hours to scan forecast (matches thermal lag)

# THE FORECAST WAS NEVER FILTERED TO THE FUTURE, AND EVERY LAYER SLICES IT POSITIONALLY.
#
# `WeatherData.forecast_hours` is documented as "Next 24-48 hours", and every consumer reads it that
# way - `forecast_hours[:3]` for the cold-snap trigger, `[:24]` for unusual-weather detection,
# `[:horizon]` for the pre-heat. But the adapter appended EVERY entry the weather entity published,
# including the ones already in the past. Many integrations publish the current period first, and a
# weather integration that has stalled holds its last forecast for as long as it stays "available".
#
# So with a forecast that starts six hours ago, `forecast_hours[:3]` is the weather from six hours
# AGO - and a cold snap an hour away sits outside every horizon anyone looks at. That is precisely
# the case the pre-heat exists for: "we need to pre-heat super early if we know a cold snap is
# coming, I mean like DAYS ahead."
#
# Entries whose hour has already ENDED are dropped. The current hour is kept - a period that began
# 40 minutes ago is still the weather now - and `WeatherData.current_temp` carries the present
# reading separately in any case. A forecast entirely in the past becomes an EMPTY one, which is
# exactly right: the layers already abstain when there is no forecast, and a frozen forecast is not
# a forecast.
WEATHER_FORECAST_PERIOD_HOURS: Final = 1.0  # each forecast entry covers one hour
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

# Minimum duration for volatility detection (Jan 4, 2026)
# Used by both price volatility (short price runs) and offset volatility (rapid reversals)
# Based on compressor dynamics: 30 min ramp-up + 15 min cool-down = 45 min
# Single source of truth - use MINUTES for offset tracking, QUARTERS for price logic
MINUTES_PER_QUARTER: Final = 15  # 15-minute price quarters
VOLATILE_MIN_DURATION_MINUTES: Final = (
    COMPRESSOR_RAMP_UP_MINUTES + COMPRESSOR_COOL_DOWN_MINUTES
)  # 45min total
VOLATILE_MIN_DURATION_QUARTERS: Final = (
    VOLATILE_MIN_DURATION_MINUTES // MINUTES_PER_QUARTER
)  # 3 quarters (45min / 15min)

# Volatile timing tolerance (Feb 2, 2026)
# Prevents floating-point rounding in time.time() from blocking at the displayed boundary.
# Jan 31 2026 incident: log showed "45min < 45min" still blocking because actual seconds
# were 2699.x which rounds to 45min display but is still < 2700 seconds.
VOLATILE_TIMING_TOLERANCE_SECONDS: Final = 2

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
PRICE_FORECAST_REDUCTION_OFFSET: Final = (
    -1.5
)  # °C - reduce heating when cheap period coming (Dec 5, 2025: strengthened from -1.0)
PRICE_FORECAST_PREHEAT_OFFSET: Final = 2.0  # °C - pre-heat when expensive period coming

# Price layer DM debt gate (Dec 13, 2025)
# When thermal debt exists, price layer should not suppress heating (fighting thermal recovery).
# Instead, contribute a gentle positive offset to aid recovery while respecting user savings intent.
# Uses WEATHER_COMP_DEFER_DM_CRITICAL (-400) as threshold for consistent behavior across layers.
PRICE_FORECAST_DM_DEBT_OFFSET: Final = 0.3  # °C - Gentle recovery offset when in debt

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
VOLATILE_WEIGHT_REDUCTION: Final = 0.3  # Retain 30% weight during volatility

# Price classification percentile thresholds (Dec 8, 2025)
# Define the boundaries between price classifications
PRICE_PERCENTILE_VERY_CHEAP: Final = 10  # Bottom 10% = VERY_CHEAP
PRICE_PERCENTILE_CHEAP: Final = 25  # 10-25% = CHEAP
PRICE_PERCENTILE_NORMAL: Final = 75  # 25-75% = NORMAL
PRICE_PERCENTILE_EXPENSIVE: Final = 90  # 75-90% = EXPENSIVE
PRICE_PERCENTILE_MEDIAN: Final = 50  # the day's midpoint; guards every band against a plateau

# A day whose prices barely move carries no signal, and percentile RANK cannot see that: it is
# scale-invariant, so it cannot tell a 130 ore spread from a 0.4 ore one. A day that ran from
# 39.80 to 40.20 ore used to earn the full VERY_CHEAP..PEAK banding - a 14 C swing in commanded
# offset - to chase four tenths of an ore.
#
# The test is RELATIVE, not an absolute number of ore, because nothing in the price layer knows
# its unit: PriceData carries none, and GE-Spot publishes whatever the owner configured. An
# absolute threshold would be a hundred times wrong for anyone reporting SEK/kWh.
PRICE_FLAT_DAY_SPREAD_FRACTION: Final = 0.05  # p90-p10 must exceed 5 % of the day's price scale
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

# Weather compensation - EN 442 emitter law (see utils/emitter.py)
#
#   EN 442-1:2014 3.31   emitter output: Phi / Phi_N = (dT / dT_N) ** n
#   EN 442-1:2014 3.23   rated point 75/65/20 => dT_N = 50 K, ARITHMETIC mean.
#                        Never pair this reference with a log-mean dT.
#   EN 1264              underfloor: q = 8.92 * dT ** 1.1 => n ~ 1.1
RADIATOR_POWER_COEFFICIENT: Final = 1.3  # EN 442 exponent n, panel/sectional radiators
RADIATOR_RATED_DT: Final = 50.0  # EN 442 rated excess temperature (75/65/20)
UFH_POWER_COEFFICIENT: Final = 1.1  # EN 1264 exponent n, underfloor (NOT 1.3)

# Design point: the outdoor temperature the emitters were sized for, and the supply they need at
# it. Defaults describe a standard Swedish low-temperature radiator system, erring WARM: a
# too-warm design point over-supplies slightly, a too-cold one silently under-heats, and degree
# minutes cannot detect under-heating that a negative offset causes (they improve as it worsens).
# INTERNAL GAINS. Bodies, appliances and the sun heat a house for free. Demand is NOT linear in
# (indoor - outdoor); it is linear in (balance_point - outdoor), and it reaches zero well before
# the outdoor air reaches room temperature. Ignoring that asks the pump to run hot in mild weather
# - where most of the season's kWh are delivered, and where OEM's measured fleet puts the COP
# penalty at 2.5-3 % per degree of excess flow.
#
# GAINS ARE WATTS, NOT DEGREES. The balance point is DERIVED, never fitted:
#
#     balance_point = indoor_setpoint - INTERNAL_GAINS_W / heat_loss_coefficient
#
# A fixed offset in degrees would make the gains scale with the house's heat loss - a leaky house
# would be credited with MORE free heat than a well-insulated one, from the same bodies and the
# same fridge. Backwards. Watts divided by W/K is the only form that gets this right, and it is
# the form both measured sources below are expressed in.
#
# Measured sources:
#   * heatpumpmonitor.org, 383 monitored systems with a fitted heat-demand line: median implied
#     gains 583 W (750 W among the systems where it fitted non-zero).
#   * OpenEnergyMonitor's SCOP tool: baseTemp 15.5 C against a 19.3 C room. Its source carries the
#     naive no-gains formula COMMENTED OUT, with the note "This approach would need to take into
#     account gains, hence use of degree days approach".
#
# NOT a source: fitting this against NIBE's published heating curve. That fit is DEGENERATE - a
# constant spread lifts the curve by (spread/2)(1 - phi^(1/n)) and the balance point drops it by
# the same basis function, so the two are not separately identifiable and any assumed spread
# manufactures a matching "gains" figure. Fitting the post-fix law to Kuhne's Vaillant curve -
# which contains PROVABLY ZERO gains - still yields 0.3 K at spread 0, 2.6 K at spread 5 and 4.9 K
# at spread 10. A curve fit cannot see gains. Only a wattage can.
INTERNAL_GAINS_W: Final = 600.0  # W of free heat; fleet median 583 W

# A mis-configured heat loss must not drive the balance point somewhere absurd (a 20 W/K entry
# would put it 30 K below the setpoint and switch heating off entirely).
BALANCE_POINT_MIN_OFFSET: Final = 1.0  # C below setpoint, floor
BALANCE_POINT_MAX_OFFSET: Final = 6.0  # C below setpoint, ceiling

DEFAULT_DESIGN_OUTDOOR_TEMP: Final = -15.0  # °C, dimensioning outdoor temperature (DUT/DVUT)
DEFAULT_DESIGN_FLOW_TEMP_RADIATOR: Final = 50.0  # °C supply at DUT for radiators
DEFAULT_DESIGN_FLOW_TEMP_UFH: Final = 35.0  # °C supply at DUT for UFH (NIBE: normally 35-45)
DEFAULT_DESIGN_SPREAD: Final = 5.0  # °C flow-return spread at design load

# Weather compensation TRIMS the pump's own curve, it does not replace it: a correctly tuned curve
# needs a near-zero correction. This bound stops a mis-configured design point from ever
# commanding a large swing in either direction.
WEATHER_COMP_MAX_OFFSET: Final = 3.0  # °C, absolute cap on the compensation offset

# Heat loss coefficient defaults (W/°C)
DEFAULT_HEAT_LOSS_COEFFICIENT: Final = 180.0  # W/°C typical value

# Fallback decay rate for the pre-heat sizing, used until learning is trusted. It was a bare 0.15
# in adaptive_learning next to a bare 180.0 - and the 180.0 was a second copy of the constant
# directly above this line.
DEFAULT_THERMAL_DECAY_RATE: Final = 0.15  # °C per hour, per °C of indoor-outdoor difference

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

# And a CEILING, for the same reason the floor exists: a reading this far outside what a house can
# physically draw is not a peak, it is a broken sensor.
#
# A Swedish domestic service is 16-25 A three-phase (11-17 kW); 35 A (24 kW) is a large villa with
# an EV charger. 100 kW cannot happen behind a domestic main fuse, so nothing real is ever refused
# here - but a mis-scaled reading is, and one of those got all the way into the tariff record:
# `power_kw_from_state` case-folded its unit and read 5000 mW (five watts) as 5 000 000 kW. The
# unit table is fixed, but a number that is persisted for a month and silently governs whether the
# house is throttled deserves a plausibility bound of its own. An astronomical recorded peak does
# not throttle the house - it makes every real quarter look safe against it, and peak protection
# goes quiet for the rest of the month.
PEAK_RECORDING_MAXIMUM: Final = 100.0  # kW - beyond any domestic main fuse; a sensor fault

# Update intervals
UPDATE_INTERVAL_MINUTES: Final = (
    5  # Coordinator update frequency + thermal predictor save throttle interval
)
# The SPOT PRICE interval. Nordpool settles in quarter-hours, and this is that - it is NOT the
# effect tariff's measurement period, which the comment here used to claim it was. See
# BILLING_PERIOD_MINUTES below. Conflating the two is what made the integration defend a peak
# nobody is billed for.
QUARTER_INTERVAL_MINUTES: Final = 15  # Nordpool spot-price settlement interval
QUARTERS_PER_DAY: Final = 96  # Quarters in a normal (non-DST-transition) day
# Native interval counts a day can have: 92 (spring DST), 96 (normal),
# 100 (autumn DST). Anything else means the source delivered a data gap.
NATIVE_DAY_QUARTER_COUNTS: Final = (92, 96, 100)
# How many consecutive update cycles the coordinator will wait for the heat pump to appear before
# it reports failure. MyUplink can take 45-50 seconds to publish its entities, so waiting is right;
# waiting FOREVER is a silent failure. Unbounded, a user who picked the wrong entity - or who has
# no NIBE at all - keeps a config entry that stays loaded and green indefinitely while the
# integration reads nothing and controls nothing.
#
# At UPDATE_INTERVAL_MINUTES this is a generous margin over any plausible MyUplink start-up.
# How old a NIBE reading may be before it stops being a reading and becomes a memory.
#
# The adapter rejects `unavailable` and `unknown`, so an upstream integration that DIES is caught -
# MyUplink and nibe_heatpump use a coordinator, and their entities go unavailable when polling
# fails. But `manifest.json` also lists mqtt and modbus, and an MQTT sensor holds its last retained
# value indefinitely: if the bridge publishing degree minutes stops, the sensor goes on reporting
# the number it was given hours ago and every other check passes. The pump keeps being driven on it,
# and the real degree minutes could be anywhere - including past the auxiliary-heat limit.
#
# 30 minutes is deliberately generous: the control loop runs every 5, so any NIBE source reporting
# less often than this cannot support five-minute heat-pump control anyway, and nothing that works
# today can be broken by the guard. (Audit F-015.)
NIBE_READING_MAX_AGE_MINUTES: Final = 30

STARTUP_MAX_GRACE_ATTEMPTS: Final = 12  # cycles (~1 hour) before a missing pump is an error

STARTUP_GRACE_UPDATES: Final = 1  # Number of full cycles to observe before active control
STARTUP_GRACE_MIN_INTERVAL: Final = 120  # Seconds - minimum lockout before observation cycles

# Thermal predictor history constants (derived from UPDATE_INTERVAL_MINUTES)
SAMPLES_PER_HOUR: Final = 60 // UPDATE_INTERVAL_MINUTES  # 12 samples/hour with 5-min intervals

# THE PREDICTION GATES COUNTED SAMPLES AND SPOKE IN HOURS, AND THE TWO DISAGREED BY 3x.
#
# SAMPLES_PER_HOUR is derived correctly above, and the predictor's own deque is sized with it. The
# gates were not: they hardcoded 4, 96 and 8, with comments claiming "1 hour", "24 hours" and
# "2+ hours". At a five-minute coordinator tick those are 20 minutes, EIGHT hours, and 40 minutes.
#
# The learned pre-heating layer therefore engaged on a THIRD of the data it believed it had, and
# eight hours of a Swedish winter night is not a representative day. The hours are named here and
# the sample counts derived from them, so the two can no longer drift apart.
PREDICTION_MIN_HISTORY_HOURS: Final = 1  # before projecting forward at all
PREDICTION_RESPONSIVENESS_MIN_HOURS: Final = 2  # before estimating how fast the house responds
PREDICTION_LEARNED_PREHEAT_MIN_HOURS: Final = 24  # before acting on learned pre-heating

# Adaptive learning parameters
# Source: POST_PHASE_5_ROADMAP.md Phase 6 - Self-Learning Capability
#
# LEARNING OBSERVES ON A DIFFERENT CLOCK FROM CONTROL, and it has to.
#
# Control runs every UPDATE_INTERVAL_MINUTES because the pump needs steering that often. Learning
# used to piggy-back on the same tick, and could therefore never learn anything: the NIBE BT1 indoor
# sensor reports to 0.1 C, and a house warming at a brisk 0.6 C/h moves 0.05 C in five minutes - half
# a sensor tick. Every observed rate quantised to 0.0 or 1.2 C/h with nothing in between, so the
# scatter that `_calculate_confidence` scores was a measurement of the SAMPLING INTERVAL, not of the
# building. Confidence sat at 0.467 against a 0.7 gate, on any house, forever (F-132).
#
# The same house, identical physics, identical sensor, watched more slowly:
#     5 min   quantum 1.20 C/h   confidence 0.467   never engages
#    15 min   quantum 0.40 C/h   confidence 0.600   never engages
#    30 min   quantum 0.20 C/h   confidence 0.707   engages, barely
#    60 min   quantum 0.10 C/h   confidence 0.811   engages, comfortably
#
# Hourly is also the honest timescale for the question being asked. A building's thermal time
# constant is hours; the concrete slab in UFH_CONCRETE_* lags six of them. Sampling that every five
# minutes is sampling noise.
#
# It fixes the memory as well. The 672-entry deque was commented "1 week" but spanned 56 HOURS at the
# 5-minute cadence - and it is a rolling window, so the model on day 90 saw exactly what it saw on
# day 3, and the README's "Day 8-14: high confidence" was unreachable by construction. At one
# observation an hour the same deque remembers 28 days.
LEARNING_OBSERVATION_INTERVAL_MINUTES: Final = (
    60  # Learning observes hourly, not every control tick
)
LEARNING_OBSERVATION_WINDOW: Final = 672  # 672 hourly observations = 28 days of memory
LEARNING_MIN_OBSERVATIONS: Final = 96  # 96 hourly observations = 4 days before learning may engage
LEARNING_CONFIDENCE_THRESHOLD: Final = 0.7  # 70% confidence to use learned params

# What it takes for the heating observations to carry any information at all.
# The indoor sensor (NIBE BT1) reports to 0.1 C. A house that moved less than one sensor tick per
# hour WHILE ACTIVELY HEATING has told us nothing measurable about itself: the signal is below the
# instrument's resolution. Such a run must score ZERO confidence, not perfect confidence - which is
# what a std/mean ratio does when every reading is identical and std collapses to 0 (F-132).
LEARNING_MIN_HEATING_RATE: Final = 0.1  # °C/h - one sensor tick per hour; below this, no signal
LEARNING_MIN_HEATING_SAMPLES: Final = 10  # observations under heating before consistency means much

# Swedish climate regions - SMHI historical data (1961-1990)
# Source: Swedish_Climate_Adaptations.md

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

# DHW (Domestic Hot Water) Optimization Constants
# Based on DHW_RESEARCH_FINDINGS.md and DHW_IMPLEMENTATION_CORRECTIONS.md
#
# Temperature hierarchy:
# - 20°C (DHW_SAFETY_CRITICAL): below this, price is no longer a reason to wait
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

# Legionella / hygiene.
#
# ⚠️ EFFEKTGUARD DOES NOT PROVIDE LEGIONELLA PROTECTION. NIBE DOES.
#
# NIBE's built-in "periodic increase" function (menu 2.9.1 on F-series, 2.4 on S-series -
# NOT 4.9.5, which is schedule blocking) is ACTIVATED FROM THE FACTORY, runs every 14 days
# (7 on S-series), targets a stop temperature of 55 C (settable 55-70, never lower), and
# explicitly uses "the compressor AND the immersion heater". EffektGuard cannot block it,
# and cannot even observe it: Home Assistant's myuplink integration excludes parameters
# 47050 (enable) and 47051 (interval) from the entities it creates.
# Source: NIBE F750 / F730 / F1155 installer manuals, menus 2.9.1 and 5.1.1.
#
# Why EffektGuard's own boost CANNOT perform a Legionella cycle:
# Temporary lux is not a setpoint - it switches the hot-water comfort mode to LUXURY for
# 3/6/12 h, so the tank is driven to the pump's configured LUXURY STOP temperature. Factory
# values, measured on BT6 (the CONTROL sensor): F750 54 C, F730 53 C, F1155 53 C - all BELOW
# the 55 C floor NIBE enforces for its Legionella function. Those setpoints are
# installer-adjustable, so their true value is UNKNOWN to us at runtime. Never hard-code it.
#
# NOTE: the DHW immersion heater (elpatron) is separate from the space-heating auxiliary
# heater. Different electrical systems, different purposes.
DHW_LEGIONELLA_DETECT: Final = 55.0
"""°C on BT7 taken as evidence that a high-temperature cycle occurred.

Unsound as proof that OUR boost completed; kept only as a best-effort observation of NIBE's
own periodic increase:
  - BT7 is "hot water, DISPLAY"; BT6 is "hot water, CONTROL". Every setpoint acts on BT6.
  - On F1155 / S1155, BT7 is OPTIONAL and may not physically exist.
  - Temporary lux stops at 53-54 C on BT6, so it will not reach this threshold.
"""

DHW_LEGIONELLA_PREVENT_TEMP: Final = 56.0
"""°C - target requested for the opportunistic high-temperature top-up.

NEVER ACTUALLY WRITTEN TO NIBE: the only DHW actuator is the temporary-lux switch, and the
pump heats to ITS OWN configured lux stop temperature, not to this value.
"""

# Days without any observed high-temperature cycle after which EffektGuard warns the user.
# NIBE's periodic increase runs every DHW_LEGIONELLA_MAX_DAYS (14) from the factory, so
# going well beyond that suggests it has been switched off on the pump. DIAGNOSTIC ONLY -
# we warn, we do not attempt to substitute for the function (we cannot: see above).
DHW_LEGIONELLA_OVERDUE_DAYS: Final = 21.0
DHW_LEGIONELLA_MAX_DAYS: Final = 14.0  # Days - Max time without high-temp cycle (hygiene)
DHW_HEATING_TIME_HOURS: Final = 1.5  # Hours to heat DHW tank (typically 1-2h)
DHW_SCHEDULING_WINDOW_MAX: Final = 24  # Max hours ahead for DHW scheduling
DHW_SCHEDULING_WINDOW_MIN: Final = 0.25  # Min hours ahead (15 min minimum for meaningful pre-heat)
DHW_SCHEDULED_WINDOW_HOURS: Final = 6  # Hours before target time when scheduling becomes active
DHW_MAX_WAIT_HOURS: Final = 36.0  # Max hours between DHW heating (hygiene/comfort)

# DHW amount-based scheduling (minutes of hot water available)
# Schedule time = when water should be AVAILABLE (ready), not when heating starts
DHW_MIN_AMOUNT_DEFAULT: Final = 5  # Default: min 5 minutes of hot water at scheduled time
DHW_MIN_AMOUNT_MIN: Final = 1  # Config minimum: 1 minute
DHW_MIN_AMOUNT_MAX: Final = 30  # Config maximum: 30 minutes
CONF_DHW_MIN_AMOUNT: Final = "dhw_min_amount"  # Config key for min hot water minutes

# DHW temperature-based heating rate (measured from real F750 data 2025-12-22)
# 19:00 → 20:00: 23.8°C → 37.8°C = 14°C in 60 min = 14°C/hour
# Used as fallback when insufficient history for dynamic calculation
# The dhw_optimizer uses calculate_heating_rate() for dynamic estimation from BT7 history
DHW_DEFAULT_HEATING_RATE: Final = 14.0  # °C/hour (measured from debug log)

# Plausible band for the DHW tank heating rate (°C/hour). Applied BOTH when a rate is learned
# from BT7 history AND when one is restored from storage: an unchecked restore can load 0.0
# (ZeroDivisionError in estimate_heating_time) or 0.1 (a 200-hour heat-up estimate, which makes
# the scheduler panic-heat immediately at any price, forever).
DHW_HEATING_RATE_MIN: Final = 5.0
DHW_HEATING_RATE_MAX: Final = 25.0

DHW_AMOUNT_HEATING_BUFFER: Final = 0.5  # Hours buffer for scheduling (arrive early, not late)

# DHW optimal window price optimization (Phase 1 fix - Jan 2026)
# When DHW temp < MIN_DHW_TARGET_TEMP within scheduled window, check if waiting
# for cheaper optimal window is worth it before heating immediately
DHW_OPTIMAL_WINDOW_MIN_SAVINGS: Final = 0.15  # 15% minimum price savings to wait for optimal window
DHW_OPTIMAL_WINDOW_MIN_TIME_BUFFER: Final = (
    0.25  # 15 min buffer before considering "in optimal window"
)

# DHW thermal debt fallback thresholds (used only if climate detector unavailable)
# These are balanced fixed values for rare fallback scenarios
DM_DHW_BLOCK_FALLBACK: Final = -340.0  # Fallback: Never start DHW below this DM
DM_DHW_ABORT_FALLBACK: Final = -500.0  # Fallback: Abort DHW if reached during run

# How far BELOW the block threshold a running DHW cycle is allowed to sink before it gives up.
# Heating hot water takes the compressor away from space heating, so degree minutes ALWAYS fall
# during a cycle. Abort must therefore sit deeper than block, or every cycle that starts near the
# block threshold is aborted on the next tick and the pump cycles (audit F-030). This 160 DM is the
# gap the fallback pair above already encodes: -340 block, -500 abort.
DM_DHW_ABORT_BUFFER: Final = 160.0

# DHW runtime safeguards (monitoring only - NIBE controls actual completion)
DHW_SAFETY_RUNTIME_MINUTES: Final = 30  # Safety minimum heating (emergency)
DHW_NORMAL_RUNTIME_MINUTES: Final = 45  # Normal DHW heating window
DHW_EXTENDED_RUNTIME_MINUTES: Final = 60  # High demand period heating
DHW_URGENT_RUNTIME_MINUTES: Final = 90  # Urgent pre-demand heating

# NIBE Power Calculation Constants (Swedish 3-phase standard)
# All NIBE heat pumps in Sweden are 3-phase systems.
#
# This was 240.0 V, and its own comment gave the reason it could not be: "400V between phases,
# 240V phase-to-neutral". Those two numbers contradict each other by a factor of sqrt(3) -
# 400 / sqrt(3) is 230.94, not 240. 240 V is the legacy UK/US figure.
#
# IEC 60038 and EN 50160 declare the European low-voltage supply as 230/400 V, and Sweden is
# 230 V phase-to-neutral. At 240 V every power figure derived from NIBE's BE1/BE2/BE3 phase
# currents came out 4.3 % HIGH - and those figures now feed the monthly peak history that drives
# peak protection, so the bias is not cosmetic.
NIBE_VOLTAGE_PER_PHASE: Final = 230.0  # V - IEC 60038: 230/400 V, phase-to-neutral
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

# The airflow COP-improvement constants that lived here are gone with the term that used them.
# Extracting more heat from more air and "improving the COP" are the same joules: at constant
# electrical input the first law gives d(Q_cond) = d(Q_evap) = P_el * d(COP). Adding both counted
# the heat twice, and made a net thermal LOSS look like a gain across the whole heating season.
# See optimization/airflow_optimizer.py.

# Temperature thresholds
AIRFLOW_OUTDOOR_TEMP_MIN: Final = -15.0  # °C - Never enhance below this (penalty exceeds gains)
AIRFLOW_INDOOR_DEFICIT_MIN: Final = 0.2  # °C - Minimum deficit to trigger enhancement

# Compressor threshold calculation
# Minimum compressor % needed for enhanced flow to be beneficial
# Colder outside → need higher compressor output to justify extra ventilation
# Formula: threshold = 61 + slope * outdoor_temp (for outdoor_temp < 0)
# Base raised from 50% to 61% (81 Hz) based on real-world observations showing
# enhancement at lower Hz caused cooling during periods when pump was struggling
AIRFLOW_COMPRESSOR_BASE_THRESHOLD: Final = 61.0  # % base threshold at 0°C (81 Hz)
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
AIRFLOW_TREND_COOLING_THRESHOLD: Final = -0.10  # °C/h - cooling despite enhanced = stop

# Configuration keys
CONF_ENABLE_AIRFLOW_OPTIMIZATION: Final = "enable_airflow_optimization"
CONF_AIRFLOW_STANDARD_RATE: Final = "airflow_standard_rate"  # m³/h
CONF_AIRFLOW_ENHANCED_RATE: Final = "airflow_enhanced_rate"  # m³/h

# NIBE Enhanced Ventilation (F750/F730)
# Controls the "Increased Ventilation" switch on exhaust air heat pumps
# Entity pattern: switch.{device}_increased_ventilation
# THE VENTILATION FAN COULD CYCLE FOREVER, AND THE GUARD MEANT TO STOP IT GUARDED NOTHING.
#
# NIBE_VENTILATION_MIN_ENHANCED_DURATION was 5 minutes - exactly UPDATE_INTERVAL_MINUTES - so a
# turn-off was permitted on the very next coordinator tick. And nothing at all guarded the turn-ON.
# A decision that oscillates around its threshold, which is what a marginal COP gain does, produced
# twelve fan state changes an hour, indefinitely. On an exhaust-air F750 every one of them perturbs
# the source air the compressor is drawing from.
#
# It was also shorter than the SHORTEST duration the airflow optimizer ever recommends
# (AIRFLOW_DURATION_SMALL_DEFICIT, 15 min) - so it could never enforce even the mildest of them.
# The optimizer's own `duration_minutes` is now the minimum run time; this is only a floor under it,
# for the case where a decision arrives with no duration at all.
NIBE_VENTILATION_MIN_ENHANCED_DURATION: Final = 15  # Minimum minutes to run enhanced

# And a minimum REST before enhancing again, which did not exist. Without it the fan can go
# ON -> hold -> OFF -> ON on the next tick, and the minimum run time above just sets the period of
# the oscillation rather than preventing it.
NIBE_VENTILATION_MIN_REST_DURATION: Final = 15  # Minimum minutes at normal before re-enhancing

# "HARD FLOOR, ALWAYS HEAT BELOW THIS" IS WHAT THIS SAID, AND IT IS NOT TRUE.
#
# Below 20 C the optimizer stops WAITING FOR A CHEAPER PRICE. It does not heat unconditionally, and
# it must not: two things still outrank the hot water, and both are deliberate.
#
#   * CRITICAL THERMAL DEBT. Running a DHW cycle takes the compressor away from space heating, and
#     doing that while the house is already in deep degree-minute debt is how a recoverable debt
#     becomes an immersion-heater one.
#   * THE HOUSE ITSELF BEING BELOW ITS SAFETY FLOOR. The owner's rule, in his own words: "DHW wins,
#     but never below safety."
#
# Verified by driving the real scheduler at 15 C - five degrees under this "hard floor":
#
#     DHW 15 C, DM  -150, indoor 21.0  ->  heats            (DHW_SAFETY_MINIMUM)
#     DHW 15 C, DM -1400, indoor 21.0  ->  does NOT heat    (CRITICAL_THERMAL_DEBT)
#     DHW 15 C, DM  -150, indoor 17.0  ->  does NOT heat    (SPACE_HEATING_EMERGENCY)
#
# The code is right. The comment was the lie, and it is the kind of lie that gets a safety rule
# "restored" by the next reader who trusts it.
DHW_SAFETY_CRITICAL: Final = 20.0  # °C - below this, price stops being a reason to defer
DHW_SAFETY_MIN: Final = 30.0  # °C - Safety minimum (can defer if 20-30°C during expensive periods)
DHW_COOLING_RATE: Final = 0.5  # °C/hour - Conservative DHW tank cooling estimate

# Unit conversion
WATTS_PER_KILOWATT: Final = 1000.0
KILOWATTS_PER_MEGAWATT: Final = 1000.0
MILLIWATTS_PER_KILOWATT: Final = 1_000_000.0

# Where a power reading came from. The monthly effect tariff may only be billed against a real
# measurement, so the value has to carry its own provenance - asking whether a power ENTITY is
# configured tells you nothing about whether it answered this cycle.
POWER_SOURCE_EXTERNAL_METER: Final = "external_meter"
POWER_SOURCE_NIBE_CURRENTS: Final = "nibe_currents"
POWER_SOURCE_ESTIMATE: Final = "estimate"
POWER_SOURCE_NONE: Final = "none"

# BILLABLE and USABLE-AS-A-CONTROL-THRESHOLD are two different questions, and conflating them broke
# the integration's headline feature for everyone without a whole-house meter.
#
# BILLABLE: the Swedish effect tariff bills whole-house grid IMPORT, so only a whole-house meter can
# produce a number that belongs in a billing figure. NIBE phase currents (BE1/BE2/BE3) measure the
# heat pump and nothing else - not the oven, not the EV charger. Estimates are fabricated. Neither
# may be reported to the owner as "your monthly peak".
BILLABLE_POWER_SOURCES: Final = frozenset({POWER_SOURCE_EXTERNAL_METER})

# PEAK CONTROL: a reading does not have to be the billed quantity to be worth acting on. The heat
# pump is the dominant CONTROLLABLE load in the house, and `should_limit_power` compares this
# quarter against the month's own recorded peaks - so a NIBE-only history compared against NIBE-only
# power is self-consistent, and it still throttles the pump when the pump is the thing spiking.
#
# The whole-house meter is OPTIONAL in the config flow. Gating peak RECORDING on billability alone
# left `_monthly_peaks` permanently empty for those users, and `should_limit_power` returns
# "OK - no peaks recorded yet" on an empty history: peak protection silently never fired at all.
# `main` allowed phase currents here (`has_external_power_sensor or phase1_current is not None`) and
# ran a winter that way.
#
# Estimates stay out of BOTH. A number derived from compressor Hz is not a measurement, and driving
# peak protection from it would throttle the house on the strength of a guess.
PEAK_CONTROL_POWER_SOURCES: Final = frozenset(
    {POWER_SOURCE_EXTERNAL_METER, POWER_SOURCE_NIBE_CURRENTS}
)

# NIBE Adapter Constants
NIBE_DEFAULT_SUPPLY_TEMP: Final = 35.0  # °C - Default supply/flow temp when sensor unavailable

# AN IMPLAUSIBLE READING IS NOT A READING.
#
# `get_current_state` already says this about a MISSING sensor: "Never substitute a plausible
# constant for a missing one: that makes a broken installation indistinguishable from a healthy
# one and still writes a curve offset to the pump." A reading that is physically impossible is the
# same thing wearing a number, and the mechanism that produces one is mundane: NIBE's Modbus
# registers hold temperatures in DECI-degrees, so a hand-written YAML that omits `scale: 0.1`
# reports BT50's 213 as 213.0 C rather than 21.3 C. The repo's own Modbus simulator documents
# exactly that register, with exactly that value.
#
# 213 C indoor is not a rounding error. The comfort layer reads a 192 C overshoot and commands
# -10.0 C at critical weight, forever - maximum heat reduction in a Swedish January - and the
# 18 C safety floor never fires, because the safety layer is reading the same 213 C.
#
# These bands are applied AFTER unit conversion to °C. They are deliberately wide: their job is to
# catch a value that cannot be a temperature at all, not to second-guess a working sensor.
INDOOR_SENSOR_PLAUSIBLE_MIN: Final = 15.0
INDOOR_SENSOR_PLAUSIBLE_MAX: Final = 30.0

# Outdoor air (BT1). Kiruna reaches -40 C; the band leaves room below it and well above any
# habitable summer. A mis-scaled BT1 reading -105 C demands a 96.8 C flow temperature and pushes
# the degree-minute warning threshold to -1450 - fifty short of the aux limit - so the immersion
# heater engages with no warning at all.
NIBE_OUTDOOR_PLAUSIBLE_MIN: Final = -50.0
NIBE_OUTDOOR_PLAUSIBLE_MAX: Final = 50.0

# Heating water (BT25/BT63 supply, BT3 return). It cannot freeze and it cannot boil.
NIBE_WATER_PLAUSIBLE_MIN: Final = 0.0
NIBE_WATER_PLAUSIBLE_MAX: Final = 100.0
NIBE_FRACTIONAL_ACCUMULATOR_THRESHOLD: Final = (
    1.0  # °C - Write to NIBE when accumulator crosses ±1.0
)

# NIBE source-entity discovery patterns (issue #18: multi-source support)
# EffektGuard reads NIBE data from entities created by ANY of these integrations:
#   - myuplink (cloud): sensor.<system>_current_outd_temp_bt1,
#     number.<system>_degree_minutes, number.<system>_heating_offset_climate_system_1
#     (writable parameters are NUMBER entities; ids carry names, not parameter ids)
#   - nibe_heatpump (local NibeGW/Modbus): ids from yozik04/nibe coil names, e.g.
#     sensor.bt1_outdoor_temperature_40004, number.degree_minutes_16_bit_43005,
#     number.heat_offset_s1_47011 (writable coils are NUMBER entities)
#   - generic modbus YAML + template numbers (user-defined names)
# Matching: lowercase substring of entity_id; keys are tried in dict order and each
# entity satisfies at most ONE key, so specific temperature keys MUST come before
# broad status keys (e.g. sensor.*_hot_water_top_bt7 belongs to dhw_top_temp, not
# hot_water_status).
# Register references (yozik04/nibe, file nibe/data/f1155_f1255.json — the map
# HA's nibe_heatpump integration uses; F-series register ids double as MyUplink
# parameter ids): BT1=40004, BT2=40008, BT3=40012, BT7=40013, BT6=40014,
# BT50=40033, BT25=40071, DM=40940 (32-bit)/43005 (16-bit, Modbus), offset S1=47011,
# compressor state=43427, compressor frequency=43136, prio=43086,
# BE1/BE2/BE3=40083/40081/40079.
# 40941 is MyUplink's DM id on some models (second word of 40940).
NIBE_DISCOVERY_PATTERNS: Final[dict[str, list[str]]] = {
    "outdoor_temp": ["_bt1", "bt1_", "outdoor_temp", "outd_temp", "40004"],
    # NOTE: no short "room_temp" - "bedroom_temp" etc. contain it as a
    # substring; BT50-suffixed ids (nibe_heatpump/myuplink) match via _bt50
    "indoor_temp": ["_bt50", "bt50_", "room_temperature", "40033"],
    # NOTE: no bare "_bt2"/"bt2_" - it would substring-match BT20/BT21/BT22
    # exhaust-air sensors; BT2 entities are covered by "supply_temp"
    # (nibe_heatpump bt2_supply_temp_s1_40008) and "supply_line" (MyUplink)
    "supply_temp": [
        "_bt25",
        "bt25_",
        "_bt63",
        "bt63_",
        "supply_temp",
        "supply_line",
        "heating_medium_supply",
        "40008",
        "40071",
    ],
    "return_temp": ["_bt3", "bt3_", "return_temp", "return_line", "40012"],
    "dhw_top_temp": ["_bt7", "bt7_", "hw_top", "hot_water_top", "40013"],
    "dhw_charging_temp": [
        "_bt6",
        "bt6_",
        "hw_bottom",
        "hw_charging",
        "hw_load",
        "hot_water_charging",
        "40014",
    ],
    "degree_minutes": ["degree_minutes", "gradminuter", "40940", "43005", "40941"],
    "offset": ["heat_offset", "heating_offset", "offset", "47011"],
    "compressor_hz": ["compressor_frequency", "43136"],
    "compressor_status": ["compressor_state", "status_compressor", "compressor_status", "43427"],
    # "prio" also matches "priority" (MyUplink naming) as a substring
    "prio": ["prio", "43086"],
    "phase1_current": ["current_be1", "_be1", "phase_1_current", "40083"],
    "phase2_current": ["current_be2", "_be2", "phase_2_current", "40081"],
    "phase3_current": ["current_be3", "_be3", "phase_3_current", "40079"],
    "dhw_amount": ["hot_water_amount", "hw_amount"],
    "hot_water_status": ["hot_water", "dhw"],
}

# Entity-id substrings that must never be matched by discovery
# 47394 = MyUplink "control room sensor syst" (configuration, not a temperature)
# (EffektGuard's own entities are excluded by registry platform, not by name)
NIBE_DISCOVERY_EXCLUDE: Final[list[str]] = ["47394", "control_room_sensor"]

# Keys that trigger re-discovery on the update cycle while missing.
# Source integrations (modbus, nibe_heatpump, myuplink) may create their
# entities AFTER EffektGuard's first discovery pass during HA startup.
NIBE_DISCOVERY_CORE_KEYS: Final[tuple[str, ...]] = ("outdoor_temp", "indoor_temp", "supply_temp")

# Re-discover fast for this many attempts, then fall back to a slow retry
# so a source integration appearing hours after HA start is still picked up
# without rescanning every cycle forever when a sensor genuinely does not
# exist (e.g. no BT50 room sensor - a legitimate configuration).
NIBE_DISCOVERY_MAX_ATTEMPTS: Final = 12
NIBE_DISCOVERY_SLOW_RETRY_CYCLES: Final = 6  # every 6th update cycle (~30 min)

# Cache keys that must be temperature sensors (validated by device_class/unit)
NIBE_TEMPERATURE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "outdoor_temp",
        "indoor_temp",
        "supply_temp",
        "return_temp",
        "dhw_top_temp",
        "dhw_charging_temp",
    }
)

# Discovery candidate ranks: lower wins. Manual overrides always win; a live
# entity backed by the entity registry (a real integration) beats a live
# entity without a registry entry (template/MQTT/modbus YAML lookalikes),
# which beats a registry entry without a state yet (stale/still-loading
# entries otherwise win by iteration order and read as defaults forever).
NIBE_DISCOVERY_RANK_MANUAL: Final = -1
NIBE_DISCOVERY_RANK_LIVE: Final = 0
NIBE_DISCOVERY_RANK_LIVE_UNREGISTERED: Final = 1
NIBE_DISCOVERY_RANK_REGISTRY_ONLY: Final = 2

# Switch entity discovery (separate domain scan)
NIBE_SWITCH_DISCOVERY_PATTERNS: Final[dict[str, list[str]]] = {
    "increased_ventilation": ["increased_ventilation"],
}

# Cache keys that hold manual config-flow overrides (override beats discovery)
# Maps discovery cache key -> CONF_* config key
NIBE_MANUAL_OVERRIDE_KEYS: Final[dict[str, str]] = {
    "outdoor_temp": CONF_OUTDOOR_TEMP_ENTITY,
    "indoor_temp": CONF_INDOOR_TEMP_ENTITY,
    "supply_temp": CONF_SUPPLY_TEMP_ENTITY,
    "return_temp": CONF_RETURN_TEMP_ENTITY,
    "dhw_top_temp": CONF_DHW_TEMP_ENTITY,
    "dhw_charging_temp": CONF_DHW_CHARGING_TEMP_ENTITY,
}

# Sensor states that mean "no valid reading": -32768 is the s16 unknown-value
# marker (MyUplink API and disconnected Modbus sensors); -3276.8 is the same
# marker after the common x10 scale factor. Treated as unavailable.
NIBE_UNKNOWN_VALUE_MARKERS: Final[frozenset[float]] = frozenset({-32768.0, -3276.8})

# Re-sync the tracked offset from the entity when it disagrees and no
# write happened within this window (external change or failed write)
NIBE_OFFSET_RESYNC_MINUTES: Final = 15

# States treated as "active" when reading status entities.
# Covers on/off style states plus nibe_heatpump mapped coil states
# ("Running"/"Starting" from compressor state 43427). Compared lowercase.
NIBE_STATUS_ACTIVE_STATES: Final[frozenset[str]] = frozenset(
    {"on", "true", "1", "yes", "running", "starting"}
)

# Priority register 43086 identifies what the compressor is serving:
# 10=Off, 20=Hot Water, 30=Heat, 40=Pool, 41=Pool 2, 50=Transfer, 60=Cooling
# (yozik04/nibe, nibe/data/f1155_f1255.json). nibe_heatpump exposes the mapped
# strings, raw Modbus the numbers, MyUplink its own enum text. A run with
# hot-water priority must NOT count as space heating. Compared lowercase;
# unknown values leave the pattern-based status reads untouched.
NIBE_PRIO_HOT_WATER_STATES: Final[frozenset[str]] = frozenset(
    {"hot water", "hot_water", "20", "20.0"}
)
NIBE_PRIO_HEATING_STATES: Final[frozenset[str]] = frozenset({"heat", "heating", "30", "30.0"})

# Compressor frequency above this means the compressor is running.
# Used to derive is_heating when no parseable status entity exists
# (raw Modbus 43427 is a numeric enum: 20=Stopped 40=Starting 60=Running).
NIBE_COMPRESSOR_ACTIVE_HZ_THRESHOLD: Final = 1.0  # Hz

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

# THE SWEDISH EFFECT TARIFF, AS A REAL COMPANY ACTUALLY BILLS IT.
#
# Every number below used to be a guess. The rate was "50.0 # Conservative average", attributed to
# "Ellevio ~55, Vattenfall/E.ON ~50" - figures that appear in no price list. The simulator
# meanwhile used 81.25 and called it "fictional-but-typical". Two different numbers for one
# quantity, neither sourced, and the one in the SEK figure shown to the owner was the wrong one.
#
# Ellevio publishes its model in full, and 81.25 is theirs:
#
#     "Genomsnittet av de tre hogsta effekttopparna under manaden" - the mean of the three highest
#     peaks of the month, at most one per day, so on three different days. The measurement uses
#     HOURLY AVERAGES. Between 22:00 and 06:00 "raknas bara halva effekttoppen" - only half the
#     peak counts. 81,25 kr per kilowatt per manad.
#     https://www.ellevio.se/abonnemang/ny-prismodell-baserad-pa-effekt/
#
# REGULATORY STATUS, AND IT IS NOT SETTLED. On 13 March 2026 the government instructed
# Energimarknadsinspektionen to repeal the requirement that grid companies levy effect charges at
# all - the stated reason being that every DSO had built its own model, with different calculations,
# different hours and different prices. EIFS 2022:1 was repealed in June 2026 and Ellevio dropped
# its effect charge on 1 June 2026. Ei must propose a new, uniform model by 12 April 2027.
#
# Effect charges are NOT prohibited and several DSOs still levy them, so the feature is not dead -
# but this rate is one company's, it is no longer that company's, and it is not configurable. That
# is a product decision and it is the owner's.
#     https://www.regeringen.se/pressmeddelanden/2026/03/krav-pa-inforande-av-effektavgifter-stoppas/
#     https://ei.se/konsument/anvand-el-smartare/elnatsavtal-med-effektavgift
SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH: Final = 81.25  # Ellevio, kr/kW/month

# THE TARIFF BILLS THE HOUR. IT DOES NOT BILL THE QUARTER-HOUR.
#
# The integration measured quarter-hour means and called them billing peaks, and the constant that
# said so called itself "Swedish Effektavgift measurement period". Ellevio: "the measurement uses
# hourly averages". Energimarknadsinspektionen: "elnatsforetagen mater din elanvandning per timme".
#
# The difference is up to fourfold. A 15-minute hot-water cycle at 9 kW inside an otherwise idle
# hour has an hourly mean of 3 kW - and EffektGuard recorded 9, then throttled the heat pump to
# defend a peak that appears on no bill.
BILLING_PERIOD_MINUTES: Final = 60
BILLING_PERIODS_PER_DAY: Final = 24
# The longest silence between two meter readings that still leaves a billing hour MEASURED.
#
# The hourly mean weights each reading by how long it stood, so a reading is implicitly extrapolated
# forward until the next one arrives. That is correct for the ordinary five-minute cadence and absurd
# across a blackout: a meter that read 9 kW at 10:00, went `unavailable`, and came back at 10:55
# reading 1 kW had that 9 kW stretched over fifty unwatched minutes and the hour was billed at
# 8.33 kW. The effect tariff bills the mean of the month's three highest hours, and this integration
# throttles the pump to defend that record - so a fabricated peak holds the heat back for weeks to
# protect a number that happened in no hour.
#
# A JUDGEMENT, NOT A CITATION. No standard says how much of an hour must be seen; what is defensible
# is the DIRECTION. Refusing an under-observed hour can miss a real peak, which costs some protection.
# Billing an invented one costs a month of throttling to defend a fiction - and the utility bills from
# ITS meter, not from ours, so our record only decides whether to hold the pump back. Missing an hour
# is recoverable; inventing one is not.
#
# Three update intervals: one or two missed cycles is jitter, which Home Assistant does routinely, and
# the reading either side of a brief blink is the same reading. Fifteen consecutive minutes of silence
# from a sensor that polls every five is an outage, and an hour containing one was not measured.
MAX_BILLING_OBSERVATION_GAP_MINUTES: Final = 15

# BASELINE_PEAK_MULTIPLIER (1.176) was deleted. It manufactured an unoptimised baseline from the
# CURRENT peak - `baseline = peak * 1.176` - so the reported effect-tariff saving reduced to
# `0.176 * peak * tariff`: a higher peak reported MORE "savings", and the sensor could never read
# zero however badly the optimiser was doing. A baseline is now either MEASURED (from the quarters
# recorded while the optimisation switch is off, when the offset is held at 0.0) or absent, and an
# absent one reports no effect saving at all.

# Price-unit handling for savings math. GE-Spot preserves whatever display
# unit the user configured; savings must convert to the MAIN currency unit
# accordingly instead of assuming öre/kWh. Prefix match on the normalized
# (lowercase, spaceless) unit string; sub-units (öre, cents) divide by 100.
PRICE_SUBUNIT_PREFIXES: Final[tuple[str, ...]] = ("öre", "ore", "cent", "snt")
PRICE_MAINUNIT_PREFIXES: Final[tuple[str, ...]] = ("sek", "eur", "nok", "dkk", "€", "kr")

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
# Based on typical NIBE F750/F2040 inverter compressor specifications:
# - Frequency range: 20-120 Hz (model-specific, see modulation_range in heat pump profiles)
# - 20 Hz (idle): ~1.5-2.0 kW
# - 50 Hz (mid): ~3.5-4.5 kW
# - 80 Hz (normal max): ~6.0-7.0 kW
# - 100-120 Hz (emergency/max): Higher output, reduced efficiency
# Compressor stress levels, as reported by CompressorHealthMonitor.assess_risk().
#
# HIGH means the compressor has been above 100 Hz for more than fifteen minutes: it is at maximum
# capacity and has nothing left to give. Asking it for more heat by raising the curve offset
# produces NONE - the offset raises the calculated supply setpoint S1, and the compressor cannot
# follow it. What it does produce is wear, and a DEEPER degree-minute deficit, because
# DM = integral(BT25 - S1) and S1 just went up while BT25 could not.
#
# The auxiliary heater exists for exactly this moment: NIBE places "start addition" where it does
# (menu 4.9.3) so the compressor need not grind at full frequency for hours. Declining its help by
# demanding more from a saturated compressor trades cheap kWh for expensive compressor life.
COMPRESSOR_RISK_HIGH: Final = "HIGH"  # >100 Hz for >15 min - at maximum, nothing left to give
COMPRESSOR_RISK_ELEVATED: Final = "ELEVATED"  # >80 Hz for >2 h
COMPRESSOR_RISK_NOTABLE: Final = "NOTABLE"
COMPRESSOR_RISK_WATCH: Final = "WATCH"
COMPRESSOR_RISK_OK: Final = "OK"

COMPRESSOR_HZ_MIN: Final = 20  # Minimum operating frequency
COMPRESSOR_HZ_MAX: Final = 120  # Maximum operating frequency (NIBE F-series inverter)
COMPRESSOR_HZ_RANGE: Final = 100  # 120-20 = operating range
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


# Classifications that benefit from heating boosts (pre-heating opportunities)
# Used by price_layer and volatile_helpers to determine when to boost heating
BENEFICIAL_CLASSIFICATIONS: Final = frozenset(
    [QuarterClassification.VERY_CHEAP, QuarterClassification.CHEAP]
)


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

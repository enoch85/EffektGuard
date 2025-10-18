"""Constants for EffektGuard integration."""

from enum import StrEnum
from typing import Final

# Domain
DOMAIN: Final = "effektguard"

# DEBUG MODE - Force outdoor temperature for testing
# Set to None to disable, or a float value to override outdoor temp
# Example: DEBUG_FORCE_OUTDOOR_TEMP = -5.0 to test cold weather behavior
# WARNING: Only for development/testing! Remove for production.
DEBUG_FORCE_OUTDOOR_TEMP: Final = None  # Set to -5.0 to test

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
CONF_CONTROL_PRIORITY: Final = "control_priority"
CONF_THERMAL_MASS: Final = "thermal_mass"
CONF_INSULATION_QUALITY: Final = "insulation_quality"
CONF_HEAT_PUMP_MODEL: Final = "heat_pump_model"
CONF_PEAK_PROTECTION_MARGIN: Final = "peak_protection_margin"
CONF_WEATHER_COMPENSATION_WEIGHT: Final = "weather_compensation_weight"  # 0.0-1.0

# Defaults
DEFAULT_TOLERANCE: Final = 0.5
DEFAULT_TARGET_TEMP: Final = 21.0
DEFAULT_INDOOR_TEMP: Final = 21.0  # Fallback when sensor unavailable
DEFAULT_THERMAL_MASS: Final = 1.0
DEFAULT_INSULATION_QUALITY: Final = 1.0
DEFAULT_HEAT_PUMP_MODEL: Final = "nibe_f750"  # Most common model
DEFAULT_OPTIMIZATION_MODE: Final = "balanced"
DEFAULT_PEAK_PROTECTION_MARGIN: Final = 0.5  # kW
DEFAULT_WEATHER_COMPENSATION_WEIGHT: Final = 0.49  # User-configurable weight (matches layer weight)
DEFAULT_INDOOR_TEMP_METHOD: Final = "median"  # median more robust to outliers than average
DEFAULT_DHW_TARGET_TEMP: Final = 50.0  # °C - Default DHW target temperature

# Climate entity temperature limits (displayed in UI)
MIN_INDOOR_TEMP: Final = 15.0  # °C - minimum settable temperature
MAX_INDOOR_TEMP: Final = 25.0  # °C - maximum settable temperature
TEMP_STEP: Final = 0.5  # °C - temperature adjustment step

# Optimization modes for climate entity presets
OPTIMIZATION_MODE_COMFORT: Final = "comfort"  # Minimize deviation, accept higher costs
OPTIMIZATION_MODE_BALANCED: Final = "balanced"  # Balance comfort and savings
OPTIMIZATION_MODE_SAVINGS: Final = "savings"  # Maximize savings, wider tolerance

# Control priority options (same values as optimization modes for consistency)
CONTROL_PRIORITY_COMFORT: Final = "comfort"
CONTROL_PRIORITY_BALANCED: Final = "balanced"
CONTROL_PRIORITY_SAVINGS: Final = "savings"
DEFAULT_CONTROL_PRIORITY: Final = "balanced"

# Config keys for internal use
CONF_TARGET_INDOOR_TEMP: Final = "target_indoor_temp"

# Limits
MIN_OFFSET: Final = -10.0
MAX_OFFSET: Final = 10.0
MIN_TARGET_TEMP: Final = 18.0
MAX_TARGET_TEMP: Final = 26.0
MIN_TEMP_LIMIT: Final = 18.0
MAX_TEMP_LIMIT: Final = 24.0

# Service call rate limiting (boost, DHW, general)
BOOST_COOLDOWN_MINUTES: Final = 45  # Prevent boost spam
DHW_BOOST_COOLDOWN_MINUTES: Final = 60  # DHW boost cooldown
DHW_CONTROL_MIN_INTERVAL_MINUTES: Final = 60  # Automatic DHW control rate limit (1 hour)
SERVICE_RATE_LIMIT_MINUTES: Final = 5  # General service call cooldown

# Decision engine layer weights
# Source: User feedback and optimization tuning (Oct 2025)
# Philosophy: "Charge heat when cheap, without peaking the peak"
LAYER_WEIGHT_SAFETY: Final = 1.0  # Absolute priority (temp limits)
LAYER_WEIGHT_EMERGENCY: Final = 0.8  # High priority (DM beyond expected)
LAYER_WEIGHT_PRICE: Final = 0.75  # Strong influence (increased from 0.6)
LAYER_WEIGHT_WEATHER_COMP: Final = 0.49  # Moderate influence
LAYER_WEIGHT_PROACTIVE_MIN: Final = 0.3  # Minimum proactive weight
LAYER_WEIGHT_PROACTIVE_MAX: Final = 0.6  # Maximum proactive weight
LAYER_WEIGHT_COMFORT_MIN: Final = 0.2  # Minimum comfort weight
LAYER_WEIGHT_COMFORT_MAX: Final = 0.5  # Maximum comfort weight

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

# Safety thresholds - Degree Minutes (DM) / Gradminuter (GM)
# Based on Swedish NIBE forum research and real-world validation
# Source: Forum_Summary.md, Swedish_NIBE_Forum_Findings.md, Swedish_Climate_Adaptations.md
#
# Swedish research shows DM -1500 is safe operating limit across all Nordic conditions.
# More conservative than UK-based -500 threshold, but appropriate for Nordic climate.
# Validated in real-world Swedish conditions: -30°C to +5°C temperature range.
#
# CLIMATE-AWARE DESIGN:
# Climate zone module (climate_zones.py) provides context-aware DM thresholds that
# automatically adapt from Arctic (-30°C) to Mild (5°C) climates without configuration.
# No hardcoded temperature bands - uses latitude-based climate zones with outdoor temp adjustment.
#
DM_THRESHOLD_START: Final = -60  # Normal compressor start (NIBE standard)
DM_THRESHOLD_ABSOLUTE_MAX: Final = -1500  # NEVER EXCEED - hard safety limit

# UFH prediction horizons based on thermal lag research
# Source: glyn.hudson F2040 case study, enoch85 extreme cold snap feedback
UFH_CONCRETE_PREDICTION_HORIZON: Final = (
    24.0  # hours - 6+ hour lag, needs 24h for extreme cold (20°C drops)
)
UFH_TIMBER_PREDICTION_HORIZON: Final = 12.0  # hours - 2-3 hour lag (increased from 6h)
UFH_RADIATOR_PREDICTION_HORIZON: Final = 6.0  # hours - <1 hour lag (increased from 2h)

# UFH comfort targets (°C)
UFH_CONCRETE_COMFORT_TOLERANCE: Final = 0.3  # ±0.3°C for concrete slab
UFH_TIMBER_COMFORT_TOLERANCE: Final = 0.3  # ±0.3°C for timber
UFH_RADIATOR_COMFORT_TOLERANCE: Final = 0.2  # ±0.2°C for radiators

# Optimal flow temperature deltas (°C above outdoor temp)
# Source: Mathematical_Enhancement_Summary.md - André Kühne formula validation
OPTIMAL_FLOW_DELTA_SPF_4: Final = 27.0  # ±3°C for SPF ≥4.0 systems
OPTIMAL_FLOW_DELTA_SPF_35: Final = 30.0  # ±4°C for SPF ≥3.5 systems

# Weather compensation mathematical constants
# Source: Mathematical_Enhancement_Summary.md - OpenEnergyMonitor research
KUEHNE_COEFFICIENT: Final = 2.55  # André Kühne's universal constant
KUEHNE_POWER: Final = 0.78  # Power coefficient for heat transfer physics
RADIATOR_POWER_COEFFICIENT: Final = 1.3  # BS EN442 standard radiator output
RADIATOR_RATED_DT: Final = 50.0  # Standard test DT (75°C flow, 65°C return, 20°C room)

# UFH flow temperature adjustments
# Source: Mathematical_Enhancement_Summary.md - Floor heating optimizations
UFH_FLOW_REDUCTION_CONCRETE: Final = 8.0  # °C reduction for concrete slab UFH
UFH_FLOW_REDUCTION_TIMBER: Final = 5.0  # °C reduction for timber/lightweight UFH
UFH_MIN_FLOW_TEMP_CONCRETE: Final = 25.0  # Minimum effective concrete slab temp
UFH_MIN_FLOW_TEMP_TIMBER: Final = 22.0  # Minimum effective timber UFH temp

# Heat loss coefficient defaults (W/°C)
# Source: Timbones' heat transfer method - typical house range
DEFAULT_HEAT_LOSS_COEFFICIENT: Final = 180.0  # W/°C typical value
HEAT_LOSS_COEFFICIENT_MIN: Final = 100.0  # W/°C well-insulated house
HEAT_LOSS_COEFFICIENT_MAX: Final = 300.0  # W/°C poorly-insulated house

# Power estimation defaults (kW)
# Used when actual power sensor unavailable - fallback values
DEFAULT_BASE_POWER: Final = 3.0  # kW typical NIBE heat pump average
ESTIMATED_POWER_BASELINE: Final = 4.0  # kW baseline for coordinator estimates
TEMP_FACTOR_MIN: Final = 0.5  # Minimum temperature scaling factor
TEMP_FACTOR_MAX: Final = 3.0  # Maximum temperature scaling factor

# Effect tariff peak tracking thresholds (kW)
# Source: Real-world NIBE heat pump consumption patterns
# Prevents recording standby/startup noise as legitimate peaks
# Start with lower threshold for initial learning phase
# System learns from actual usage patterns even during idle periods
PEAK_RECORDING_MINIMUM: Final = 0.5  # kW - lowered from 1.0 for better learning
# Typical NIBE consumption: standby 0.05-0.1 kW, heating 2.5-6.0 kW

# Pump configuration - open-loop UFH requirements
# Source: Forum_Summary.md - glyn.hudson case study (8-hour off periods)
PUMP_MIN_SPEED_ASHP: Final = 10  # % for ASHP open-loop systems
PUMP_MIN_SPEED_GSHP: Final = 20  # % for GSHP open-loop systems

# Update intervals
UPDATE_INTERVAL_MINUTES: Final = (
    5  # Coordinator update frequency + thermal predictor save throttle interval
)
QUARTER_INTERVAL_MINUTES: Final = 15  # Swedish Effektavgift measurement period

# Adaptive learning parameters
# Source: POST_PHASE_5_ROADMAP.md Phase 6 - Self-Learning Capability
LEARNING_OBSERVATION_WINDOW: Final = 672  # 1 week of 15-minute observations
LEARNING_MIN_OBSERVATIONS: Final = 96  # 24 hours minimum for basic learning
LEARNING_CONFIDENCE_THRESHOLD: Final = 0.7  # 70% confidence to use learned params
LEARNING_UPDATE_INTERVAL_HOURS: Final = 24  # Re-calculate learned params daily

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
# - 30°C (NIBE_DHW_SAFETY_CRITICAL): Emergency override, always heat
# - 35°C (NIBE_DHW_SAFETY_MIN): Legionella prevention minimum
# - 40°C (DHW_MIN_TEMP): User-configurable minimum (validation)
# - 45°C (MIN_DHW_TARGET_TEMP): Minimum user target / NIBE start threshold
# - 50°C (NIBE_DHW_NORMAL_TARGET): Normal comfort target
# - 60°C (DHW_MAX_TEMP): Maximum comfort temperature
#
DHW_MIN_TEMP: Final = 40.0  # °C - Minimum safe DHW temperature (user validation)
DHW_MAX_TEMP: Final = 60.0  # °C - Maximum normal DHW temperature (comfort)
DHW_PREHEAT_TARGET_OFFSET: Final = 5.0  # °C - Extra heating above target for optimal windows
MIN_DHW_TARGET_TEMP: Final = (
    45.0  # °C - Minimum user-configurable DHW target (safety + comfort threshold)
)
DHW_LEGIONELLA_DETECT: Final = 63.0  # °C - BT7 temp indicating Legionella boost
DHW_TARGET_HIGH_DEMAND: Final = 55.0  # °C - Extra comfort target for high demand periods
DHW_TARGET_NORMAL: Final = 50.0  # °C - Normal DHW comfort target temperature
DHW_HEATING_TIME_HOURS: Final = 1.5  # Hours to heat DHW tank (typically 1-2h)
DHW_SCHEDULING_WINDOW_MAX: Final = 24  # Max hours ahead for DHW scheduling
DHW_SCHEDULING_WINDOW_MIN: Final = 1  # Min hours ahead for DHW scheduling
DHW_MAX_WAIT_HOURS: Final = 36.0  # Max hours between DHW heating (hygiene/comfort)

# DHW thermal debt thresholds (climate-aware via spare capacity calculation)
# Instead of hardcoded DM thresholds, we calculate spare capacity as percentage
# above the climate-aware warning threshold for current outdoor temperature
DHW_SPARE_CAPACITY_PERCENT: Final = 50.0  # Require 50% spare capacity above warning threshold
# Ensures DHW heating only when heat pump has significant spare capacity
# Example: Stockholm at -10°C has warning=-700, so require DM > -350 (-700 * 0.5)
# Example: Kiruna at -30°C has warning=-1200, so require DM > -600 (-1200 * 0.5)
# This keeps DHW heating within the normal operating range, not near thermal debt warning

# DHW thermal debt fallback thresholds (used only if climate detector unavailable)
# These are balanced fixed values for rare fallback scenarios
DM_DHW_BLOCK_FALLBACK: Final = -340.0  # Fallback: Never start DHW below this DM
DM_DHW_ABORT_FALLBACK: Final = -500.0  # Fallback: Abort DHW if reached during run
DM_DHW_SPARE_CAPACITY_FALLBACK: Final = -80.0  # Fallback: Spare capacity threshold

# DHW runtime safeguards (monitoring only - NIBE controls actual completion)
DHW_SAFETY_RUNTIME_MINUTES: Final = 30  # Safety minimum heating (emergency)
DHW_NORMAL_RUNTIME_MINUTES: Final = 45  # Normal DHW heating window
DHW_EXTENDED_RUNTIME_MINUTES: Final = 60  # High demand period heating
DHW_URGENT_RUNTIME_MINUTES: Final = 90  # Urgent pre-demand heating

# MyUplink DHW control entities (NIBE parameter IDs)
NIBE_TEMP_LUX_ENTITY_ID: Final = "switch.temporary_lux_50004"  # Temporary lux boost
NIBE_BT7_SENSOR_ID: Final = "sensor.bt7_hw_top_40013"  # Hot water top temperature

# NIBE Power Calculation Constants (Swedish 3-phase standard)
# All NIBE heat pumps in Sweden are 3-phase systems
NIBE_VOLTAGE_PER_PHASE: Final = (
    240.0  # V - Swedish 3-phase: 400V between phases, 240V phase-to-neutral
)
NIBE_POWER_FACTOR: Final = 0.95  # Conservative for inverter compressor (real likely 0.96-0.98)
DHW_SAFETY_CRITICAL: Final = 30.0  # °C - Below this, always heat (safety override)
DHW_SAFETY_MIN: Final = 35.0  # °C - Safety minimum (can defer if 30-35°C during expensive periods)
NIBE_DHW_START_THRESHOLD: Final = 45.0  # °C - Typical NIBE DHW heating trigger setpoint
NIBE_DHW_COOLING_RATE: Final = 0.5  # °C/hour - Conservative DHW tank cooling estimate

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

# Heating impact factors
HEATING_FACTOR_PER_DEGREE: Final = 0.1  # 10% power change per °C offset change
CHEAP_PERIOD_BONUS_MULTIPLIER: Final = 1.2  # 20% bonus for strategic preheating
EXPENSIVE_AVOIDANCE_BONUS_MULTIPLIER: Final = 1.3  # 30% bonus for avoiding expensive periods
EMERGENCY_HEATING_COST_FACTOR: Final = 0.7  # 30% cost recognition for unavoidable heating

# Baseline tracking - exponential moving average weights
BASELINE_EMA_WEIGHT_OLD: Final = 0.8  # 80% weight on existing baseline
BASELINE_EMA_WEIGHT_NEW: Final = 0.2  # 20% weight on new observation

# Default heat pump power for savings estimation (system-specific, will vary)
# F750: typically 2-7 kW depending on outdoor temp and compressor frequency
DEFAULT_HEAT_PUMP_POWER_KW: Final = 4.0  # Mid-range estimate for average conditions


class OptimizationMode(StrEnum):
    """Optimization mode options."""

    BASIC = "basic"
    PRICE = "price"
    ADVANCED = "advanced"
    EXPERT = "expert"


class QuarterClassification(StrEnum):
    """Quarter-hour classification based on GE-Spot prices."""

    CHEAP = "cheap"
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

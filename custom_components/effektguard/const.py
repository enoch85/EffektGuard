"""Constants for EffektGuard integration."""

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
DEFAULT_THERMAL_MASS: Final = 1.0
DEFAULT_INSULATION_QUALITY: Final = 1.0
DEFAULT_HEAT_PUMP_MODEL: Final = "nibe_f750"  # Most common model
DEFAULT_OPTIMIZATION_MODE: Final = "balanced"
DEFAULT_PEAK_PROTECTION_MARGIN: Final = 0.5  # kW
DEFAULT_WEATHER_COMPENSATION_WEIGHT: Final = 0.75  # Moderate influence

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
SERVICE_RATE_LIMIT_MINUTES: Final = 5  # General service call cooldown

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
# Source: Enhancement_Proposals.md, Floor_Heating_Enhancements.md
UFH_CONCRETE_PREDICTION_HORIZON: Final = 12.0  # hours - 6+ hour lag
UFH_TIMBER_PREDICTION_HORIZON: Final = 6.0  # hours - 2-3 hour lag
UFH_RADIATOR_PREDICTION_HORIZON: Final = 2.0  # hours - <1 hour lag

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

# Pump configuration - open-loop UFH requirements
# Source: Forum_Summary.md - glyn.hudson case study (8-hour off periods)
PUMP_MIN_SPEED_ASHP: Final = 10  # % for ASHP open-loop systems
PUMP_MIN_SPEED_GSHP: Final = 20  # % for GSHP open-loop systems

# Update intervals
UPDATE_INTERVAL_MINUTES: Final = 5  # Coordinator update frequency
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
# Note: This replaces old "arctic", "subarctic", etc. names with heating-focused names.
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
SERVICE_CALCULATE_OPTIMAL_SCHEDULE: Final = "calculate_optimal_schedule"

# Service parameters
ATTR_OFFSET: Final = "offset"
ATTR_DURATION: Final = "duration"

# Attributes
ATTR_CURRENT_OFFSET: Final = "current_offset"
ATTR_DECISION_REASONING: Final = "decision_reasoning"
ATTR_LAYER_VOTES: Final = "layer_votes"
ATTR_PEAK_TODAY: Final = "peak_today"
ATTR_PEAK_THIS_MONTH: Final = "peak_this_month"
ATTR_THERMAL_DEBT: Final = "thermal_debt"
ATTR_QUARTER_OF_DAY: Final = "quarter_of_day"
ATTR_OPTIONAL_FEATURES: Final = "optional_features_status"


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
    """Underfloor heating system type."""

    CONCRETE_SLAB = "concrete_slab"
    TIMBER = "timber"
    RADIATOR = "radiator"
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

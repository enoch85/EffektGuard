"""Constants for EffektGuard integration."""

from enum import StrEnum
from typing import Final

# Domain
DOMAIN: Final = "effektguard"

# Configuration keys
CONF_NIBE_ENTITY: Final = "nibe_entity"
CONF_GESPOT_ENTITY: Final = "gespot_entity"
CONF_WEATHER_ENTITY: Final = "weather_entity"
CONF_ENABLE_PRICE_OPTIMIZATION: Final = "enable_price_optimization"
CONF_ENABLE_PEAK_PROTECTION: Final = "enable_peak_protection"
CONF_TARGET_TEMPERATURE: Final = "target_temperature"
CONF_TOLERANCE: Final = "tolerance"
CONF_OPTIMIZATION_MODE: Final = "optimization_mode"
CONF_THERMAL_MASS: Final = "thermal_mass"
CONF_INSULATION_QUALITY: Final = "insulation_quality"

# Defaults
DEFAULT_TOLERANCE: Final = 0.5
DEFAULT_TARGET_TEMP: Final = 21.0
DEFAULT_THERMAL_MASS: Final = 1.0
DEFAULT_INSULATION_QUALITY: Final = 1.0
DEFAULT_OPTIMIZATION_MODE: Final = "balanced"

# Optimization modes for climate entity presets
OPTIMIZATION_MODE_COMFORT: Final = "comfort"  # Minimize deviation, accept higher costs
OPTIMIZATION_MODE_BALANCED: Final = "balanced"  # Balance comfort and savings
OPTIMIZATION_MODE_SAVINGS: Final = "savings"  # Maximize savings, wider tolerance

# Config keys for internal use
CONF_TARGET_INDOOR_TEMP: Final = "target_indoor_temp"

# Limits
MIN_OFFSET: Final = -10.0
MAX_OFFSET: Final = 10.0
MIN_TARGET_TEMP: Final = 18.0
MAX_TARGET_TEMP: Final = 26.0
MIN_TEMP_LIMIT: Final = 18.0
MAX_TEMP_LIMIT: Final = 24.0

# Safety thresholds - Degree Minutes (DM) / Gradminuter (GM)
# Based on real-world NIBE F2040/F750 research and forum validation
# Source: Forum_Summary.md, Swedish_NIBE_Forum_Findings.md
DM_THRESHOLD_START: Final = -60  # Normal compressor start
DM_THRESHOLD_EXTENDED: Final = -240  # Extended runs, acceptable (stevedvo custom)
DM_THRESHOLD_WARNING: Final = -400  # Approaching danger, prevent reductions
DM_THRESHOLD_CRITICAL: Final = -500  # Catastrophic: 15kW spikes, 10°K overshoot
DM_THRESHOLD_AUX_SWEDISH: Final = -1500  # Swedish F750 auxiliary optimization

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

# Pump configuration - open-loop UFH requirements
# Source: Forum_Summary.md - glyn.hudson case study (8-hour off periods)
PUMP_MIN_SPEED_ASHP: Final = 10  # % for ASHP open-loop systems
PUMP_MIN_SPEED_GSHP: Final = 20  # % for GSHP open-loop systems

# Update intervals
UPDATE_INTERVAL_MINUTES: Final = 5  # Coordinator update frequency
QUARTER_INTERVAL_MINUTES: Final = 15  # Swedish Effektavgift measurement period

# Storage
STORAGE_VERSION: Final = 1
STORAGE_KEY: Final = f"{DOMAIN}_state"

# Services
SERVICE_SET_OPTIMIZATION_MODE: Final = "set_optimization_mode"
SERVICE_FORCE_UPDATE: Final = "force_update"
SERVICE_RESET_PEAKS: Final = "reset_peaks"

# Attributes
ATTR_CURRENT_OFFSET: Final = "current_offset"
ATTR_DECISION_REASONING: Final = "decision_reasoning"
ATTR_LAYER_VOTES: Final = "layer_votes"
ATTR_PEAK_TODAY: Final = "peak_today"
ATTR_PEAK_THIS_MONTH: Final = "peak_this_month"
ATTR_THERMAL_DEBT: Final = "thermal_debt"
ATTR_QUARTER_OF_DAY: Final = "quarter_of_day"


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

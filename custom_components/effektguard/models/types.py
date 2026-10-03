"""Type definitions for EffektGuard configuration.

TypedDicts for configuration dictionaries to avoid using Any type.
These provide better type checking and IDE support.
"""

from typing import TypedDict


class EffektGuardConfigDict(TypedDict, total=False):
    """Configuration dictionary for EffektGuard components.

    This TypedDict defines the expected keys and types for configuration
    passed to adapters and the decision engine. Using total=False allows
    all keys to be optional (adapters handle missing keys with .get()).
    """

    # Entity IDs (from config flow)
    nibe_entity: str
    gespot_entity: str
    weather_entity: str
    degree_minutes_entity: str
    power_sensor_entity: str
    dhw_temp_entity: str
    nibe_temp_lux_entity: str
    additional_indoor_sensors: list[str]
    indoor_temp_method: str

    # Manual sensor overrides (issue #18: Modbus/renamed entities)
    outdoor_temp_entity: str
    indoor_temp_entity: str
    supply_temp_entity: str
    return_temp_entity: str
    dhw_charging_temp_entity: str

    # Feature flags
    enable_price_optimization: bool
    enable_peak_protection: bool
    enable_weather_prediction: bool
    enable_hot_water_optimization: bool
    enable_weather_compensation: bool
    enable_optimization: bool

    # Thermal settings
    target_indoor_temp: float
    tolerance: float
    thermal_mass: float
    insulation_quality: float
    heat_loss_coefficient: float
    radiator_rated_output: float
    heating_type: str

    # Optimization settings
    optimization_mode: str
    weather_compensation_weight: float

    # Location
    latitude: float
    longitude: float

    # Heat pump model
    heat_pump_model: str


class AdapterConfigDict(TypedDict, total=False):
    """Minimal configuration for adapters.

    Subset of EffektGuardConfigDict focused on what adapters need.
    """

    # Required entity IDs
    nibe_entity: str
    gespot_entity: str
    weather_entity: str

    # Optional entity IDs
    degree_minutes_entity: str
    power_sensor_entity: str
    additional_indoor_sensors: list[str]
    indoor_temp_method: str

    # Manual sensor overrides (issue #18: Modbus/renamed entities)
    outdoor_temp_entity: str
    indoor_temp_entity: str
    supply_temp_entity: str
    return_temp_entity: str
    dhw_temp_entity: str
    dhw_charging_temp_entity: str


class DiagnosticsLayerDict(TypedDict):
    """One layer's vote in the decision that was made."""

    name: str | None
    offset: float | None
    weight: float | None
    reason: str | None


class DiagnosticsDecisionDict(TypedDict, total=False):
    """The offset commanded, and the votes behind it."""

    offset: float | None
    reasoning: str | None
    is_emergency: bool | None
    is_manual_override: bool | None
    anti_windup_active: bool | None
    layers: list[DiagnosticsLayerDict]


class DiagnosticsDict(TypedDict, total=False):
    """What a bug report about this heat pump needs to contain.

    The decision, the state it was made from, the degree-minute band actually in force, and whether
    the price and weather sources were even live. NOT the home's coordinates: the decision engine
    holds the latitude (it is how the climate zone is detected) and this file gets pasted into
    public issue trackers. The climate ZONE goes in instead - it is what the thresholds derive
    from, and it identifies nobody.
    """

    error: str
    config: dict[str, object]
    sources: dict[str, str]
    nibe: dict[str, object]
    decision: DiagnosticsDecisionDict
    dm_thresholds: dict[str, object]
    compressor_risk: str | None
    peaks: dict[str, object]

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

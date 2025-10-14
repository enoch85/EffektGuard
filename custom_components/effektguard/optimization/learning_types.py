"""Shared types for learning and prediction modules.

Dataclasses used across adaptive learning, thermal prediction, and weather learning.
Keeps type definitions centralized and consistent.
"""

from dataclasses import dataclass
from datetime import datetime

from ..const import UFHType


@dataclass
class AdaptiveThermalObservation:
    """Single thermal response observation for adaptive learning.

    Enhanced version of ThermalObservation with calculated metrics
    for learning thermal characteristics.
    """

    timestamp: datetime
    indoor_temp: float
    outdoor_temp: float
    heating_offset: float
    temp_change: float  # °C change since last observation
    time_delta_hours: float  # Hours since last observation


@dataclass
class LearnedThermalParameters:
    """Thermal parameters learned from observations."""

    thermal_mass: float  # Relative scale (1.0 = normal)
    heat_loss_coefficient: float  # W/°C
    heating_efficiency: float  # °C per offset per hour
    thermal_decay_rate: float  # °C/hour natural cooling
    ufh_type: UFHType  # Detected UFH type
    last_updated: datetime
    confidence: float  # 0-1 based on data quality
    observation_count: int


@dataclass
class ThermalSnapshot:
    """Snapshot of thermal state at a point in time."""

    timestamp: datetime
    indoor_temp: float
    outdoor_temp: float
    heating_offset: float
    flow_temp: float
    degree_minutes: float


@dataclass
class TempPrediction:
    """Temperature prediction result."""

    predicted_temps: list[float]  # Predicted temperatures for each hour
    hours_ahead: int
    confidence: float  # 0-1 based on model accuracy
    trajectory_trend: str  # "rising", "falling", "stable"


@dataclass
class PreHeatDecision:
    """Pre-heating decision with reasoning."""

    should_preheat: bool
    recommended_offset: float
    reason: str
    hours_until_event: int  # Hours until cold spell/peak/etc
    target_temp: float  # Target indoor temp for pre-heating


@dataclass
class WeatherPattern:
    """Historical weather pattern for a time period."""

    date: datetime
    avg_temp: float
    min_temp: float
    max_temp: float
    temp_range: float  # max - min
    volatility: float  # Standard deviation
    year: int


@dataclass
class WeatherExpectation:
    """Expected typical weather for a time period."""

    period_key: str  # e.g., "1-2" for January week 2
    typical_low: float  # 25th percentile
    typical_avg: float  # Median
    typical_high: float  # 75th percentile
    confidence: float  # 0-1 based on historical data consistency
    years_of_data: int


@dataclass
class UnusualWeatherAlert:
    """Alert for unusual weather pattern detection."""

    is_unusual: bool
    severity: str  # "normal", "unusual", "extreme"
    forecast_temps: list[float]
    deviation_from_typical: float  # °C difference from typical
    recommendation: str  # e.g., "aggressive pre-heating recommended"

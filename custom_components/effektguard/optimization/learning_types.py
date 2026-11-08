"""Shared types for learning and prediction modules.

Dataclasses used across adaptive learning, thermal prediction, and weather learning.
Keeps type definitions centralized and consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import Sequence
from math import isfinite

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


@dataclass(frozen=True)
class ThermalForecast:
    """A direct-horizon delta model and its untouched later-block validation."""

    hours: int
    coefficients: tuple[float, ...]
    skill: float
    validation_mae: float
    persistence_mae: float
    max_validation_error: float
    room_range: tuple[float, float]
    outdoor_range: tuple[float, float]
    offset_range: tuple[float, float]

    def predict(
        self, room: float, outdoor: Sequence[float], offsets: Sequence[float]
    ) -> float | None:
        """Refuse extrapolation of inputs or an incomplete conditional trajectory."""
        if len(outdoor) < self.hours or len(offsets) < self.hours:
            return None
        outside = outdoor[: self.hours]
        heating = offsets[: self.hours]
        domains = [(room, self.room_range)]
        domains += [(value, self.outdoor_range) for value in outside]
        domains += [(value, self.offset_range) for value in heating]
        for value, (lower, upper) in domains:
            if not isfinite(value) or not lower <= value <= upper:
                return None
        features = [1.0, room, sum(outside) / self.hours, sum(heating) / self.hours]
        prediction = room + sum(c * x for c, x in zip(self.coefficients, features, strict=True))
        return prediction if isfinite(prediction) else None


@dataclass
class LearnedThermalParameters:
    """Thermal parameters learned from observations."""

    thermal_mass: float  # Relative scale (1.0 = normal)
    heat_loss_coefficient: float  # Diagnostic relative index; never a physical control coefficient
    heating_efficiency: float  # °C per offset per hour
    thermal_decay_rate: float  # °C/hour natural cooling
    ufh_type: UFHType  # Detected UFH type
    last_updated: datetime
    confidence: float  # Best held-out horizon skill; forecasts carry the individual scores
    observation_count: int
    forecasts: dict[int, ThermalForecast] = field(default_factory=dict)
    observations_end: datetime | None = None


@dataclass
class ThermalSnapshot:
    """Snapshot of thermal state at a point in time."""

    timestamp: datetime
    indoor_temp: float
    outdoor_temp: float
    heating_offset: float
    flow_temp: float
    degree_minutes: float
    # False when the room reading was the adapter's placeholder (no room sensor), not a measurement
    indoor_measured: bool = True


@dataclass
class TempPrediction:
    """Temperature prediction result."""

    predicted_temps: list[float]  # Predicted temperatures for each hour
    hours_ahead: int
    confidence: float  # 0-1 based on model accuracy
    trajectory_trend: str  # "rising", "falling", "stable"
    hourly_confidence: list[float] = field(default_factory=list)


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

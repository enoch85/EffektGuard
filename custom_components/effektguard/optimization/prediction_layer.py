"""Measured thermal trends and conditional room forecasts for gated preheating.

Horizon models must beat persistence on later observations. Runtime inputs use real future
weather instants; unsupported, stale or ambiguous trajectories abstain.
"""

import logging
from bisect import bisect_left
from math import isfinite
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypedDict

import numpy as np
from homeassistant.util import dt as dt_util

from ..const import (
    LAYER_WEIGHT_PREDICTION,
    LEARNING_CONFIDENCE_THRESHOLD,
    LEARNING_OBSERVATION_INTERVAL_MINUTES,
    PREDICTION_COMFORT_THRESHOLD_COLD,
    PREDICTION_COMFORT_THRESHOLD_EXTREME_COLD,
    PREDICTION_COMFORT_THRESHOLD_MILD,
    PREDICTION_OUTDOOR_COLD,
    PREDICTION_OUTDOOR_EXTREME_COLD,
    PREDICTION_OVERSHOOT_LOG_THRESHOLD,
    PREDICTION_PREHEAT_MAX_COLD,
    PREDICTION_PREHEAT_MAX_EXTREME_COLD,
    PREDICTION_PREHEAT_MAX_MILD,
    PREDICTION_TREND_FALLING_THRESHOLD,
    PREDICTION_TREND_RISING_THRESHOLD,
    PREDICTION_LEARNED_PREHEAT_MIN_HOURS,
    PREDICTION_MIN_HISTORY_HOURS,
    PREDICTION_MAX_SAMPLE_GAP_MINUTES,
    UPDATE_INTERVAL_MINUTES,
    SAMPLES_PER_HOUR,
)
from ..utils.time_utils import seconds_between
from .learning_types import (
    LearnedThermalParameters,
    PreHeatDecision,
    TempPrediction,
    ThermalSnapshot,
)

_LOGGER = logging.getLogger(__name__)


class ThermalTrendDict(TypedDict):
    """Temperature trend analysis result."""

    trend: str  # "rising", "falling", "stable", "unknown"
    rate_per_hour: float
    confidence: float
    samples: int


class OutdoorTrendDict(TypedDict):
    """Outdoor temperature trend from BT1 sensor."""

    trend: str  # "warming", "cooling", "stable", "unknown"
    rate_per_hour: float
    confidence: float
    samples: int
    temp_change_last_2h: float


class ThermalSnapshotDict(TypedDict):
    """Dictionary representation of a ThermalSnapshot for serialization."""

    timestamp: str  # ISO format
    indoor_temp: float
    outdoor_temp: float
    heating_offset: float
    flow_temp: float
    degree_minutes: float
    indoor_measured: bool


class PredictorStateDict(TypedDict):
    """Serialized predictor state for persistence."""

    lookback_hours: int
    state_history: list[ThermalSnapshotDict]
    thermal_responsiveness: float


@dataclass
class PredictionLayerDecision:
    """Decision from the learned prediction layer.

    Encapsulates the pre-heating recommendation based on learned
    building thermal characteristics.
    """

    name: str
    offset: float
    weight: float
    reason: str


class ThermalStatePredictor:
    """Predict future temperature based on thermal state trajectory.

    Uses historical state to project forward and make proactive decisions:
    - Cold spell pre-heating (6-12 hours before cold weather)
    - Morning warm-up (pre-heat before wake time)
    - Peak avoidance (heat during cheap periods before expensive peaks)
    - DHW coordination (pre-heat before DHW cycles)

    Adapts to Swedish climate with temperature-aware thresholds.
    """

    def __init__(self, lookback_hours: int = 24):
        """Initialize thermal state predictor.

        Args:
            lookback_hours: Hours of history to maintain (default 24)
        """
        self.lookback_hours = lookback_hours
        self.state_history: deque[ThermalSnapshot] = deque(
            maxlen=int(lookback_hours * SAMPLES_PER_HOUR) + 2
        )
        self._thermal_responsiveness: float | None = None

    def record_state(
        self,
        timestamp: datetime,
        indoor_temp: float,
        outdoor_temp: float,
        heating_offset: float,
        flow_temp: float,
        degree_minutes: float,
        indoor_measured: bool = True,
    ) -> None:
        """Record current thermal state.

        Args:
            timestamp: State timestamp
            indoor_temp: Current indoor temperature (°C)
            outdoor_temp: Current outdoor temperature (°C)
            heating_offset: Current heating curve offset (°C)
            flow_temp: Current flow temperature (°C)
            degree_minutes: Current thermal debt (DM/GM)
            indoor_measured: False when indoor_temp is a placeholder (no room sensor)
        """
        snapshot = ThermalSnapshot(
            timestamp=timestamp,
            indoor_temp=indoor_temp,
            outdoor_temp=outdoor_temp,
            heating_offset=heating_offset,
            flow_temp=flow_temp,
            degree_minutes=degree_minutes,
            indoor_measured=indoor_measured,
        )

        if self.state_history:
            previous = self.state_history[-1].timestamp.timestamp()
            if timestamp.timestamp() < previous:
                return
            if timestamp.timestamp() == previous:
                self.state_history.pop()
        self.state_history.append(snapshot)
        # Retain both endpoints and one cycle of scheduling jitter around the lookback window.
        cutoff = timestamp.timestamp() - self.lookback_hours * 3600 - UPDATE_INTERVAL_MINUTES * 60
        while self.state_history and self.state_history[0].timestamp.timestamp() < cutoff:
            self.state_history.popleft()

        _LOGGER.debug(
            "Recorded thermal state: indoor=%.1f°C, outdoor=%.1f°C, "
            "offset=%.1f°C, flow=%.1f°C, DM=%.0f",
            indoor_temp,
            outdoor_temp,
            heating_offset,
            flow_temp,
            degree_minutes,
        )

    def _measured_room(self) -> list[ThermalSnapshot]:
        """The snapshots whose room temperature was measured: the only ones a room figure may use.

        Without a room sensor the adapter shows DEFAULT_INDOOR_TEMP. Learned from as a reading, a
        constant 21 °C under a 22 °C target made the learned pre-heat vote +3.0 at weight 0.65 for a
        room nothing measured. The outdoor readings of the same snapshots are real, and stay.
        """
        if not self.state_history or not self.state_history[-1].indoor_measured:
            return []
        return self._recent_history(
            [snapshot for snapshot in self.state_history if snapshot.indoor_measured]
        )

    def _recent_history(self, snapshots: list[ThermalSnapshot]) -> list[ThermalSnapshot]:
        """Only the continuous suffix ending in a recent, non-future measurement."""
        if not snapshots:
            return []
        limit = PREDICTION_MAX_SAMPLE_GAP_MINUTES * 60
        age = dt_util.utcnow().timestamp() - snapshots[-1].timestamp.timestamp()
        if age < 0 or age > limit:
            return []
        start = len(snapshots) - 1
        while start > 0:
            elapsed = seconds_between(snapshots[start - 1].timestamp, snapshots[start].timestamp)
            if elapsed <= 0 or elapsed > limit:
                break
            start -= 1
        return snapshots[start:]

    def _has_history_for(self, room: list[ThermalSnapshot], hours: int) -> bool:
        """N intervals need N+1 measurements and the claimed real elapsed duration."""
        return (
            len(room) >= hours * SAMPLES_PER_HOUR + 1
            and seconds_between(room[0].timestamp, room[-1].timestamp) >= hours * 3600
        )

    def predict_temperature(
        self,
        hours_ahead: int,
        future_outdoor_temps: list[float],
        planned_offsets: list[float] | None = None,
        thermal_mass: float = 1.0,
        insulation_quality: float = 1.0,
        learned_params: LearnedThermalParameters | None = None,
        current_indoor_temp: float | None = None,
    ) -> TempPrediction:
        """Predict a conditional trajectory using independently validated horizon models.

        Native heating is included in each model's fitted intercept. Without a validated model,
        holding the measured room temperature is the explicit zero-confidence fallback. Manual
        mass/insulation settings cannot turn a heuristic into evidence of forecast skill.
        """
        room = self._measured_room()
        current_temp = (
            current_indoor_temp
            if current_indoor_temp is not None
            else (room[-1].indoor_temp if room else 21.0)
        )
        temperatures = [current_temp] * hours_ahead
        skills = [0.0] * hours_ahead
        model_is_recent = False
        if (
            learned_params is not None
            and learned_params.observations_end is not None
            and learned_params.observations_end.tzinfo is not None
        ):
            model_age = seconds_between(learned_params.observations_end, dt_util.utcnow())
            model_is_recent = (
                0
                <= model_age
                <= (LEARNING_OBSERVATION_INTERVAL_MINUTES + PREDICTION_MAX_SAMPLE_GAP_MINUTES) * 60
            )
        if self._has_history_for(room, PREDICTION_MIN_HISTORY_HOURS) and model_is_recent:
            current_offset = self.state_history[-1].heating_offset
            offsets = (
                planned_offsets if planned_offsets is not None else [current_offset] * hours_ahead
            )
            for hour in range(1, hours_ahead + 1):
                forecast = learned_params.forecasts.get(hour)
                if forecast is None or forecast.skill < LEARNING_CONFIDENCE_THRESHOLD:
                    continue
                predicted = forecast.predict(current_temp, future_outdoor_temps, offsets)
                if predicted is not None:
                    temperatures[hour - 1] = predicted
                    skills[hour - 1] = forecast.skill
        change = temperatures[-1] - current_temp if temperatures else 0.0
        trend = "stable"
        if change > PREDICTION_TREND_RISING_THRESHOLD:
            trend = "rising"
        elif change < PREDICTION_TREND_FALLING_THRESHOLD:
            trend = "falling"
        return TempPrediction(
            predicted_temps=temperatures,
            hours_ahead=hours_ahead,
            confidence=skills[-1] if skills else 0.0,
            trajectory_trend=trend,
            hourly_confidence=skills,
        )

    def should_pre_heat(
        self,
        target_temp: float,
        hours_ahead: int,
        future_outdoor_temps: list[float],
        current_outdoor_temp: float,
        current_indoor_temp: float,
        thermal_mass: float = 1.0,
        insulation_quality: float = 1.0,
        learned_params: LearnedThermalParameters | None = None,
    ) -> PreHeatDecision:
        """Determine if pre-heating is needed.

        Predicts if temperature will drop below comfort threshold and
        recommends proactive heating.

        Args:
            target_temp: Target indoor temperature (°C)
            hours_ahead: Hours to look ahead
            future_outdoor_temps: Forecasted outdoor temperatures
            current_outdoor_temp: Current outdoor temperature
            current_indoor_temp: Current indoor temperature (°C, from nibe_state)
            thermal_mass: Building thermal mass
            insulation_quality: Insulation quality
            learned_params: Learned thermal parameters from AdaptiveThermalModel

        Returns:
            Pre-heating decision with reasoning
        """
        # Predict temperature without additional heating (offset = 0)
        no_heat_prediction = self.predict_temperature(
            hours_ahead=hours_ahead,
            future_outdoor_temps=future_outdoor_temps,
            planned_offsets=[0.0] * hours_ahead,
            thermal_mass=thermal_mass,
            insulation_quality=insulation_quality,
            learned_params=learned_params,
            current_indoor_temp=current_indoor_temp,
        )

        eligible = [
            (hour, temp)
            for hour, (temp, skill) in enumerate(
                zip(no_heat_prediction.predicted_temps, no_heat_prediction.hourly_confidence),
                start=1,
            )
            if skill >= LEARNING_CONFIDENCE_THRESHOLD
        ]
        if no_heat_prediction.confidence < LEARNING_CONFIDENCE_THRESHOLD or not eligible:
            return PreHeatDecision(False, 0.0, "No validated room forecast", 0, target_temp)
        event_hour, min_predicted_temp = min(eligible, key=lambda pair: pair[1])
        temp_deficit = target_temp - min_predicted_temp
        # The forecast starts at the measured room; its existing warmth is already included.
        current_overshoot = max(0.0, current_indoor_temp - target_temp)
        error_bound = learned_params.forecasts[event_hour].max_validation_error
        effective_deficit = max(0.0, temp_deficit - error_bound)

        # Swedish climate-aware comfort thresholds
        # Colder weather = wider tolerance before pre-heating
        if current_outdoor_temp < PREDICTION_OUTDOOR_EXTREME_COLD:
            # Extreme cold: Allow larger deficit before pre-heating
            comfort_threshold = PREDICTION_COMFORT_THRESHOLD_EXTREME_COLD
        elif current_outdoor_temp < PREDICTION_OUTDOOR_COLD:
            # Cold: Allow moderate deficit
            comfort_threshold = PREDICTION_COMFORT_THRESHOLD_COLD
        else:
            # Mild: Strict deficit
            comfort_threshold = PREDICTION_COMFORT_THRESHOLD_MILD

        if effective_deficit > comfort_threshold:
            forecast = learned_params.forecasts[event_hour]
            if current_outdoor_temp < PREDICTION_OUTDOOR_EXTREME_COLD:
                maximum = PREDICTION_PREHEAT_MAX_EXTREME_COLD
            elif current_outdoor_temp < PREDICTION_OUTDOOR_COLD:
                maximum = PREDICTION_PREHEAT_MAX_COLD
            else:
                maximum = PREDICTION_PREHEAT_MAX_MILD
            recommended_offset = max(
                0.0,
                min(
                    effective_deficit / forecast.coefficients[-1], maximum, forecast.offset_range[1]
                ),
            )
            reason = f"validated room forecast {temp_deficit:.1f}°C under target"

            # Add strategic context if pre-heating when indoor already above target
            # This explains thermal storage strategy to user
            if current_overshoot > PREDICTION_OVERSHOOT_LOG_THRESHOLD:
                reason += f", {current_overshoot:.1f}°C stored"

            # Calculate hours until temperature drops below threshold
            hours_until_cold = next(
                (hour for hour, temp in eligible if temp < target_temp - comfort_threshold),
                event_hour,
            )

            # Calculate pre-heat target temperature
            # Higher thermal mass = building holds heat longer = LESS safety margin needed
            # thermal_mass 0.5 (light) → 1.0°C margin, 2.0 (heavy) → 0.25°C margin
            safety_margin = 0.5 / max(thermal_mass, 0.5)
            preheat_target = target_temp + safety_margin

            return PreHeatDecision(
                should_preheat=recommended_offset > 0.0,
                recommended_offset=recommended_offset,
                reason=reason,
                hours_until_event=hours_until_cold,
                target_temp=preheat_target,
            )
        else:
            # No pre-heating needed
            return PreHeatDecision(
                should_preheat=False,
                recommended_offset=0.0,
                reason=f"Temperature predicted to remain adequate (min {min_predicted_temp:.1f}°C)",
                hours_until_event=0,
                target_temp=target_temp,
            )

    def _weather_for_hours(self, weather_data, hours: int) -> list[float]:
        """Sample actual future instants; missing hourly coverage cannot become a trusted input.

        Forecast timestamps are absolute instants, including both occurrences of a repeated local
        hour. Linear interpolation is allowed between adjacent hourly points, never across a gap
        or beyond coverage. A naive, duplicate or non-finite point makes the trajectory ambiguous.
        """
        points = []
        for hour in weather_data.forecast_hours:
            if hour.datetime.tzinfo is None or not isfinite(hour.temperature):
                return []
            points.append((hour.datetime.timestamp(), hour.temperature))
        points.sort()
        times = [instant for instant, _ in points]
        if not times or any(right <= left for left, right in zip(times, times[1:])):
            return []
        now = dt_util.utcnow()
        maximum_gap = (
            LEARNING_OBSERVATION_INTERVAL_MINUTES + PREDICTION_MAX_SAMPLE_GAP_MINUTES
        ) * 60
        temperatures = []
        for h in range(1, hours + 1):
            instant = (now + timedelta(hours=h)).timestamp()
            index = bisect_left(times, instant)
            if index < len(times) and times[index] == instant:
                temperatures.append(points[index][1])
                continue
            if index == 0 or index == len(times):
                return []
            before, after = points[index - 1], points[index]
            elapsed = after[0] - before[0]
            if elapsed > maximum_gap:
                return []
            fraction = (instant - before[0]) / elapsed
            temperatures.append(before[1] + fraction * (after[1] - before[1]))
        return temperatures

    def evaluate_layer(
        self,
        nibe_state,
        weather_data,
        target_temp: float,
        thermal_model,
    ) -> PredictionLayerDecision:
        """Prediction layer: Learned pre-heating using thermal state predictor.

        Uses learned building thermal characteristics to make intelligent
        pre-heating decisions based on predicted temperature evolution.

        Each requested horizon needs validated conditional skill, recent measurements and
        complete timestamped weather coverage before contributing a bounded heating request.

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast data
            target_temp: Target indoor temperature
            thermal_model: ThermalModel instance for prediction horizon

        Returns:
            PredictionLayerDecision with learned pre-heating recommendation
        """
        # Skip until a full day of history exists. Derive the sample count from SAMPLES_PER_HOUR: a
        # hardcoded 96 is only EIGHT hours at the 5-minute tick, engaging the learned pre-heat on a
        # third of a day - and eight hours of a Swedish winter night is not a representative day.
        # Without a measured room there is nothing to predict from: the reading is a placeholder.
        if not getattr(nibe_state, "indoor_temp_valid", True):
            return PredictionLayerDecision(
                name="Learned Pre-heat",
                offset=0.0,
                weight=0.0,
                reason="No room sensor",
            )
        required = PREDICTION_LEARNED_PREHEAT_MIN_HOURS * SAMPLES_PER_HOUR + 1
        room = self._measured_room()
        measured = len(room)
        if not self._has_history_for(room, PREDICTION_LEARNED_PREHEAT_MIN_HOURS):
            return PredictionLayerDecision(
                name="Learned Pre-heat",
                offset=0.0,
                weight=0.0,
                reason=f"Learning: {measured}/{required} observations",
            )

        # Skip if no weather forecast available
        if not weather_data or not weather_data.forecast_hours:
            return PredictionLayerDecision(
                name="Learned Pre-heat",
                offset=0.0,
                weight=0.0,
                reason="No weather forecast",
            )

        try:
            # The horizon of the owner's thermal mass setting (prediction_horizon_for)
            prediction_horizon = int(thermal_model.get_prediction_horizon())
            # Half of the configured horizon, measured in real elapsed hours.
            hours_ahead = prediction_horizon // 2
            forecast_temps = self._weather_for_hours(weather_data, hours_ahead)

            if not forecast_temps:
                return PredictionLayerDecision(
                    name="Learned Pre-heat",
                    offset=0.0,
                    weight=0.0,
                    reason="Empty weather forecast",
                )

            # Get learned parameters from AdaptiveThermalModel if available
            learned_params = None
            if hasattr(thermal_model, "get_parameters"):
                learned_params = thermal_model.get_parameters()

            preheat_decision = self.should_pre_heat(
                target_temp=target_temp,
                hours_ahead=hours_ahead,
                future_outdoor_temps=forecast_temps,
                current_outdoor_temp=nibe_state.outdoor_temp,
                current_indoor_temp=nibe_state.indoor_temp,
                thermal_mass=thermal_model.thermal_mass,
                insulation_quality=thermal_model.insulation_quality,
                learned_params=learned_params,
            )

            if preheat_decision.should_preheat:
                # Thermal predictor now accounts for current overshoot as stored thermal energy
                # The recommended_offset is already adjusted for overshoot in should_pre_heat()
                # Weight 0.65 - slightly higher than base price layer (0.6)
                # but lower than effect/weather layers (0.7-0.8)
                return PredictionLayerDecision(
                    name="Learned Pre-heat",
                    offset=preheat_decision.recommended_offset,
                    weight=LAYER_WEIGHT_PREDICTION,
                    reason=preheat_decision.reason,
                )
            else:
                return PredictionLayerDecision(
                    name="Learned Pre-heat",
                    offset=0.0,
                    weight=0.0,
                    reason="No pre-heat needed",
                )

        except AttributeError as err:
            _LOGGER.error(
                "Thermal model API compatibility error: %s. "
                "This indicates the thermal model is missing required attributes. "
                "Check that AdaptiveThermalModel has insulation_quality property. "
                "Falling back to basic optimization without pre-heating.",
                err,
                exc_info=True,
            )
            return PredictionLayerDecision(
                name="Learned Pre-heat",
                offset=0.0,
                weight=0.0,
                reason="Thermal model API error",
            )
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as err:
            _LOGGER.warning("Prediction calculation failed: %s", err)
            return PredictionLayerDecision(
                name="Learned Pre-heat",
                offset=0.0,
                weight=0.0,
                reason=f"Calculation error: {err}",
            )

    def _calculate_prediction_confidence(
        self, snapshots: list[ThermalSnapshot] | None = None
    ) -> float:
        """Calculate confidence in predictions based on history quality.

        Args:
            snapshots: The history to judge; the measured room by default. The outdoor trend passes
                every snapshot, as it always did.

        Returns:
            Confidence 0-1
        """
        room = self._measured_room() if snapshots is None else self._recent_history(snapshots)
        if len(room) < 4:
            return 0.0

        # Base confidence on history length
        history_confidence = min(
            seconds_between(room[0].timestamp, room[-1].timestamp) / (24 * 3600), 1.0
        )

        # Data consistency (low variance in last hour = higher confidence)
        recent_temps = [s.indoor_temp for s in room[-SAMPLES_PER_HOUR:]]
        if len(recent_temps) > 4:
            temp_variance = np.std(recent_temps)
            # Low variance in reasonable range = good
            consistency = 1.0 - min(temp_variance / 2.0, 1.0)
        else:
            consistency = 0.5

        # Weighted combination
        return history_confidence * 0.6 + consistency * 0.4

    def get_current_trend(self) -> ThermalTrendDict:
        """Get current temperature trend analysis.

        Returns:
            Dictionary with trend information
        """
        room = self._measured_room()
        if len(room) < 8:
            return {
                "trend": "unknown",
                "rate_per_hour": 0.0,
                "confidence": 0.0,
                "samples": 0,
            }

        # Calculate temperature change over last 2 hours
        current = room[-1]
        two_hours_ago = (
            room[-(SAMPLES_PER_HOUR * 2)] if len(room) >= SAMPLES_PER_HOUR * 2 else room[0]
        )

        time_delta = seconds_between(two_hours_ago.timestamp, current.timestamp) / 3600
        temp_delta = current.indoor_temp - two_hours_ago.indoor_temp

        if time_delta > 0:
            rate = temp_delta / time_delta
        else:
            rate = 0.0

        if rate > 0.1:
            trend = "rising"
        elif rate < -0.1:
            trend = "falling"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "rate_per_hour": rate,
            "confidence": self._calculate_prediction_confidence(room),
            "samples": len(room),
        }

    def get_outdoor_trend(self) -> OutdoorTrendDict:
        """Get outdoor temperature trend from BT1 sensor history.

        Real-time outdoor trend can detect weather changes BEFORE forecast updates.
        Useful for proactive pre-heating when outdoor temp dropping rapidly.

        Returns:
            Dictionary with outdoor trend information
        """
        history = self._recent_history(list(self.state_history))
        if len(history) < 8:
            return {
                "trend": "unknown",
                "rate_per_hour": 0.0,
                "confidence": 0.0,
                "samples": 0,
                "temp_change_last_2h": 0.0,
            }

        # Calculate outdoor temperature change over last 2 hours
        current = history[-1]
        two_hours_ago = (
            history[-(SAMPLES_PER_HOUR * 2)] if len(history) >= SAMPLES_PER_HOUR * 2 else history[0]
        )

        time_delta = seconds_between(two_hours_ago.timestamp, current.timestamp) / 3600
        outdoor_delta = current.outdoor_temp - two_hours_ago.outdoor_temp

        if time_delta > 0:
            rate = outdoor_delta / time_delta
        else:
            rate = 0.0

        # Classify trend (outdoor changes faster than indoor)
        if rate > 0.5:
            trend = "warming_rapidly"
        elif rate > 0.2:
            trend = "warming"
        elif rate < -0.5:
            trend = "cooling_rapidly"
        elif rate < -0.2:
            trend = "cooling"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "rate_per_hour": rate,
            "confidence": self._calculate_prediction_confidence(list(history)),
            "samples": len(history),
            "temp_change_last_2h": outdoor_delta,
        }

    def to_dict(self) -> PredictorStateDict:
        """Serialize predictor to dictionary for persistence.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "lookback_hours": self.lookback_hours,
            "state_history": [
                {
                    "timestamp": snapshot.timestamp.isoformat(),
                    "indoor_temp": snapshot.indoor_temp,
                    "outdoor_temp": snapshot.outdoor_temp,
                    "heating_offset": snapshot.heating_offset,
                    "flow_temp": snapshot.flow_temp,
                    "degree_minutes": snapshot.degree_minutes,
                    "indoor_measured": snapshot.indoor_measured,
                }
                for snapshot in self.state_history
            ],
            "thermal_responsiveness": self._thermal_responsiveness,
        }

    @classmethod
    def from_dict(cls, data: PredictorStateDict) -> "ThermalStatePredictor":
        """Deserialize predictor from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Restored ThermalStatePredictor instance
        """
        predictor = cls(lookback_hours=data.get("lookback_hours", 24))

        # Restore state history
        for snapshot_data in sorted(
            data.get("state_history", []),
            key=lambda row: datetime.fromisoformat(row["timestamp"]).timestamp(),
        ):
            snapshot = ThermalSnapshot(
                timestamp=datetime.fromisoformat(snapshot_data["timestamp"]),
                indoor_temp=snapshot_data["indoor_temp"],
                outdoor_temp=snapshot_data["outdoor_temp"],
                heating_offset=snapshot_data["heating_offset"],
                flow_temp=snapshot_data["flow_temp"],
                degree_minutes=snapshot_data["degree_minutes"],
                # Stored before the flag existed: read as measured, as it was. A placeholder among
                # them leaves the 24-hour history within a day.
                indoor_measured=snapshot_data.get("indoor_measured", True),
            )
            predictor.record_state(
                snapshot.timestamp,
                snapshot.indoor_temp,
                snapshot.outdoor_temp,
                snapshot.heating_offset,
                snapshot.flow_temp,
                snapshot.degree_minutes,
                indoor_measured=snapshot.indoor_measured,
            )

        # Restore cached thermal responsiveness
        predictor._thermal_responsiveness = data.get("thermal_responsiveness")

        return predictor

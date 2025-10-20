"""Thermal state prediction for proactive heating control.

Predicts future indoor temperature and makes pre-heating decisions:
- Tracks 24-hour thermal state history
- Projects temperature trajectory hours ahead
- Identifies need for pre-heating
- Swedish climate-aware thresholds

Based on POST_PHASE_5_ROADMAP.md Phase 6.2 and Forum_Summary.md research.
"""

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from ..const import DM_THRESHOLD_ABSOLUTE_MAX
from .learning_types import PreHeatDecision, TempPrediction, ThermalSnapshot

_LOGGER = logging.getLogger(__name__)


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
            maxlen=int(lookback_hours * 4)  # 15-minute intervals
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
    ) -> None:
        """Record current thermal state.

        Args:
            timestamp: State timestamp
            indoor_temp: Current indoor temperature (°C)
            outdoor_temp: Current outdoor temperature (°C)
            heating_offset: Current heating curve offset (°C)
            flow_temp: Current flow temperature (°C)
            degree_minutes: Current thermal debt (DM/GM)
        """
        snapshot = ThermalSnapshot(
            timestamp=timestamp,
            indoor_temp=indoor_temp,
            outdoor_temp=outdoor_temp,
            heating_offset=heating_offset,
            flow_temp=flow_temp,
            degree_minutes=degree_minutes,
        )

        self.state_history.append(snapshot)

        _LOGGER.debug(
            "Recorded thermal state: indoor=%.1f°C, outdoor=%.1f°C, "
            "offset=%.1f°C, flow=%.1f°C, DM=%.0f",
            indoor_temp,
            outdoor_temp,
            heating_offset,
            flow_temp,
            degree_minutes,
        )

    def predict_temperature(
        self,
        hours_ahead: int,
        future_outdoor_temps: list[float],
        planned_offsets: list[float] | None = None,
        thermal_mass: float = 1.0,
        insulation_quality: float = 1.0,
    ) -> TempPrediction:
        """Predict indoor temperature N hours ahead.

        Args:
            hours_ahead: Number of hours to predict forward
            future_outdoor_temps: Forecasted outdoor temps (one per hour)
            planned_offsets: Planned heating offsets (one per hour), or None for current
            thermal_mass: Building thermal mass (learned or configured)
            insulation_quality: Insulation quality (learned or configured)

        Returns:
            Temperature prediction with confidence
        """
        if len(self.state_history) < 4:  # Need at least 1 hour of history
            # Insufficient data - return simple projection
            current_temp = self.state_history[-1].indoor_temp if self.state_history else 21.0
            return TempPrediction(
                predicted_temps=[current_temp] * hours_ahead,
                hours_ahead=hours_ahead,
                confidence=0.0,
                trajectory_trend="stable",
            )

        # Get current state
        current = self.state_history[-1]
        current_temp = current.indoor_temp
        current_outdoor = current.outdoor_temp
        current_offset = current.heating_offset

        # Calculate thermal responsiveness if not cached
        if self._thermal_responsiveness is None:
            self._thermal_responsiveness = self._calculate_thermal_responsiveness()

        # Use planned offsets or assume current offset continues
        if planned_offsets is None:
            planned_offsets = [current_offset] * hours_ahead

        # Predict trajectory
        predicted_temps = []
        temp = current_temp

        for hour in range(hours_ahead):
            outdoor_temp = (
                future_outdoor_temps[hour] if hour < len(future_outdoor_temps) else current_outdoor
            )
            offset = planned_offsets[hour] if hour < len(planned_offsets) else current_offset

            # Calculate heat loss rate (natural cooling)
            temp_diff = temp - outdoor_temp
            heat_loss_rate = temp_diff / (insulation_quality * 10)  # °C per hour

            # Calculate heat input from heating system
            heat_input_rate = offset * self._thermal_responsiveness

            # Net temperature change (higher thermal mass = slower change)
            dt_per_hour = (heat_input_rate - heat_loss_rate) / thermal_mass

            # Update temperature
            temp = temp + dt_per_hour
            predicted_temps.append(temp)

        # Calculate confidence based on history quality
        confidence = self._calculate_prediction_confidence()

        # Determine trajectory trend
        if len(predicted_temps) >= 2:
            temp_delta = predicted_temps[-1] - predicted_temps[0]
            if temp_delta > 0.3:
                trend = "rising"
            elif temp_delta < -0.3:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return TempPrediction(
            predicted_temps=predicted_temps,
            hours_ahead=hours_ahead,
            confidence=confidence,
            trajectory_trend=trend,
        )

    def should_pre_heat(
        self,
        target_temp: float,
        hours_ahead: int,
        future_outdoor_temps: list[float],
        current_outdoor_temp: float,
        thermal_mass: float = 1.0,
        insulation_quality: float = 1.0,
    ) -> PreHeatDecision:
        """Determine if pre-heating is needed.

        Predicts if temperature will drop below comfort threshold and
        recommends proactive heating.

        Args:
            target_temp: Target indoor temperature (°C)
            hours_ahead: Hours to look ahead
            future_outdoor_temps: Forecasted outdoor temperatures
            current_outdoor_temp: Current outdoor temperature
            thermal_mass: Building thermal mass
            insulation_quality: Insulation quality

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
        )

        # Check minimum predicted temperature
        min_predicted_temp = min(no_heat_prediction.predicted_temps)
        temp_deficit = target_temp - min_predicted_temp

        # Swedish climate-aware comfort thresholds
        # Colder weather = wider tolerance before pre-heating
        if current_outdoor_temp < -15:
            # Extreme cold: Allow 1.0°C deficit before pre-heating
            comfort_threshold = 1.0
        elif current_outdoor_temp < -5:
            # Cold: Allow 0.7°C deficit
            comfort_threshold = 0.7
        else:
            # Mild: Strict 0.5°C deficit
            comfort_threshold = 0.5

        if temp_deficit > comfort_threshold:
            # Pre-heating recommended
            # Calculate required offset to maintain comfort
            # More aggressive in extreme cold to compensate for thermal debt risk
            if current_outdoor_temp < -15:
                # Extreme cold: Conservative pre-heating (risk of DM -1500)
                recommended_offset = min(temp_deficit * 1.2, 2.0)
                reason = f"Extreme cold pre-heating: predicted drop {temp_deficit:.1f}°C"
            elif current_outdoor_temp < -5:
                # Cold: Moderate pre-heating
                recommended_offset = min(temp_deficit * 1.5, 2.5)
                reason = f"Cold spell pre-heating: predicted drop {temp_deficit:.1f}°C"
            else:
                # Mild: Normal pre-heating
                recommended_offset = min(temp_deficit * 2.0, 3.0)
                reason = f"Pre-heating: predicted drop {temp_deficit:.1f}°C"

            # Calculate hours until temperature drops below threshold
            hours_until_cold = 0
            for i, temp in enumerate(no_heat_prediction.predicted_temps):
                if temp < target_temp - comfort_threshold:
                    hours_until_cold = i + 1
                    break

            # Calculate pre-heat target temperature
            # Add safety margin proportional to thermal mass
            safety_margin = 1.0 / thermal_mass  # 0.5-2.0°C range
            preheat_target = target_temp + safety_margin

            return PreHeatDecision(
                should_preheat=True,
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

    def _calculate_thermal_responsiveness(self) -> float:
        """Calculate how responsive system is to offset changes.

        Based on historical temperature response to heating changes.

        Returns:
            Responsiveness factor (°C per offset per hour)
        """
        if len(self.state_history) < 8:  # Need 2+ hours
            return 0.5  # Default

        # Analyze temperature changes relative to offset
        responsiveness_samples = []

        for i in range(4, len(self.state_history)):
            current = self.state_history[i]
            prev = self.state_history[i - 4]  # 1 hour ago

            time_delta = (current.timestamp - prev.timestamp).total_seconds() / 3600

            if time_delta > 0 and current.heating_offset > 0.5:
                temp_change = current.indoor_temp - prev.indoor_temp
                responsiveness = temp_change / (time_delta * current.heating_offset)

                # Sanity check
                if 0 < responsiveness < 1.0:
                    responsiveness_samples.append(responsiveness)

        if responsiveness_samples:
            return np.median(responsiveness_samples)
        else:
            return 0.5  # Default

    def _calculate_prediction_confidence(self) -> float:
        """Calculate confidence in predictions based on history quality.

        Returns:
            Confidence 0-1
        """
        if len(self.state_history) < 4:
            return 0.0

        # Base confidence on history length
        history_confidence = min(len(self.state_history) / (24 * 4), 1.0)

        # Data consistency (low variance in trends = higher confidence)
        recent_temps = [s.indoor_temp for s in list(self.state_history)[-12:]]
        if len(recent_temps) > 4:
            temp_variance = np.std(recent_temps)
            # Low variance in reasonable range = good
            consistency = 1.0 - min(temp_variance / 2.0, 1.0)
        else:
            consistency = 0.5

        # Weighted combination
        return history_confidence * 0.6 + consistency * 0.4

    def get_current_trend(self) -> dict[str, Any]:
        """Get current temperature trend analysis.

        Returns:
            Dictionary with trend information
        """
        if len(self.state_history) < 8:
            return {
                "trend": "unknown",
                "rate_per_hour": 0.0,
                "confidence": 0.0,
            }

        # Calculate temperature change over last 2 hours
        current = self.state_history[-1]
        two_hours_ago = (
            self.state_history[-8] if len(self.state_history) >= 8 else self.state_history[0]
        )

        time_delta = (current.timestamp - two_hours_ago.timestamp).total_seconds() / 3600
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
            "confidence": self._calculate_prediction_confidence(),
            "samples": len(self.state_history),
        }

    def get_outdoor_trend(self) -> dict[str, Any]:
        """Get outdoor temperature trend from BT1 sensor history.

        Real-time outdoor trend can detect weather changes BEFORE forecast updates.
        Useful for proactive pre-heating when outdoor temp dropping rapidly.

        Returns:
            Dictionary with outdoor trend information
        """
        if len(self.state_history) < 8:
            return {
                "trend": "unknown",
                "rate_per_hour": 0.0,
                "confidence": 0.0,
            }

        # Calculate outdoor temperature change over last 2 hours
        current = self.state_history[-1]
        two_hours_ago = (
            self.state_history[-8] if len(self.state_history) >= 8 else self.state_history[0]
        )

        time_delta = (current.timestamp - two_hours_ago.timestamp).total_seconds() / 3600
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
            "confidence": self._calculate_prediction_confidence(),
            "samples": len(self.state_history),
            "temp_change_2h": outdoor_delta,
        }

    def to_dict(self) -> dict[str, Any]:
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
                }
                for snapshot in self.state_history
            ],
            "thermal_responsiveness": self._thermal_responsiveness,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThermalStatePredictor":
        """Deserialize predictor from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Restored ThermalStatePredictor instance
        """
        predictor = cls(lookback_hours=data.get("lookback_hours", 24))

        # Restore state history
        for snapshot_data in data.get("state_history", []):
            snapshot = ThermalSnapshot(
                timestamp=datetime.fromisoformat(snapshot_data["timestamp"]),
                indoor_temp=snapshot_data["indoor_temp"],
                outdoor_temp=snapshot_data["outdoor_temp"],
                heating_offset=snapshot_data["heating_offset"],
                flow_temp=snapshot_data["flow_temp"],
                degree_minutes=snapshot_data["degree_minutes"],
            )
            predictor.state_history.append(snapshot)

        # Restore cached thermal responsiveness
        predictor._thermal_responsiveness = data.get("thermal_responsiveness")

        return predictor

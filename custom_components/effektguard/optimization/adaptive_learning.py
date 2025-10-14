"""Adaptive thermal learning for building characteristics.

Automatically learns building thermal characteristics without manual configuration:
- Thermal mass (heat capacity)
- Heat loss coefficient (insulation quality)
- Heating efficiency (temperature rise per offset)
- UFH type detection (concrete slab, timber, radiator)
- Thermal decay rate (natural cooling speed)

Based on POST_PHASE_5_ROADMAP.md Phase 6.1 and Swedish NIBE research.
"""

import logging
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

from ..const import (
    LEARNING_CONFIDENCE_THRESHOLD,
    LEARNING_MIN_OBSERVATIONS,
    LEARNING_OBSERVATION_WINDOW,
    UFH_CONCRETE_PREDICTION_HORIZON,
    UFH_RADIATOR_PREDICTION_HORIZON,
    UFH_TIMBER_PREDICTION_HORIZON,
    UFHType,
)
from .learning_types import AdaptiveThermalObservation, LearnedThermalParameters

_LOGGER = logging.getLogger(__name__)


class AdaptiveThermalModel:
    """Self-learning thermal model that adapts to building characteristics.

    Learns from real system behavior without manual configuration:
    - Observes thermal response to heating changes
    - Detects UFH type from lag time
    - Calculates thermal mass from heat retention
    - Measures insulation quality from heat loss rate
    - Builds confidence over time

    Swedish climate awareness:
    - Handles -30°C to +5°C range
    - Validates against DM -1500 absolute maximum
    - Adapts to Nordic thermal dynamics
    """

    def __init__(self, initial_thermal_mass: float = 1.0):
        """Initialize adaptive thermal model.

        Args:
            initial_thermal_mass: Starting thermal mass estimate (1.0 = normal)
        """
        self.thermal_mass = initial_thermal_mass
        self.observations: deque[AdaptiveThermalObservation] = deque(
            maxlen=LEARNING_OBSERVATION_WINDOW
        )
        self.learned_parameters: dict[str, float] = {}
        self.ufh_type: UFHType = UFHType.UNKNOWN
        self._last_update: datetime | None = None

    def record_observation(
        self,
        timestamp: datetime,
        indoor_temp: float,
        outdoor_temp: float,
        heating_offset: float,
    ) -> None:
        """Record thermal response observation for learning.

        Args:
            timestamp: Observation timestamp
            indoor_temp: Current indoor temperature (°C)
            outdoor_temp: Current outdoor temperature (°C)
            heating_offset: Current heating curve offset (°C)
        """
        # Calculate change from previous observation
        temp_change = 0.0
        time_delta_hours = 0.0

        if len(self.observations) > 0:
            prev = self.observations[-1]
            temp_change = indoor_temp - prev.indoor_temp
            time_delta_hours = (timestamp - prev.timestamp).total_seconds() / 3600

        observation = AdaptiveThermalObservation(
            timestamp=timestamp,
            indoor_temp=indoor_temp,
            outdoor_temp=outdoor_temp,
            heating_offset=heating_offset,
            temp_change=temp_change,
            time_delta_hours=time_delta_hours,
        )

        self.observations.append(observation)

        _LOGGER.debug(
            "Recorded observation: indoor=%.1f°C, outdoor=%.1f°C, "
            "offset=%.1f°C, change=%.2f°C, Δt=%.2fh",
            indoor_temp,
            outdoor_temp,
            heating_offset,
            temp_change,
            time_delta_hours,
        )

    def update_learned_parameters(self) -> LearnedThermalParameters | None:
        """Calculate learned parameters from observations.

        Runs analysis on accumulated observations to extract:
        - Thermal mass from temperature response
        - Heat loss coefficient from cooling rate
        - Heating efficiency from heating rate
        - UFH type from lag analysis
        - Thermal decay rate

        Returns:
            Learned parameters with confidence score, or None if insufficient data
        """
        if len(self.observations) < LEARNING_MIN_OBSERVATIONS:
            _LOGGER.debug(
                "Insufficient observations for learning: %d < %d",
                len(self.observations),
                LEARNING_MIN_OBSERVATIONS,
            )
            return None

        try:
            # Detect UFH type first (affects other calculations)
            ufh_type = self._detect_ufh_type()

            # Calculate thermal characteristics
            thermal_mass = self._calculate_thermal_mass()
            heat_loss_coef = self._calculate_heat_loss_coefficient()
            heating_efficiency = self._calculate_heating_efficiency()
            thermal_decay_rate = self._calculate_thermal_decay_rate()

            # Calculate confidence based on data quality
            confidence = self._calculate_confidence()

            learned = LearnedThermalParameters(
                thermal_mass=thermal_mass,
                heat_loss_coefficient=heat_loss_coef,
                heating_efficiency=heating_efficiency,
                thermal_decay_rate=thermal_decay_rate,
                ufh_type=ufh_type,
                last_updated=datetime.now(),
                confidence=confidence,
                observation_count=len(self.observations),
            )

            self.ufh_type = ufh_type
            self.thermal_mass = thermal_mass
            self._last_update = datetime.now()

            _LOGGER.info(
                "Updated learned parameters: thermal_mass=%.2f, "
                "heat_loss=%.1fW/°C, efficiency=%.2f°C/offset/h, "
                "decay=%.3f°C/h, ufh=%s, confidence=%.1f%%",
                thermal_mass,
                heat_loss_coef,
                heating_efficiency,
                thermal_decay_rate,
                ufh_type,
                confidence * 100,
            )

            return learned

        except Exception as e:
            _LOGGER.error("Failed to update learned parameters: %s", e, exc_info=True)
            return None

    def _detect_ufh_type(self) -> UFHType:
        """Detect UFH type from thermal response lag.

        Analyzes step response to offset changes:
        - 6+ hour lag = concrete slab
        - 2-3 hour lag = timber
        - <1 hour lag = radiators

        Based on Forum_Summary.md and Floor_Heating_Enhancements.md research.

        Returns:
            Detected UFH type
        """
        if len(self.observations) < LEARNING_MIN_OBSERVATIONS * 2:
            return UFHType.UNKNOWN

        # Find significant offset changes (>1.0°C)
        offset_changes = []
        for i in range(1, len(self.observations)):
            prev = self.observations[i - 1]
            curr = self.observations[i]
            offset_delta = abs(curr.heating_offset - prev.heating_offset)

            if offset_delta > 1.0:
                # Track temperature response over next several hours
                response_lag = self._measure_response_lag(i)
                if response_lag is not None:
                    offset_changes.append(response_lag)

        if not offset_changes:
            return UFHType.UNKNOWN

        # Calculate median lag time
        median_lag = np.median(offset_changes)

        # Classify based on research thresholds
        if median_lag >= 6.0:
            return UFHType.CONCRETE_SLAB
        elif median_lag >= 2.0:
            return UFHType.TIMBER
        elif median_lag >= 0.5:
            return UFHType.RADIATOR
        else:
            return UFHType.RADIATOR  # Very fast response = radiators

    def _measure_response_lag(self, change_index: int) -> float | None:
        """Measure thermal lag time after offset change.

        Args:
            change_index: Index where offset change occurred

        Returns:
            Lag time in hours, or None if cannot measure
        """
        if change_index + 12 >= len(self.observations):  # Need 3+ hours after
            return None

        baseline_temp = self.observations[change_index].indoor_temp
        offset_direction = (
            self.observations[change_index].heating_offset
            - self.observations[change_index - 1].heating_offset
        )

        # Find when temperature starts responding (>0.1°C change in expected direction)
        cumulative_time = 0.0
        for i in range(change_index + 1, min(change_index + 48, len(self.observations))):
            obs = self.observations[i]
            temp_change = obs.indoor_temp - baseline_temp

            cumulative_time += obs.time_delta_hours

            # Check if responding in expected direction
            if offset_direction > 0 and temp_change > 0.1:
                return cumulative_time
            elif offset_direction < 0 and temp_change < -0.1:
                return cumulative_time

            # Maximum lag detection window (12 hours)
            if cumulative_time > 12.0:
                break

        return None

    def _calculate_thermal_mass(self) -> float:
        """Calculate building thermal mass from temperature response.

        Higher thermal mass = slower temperature changes for same heating input.

        Returns:
            Thermal mass relative to normal (1.0 = normal, 0.5-2.0 range)
        """
        if len(self.observations) < LEARNING_MIN_OBSERVATIONS:
            return 1.0

        # Analyze heating periods (positive offset)
        heating_rates = []
        for obs in self.observations:
            if obs.heating_offset > 0.5 and obs.time_delta_hours > 0:
                # Temperature change per hour per offset
                rate = obs.temp_change / (obs.time_delta_hours * obs.heating_offset)
                if abs(rate) < 1.0:  # Sanity check
                    heating_rates.append(rate)

        if not heating_rates:
            return 1.0

        # Average heating rate
        avg_rate = np.mean(heating_rates)

        # Thermal mass inversely proportional to heating rate
        # Baseline: 0.5°C/h per 1.0 offset = thermal mass 1.0
        baseline_rate = 0.5
        thermal_mass = baseline_rate / max(abs(avg_rate), 0.1)

        # Clamp to reasonable range
        return np.clip(thermal_mass, 0.5, 2.0)

    def _calculate_heat_loss_coefficient(self) -> float:
        """Calculate heat loss coefficient (insulation quality).

        Measured in W/°C - heat loss per degree temperature difference.

        Returns:
            Heat loss coefficient (typical range 100-300 W/°C)
        """
        if len(self.observations) < LEARNING_MIN_OBSERVATIONS:
            return 180.0  # Default typical value

        # Analyze cooling periods (low/no offset)
        cooling_rates = []
        for obs in self.observations:
            if obs.heating_offset < 0.5 and obs.time_delta_hours > 0:
                temp_diff = obs.indoor_temp - obs.outdoor_temp
                if temp_diff > 0:  # Indoor warmer than outdoor
                    # Cooling rate per temperature difference
                    rate = abs(obs.temp_change) / (obs.time_delta_hours * temp_diff)
                    if 0.001 < rate < 0.1:  # Sanity check
                        cooling_rates.append((temp_diff, abs(obs.temp_change)))

        if not cooling_rates:
            return 180.0

        # Estimate heat loss coefficient
        # Q = U × A × ΔT, where Q is heat loss rate
        # For typical house, convert cooling rate to W/°C
        avg_temp_diff = np.mean([td for td, _ in cooling_rates])
        avg_cooling = np.mean([cool for _, cool in cooling_rates])

        # Rough conversion: cooling rate to heat loss coefficient
        # This is simplified - real calculation would need house volume
        heat_loss_coef = (avg_cooling / max(avg_temp_diff, 1.0)) * 3600 * 50

        # Clamp to reasonable range
        return np.clip(heat_loss_coef, 100.0, 300.0)

    def _calculate_heating_efficiency(self) -> float:
        """Calculate heating efficiency (°C per offset per hour).

        Returns:
            Heating efficiency (typical range 0.2-0.6 °C/offset/hour)
        """
        if len(self.observations) < LEARNING_MIN_OBSERVATIONS:
            return 0.42  # Default from roadmap

        heating_efficiencies = []
        for obs in self.observations:
            if obs.heating_offset > 0.5 and obs.time_delta_hours > 0:
                efficiency = obs.temp_change / (obs.time_delta_hours * obs.heating_offset)
                if 0 < efficiency < 1.0:  # Sanity check
                    heating_efficiencies.append(efficiency)

        if not heating_efficiencies:
            return 0.42

        return np.clip(np.mean(heating_efficiencies), 0.2, 0.6)

    def _calculate_thermal_decay_rate(self) -> float:
        """Calculate thermal decay rate (natural cooling speed).

        Measured in °C/hour when heating is off.

        Returns:
            Thermal decay rate (typically -0.05 to -0.15 °C/hour)
        """
        if len(self.observations) < LEARNING_MIN_OBSERVATIONS:
            return -0.08  # Default from roadmap

        decay_rates = []
        for obs in self.observations:
            if obs.heating_offset < 0.5 and obs.time_delta_hours > 0:
                decay_rate = obs.temp_change / obs.time_delta_hours
                if -0.5 < decay_rate < 0:  # Sanity check (should be negative)
                    decay_rates.append(decay_rate)

        if not decay_rates:
            return -0.08

        return np.clip(np.mean(decay_rates), -0.2, -0.02)

    def _calculate_confidence(self) -> float:
        """Calculate confidence in learned parameters.

        Based on:
        - Number of observations
        - Data consistency (low variance)
        - Time span covered

        Returns:
            Confidence score 0-1
        """
        if len(self.observations) < LEARNING_MIN_OBSERVATIONS:
            return 0.0

        # Base confidence from observation count
        obs_confidence = min(len(self.observations) / LEARNING_OBSERVATION_WINDOW, 1.0)

        # Data consistency (low variance in rates = higher confidence)
        heating_rates = []
        for obs in self.observations:
            if obs.heating_offset > 0.5 and obs.time_delta_hours > 0:
                rate = obs.temp_change / obs.time_delta_hours
                heating_rates.append(rate)

        if len(heating_rates) > 10:
            consistency = 1.0 - min(np.std(heating_rates) / max(np.mean(heating_rates), 0.1), 1.0)
        else:
            consistency = 0.5

        # Time span (prefer observations over longer period)
        if len(self.observations) > 1:
            time_span_hours = (
                self.observations[-1].timestamp - self.observations[0].timestamp
            ).total_seconds() / 3600
            time_confidence = min(time_span_hours / (24 * 7), 1.0)  # 1 week = full confidence
        else:
            time_confidence = 0.0

        # Weighted combination
        confidence = obs_confidence * 0.4 + consistency * 0.4 + time_confidence * 0.2

        return confidence

    def get_prediction_horizon(self) -> float:
        """Get appropriate prediction horizon based on detected UFH type.

        Returns:
            Prediction horizon in hours
        """
        if self.ufh_type == UFHType.CONCRETE_SLAB:
            return UFH_CONCRETE_PREDICTION_HORIZON
        elif self.ufh_type == UFHType.TIMBER:
            return UFH_TIMBER_PREDICTION_HORIZON
        elif self.ufh_type == UFHType.RADIATOR:
            return UFH_RADIATOR_PREDICTION_HORIZON
        else:
            # Unknown - use moderate horizon
            return UFH_TIMBER_PREDICTION_HORIZON

    def should_use_learned_parameters(self) -> bool:
        """Check if learned parameters are reliable enough to use.

        Returns:
            True if confidence exceeds threshold
        """
        if not self.learned_parameters:
            return False

        confidence = self.learned_parameters.get("confidence", 0.0)
        return confidence >= LEARNING_CONFIDENCE_THRESHOLD

    def get_thermal_characteristics(self) -> dict[str, Any]:
        """Get current thermal model characteristics.

        Returns:
            Dictionary with thermal parameters and learning status
        """
        return {
            "thermal_mass": self.thermal_mass,
            "ufh_type": self.ufh_type,
            "observations": len(self.observations),
            "learned_parameters": self.learned_parameters,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "prediction_horizon": self.get_prediction_horizon(),
        }

    def get_parameters(self) -> LearnedThermalParameters | None:
        """Get current learned parameters.

        Alias for update_learned_parameters() for API compatibility.

        Returns:
            Current learned parameters or None if insufficient data
        """
        return self.update_learned_parameters()

    def to_dict(self) -> dict[str, Any]:
        """Serialize model state to dictionary for storage.

        Returns:
            Dictionary with all model state
        """
        # Get current learned parameters
        params = self.update_learned_parameters()

        return {
            "thermal_mass": self.thermal_mass,
            "ufh_type": (
                str(self.ufh_type.value)
                if isinstance(self.ufh_type, UFHType)
                else str(self.ufh_type)
            ),
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "observation_count": len(self.observations),
            "learned_parameters": (
                {
                    "thermal_mass": params.thermal_mass if params else self.thermal_mass,
                    "heat_loss_coefficient": params.heat_loss_coefficient if params else 180.0,
                    "heating_efficiency": params.heating_efficiency if params else 0.42,
                    "thermal_decay_rate": params.thermal_decay_rate if params else -0.08,
                    "confidence": params.confidence if params else 0.0,
                }
                if params
                else None
            ),
            "observations": [
                {
                    "timestamp": obs.timestamp.isoformat(),
                    "indoor_temp": obs.indoor_temp,
                    "outdoor_temp": obs.outdoor_temp,
                    "heating_offset": obs.heating_offset,
                    "temp_change": obs.temp_change,
                    "time_delta_hours": obs.time_delta_hours,
                }
                for obs in list(self.observations)[-100:]  # Keep last 100 for storage
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdaptiveThermalModel":
        """Restore model state from dictionary.

        Args:
            data: Serialized model state

        Returns:
            Restored AdaptiveThermalModel instance
        """
        model = cls(initial_thermal_mass=data.get("thermal_mass", 1.0))

        # Restore UFH type
        ufh_type_str = data.get("ufh_type", "unknown")
        try:
            model.ufh_type = UFHType(ufh_type_str)
        except ValueError:
            model.ufh_type = UFHType.UNKNOWN

        # Restore last update time
        if data.get("last_update"):
            try:
                model._last_update = datetime.fromisoformat(data["last_update"])
            except (ValueError, TypeError):
                model._last_update = None

        # Restore learned parameters
        if data.get("learned_parameters"):
            model.learned_parameters = data["learned_parameters"]

        # Restore observations (limited set)
        if data.get("observations"):
            for obs_data in data["observations"]:
                try:
                    obs = AdaptiveThermalObservation(
                        timestamp=datetime.fromisoformat(obs_data["timestamp"]),
                        indoor_temp=obs_data["indoor_temp"],
                        outdoor_temp=obs_data["outdoor_temp"],
                        heating_offset=obs_data["heating_offset"],
                        temp_change=obs_data.get("temp_change", 0.0),
                        time_delta_hours=obs_data.get("time_delta_hours", 0.0),
                    )
                    model.observations.append(obs)
                except (KeyError, ValueError, TypeError) as e:
                    _LOGGER.warning("Failed to restore observation: %s", e)
                    continue

        return model

"""Thermal model for building thermal behavior prediction.

Models heat storage and loss characteristics for predictive control.
Enables pre-heating and thermal energy banking strategies.
"""

import logging
from dataclasses import dataclass
from typing import Any

_LOGGER = logging.getLogger(__name__)


@dataclass
class ThermalObservation:
    """Observation for thermal model learning."""

    timestamp: float
    indoor_temp: float
    outdoor_temp: float
    heating_active: bool
    heating_offset: float


class ThermalModel:
    """Model building thermal characteristics for predictive control.

    Implements simplified thermal dynamics:
    - Heat capacity (thermal mass)
    - Heat loss (insulation quality)
    - Temperature trajectory prediction
    - Pre-heating target calculation
    """

    def __init__(
        self,
        thermal_mass: float = 1.0,  # Relative scale 0.5-2.0
        insulation_quality: float = 1.0,  # Relative scale 0.5-2.0
    ):
        """Initialize thermal model.

        Args:
            thermal_mass: Relative thermal mass (1.0 = normal)
                - 0.5 = low mass (timber frame)
                - 1.0 = normal mass (mixed construction)
                - 2.0 = high mass (concrete/masonry)
            insulation_quality: Relative insulation (1.0 = normal)
                - 0.5 = poor insulation
                - 1.0 = standard insulation
                - 2.0 = excellent insulation
        """
        self.thermal_mass = thermal_mass
        self.insulation_quality = insulation_quality
        self._learning_data: list[ThermalObservation] = []

    def predict_temperature_trajectory(
        self,
        current_temp: float,
        outdoor_temp: float,
        heating_offset: float,
        hours_ahead: int = 3,
    ) -> list[float]:
        """Predict indoor temperature trajectory.

        Simplified thermal model:
        dT/dt = (heat_input - heat_loss) / thermal_mass
        heat_loss = (indoor - outdoor) / insulation

        Args:
            current_temp: Current indoor temperature (°C)
            outdoor_temp: Current outdoor temperature (°C)
            heating_offset: Heating curve offset (°C)
            hours_ahead: Number of hours to predict

        Returns:
            List of predicted temperatures for next N hours
        """
        trajectory = [current_temp]
        temp = current_temp

        for hour in range(hours_ahead):
            # Calculate heat loss rate
            # Higher temp difference = more loss
            # Better insulation = less loss
            temp_diff = temp - outdoor_temp
            heat_loss_rate = temp_diff / (self.insulation_quality * 10)  # °C per hour

            # Calculate heat input from heating system
            # Positive offset = more heat input
            # Negative offset = less heat input
            heat_input_rate = heating_offset * 0.5  # Simplified

            # Net temperature change
            # Higher thermal mass = slower change
            dt_per_hour = (heat_input_rate - heat_loss_rate) / self.thermal_mass

            # Update temperature
            temp = temp + dt_per_hour
            trajectory.append(temp)

        return trajectory

    def calculate_preheating_target(
        self,
        current_temp: float,
        desired_temp: float,
        hours_until_peak: int,
        outdoor_temp: float,
        forecast_min_temp: float,
    ) -> float:
        """Calculate target temperature for pre-heating before expensive/cold period.

        Original algorithm accounting for thermal decay during no-heating period
        and rate of outdoor temperature change.

        Based on research showing load shifting without battery is ineffective.
        Uses moderate pre-heating to prevent thermal debt (DM < -500 catastrophic).

        Args:
            current_temp: Current indoor temperature (°C)
            desired_temp: Target indoor temperature during expensive hours (°C)
            hours_until_peak: Hours until expensive period begins
            outdoor_temp: Current outdoor temperature (°C)
            forecast_min_temp: Minimum outdoor temperature in forecast period (°C)

        Returns:
            Target indoor temperature for pre-heating phase (°C)

        References:
            - Forum_Summary.md: stevedvo's thermal debt case study
            - Enhancement_Proposals.md: Thermal model mathematics
        """
        # Account for thermal decay during forecast period
        # Buildings lose heat proportional to temp difference and inversely to insulation
        temp_diff = current_temp - forecast_min_temp
        heat_loss_rate = temp_diff / (self.insulation_quality * 10)  # °C per hour

        # Calculate expected temperature at end of forecast period
        # if no heating occurs
        expected_temp_end = current_temp - (heat_loss_rate * hours_until_peak)

        # Calculate how much extra heat needed to maintain comfort
        deficit = desired_temp - expected_temp_end

        # Pre-heat target: current desired + deficit + safety margin
        # Higher thermal mass = less aggressive pre-heating needed
        # (thermal mass stores heat better)
        safety_margin = 1.0 / self.thermal_mass  # Range: 0.5-2.0°C

        target = desired_temp + deficit + safety_margin

        # Cap at reasonable limits to avoid excessive pre-heating
        # (which could cause thermal debt if outdoor temp drops faster than expected)
        max_preheat = desired_temp + 3.0
        target = max(desired_temp, min(target, max_preheat))

        _LOGGER.debug(
            "Pre-heat target: %.1f°C (deficit: %.1f°C, safety: %.1f°C)",
            target,
            deficit,
            safety_margin,
        )

        return target

    def add_observation(
        self,
        timestamp: float,
        indoor_temp: float,
        outdoor_temp: float,
        heating_active: bool,
        heating_offset: float,
    ) -> None:
        """Add observation for future learning enhancement.

        Args:
            timestamp: Unix timestamp
            indoor_temp: Indoor temperature (°C)
            outdoor_temp: Outdoor temperature (°C)
            heating_active: Whether heating was active
            heating_offset: Heating curve offset (°C)
        """
        # Store observation for future ML enhancement
        observation = ThermalObservation(
            timestamp=timestamp,
            indoor_temp=indoor_temp,
            outdoor_temp=outdoor_temp,
            heating_active=heating_active,
            heating_offset=heating_offset,
        )
        self._learning_data.append(observation)

        # Keep only last 1000 observations
        if len(self._learning_data) > 1000:
            self._learning_data = self._learning_data[-1000:]

    def get_prediction_horizon(self) -> float:
        """Get prediction horizon for weather forecasting.

        Base implementation returns default 12 hours.
        AdaptiveThermalModel overrides this with UFH-type-specific values.

        Returns:
            Prediction horizon in hours (default 12.0)
        """
        return 12.0  # Default medium horizon

    def get_thermal_characteristics(self) -> dict[str, Any]:
        """Get current thermal model characteristics.

        Returns:
            Dictionary with thermal mass, insulation, and observation count
        """
        return {
            "thermal_mass": self.thermal_mass,
            "insulation_quality": self.insulation_quality,
            "observations": len(self._learning_data),
        }

"""Base classes for heat pump model profiles.

Defines the abstract interface that all heat pump models must implement.
Model profiles contain manufacturer-specific characteristics for optimal
performance and validation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of power consumption validation."""

    valid: bool
    severity: str  # "info", "warning", "error"
    message: str
    suggestions: list[str]


@dataclass
class HeatPumpProfile(ABC):
    """Abstract base class for heat pump model profiles.

    Each heat pump model should subclass this and provide:
    - Power characteristics (rated power, modulation range)
    - Efficiency curves (COP vs outdoor/flow temp)
    - Optimization parameters (DM thresholds, cycling protection)
    - Validation logic (verify power consumption is normal)
    """

    # Identity
    model_name: str
    manufacturer: str
    model_type: str  # e.g., "F-series ASHP", "S-series GSHP"

    # Power characteristics
    rated_power_kw: tuple[float, float]  # (min, max) heat output
    typical_electrical_range_kw: tuple[float, float]  # Typical electrical consumption
    modulation_range: tuple[int, int]  # Hz or %
    modulation_type: str  # "inverter", "on_off", "staged"

    # Efficiency characteristics
    typical_cop_range: tuple[float, float]  # (min, max) COP
    optimal_flow_delta: float  # °C above outdoor for max efficiency
    cop_curve: dict[float, float]  # outdoor_temp → COP mapping

    # System capabilities
    supports_aux_heating: bool
    supports_modulation: bool
    supports_weather_compensation: bool
    max_flow_temp: float
    min_flow_temp: float

    # Optimization parameters (Swedish NIBE research)
    dm_threshold_start: float = -60  # Normal compressor start
    dm_threshold_extended: float = -240  # Extended runs acceptable
    dm_threshold_warning: float = -400  # Approaching danger
    dm_threshold_critical: float = -500  # Emergency recovery
    dm_threshold_aux_swedish: float = -1500  # Swedish aux optimization

    # Cycling protection
    min_runtime_minutes: int = 30
    min_rest_minutes: int = 10

    # Exhaust air heat pump features
    # Only EAHP models (F730, F750) support airflow optimization
    supports_exhaust_airflow: bool = False
    standard_airflow_m3h: float = 0.0  # Normal ventilation rate
    enhanced_airflow_m3h: float = 0.0  # Maximum ventilation rate

    @abstractmethod
    def calculate_optimal_flow_temp(
        self,
        outdoor_temp: float,
        indoor_target: float,
        heat_demand_kw: float,
    ) -> float:
        """Calculate optimal flow temperature for conditions.

        Args:
            outdoor_temp: Current outdoor temperature (°C)
            indoor_target: Target indoor temperature (°C)
            heat_demand_kw: Required heat output (kW)

        Returns:
            Optimal flow temperature (°C) for maximum efficiency
        """
        pass

    @abstractmethod
    def validate_power_consumption(
        self,
        current_power_kw: float,
        outdoor_temp: float,
        flow_temp: float,
    ) -> ValidationResult:
        """Validate if current power consumption is normal.

        Args:
            current_power_kw: Current electrical consumption (kW)
            outdoor_temp: Outdoor temperature (°C)
            flow_temp: Flow temperature (°C)

        Returns:
            ValidationResult with status and recommendations
        """
        pass

    def get_cop_at_temperature(self, outdoor_temp: float) -> float:
        """Get COP for given outdoor temperature using interpolation.

        Args:
            outdoor_temp: Outdoor temperature (°C)

        Returns:
            Estimated COP
        """
        temps = sorted(self.cop_curve.keys())

        if outdoor_temp >= temps[-1]:
            return self.cop_curve[temps[-1]]
        if outdoor_temp <= temps[0]:
            return self.cop_curve[temps[0]]

        # Linear interpolation
        for i in range(len(temps) - 1):
            if temps[i] <= outdoor_temp <= temps[i + 1]:
                t1, t2 = temps[i], temps[i + 1]
                cop1, cop2 = self.cop_curve[t1], self.cop_curve[t2]

                ratio = (outdoor_temp - t1) / (t2 - t1)
                return cop1 + (cop2 - cop1) * ratio

        return 3.0  # Conservative fallback

    def estimate_electrical_consumption(
        self,
        heat_demand_kw: float,
        outdoor_temp: float,
    ) -> float:
        """Estimate electrical consumption for heat demand.

        Args:
            heat_demand_kw: Required heat output (kW)
            outdoor_temp: Outdoor temperature (°C)

        Returns:
            Estimated electrical consumption (kW)
        """
        cop = self.get_cop_at_temperature(outdoor_temp)
        electrical_kw = heat_demand_kw / cop

        # Cap at max electrical power
        return min(electrical_kw, self.typical_electrical_range_kw[1])

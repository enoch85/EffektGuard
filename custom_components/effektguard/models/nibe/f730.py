"""NIBE F730 heat pump profile.

6kW ASHP - Smaller model for 80-120m² houses or well-insulated properties.
"""

from dataclasses import dataclass

from ...const import KUEHNE_COEFFICIENT, KUEHNE_POWER
from ..base import HeatPumpProfile, ValidationResult
from ..registry import HeatPumpModelRegistry


@HeatPumpModelRegistry.register("nibe_f730")
@dataclass
class NibeF730Profile(HeatPumpProfile):
    """NIBE F730 6kW Air Source Heat Pump.

    **Target Market**: 80-120m² or well-insulated houses
    **Electrical**: Single-phase 16A or 3-phase
    **Power**: 1.0-2.0kW electrical typical
    """

    model_name: str = "F730"
    manufacturer: str = "NIBE"
    model_type: str = "F-series ASHP"

    rated_power_kw: tuple[float, float] = (1.5, 6.0)
    typical_electrical_range_kw: tuple[float, float] = (1.0, 4.5)
    modulation_range: tuple[int, int] = (70, 120)
    modulation_type: str = "inverter"

    typical_cop_range: tuple[float, float] = (2.0, 5.0)
    optimal_flow_delta: float = 27.0
    cop_curve: dict[float, float] = None

    supports_aux_heating: bool = True
    supports_modulation: bool = True
    supports_weather_compensation: bool = True
    max_flow_temp: float = 58.0
    min_flow_temp: float = 20.0

    min_runtime_minutes: int = 30
    min_rest_minutes: int = 10

    def __post_init__(self):
        """Initialize COP curve."""
        # Same curve as F750 (same technology, different size)
        self.cop_curve = {
            7: 5.0,
            5: 4.5,
            0: 4.0,
            -5: 3.5,
            -10: 3.0,
            -15: 2.7,
            -20: 2.3,
            -25: 2.0,
            -30: 1.8,
        }

    def calculate_optimal_flow_temp(
        self, outdoor_temp: float, indoor_target: float, heat_demand_kw: float
    ) -> float:
        """Calculate optimal flow temp for F730."""
        heat_loss_coefficient = 150.0  # W/°C smaller house
        temp_diff = indoor_target - outdoor_temp

        flow_from_formula = (
            KUEHNE_COEFFICIENT * (heat_loss_coefficient * temp_diff) ** KUEHNE_POWER + indoor_target
        )
        flow_from_efficiency = outdoor_temp + self.optimal_flow_delta

        optimal = min(flow_from_formula, flow_from_efficiency + 3.0)
        return max(self.min_flow_temp, min(optimal, self.max_flow_temp))

    def validate_power_consumption(
        self, current_power_kw: float, outdoor_temp: float, flow_temp: float
    ) -> ValidationResult:
        """Validate F730 power consumption."""
        cop = self.get_cop_at_temperature(outdoor_temp)
        indoor_target = 21.0
        temp_diff = indoor_target - outdoor_temp
        heat_loss_coefficient = 150.0  # Smaller house
        expected_heat_demand_kw = heat_loss_coefficient * temp_diff / 1000
        expected_power_kw = expected_heat_demand_kw / cop

        if current_power_kw > self.typical_electrical_range_kw[1]:
            return ValidationResult(
                valid=False,
                severity="warning",
                message=f"High power: {current_power_kw:.1f}kW exceeds F730 typical",
                suggestions=[
                    "F730 may be undersized for this house",
                    "Consider upgrading to F750 or F1145",
                    "Check auxiliary heating activation",
                ],
            )

        return ValidationResult(
            valid=True,
            severity="info",
            message=f"Power consumption OK: {current_power_kw:.1f}kW (COP {cop:.1f})",
            suggestions=[],
        )

"""NIBE F2040 heat pump profile.

12-16kW ASHP - Large model for 180-250m² houses or poorly insulated properties.
"""

from dataclasses import dataclass

from ...const import KUEHNE_COEFFICIENT, KUEHNE_POWER
from ..base import HeatPumpProfile, ValidationResult
from ..registry import HeatPumpModelRegistry


@HeatPumpModelRegistry.register("nibe_f2040")
@dataclass
class NibeF2040Profile(HeatPumpProfile):
    """NIBE F2040 12-16kW Air Source Heat Pump.

    **Target Market**: 180-250m² houses or older/poorly insulated
    **Electrical**: 3-phase 16A/20A (higher loads)
    **Power**: 2.5-6.5kW electrical typical (can spike to 10kW+ in extreme cold)
    """

    model_name: str = "F2040"
    manufacturer: str = "NIBE"
    model_type: str = "F-series ASHP"

    rated_power_kw: tuple[float, float] = (3.0, 16.0)
    typical_electrical_range_kw: tuple[float, float] = (2.5, 10.0)
    modulation_range: tuple[int, int] = (70, 120)
    modulation_type: str = "inverter"

    typical_cop_range: tuple[float, float] = (1.8, 4.8)
    optimal_flow_delta: float = 30.0  # Slightly higher for larger systems
    cop_curve: dict[float, float] = None

    supports_aux_heating: bool = True  # Larger immersion heaters
    supports_modulation: bool = True
    supports_weather_compensation: bool = True
    max_flow_temp: float = 63.0
    min_flow_temp: float = 20.0

    min_runtime_minutes: int = 35
    min_rest_minutes: int = 12

    def __post_init__(self):
        """Initialize COP curve - slightly lower than F750 (larger unit)."""
        self.cop_curve = {
            7: 4.8,
            5: 4.3,
            0: 3.8,
            -5: 3.3,
            -10: 2.8,
            -15: 2.5,
            -20: 2.1,
            -25: 1.9,
            -30: 1.7,
        }

    def calculate_optimal_flow_temp(
        self, outdoor_temp: float, indoor_target: float, heat_demand_kw: float
    ) -> float:
        """Calculate optimal flow temp for F2040."""
        heat_loss_coefficient = 250.0  # W/°C large/poorly insulated house
        temp_diff = indoor_target - outdoor_temp

        flow_from_formula = (
            KUEHNE_COEFFICIENT * (heat_loss_coefficient * temp_diff) ** KUEHNE_POWER + indoor_target
        )
        flow_from_efficiency = outdoor_temp + self.optimal_flow_delta

        optimal = min(flow_from_formula, flow_from_efficiency + 4.0)
        return max(self.min_flow_temp, min(optimal, self.max_flow_temp))

    def validate_power_consumption(
        self, current_power_kw: float, outdoor_temp: float, flow_temp: float
    ) -> ValidationResult:
        """Validate F2040 power consumption."""
        cop = self.get_cop_at_temperature(outdoor_temp)
        indoor_target = 21.0
        temp_diff = indoor_target - outdoor_temp
        heat_loss_coefficient = 250.0  # Large house
        expected_heat_demand_kw = heat_loss_coefficient * temp_diff / 1000
        expected_power_kw = expected_heat_demand_kw / cop

        if current_power_kw > self.typical_electrical_range_kw[1]:
            return ValidationResult(
                valid=False,
                severity="warning",
                message=f"Very high power: {current_power_kw:.1f}kW",
                suggestions=[
                    "F2040 running at high load - check house insulation",
                    "Auxiliary heating likely active",
                    "Consider insulation upgrades to reduce demand",
                    f"At {outdoor_temp:.1f}°C, this is extreme",
                ],
            )

        return ValidationResult(
            valid=True,
            severity="info",
            message=f"Power consumption OK: {current_power_kw:.1f}kW (COP {cop:.1f})",
            suggestions=[],
        )

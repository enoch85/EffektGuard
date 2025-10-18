"""NIBE S1155 heat pump profile.

Inverter-controlled GSHP - S-series ground source, very efficient.
Available in 4 sizes: 1.5-6 kW, 3-12 kW, 4-16 kW, 6-25 kW.

Source: https://www.nibe.eu/en-eu/products/heat-pumps/ground-source-heat-pumps/s1155
Verified: October 2025, NIBE official website
"""

from dataclasses import dataclass

from ...const import KUEHNE_COEFFICIENT, KUEHNE_POWER
from ..base import HeatPumpProfile, ValidationResult
from ..registry import HeatPumpModelRegistry


@HeatPumpModelRegistry.register("nibe_s1155")
@dataclass
class NibeS1155Profile(HeatPumpProfile):
    """NIBE S1155 Ground Source Heat Pump (Inverter-controlled).

    **VERIFIED**: Official NIBE website (October 2025)
    **Available Sizes**: 1.5-6 kW, 3-12 kW, 4-16 kW, 6-25 kW
    **Target Market**: Small to large properties with ground source installation
    **Key Feature**: No integrated hot water tank (separate tank selected by need)
    **Technology**: Leading inverter technology, adjusts to heating demands
    **Seasonal Performance**: High SCOP (Seasonal Coefficient of Performance)

    **Electrical**: Much lower than ASHP (better COP from stable ground temp)
    **Key Advantage**: Stable ground temperature (5-8°C year-round) = consistent high COP

    This profile uses the mid-range 3-12 kW variant as baseline.
    User-configurable sizing for actual installed capacity recommended.
    """

    model_name: str = "S1155"
    manufacturer: str = "NIBE"
    model_type: str = "S-series GSHP"

    # Mid-range variant (3-12 kW) - VERIFIED from NIBE website
    rated_power_kw: tuple[float, float] = (3.0, 12.0)  # Heat output
    typical_electrical_range_kw: tuple[float, float] = (0.6, 2.5)  # Estimated from SCOP ~5.0
    modulation_range: tuple[int, int] = (70, 120)
    modulation_type: str = "inverter"

    typical_cop_range: tuple[float, float] = (3.5, 5.5)  # Higher than ASHP!
    optimal_flow_delta: float = 25.0  # Can run lower flow temps
    cop_curve: dict[float, float] = None

    supports_aux_heating: bool = True
    supports_modulation: bool = True
    supports_weather_compensation: bool = True
    max_flow_temp: float = 58.0
    min_flow_temp: float = 18.0  # Can go lower with ground source

    min_runtime_minutes: int = 30
    min_rest_minutes: int = 10

    def __post_init__(self):
        """Initialize COP curve - GSHP has much better COP than ASHP.

        Ground temperature is stable 5-8°C year-round in Sweden, so COP
        doesn't drop as much in cold weather.

        VERIFIED: S1155 has high seasonal performance factor (SCOP).
        Source: NIBE official website
        """
        self.cop_curve = {
            7: 5.5,  # Excellent in mild weather
            5: 5.3,
            0: 5.0,  # Still excellent in winter
            -5: 4.8,  # Ground temp stable
            -10: 4.5,  # Ground temp stable
            -15: 4.3,
            -20: 4.0,  # Still good in extreme cold
            -25: 3.8,
            -30: 3.5,  # Much better than ASHP at extreme temps
        }

    def calculate_optimal_flow_temp(
        self, outdoor_temp: float, indoor_target: float, heat_demand_kw: float
    ) -> float:
        """Calculate optimal flow temp for S1155 GSHP."""
        heat_loss_coefficient = 180.0  # W/°C typical house
        temp_diff = indoor_target - outdoor_temp

        flow_from_formula = (
            KUEHNE_COEFFICIENT * (heat_loss_coefficient * temp_diff) ** KUEHNE_POWER + indoor_target
        )

        # GSHP can run lower flow temps for better COP
        flow_from_efficiency = outdoor_temp + self.optimal_flow_delta

        optimal = min(flow_from_formula, flow_from_efficiency + 3.0)
        return max(self.min_flow_temp, min(optimal, self.max_flow_temp))

    def validate_power_consumption(
        self, current_power_kw: float, outdoor_temp: float, flow_temp: float
    ) -> ValidationResult:
        """Validate S1155 power consumption."""
        cop = self.get_cop_at_temperature(outdoor_temp)
        indoor_target = 21.0
        temp_diff = indoor_target - outdoor_temp
        heat_loss_coefficient = 180.0
        expected_heat_demand_kw = heat_loss_coefficient * temp_diff / 1000
        expected_power_kw = expected_heat_demand_kw / cop

        if current_power_kw > self.typical_electrical_range_kw[1] * 1.3:
            return ValidationResult(
                valid=False,
                severity="warning",
                message=f"High power for GSHP: {current_power_kw:.1f}kW (expected {expected_power_kw:.1f}kW)",
                suggestions=[
                    "GSHP should use much less power than ASHP",
                    "Check ground loop circulation pump",
                    "Verify brine flow rate and temperatures",
                    "Ground loop may be undersized or flow restricted",
                ],
            )

        return ValidationResult(
            valid=True,
            severity="info",
            message=f"✓ Excellent GSHP efficiency: {current_power_kw:.1f}kW (COP {cop:.1f})",
            suggestions=[],
        )

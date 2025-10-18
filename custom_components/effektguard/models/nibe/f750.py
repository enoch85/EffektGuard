"""NIBE F750 heat pump profile.

8kW ASHP - Most common model for 100-150m² houses in Sweden.
Based on NIBE official specifications and Swedish forum validation.
"""

from dataclasses import dataclass

from ...const import KUEHNE_COEFFICIENT, KUEHNE_POWER
from ..base import HeatPumpProfile, ValidationResult
from ..registry import HeatPumpModelRegistry


@HeatPumpModelRegistry.register("nibe_f750")
@dataclass
class NibeF750Profile(HeatPumpProfile):
    """NIBE F750 8kW Air Source Heat Pump.

    **Target Market**: 100-150m² standard insulation houses
    **Typical Application**: Single-family homes, floor heating + radiators
    **Electrical**: 3-phase 16A (11kW) or 3-phase 20A (13.8kW)
    **Heating Medium**: Optimized for UFH (25-35°C flow), OK for radiators (45-55°C)

    **Power Characteristics**:
    - Rated: 8kW heat at 7°C outdoor, 45°C flow
    - Modulation: 1.2-6.5kW electrical (3-phase)
    - Typical: 1.5-2.5kW electrical for well-matched system

    **COP Performance**:
    - Best: 5.0 at 7°C outdoor (mild Swedish winter)
    - Good: 4.0 at 0°C (Malmö/Gothenburg average)
    - Acceptable: 3.0 at -10°C (Stockholm cold spell)
    - Survival: 2.0 at -25°C (Kiruna extreme)

    **Source**: NIBE F750 datasheet, Swedish NIBE forum validation
    """

    # Identity
    model_name: str = "F750"
    manufacturer: str = "NIBE"
    model_type: str = "F-series ASHP"

    # Power characteristics
    rated_power_kw: tuple[float, float] = (2.0, 8.0)  # Heat output range
    typical_electrical_range_kw: tuple[float, float] = (1.2, 6.5)  # 3-phase
    modulation_range: tuple[int, int] = (70, 120)  # Hz (inverter compressor)
    modulation_type: str = "inverter"

    # Efficiency - Real-world COP curve for F750
    typical_cop_range: tuple[float, float] = (2.0, 5.0)
    optimal_flow_delta: float = 27.0  # SPF 4.0+ target (outdoor + 27°C)
    cop_curve: dict[float, float] = None  # Set in __post_init__

    # System capabilities
    supports_aux_heating: bool = True  # 3-9kW immersion heater
    supports_modulation: bool = True  # Inverter compressor
    supports_weather_compensation: bool = True  # Built-in weather curve
    max_flow_temp: float = 60.0
    min_flow_temp: float = 20.0

    # Swedish optimization parameters (validated in Swedish NIBE forum)
    dm_threshold_start: float = -60  # Standard NIBE compressor start
    dm_threshold_extended: float = -240  # Extended runs (custom stevedvo setting)
    dm_threshold_warning: float = -400  # Approaching thermal debt danger
    dm_threshold_critical: float = -500  # Emergency recovery needed
    dm_threshold_aux_swedish: float = -1500  # Swedish aux delay optimization

    # Cycling protection (prevents compressor wear)
    min_runtime_minutes: int = 30  # NIBE recommendation
    min_rest_minutes: int = 10  # Minimum off time between cycles

    def __post_init__(self):
        """Initialize COP curve after dataclass creation."""
        # Real-world F750 COP curve (tested and validated)
        self.cop_curve = {
            7: 5.0,  # Rated conditions (mild winter)
            5: 4.5,  # Mild
            0: 4.0,  # Average Swedish winter (Malmö, Gothenburg)
            -5: 3.5,  # Common cold (Stockholm, Uppsala)
            -10: 3.0,  # Cold winter (most of Sweden)
            -15: 2.7,  # Design temperature (Northern Sweden)
            -20: 2.3,  # Very cold (Luleå, Umeå)
            -25: 2.0,  # Extreme cold (Kiruna)
            -30: 1.8,  # Survival mode (rare extreme)
        }

    def calculate_optimal_flow_temp(
        self,
        outdoor_temp: float,
        indoor_target: float,
        heat_demand_kw: float,
    ) -> float:
        """Calculate optimal flow temperature for F750.

        Uses André Kühne's universal formula validated across manufacturers
        combined with F750-specific efficiency targets.

        Args:
            outdoor_temp: Current outdoor temperature (°C)
            indoor_target: Target indoor temperature (°C)
            heat_demand_kw: Required heat output (kW)

        Returns:
            Optimal flow temperature (°C) for maximum efficiency
        """
        # André Kühne formula (validated universal formula)
        # Source: Mathematical_Enhancement_Summary.md
        heat_loss_coefficient = 180.0  # W/°C typical Swedish house
        temp_diff = indoor_target - outdoor_temp

        flow_from_formula = (
            KUEHNE_COEFFICIENT * (heat_loss_coefficient * temp_diff) ** KUEHNE_POWER + indoor_target
        )

        # F750 efficiency target: outdoor + 27°C for SPF 4.0+
        flow_from_efficiency = outdoor_temp + self.optimal_flow_delta

        # Return lower value (more efficient while meeting demand)
        optimal = min(flow_from_formula, flow_from_efficiency + 3.0)

        # Clamp to F750 limits
        return max(self.min_flow_temp, min(optimal, self.max_flow_temp))

    def validate_power_consumption(
        self,
        current_power_kw: float,
        outdoor_temp: float,
        flow_temp: float,
    ) -> ValidationResult:
        """Validate F750 power consumption is normal.

        Args:
            current_power_kw: Current electrical consumption (kW)
            outdoor_temp: Outdoor temperature (°C)
            flow_temp: Flow temperature (°C)

        Returns:
            ValidationResult with status and recommendations
        """
        # Estimate expected power based on conditions
        cop = self.get_cop_at_temperature(outdoor_temp)

        # Typical heat demand for 150m² house
        indoor_target = 21.0  # °C
        temp_diff = indoor_target - outdoor_temp
        heat_loss_coefficient = 180.0  # W/°C typical
        expected_heat_demand_kw = heat_loss_coefficient * temp_diff / 1000

        # Expected electrical consumption
        expected_power_kw = expected_heat_demand_kw / cop

        # Validate against F750 typical range
        min_expected = expected_power_kw * 0.7  # Allow 30% lower
        max_expected = expected_power_kw * 1.5  # Allow 50% higher

        # Check if within reasonable range
        if current_power_kw < min_expected:
            return ValidationResult(
                valid=True,
                severity="info",
                message=f"Power consumption low: {current_power_kw:.1f}kW (expected {expected_power_kw:.1f}kW)",
                suggestions=[
                    "System may be oversized for house (good for efficiency)",
                    "Heat pump modulating well to match demand",
                    "Could indicate excellent insulation",
                ],
            )
        elif current_power_kw > max_expected:
            # High power - could be problem
            if current_power_kw > self.typical_electrical_range_kw[1]:
                return ValidationResult(
                    valid=False,
                    severity="warning",
                    message=f"⚠️ High power: {current_power_kw:.1f}kW exceeds F750 typical max ({self.typical_electrical_range_kw[1]}kW)",
                    suggestions=[
                        "Check if auxiliary heating is active (3-9kW immersion)",
                        f"Flow temp {flow_temp:.1f}°C may be too high for outdoor {outdoor_temp:.1f}°C",
                        "System may be undersized for house size/insulation",
                        "Verify heat loss coefficient (poor insulation?)",
                        "Check for thermal debt (DM < -400)",
                    ],
                )
            else:
                return ValidationResult(
                    valid=True,
                    severity="info",
                    message=f"Power slightly high: {current_power_kw:.1f}kW (expected {expected_power_kw:.1f}kW)",
                    suggestions=[
                        "Within F750 normal range",
                        "May be catching up from setback or cold spell",
                        "Monitor for sustained high consumption",
                    ],
                )
        else:
            # Normal range
            return ValidationResult(
                valid=True,
                severity="info",
                message=f"✓ Power consumption normal: {current_power_kw:.1f}kW (expected {expected_power_kw:.1f}kW, COP {cop:.1f})",
                suggestions=[],
            )

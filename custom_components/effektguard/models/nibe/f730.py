"""NIBE F730 heat pump profile.

6kW ASHP - Smaller model for 80-120m² houses or well-insulated properties.
"""

from dataclasses import dataclass

from ..base import HeatPumpProfile, RatingPoint, ValidationResult
from ..registry import HeatPumpModelRegistry

# NIBE F730 product data sheet, "Output data according to EN 14511". VERBATIM.
# The only three performance figures NIBE publishes for this machine.
F730_DATASHEET = (
    RatingPoint(
        "A20(12)W35, exhaust air flow 90 m3/h (25 l/s) min compressor frequency",
        source_temp_c=20.0,
        flow_temp_c=35.0,
        heat_output_kw=1.27,
        cop=4.79,
        airflow_m3h=90.0,
    ),
    RatingPoint(
        "A20(12)W35, exhaust air flow 252 m3/h (70 l/s) min compressor frequency",
        source_temp_c=20.0,
        flow_temp_c=35.0,
        heat_output_kw=1.53,
        cop=5.32,
        airflow_m3h=252.0,
    ),
    RatingPoint(
        "A20(12)W45, exhaust air flow 252 m3/h (70 l/s) max compressor frequency",
        source_temp_c=20.0,
        flow_temp_c=45.0,
        heat_output_kw=5.35,
        cop=2.43,
        airflow_m3h=252.0,
    ),
)
F730_SOURCE = (
    "NIBE F730 product data sheet 639853 CIL EN 1904-1, 'Output data according to EN 14511'. "
    "https://assetstore.nibe.se/hcms/v2.3/entity/document/22472/storage/MDIyNDcyLzAvbWFzdGVy "
    "(NOTE: the 1x230 V variant, IHB EN 2003-3/531384, publishes DIFFERENT points and a 3.5 kW "
    "immersion heater. This profile is the 3x400 V machine.)"
)


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

    datasheet_points: tuple[RatingPoint, ...] = F730_DATASHEET
    datasheet_source: str = F730_SOURCE

    heating_capacity_range_kw: tuple[float, float] = (
        1.27,
        5.35,
    )  # the max-compressor-frequency point IS the maximum
    rated_power_kw: tuple[float, float] = (1.27, 5.35)  # PH min..max, EN 14511
    typical_electrical_range_kw: tuple[float, float] = (1.0, 4.5)
    modulation_range: tuple[int, int] = (70, 120)
    modulation_type: str = "inverter"

    typical_cop_range: tuple[float, float] = (2.43, 5.32)  # the published COPs
    optimal_flow_delta: float = 27.0
    cop_curve: dict[float, float] = None

    supports_aux_heating: bool = True
    supports_modulation: bool = True
    supports_weather_compensation: bool = True
    max_flow_temp: float = 58.0
    min_flow_temp: float = 20.0

    min_runtime_minutes: int = 30
    min_rest_minutes: int = 10

    # Exhaust air heat pump features
    # F730 is an EAHP - supports airflow optimization for heat extraction
    supports_exhaust_airflow: bool = True
    standard_airflow_m3h: float = 120.0  # Normal ventilation rate (smaller unit)
    enhanced_airflow_m3h: float = 200.0  # Maximum ventilation rate

    # MODELING LIMITATION (review 2026-07): this is an exhaust-air heat pump;
    # its COP depends primarily on exhaust-air (source) and flow (sink)
    # temperatures, not outdoor temperature. The outdoor-keyed curve below is
    # an indirect approximation - adequate for relative decisions, NOT
    # validated for absolute energy/savings claims.
    def __post_init__(self):
        """Initialize COP curve."""
        # Same curve as F750 (same technology, different size)
        # A DISPLAY PROXY, anchored on this machine's own two published endpoints. Nothing
        # computes from it - see the note in f750.py, which shipped a byte-identical curve to this
        # one despite being a different machine with a different published output. That is what
        # gave the fiction away.
        best = max(point.cop for point in self.datasheet_points)  # 5.32, min freq, W35
        worst = min(point.cop for point in self.datasheet_points)  # 2.43, max freq, W45
        self.cop_curve = {
            temp: round(worst + (best - worst) * (temp + 20.0) / 27.0, 2)
            for temp in (7, 5, 0, -5, -10, -15, -20)
        }

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

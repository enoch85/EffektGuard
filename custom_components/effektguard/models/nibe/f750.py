"""NIBE F750 heat pump profile.

EXHAUST-AIR heat pump. Its heat source is the house's own ventilation air, not the outdoor air, and
its output is bounded by the airflow it breathes.

This file used to open with "8kW ASHP" and cite "NIBE official specifications and Swedish forum
validation". It is not an ASHP, it cannot make 8 kW, and the numbers were not from the datasheet.
See RatingPoint in models/base.py.
"""

from dataclasses import dataclass, field

from ..base import HeatPumpProfile, RatingPoint, ValidationResult
from ..registry import HeatPumpModelRegistry
from ...const import DM_THRESHOLD_AUX_LIMIT

# NIBE F750, "Output data according to EN 14 511", part no. 066 063 / 066 061. VERBATIM.
# The only three performance figures NIBE publishes for this machine.
F750_DATASHEET = (
    RatingPoint(
        "A20(12)W35, exhaust air flow 108 m3/h (30 l/s) min compressor frequency",
        source_temp_c=20.0,
        flow_temp_c=35.0,
        heat_output_kw=1.144,
        cop=4.20,
        airflow_m3h=108.0,
    ),
    RatingPoint(
        "A20(12)W35, exhaust air flow 252 m3/h (70 l/s) min compressor frequency",
        source_temp_c=20.0,
        flow_temp_c=35.0,
        heat_output_kw=1.498,
        cop=4.72,
        airflow_m3h=252.0,
    ),
    RatingPoint(
        "A20(12)W45, exhaust air flow 252 m3/h (70 l/s) max compressor frequency",
        source_temp_c=20.0,
        flow_temp_c=45.0,
        heat_output_kw=4.994,
        cop=2.43,
        airflow_m3h=252.0,
    ),
)
F750_SOURCE = (
    "NIBE F750 product data sheet, 'Output data according to EN 14 511', part no. 066 063/066 061. "
    "https://www.teplounion.com/doc/NIBE-F750-information.pdf"
)


@HeatPumpModelRegistry.register("nibe_f750")
@dataclass
class NibeF750Profile(HeatPumpProfile):
    """NIBE F750 EXHAUST-AIR heat pump.

    THE PERFORMANCE FIGURES THAT USED TO BE IN THIS DOCSTRING WERE INVENTED. It claimed:

        Rated: 8kW heat at 7C outdoor, 45C flow
        Best COP: 5.0 at 7C outdoor ... Survival: 2.0 at -25C
        **Source**: NIBE F750 datasheet, Swedish NIBE forum validation

    NIBE's datasheet publishes three EN 14511 points and no others. Its maximum specified heating
    output is 4.994 kW, not 8. The number 5.0 does not appear as a COP anywhere in it. And "at 7C
    outdoor" is not a condition this machine's performance is measured at, because its heat source
    is 20 C extract air from inside the house - the rating points say A20(12), and the outdoor air
    never touches the evaporator.

    What the datasheet actually says is in F750_DATASHEET above, verbatim, with the condition
    strings. Everything the simulator believes is derived from those and from nothing else.

    Pdesign 5 kW. Immersion heater 0.5-6.5 kW. SCOP(EN 14825) 4.5/4.7 average/cold at 35 C.
    """

    # Identity
    model_name: str = "F750"
    manufacturer: str = "NIBE"
    model_type: str = "F-series ASHP"

    # THE DATASHEET. Everything below that is a number is derived from it in __post_init__.
    datasheet_points: tuple[RatingPoint, ...] = F750_DATASHEET
    datasheet_source: str = F750_SOURCE

    # Power characteristics - DERIVED from the rating points, not restated.
    design_heat_load_kw: float = 5.0  # Pdesignh - datasheet "Nominal heating output (Pdesign) 5 kW"
    immersion_heater_kw: float = 3.5  # datasheet: "6.5 (3.5) kW" - max 6.5, delivery setting 3.5
    heating_capacity_range_kw: tuple[float, float] = (
        1.144,
        4.994,
    )  # the max-compressor-frequency point IS the maximum
    rated_power_kw: tuple[float, float] = (1.144, 4.994)  # PH min..max, EN 14511
    typical_electrical_range_kw: tuple[float, float] = (0.27, 2.06)  # PH/COP at those same points
    modulation_range: tuple[int, int] = (70, 120)  # Hz (inverter compressor)
    modulation_type: str = "inverter"

    typical_cop_range: tuple[float, float] = (2.43, 4.72)  # the published COPs, min..max
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
    # The simulator reads this so the plant model tracks what the integration believes.
    # It cannot do that while the profile restates the number, so it references it (F-076).
    dm_threshold_aux_swedish: float = DM_THRESHOLD_AUX_LIMIT

    # Cycling protection (prevents compressor wear)
    min_runtime_minutes: int = 30  # NIBE recommendation
    min_rest_minutes: int = 10  # Minimum off time between cycles

    # Exhaust air heat pump features
    # F750 is an EAHP - supports airflow optimization for heat extraction
    supports_exhaust_airflow: bool = True
    standard_airflow_m3h: float = 150.0  # Normal ventilation rate
    enhanced_airflow_m3h: float = 252.0  # Maximum ventilation rate

    def __post_init__(self):
        """The outdoor-keyed COP curve is a DISPLAY approximation and is labelled as one.

        THIS FILE'S OWN COMMENT ALREADY SAID SO, and I used the curve for absolute energy claims
        anyway:

            "MODELING LIMITATION: this is an exhaust-air heat pump; its COP depends primarily on
             exhaust-air (source) and flow (sink) temperatures, not outdoor temperature. The
             outdoor-keyed curve below is an indirect approximation - adequate for relative
             decisions, NOT validated for absolute energy/savings claims."

        The simulator then produced a month of kWh and SEK from it and I published the savings.

        Nothing computes from this curve any more. The simulator takes its COP from
        `datasheet_points` via the exergy-efficiency model (see scripts/simulation/sim_harness.py),
        which needs the SOURCE temperature - a constant 20 C for this machine - and the flow
        temperature, and never the weather.

        What survives here is an honest seasonal PROXY for the dashboard: as it gets colder the
        house asks for hotter water and a higher compressor frequency, and both cost efficiency. It
        is anchored on the two published endpoints (COP 4.72 at min frequency / W35, COP 2.43 at
        max frequency / W45) instead of on invented numbers.
        """
        best = max(point.cop for point in self.datasheet_points)  # 4.72, min freq, W35
        worst = min(point.cop for point in self.datasheet_points)  # 2.43, max freq, W45

        # A linear walk between the machine's own two published COPs across the Swedish range.
        # It is a PROXY for load, not a measurement against outdoor temperature - the source air is
        # 20 C whatever the weather - and no physics is computed from it.
        self.cop_curve = {
            temp: round(worst + (best - worst) * (temp + 20.0) / 27.0, 2)
            for temp in (7, 5, 0, -5, -10, -15, -20)
        }

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

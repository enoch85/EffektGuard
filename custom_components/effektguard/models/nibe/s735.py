"""NIBE S735 heat pump profile.

EXHAUST-AIR heat pump, S-series: its heat source is the house's own ventilation air, so its output
is bounded by the airflow it breathes. Performance derives from the EN 14511 rating points below;
see RatingPoint in models/base.py. Every figure is in const.py, with its source.
"""

from dataclasses import dataclass

from ..base import HeatPumpProfile, RatingPoint, ValidationResult, seasonal_cop_proxy
from ..registry import HeatPumpModelRegistry
from .f750 import NibeF750Profile
from ...const import (
    DM_THRESHOLD_START,
    POWER_VALIDATION_MARGIN,
    S735_BLOCK_FREQ_HZ,
    S735_EN14511_POINTS,
    S735_EXTRACT_AIR_C,
    S735_IMMERSION_HEATER_KW,
    S735_PDESIGNH_KW,
    S735_SUPPLY_MAX_C,
    S735_SUPPLY_MIN_C,
    S_SERIES_RELATIVE_DM_START_ADDITIONAL_HEAT,
)

# NIBE S735-7, "Output data according to EN 14511". NIBE publishes no other performance points.
S735_7_DATASHEET = tuple(
    RatingPoint(
        condition,
        source_temp_c=S735_EXTRACT_AIR_C,
        flow_temp_c=flow_c,
        heat_output_kw=output_kw,
        cop=cop,
        airflow_m3h=airflow_m3h,
    )
    for condition, flow_c, output_kw, cop, airflow_m3h in S735_EN14511_POINTS
)
S735_7_SOURCE = (
    "NIBE S735 installer manual IHB EN 2647-4 (M13212, S735 Stainless 3x400 V, 3x230 V), "
    "'Technical specifications', 'Output data according to EN 14511', S735-7 column. "
    "https://assetstore.nibe.se/hcms/v2.3/entity/document/922078/storage/OTIyMDc4LzAvbWFzdGVy "
    "(NIBE's S735 data sheet, CIL EN 2218-1/M12836, prints the same three figures. The S735-4 "
    "column publishes other points: 1.00/3.49, 1.38/4.76 and 4.21/3.25 at A20(12)W35.)"
)

# The least and the most heat NIBE publishes for it, taken from the points themselves so that no
# second copy of a figure can drift.
_MIN_POINT = min(S735_7_DATASHEET, key=lambda point: point.heat_output_kw)
_MAX_POINT = max(S735_7_DATASHEET, key=lambda point: point.heat_output_kw)
_OUTPUT_RANGE_KW = (_MIN_POINT.heat_output_kw, _MAX_POINT.heat_output_kw)


@HeatPumpModelRegistry.register("nibe_s735")
@dataclass
class NibeS735Profile(HeatPumpProfile):
    """NIBE S735 EXHAUST-AIR heat pump, the S735-7 (Pdesignh 6 kW), the size NIBE's data sheet shows.

    Integrated immersion heater of 9 kW (6.5 kW in the copper and 1x230 V versions), at its
    maximum from the factory. SCOP (EN 14825) 4.75/3.81 cold and 4.50/3.67 average climate at
    35/55 C.
    """

    model_name: str = "S735"
    manufacturer: str = "NIBE"
    model_type: str = "S-series EAHP"

    # The S-series start, as the S1155's (S_SERIES_RELATIVE_DM_START_ADDITIONAL_HEAT). A pump set
    # deeper - NIBE's S-series installer guide (2435-1) sets "Start compressor" to -120 - is learned
    # from its immersion heater's power; a start shallower than this one is not
    # (DecisionEngine.immersion_start_in_force).
    immersion_start_dm: float = float(
        DM_THRESHOLD_START - S_SERIES_RELATIVE_DM_START_ADDITIONAL_HEAT
    )

    datasheet_points: tuple[RatingPoint, ...] = S735_7_DATASHEET
    datasheet_source: str = S735_7_SOURCE

    design_heat_load_kw: float = S735_PDESIGNH_KW
    immersion_heater_kw: float = S735_IMMERSION_HEATER_KW
    heating_capacity_range_kw: tuple[float, float] = _OUTPUT_RANGE_KW
    rated_power_kw: tuple[float, float] = _OUTPUT_RANGE_KW
    typical_electrical_range_kw: tuple[float, float] = (
        round(_MIN_POINT.heat_output_kw / _MIN_POINT.cop, 2),
        round(_MAX_POINT.heat_output_kw / _MAX_POINT.cop, 2),
    )  # PH/COP at those same points
    modulation_range: tuple[int, int] = S735_BLOCK_FREQ_HZ
    publishes_max_frequency: bool = True  # "Max frequency compressor"
    modulation_type: str = "inverter"  # data sheet: "inverter-controlled exhaust air heat pump"

    typical_cop_range: tuple[float, float] = (
        min(point.cop for point in S735_7_DATASHEET),
        max(point.cop for point in S735_7_DATASHEET),
    )
    # Not a NIBE figure: the F750's, shown only as a sensor attribute.
    optimal_flow_delta: float = NibeF750Profile.optimal_flow_delta
    cop_curve: dict[float, float] = None

    supports_immersion_heating: bool = True
    supports_modulation: bool = True
    supports_weather_compensation: bool = True
    max_flow_temp: float = S735_SUPPLY_MAX_C
    min_flow_temp: float = S735_SUPPLY_MIN_C

    # An exhaust-air pump, but EffektGuard drives ventilation through an increased-ventilation
    # switch, and no S735 is known to offer one. So no airflow control here.
    supports_exhaust_airflow: bool = False

    def __post_init__(self):
        """Display-only seasonal COP proxy, as the F750's (f750.py __post_init__)."""
        self.cop_curve = seasonal_cop_proxy(self.datasheet_points)

    def validate_power_consumption(
        self, current_power_kw: float, outdoor_temp: float, flow_temp: float
    ) -> ValidationResult:
        """A draw the machine cannot make: its compressor's published maximum plus its heater."""
        ceiling = (self.typical_electrical_range_kw[1] + self.immersion_heater_kw) * (
            POWER_VALIDATION_MARGIN
        )
        if current_power_kw > ceiling:
            return ValidationResult(
                valid=False,
                severity="warning",
                message=(
                    f"Power {current_power_kw:.1f} kW is more than an S735 can draw "
                    f"({ceiling:.1f} kW with its immersion heater)"
                ),
                suggestions=["Check the power sensor's unit and scaling"],
            )
        return ValidationResult(
            valid=True,
            severity="info",
            message=f"Power {current_power_kw:.1f} kW is within what an S735 can draw",
            suggestions=[],
        )

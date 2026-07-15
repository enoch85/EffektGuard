"""NIBE F1155 heat pump profile.

Inverter-controlled GSHP - F-series ground source, predecessor of the S1155.
Available in 3 sizes: 1.5-6 kW, 4-12 kW, 4-16 kW.

F1155 owners connect via local Modbus (nibe_heatpump integration / MODBUS40).

Heat source is brine from a borehole, so performance is keyed on INCOMING BRINE temperature, not
outdoor air. The F1155 and S1155 publish IDENTICAL EN 14511 data at every size (below, verbatim).
"""

from dataclasses import dataclass

from ..base import HeatPumpProfile, RatingPoint, ValidationResult, seasonal_cop_proxy
from ..registry import HeatPumpModelRegistry

# F1155-12. EN 14511 rating points, VERBATIM.
#
# Keyed on INCOMING BRINE temperature, not outdoor air (datasheet capacity chart x-axis is
# "Incoming brine temp, C"). All four points are at NOMINAL (50 Hz) frequency; NIBE publishes no
# min-/max-frequency COP for these pumps, only the modulation envelope ("Heating capacity (PH)"),
# so load dependence of efficiency is NOT measurable here - see HouseConfig.exergy_efficiency.
F1155_12_DATASHEET = (
    RatingPoint(
        "0/35 nominal (50 Hz), incoming brine 0 C",
        source_temp_c=0.0,
        flow_temp_c=35.0,
        heat_output_kw=5.06,
        cop=4.87,
    ),
    RatingPoint(
        "0/45 nominal (50 Hz), incoming brine 0 C",
        source_temp_c=0.0,
        flow_temp_c=45.0,
        heat_output_kw=4.78,
        cop=3.75,
    ),
    RatingPoint(
        "10/35 nominal (50 Hz), incoming brine 10 C",
        source_temp_c=10.0,
        flow_temp_c=35.0,
        heat_output_kw=6.33,
        cop=6.12,
    ),
    RatingPoint(
        "10/45 nominal (50 Hz), incoming brine 10 C",
        source_temp_c=10.0,
        flow_temp_c=45.0,
        heat_output_kw=5.98,
        cop=4.59,
    ),
)
F1155_12_SOURCE = (
    "NIBE F1155 installer manual, Output data according to EN 14511, "
    "F1155_12 column. https://installer.nibe.eu/ "
    "(F1155: IHB EN 2008-5/331379 p.69; S1155: IHB EN 2001-1/531210 p.70. The two "
    "publish IDENTICAL EN 14511 data at every size - same platform.)"
)
from .s1155 import NibeS1155Profile


@HeatPumpModelRegistry.register("nibe_f1155")
@dataclass
class NibeF1155Profile(NibeS1155Profile):
    """NIBE F1155 Ground Source Heat Pump (Inverter-controlled).

    **Available Sizes**: 1.5-6 kW, 4-12 kW, 4-16 kW
    **Target Market**: Small to large properties with ground source installation
    **Technology**: Inverter-controlled compressor, adjusts to heating demand
    **Key Advantage**: Stable ground temperature (5-8°C year-round) = consistent high COP

    Inherits flow-temperature and power-validation behavior from the S1155
    sibling profile (same hydraulic family); only identity, sizing, and the
    slightly lower COP curve differ. Baseline: mid-range 4-12 kW variant.
    """

    model_name: str = "F1155"
    model_type: str = "F-series GSHP"

    # Mid-range variant (4-12 kW)
    datasheet_points: tuple[RatingPoint, ...] = F1155_12_DATASHEET
    datasheet_source: str = F1155_12_SOURCE
    design_heat_load_kw: float = (
        12.0  # Pdesignh - installer manual "Nominal heating output (Pdesignh) 12 kW" (F1155-12)
    )
    immersion_heater_kw: float = 7.0  # datasheet: additional power 1/2/3/4/5/6/7 kW
    heating_capacity_range_kw: tuple[float, float] = (
        3.0,
        12.0,
    )  # datasheet "Heating capacity (PH)"

    rated_power_kw: tuple[float, float] = (3.0, 12.0)  # the PH modulation envelope, EN 14511
    typical_electrical_range_kw: tuple[float, float] = (1.04, 2.5)  # PE at the rating points
    modulation_range: tuple[int, int] = (20, 120)

    typical_cop_range: tuple[float, float] = (3.75, 6.12)  # published COPs, 0/45 .. 10/35

    def __post_init__(self):
        """Initialize display-only COP proxy.

        Nothing computes from it: COP is a function of brine + flow temperature, not the weather,
        and the simulator takes it from `datasheet_points`. The proxy is a dashboard seasonal curve
        anchored on the two published W35/W45 COPs at 0 C brine.
        """
        self.cop_curve = seasonal_cop_proxy(self.datasheet_points, source_temp_c=0.0)

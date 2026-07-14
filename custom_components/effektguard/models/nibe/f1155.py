"""NIBE F1155 heat pump profile.

Inverter-controlled GSHP - F-series ground source, predecessor of the S1155.
Available in 3 sizes: 1.5-6 kW, 4-12 kW, 4-16 kW.

Added for issue #18: F1155 owners connect via local Modbus (nibe_heatpump
integration / MODBUS40) and previously had to pick the S1155 profile.

THE COP CURVE THAT USED TO BE HERE WAS AUTHORED, AND THIS DOCSTRING SAID SO IN PLAIN WORDS:

    "Physics inherit from the S1155 (same GSHP family); COP curve SET SLIGHTLY BELOW the S1155
     (older inverter platform, SCOP ~5.0)."

Set. Not measured. And it was set against OUTDOOR temperature, for a machine whose heat source is
brine from a borehole - NIBE's own capacity chart plots this pump's output against an x-axis
labelled "Incoming brine temp, C", and there is no air-temperature rating point anywhere in its
datasheet. The curve ran 5.3 at +7 C down to 3.3 at -30 C, describing a machine whose heat source
freezes with the weather. A ground-source pump's does not.

The real EN 14511 data is below, verbatim. It turns out the F1155 and the S1155 publish IDENTICAL
figures at every size, so "slightly below the S1155" was not merely unsourced - it was wrong.
"""

from dataclasses import dataclass

from ..base import HeatPumpProfile, RatingPoint, ValidationResult
from ..registry import HeatPumpModelRegistry

# F1155-12. EN 14511 rating points, VERBATIM.
#
# EVERY POINT IS KEYED ON INCOMING BRINE TEMPERATURE, not outdoor air. The datasheet's own
# capacity chart plots output against an x-axis labelled "Incoming brine temp, C". This machine
# does not know what the weather is doing, and the outdoor-keyed COP curve this profile used to
# carry - 5.3 at +7 C falling to 3.3 at -30 C - described a machine that does not exist.
#
# All four points are at NOMINAL (50 Hz) frequency. NIBE publishes no min- or max-frequency COP for
# these pumps, only the modulation envelope (the "Heating capacity (PH)" row). So the load
# dependence of the efficiency is NOT measurable from this datasheet, and the model does not
# pretend otherwise - see HouseConfig.exergy_efficiency.
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
    heating_capacity_range_kw: tuple[float, float] = (
        3.0,
        12.0,
    )  # datasheet "Heating capacity (PH)"

    rated_power_kw: tuple[float, float] = (3.0, 12.0)  # the PH modulation envelope, EN 14511
    typical_electrical_range_kw: tuple[float, float] = (1.04, 2.5)  # PE at the rating points
    modulation_range: tuple[int, int] = (20, 120)

    typical_cop_range: tuple[float, float] = (3.75, 6.12)  # published COPs, 0/45 .. 10/35

    def __post_init__(self):
        """Initialize COP curve - S1155 shape shifted slightly down for the
        older F-series inverter platform."""
        # A DISPLAY PROXY ONLY, and it is now honest about that.
        #
        # This machine's COP is a function of BRINE temperature and flow temperature. It has no
        # opinion about the weather. The curve that used to be here ran from 5.3 at +7 C outdoor
        # down to 3.3 at -30 C, which described a machine whose heat source freezes with the air -
        # and a ground-source pump's does not. Nothing computes from this; the simulator takes its
        # COP from `datasheet_points`.
        #
        # What is left is a seasonal proxy for the dashboard, anchored on the two published W35/W45
        # COPs at 0 C brine, because in a colder month the house asks for hotter water.
        warm = max(p.cop for p in self.datasheet_points if p.source_temp_c == 0.0)  # 0/35
        cold = min(p.cop for p in self.datasheet_points if p.source_temp_c == 0.0)  # 0/45
        self.cop_curve = {
            temp: round(cold + (warm - cold) * (temp + 20.0) / 27.0, 2)
            for temp in (7, 5, 0, -5, -10, -15, -20)
        }

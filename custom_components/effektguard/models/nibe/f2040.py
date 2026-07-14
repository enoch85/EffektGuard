"""NIBE F2040 heat pump profile.

AIR/WATER heat pump (outdoor monobloc). The ONLY machine in this package whose heat source really
is the outdoor air - which is why it is the only one for which an outdoor-keyed COP curve means
anything at all.

THE SIZES ARE DIFFERENT MACHINES. NIBE ships the F2040 as a 6, an 8, a 12 and a 16, and their
published outputs differ by a factor of nearly three (A7/W35: 2.67 / 3.86 / 5.21 / 7.03 kW). This
profile carries the **F2040-8**, and says so. A single profile cannot honestly stand for all four;
offering the size in the config flow is an owner decision and is not made here.
"""

from dataclasses import dataclass

from ..base import HeatPumpProfile, RatingPoint, ValidationResult
from ..registry import HeatPumpModelRegistry

# NIBE F2040-8, installer manual IHB EN 1848-8 / 231846, p.65:
# "Output data according to EN 14511 dT5K - Capacity / power input / COP (kW/kW/-) at nominal flow"
# VERBATIM. Note what these say and what I claimed they said.
F2040_8_DATASHEET = (
    RatingPoint(
        "A7/W35, EN 14511 dT5K at nominal flow (floor heating)",
        source_temp_c=7.0,
        flow_temp_c=35.0,
        heat_output_kw=3.86,
        cop=4.65,
    ),
    RatingPoint(
        "A2/W35, EN 14511 dT5K at nominal flow (floor heating)",
        source_temp_c=2.0,
        flow_temp_c=35.0,
        heat_output_kw=5.11,
        cop=3.76,
    ),
    RatingPoint(
        "A-7/W35, EN 14511 dT5K at nominal flow (floor heating)",
        source_temp_c=-7.0,
        flow_temp_c=35.0,
        heat_output_kw=6.60,
        cop=2.68,
    ),
    RatingPoint(
        "A7/W45, EN 14511 dT5K at nominal flow",
        source_temp_c=7.0,
        flow_temp_c=45.0,
        heat_output_kw=3.70,
        cop=3.70,
    ),
    RatingPoint(
        "A2/W45, EN 14511 dT5K at nominal flow",
        source_temp_c=2.0,
        flow_temp_c=45.0,
        heat_output_kw=5.03,
        cop=2.96,
    ),
)
F2040_SOURCE = (
    "NIBE F2040 installer manual IHB EN 1848-8/231846 p.65, 'Output data according to EN 14511' "
    "(F2040-8 column); cross-checked against the F2040 Specification Sheet. "
    "https://installer.nibe.eu/download/18.69d23679185eaf109d92e20/1676544181832/"
    "F2040-IHB-231846-8.pdf"
)

# CAPACITY RISES AS IT GETS COLDER. It does not fall.
#
# The simulator derated this machine's output by 2.5% per degree below +7 C and attributed that to
# "the EN 14511 rating points (A7/W35, A2/W35, A-7/W35, A-15/W35)", which "trace a near-linear
# decline". They trace a near-linear RISE - 3.86 -> 5.11 -> 6.60 kW from +7 to -7 C - because this
# is an INVERTER: at +7/W35 it is throttled back to part load, and as the weather cools it simply
# ramps the compressor UP. What collapses with the cold is the COP (4.65 -> 3.76 -> 2.68), not the
# capacity. There is no derating table in the datasheet because there is no derating.
#
# I invented that citation, and I got the sign of the effect backwards, and the entire
# saturated-compressor finding (F-124) was built on it.
#
# NIBE publishes no TABULATED capacity below -7 C - only a "Max specified output" graph - so the
# model holds capacity at the -7 C figure below that point and says so, rather than inventing a
# slope. Operating limits, from the same manual: "Min. / Max. air temp: -20 / 43 C".
MIN_AIR_TEMP_C = -20.0
MAX_AIR_TEMP_C = 43.0


@HeatPumpModelRegistry.register("nibe_f2040")
@dataclass
class NibeF2040Profile(HeatPumpProfile):
    """NIBE F2040-8 air/water heat pump.

    THIS DOCSTRING USED TO READ "12-16kW ASHP ... Power: 2.5-6.5kW electrical (can spike to 10kW+)"
    and the profile carried rated_power_kw = (3.0, 16.0), max_flow_temp = 63.0 and
    supports_aux_heating = True.

    The datasheet says the -8 makes 3.86 kW at its A7/W35 rating point and 6.60 kW at -7/W35; that
    it supplies at most 58 C ("Min. / Max. HM temp continuous operation: 25 / 58 C"); and that it
    has NO immersion heater at all - it is an outdoor monobloc, and the electric backup lives in the
    indoor module. Three fields, three fictions.

    SCOP(EN 14825) cold climate 35 C: 3.55, Pdesignh 9 kW.
    """

    model_name: str = "F2040"
    manufacturer: str = "NIBE"
    model_type: str = "F-series air/water"

    # SOURCED: the F2040 is an outdoor monobloc with no additive heat of its own; the indoor
    # controller owns it. VVM 320 (its standard pairing) ships "start diff additional heat" 700
    # below the compressor start (-60): additive heat engages at about -760.
    aux_start_dm: float = -760.0

    datasheet_points: tuple[RatingPoint, ...] = F2040_8_DATASHEET
    datasheet_source: str = F2040_SOURCE

    rated_power_kw: tuple[float, float] = (3.86, 6.60)  # PH at A7/W35 .. A-7/W35, EN 14511
    typical_electrical_range_kw: tuple[float, float] = (0.83, 2.46)  # Pin at those same points
    modulation_range: tuple[int, int] = (70, 120)
    modulation_type: str = "inverter"

    typical_cop_range: tuple[float, float] = (2.68, 4.65)  # the published COPs at W35
    optimal_flow_delta: float = 30.0
    cop_curve: dict[float, float] = None

    # NO IMMERSION HEATER. The F2040 is an outdoor monobloc; its technical-specifications table has
    # no immersion-heater row. Electric addition belongs to the paired indoor module (VVM/SMO).
    # NO IMMERSION HEATER. Not "0 kW as a default" - the machine physically does not have one.
    # ErP declaration, F2040-8: "Tbiv Bivalent temperature -9 C", "TOL Min. outdoor air
    # temperature -10 C", "Psup Rated heat output 1.1 kW", "Pdh Tj = biv 6.6 kW".
    # Below -9 C this machine is DESIGNED to need supplementary heat.
    bivalent_temp_c: float = -9.0
    supplementary_heat_kw: float = 1.1  # ErP: "Psup Rated heat output 1.1 kW"
    # Pdesignh at the EN 14825 COLD climate, 35 C application (spec sheet): 9.0 kW. The harness
    # sizes every house at the COLD design temperature (-22 C), so the cold figure is the one that
    # belongs here. The AVERAGE-climate declaration (design temp -10 C) is carried separately
    # below, because Tbiv and Psup above belong to IT - splicing the cold Pdesignh onto the
    # average Psup manufactured a 7.9 kW compressor that appears in no NIBE document.
    design_heat_load_kw: float = 9.0
    design_heat_load_average_kw: float = 8.2  # spec sheet, average/35
    immersion_heater_kw: float = 0.0
    supports_aux_heating: bool = False
    supports_modulation: bool = True
    supports_weather_compensation: bool = True
    max_flow_temp: float = 58.0  # "Min. / Max. HM temp continuous operation: 25 / 58 C"
    min_flow_temp: float = 25.0

    min_runtime_minutes: int = 35
    min_rest_minutes: int = 12

    def __post_init__(self):
        """The outdoor-keyed COP curve, and for THIS machine it is a real measurement.

        The F2040's heat source IS the outdoor air, so its COP genuinely is a function of outdoor
        temperature - unlike the four other profiles in this package, which shipped outdoor-keyed
        curves for machines that breathe 20 C house air or 0 C brine.

        These are the datasheet's own W35 COPs. Below -7 C, NIBE tabulates nothing, so the curve
        stops where the evidence stops rather than being extended to -30 C as it was before.
        """
        self.cop_curve = {
            int(point.source_temp_c): point.cop
            for point in self.datasheet_points
            if point.flow_temp_c == 35.0
        }

    def validate_power_consumption(
        self, current_power_kw: float, outdoor_temp: float, flow_temp: float
    ) -> ValidationResult:
        """Validate F2040 power consumption against the published electrical input."""
        max_electrical = self.typical_electrical_range_kw[1]

        if current_power_kw > max_electrical:
            return ValidationResult(
                valid=False,
                severity="warning",
                message=(
                    f"Power {current_power_kw:.1f} kW exceeds the F2040-8's published input "
                    f"({max_electrical:.2f} kW at -7/35)"
                ),
                suggestions=[
                    "Electric addition in the indoor module is probably running",
                    "The F2040 itself has no immersion heater",
                    f"At {outdoor_temp:.1f} C with {flow_temp:.1f} C flow, check the curve",
                ],
            )

        return ValidationResult(
            valid=True,
            severity="info",
            message=f"Power consumption OK: {current_power_kw:.1f} kW",
            suggestions=[],
        )

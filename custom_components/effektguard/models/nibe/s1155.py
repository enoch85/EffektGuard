"""NIBE S1155 heat pump profile.

Inverter-controlled GSHP - S-series ground source, very efficient.
Available in 4 sizes: 1.5-6 kW, 3-12 kW, 4-16 kW, 6-25 kW.

Source: https://www.nibe.eu/en-eu/products/heat-pumps/ground-source-heat-pumps/s1155
Verified: October 2025, NIBE official website
"""

from dataclasses import dataclass

from ..base import HeatPumpProfile, RatingPoint, ValidationResult, seasonal_cop_proxy
from ..registry import HeatPumpModelRegistry

# S1155-12. EN 14511 rating points, VERBATIM.
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
S1155_12_DATASHEET = (
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
S1155_12_SOURCE = (
    "NIBE S1155 installer manual, Output data according to EN 14511, "
    "S1155_12 column. https://installer.nibe.eu/ "
    "(F1155: IHB EN 2008-5/331379 p.69; S1155: IHB EN 2001-1/531210 p.70. The two "
    "publish IDENTICAL EN 14511 data at every size - same platform.)"
)


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

    # SOURCED: S1155 register 40680 "start diff additional heat" factory default 400, applied
    # below the compressor start (-60): the controller engages additive heat at about -460.
    aux_start_dm: float = -460.0

    # Mid-range variant (3-12 kW) - VERIFIED from NIBE website
    datasheet_points: tuple[RatingPoint, ...] = S1155_12_DATASHEET
    datasheet_source: str = S1155_12_SOURCE
    design_heat_load_kw: float = (
        12.0  # Pdesignh - installer manual, S1155-12: "Rated heating output (Pdesignh) 12 kW"
    )
    immersion_heater_kw: float = (
        7.0  # datasheet: 7 kW integrated electric heater, seven automatic steps
    )
    heating_capacity_range_kw: tuple[float, float] = (
        3.0,
        12.0,
    )  # datasheet "Heating capacity (PH)"

    rated_power_kw: tuple[float, float] = (3.0, 12.0)  # the PH modulation envelope, EN 14511
    typical_electrical_range_kw: tuple[float, float] = (1.04, 2.5)  # PE at the rating points
    modulation_range: tuple[int, int] = (70, 120)
    modulation_type: str = "inverter"

    typical_cop_range: tuple[float, float] = (3.75, 6.12)  # published COPs, 0/45 .. 10/35
    optimal_flow_delta: float = 25.0  # Can run lower flow temps
    cop_curve: dict[float, float] = None

    supports_aux_heating: bool = True
    supports_modulation: bool = True
    supports_weather_compensation: bool = True
    max_flow_temp: float = 65.0  # "compressor provides a supply temperature up to 65 C"
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
        self.cop_curve = seasonal_cop_proxy(self.datasheet_points, source_temp_c=0.0)

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

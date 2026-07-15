"""Base classes for heat pump model profiles.

Defines the abstract interface that all heat pump models must implement.
Model profiles contain manufacturer-specific characteristics for optimal
performance and validation.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..const import (
    DISPLAY_COP_CURVE_COLD_C,
    DISPLAY_COP_CURVE_SPAN_K,
    DISPLAY_COP_CURVE_TEMPS,
    DM_THRESHOLD_AUX_LIMIT,
)


@dataclass
class ValidationResult:
    """Result of power consumption validation."""

    valid: bool
    severity: str  # "info", "warning", "error"
    message: str
    suggestions: list[str]


@dataclass(frozen=True)
class RatingPoint:
    """One EN 14511 rating point, verbatim from the manufacturer - a measurement at a stated
    condition, not a fitted curve. `condition` is the datasheet's own string, and
    test_the_pump_models_match_their_datasheets reproduces the COP at every point.

    NOTE `source_temp_c` is the HEAT SOURCE temperature, which is NOT outdoor air for four of the
    five machines: A20(12) is 20 C extract air (exhaust-air pump), B0 is 0 C brine (ground-source),
    A7 is 7 C outdoor. Only an air/water pump like the F2040 has outdoor air as its source, so only
    for it is an outdoor-keyed curve meaningful.
    """

    condition: str  # verbatim from the datasheet, e.g. "A20(12)W35, 252 m3/h, min compressor freq"
    source_temp_c: float  # the HEAT SOURCE temperature at this point (A20 -> 20, B0 -> 0, A7 -> 7)
    flow_temp_c: float  # W35 -> 35.0
    heat_output_kw: float  # PH, the specified heating output
    cop: float
    # Ventilation rate is part of an exhaust-air pump's source condition, not a detail. The F750's
    # two minimum-frequency points differ only in airflow (108 vs 252 m3/h): more air -> higher
    # output AND higher COP. Treated as a load pair, efficiency appears to RISE with load (backwards)
    # and the fit extrapolates to COP 9.86 at full load, so the two must be told apart.
    airflow_m3h: float | None = None


def seasonal_cop_proxy(
    points: Sequence["RatingPoint"], source_temp_c: float | None = None
) -> dict[int, float]:
    """A DISPLAY-ONLY seasonal COP curve, interpolated between a machine's published extremes.

    NOTHING COMPUTES FROM THIS - it is a dashboard proxy. The simulator takes COP from
    `datasheet_points` via the exergy model, which uses SOURCE and FLOW temperatures, never the
    weather (four of the five machines do not have outdoor air as their heat source).

    `source_temp_c` filters to one source condition (how a brine machine is anchored on its two
    published W35/W45 COPs at 0 C). Left None, every published point is used.
    """
    cops = [
        point.cop
        for point in points
        if source_temp_c is None or point.source_temp_c == source_temp_c
    ]
    best, worst = max(cops), min(cops)
    return {
        temp: round(
            worst + (best - worst) * (temp - DISPLAY_COP_CURVE_COLD_C) / DISPLAY_COP_CURVE_SPAN_K, 2
        )
        for temp in DISPLAY_COP_CURVE_TEMPS
    }


@dataclass
class HeatPumpProfile(ABC):
    """Abstract base class for heat pump model profiles.

    Each heat pump model should subclass this and provide:
    - Power characteristics (rated power, modulation range)
    - Efficiency curves (COP vs outdoor/flow temp)
    - Optimization parameters (DM thresholds, cycling protection)
    - Validation logic (verify power consumption is normal)

    A profile deliberately does NOT calculate the flow temperature the house needs: that is a
    property of the HOUSE's emitters (type, sizing, design point), which a profile describing the
    PUMP cannot know. Flow temperature belongs to optimization/weather_layer.py, via the EN 442
    emitter law in utils/emitter.py.
    """

    # Identity
    model_name: str
    manufacturer: str
    model_type: str  # e.g., "F-series ASHP", "S-series GSHP"

    # Power characteristics
    rated_power_kw: tuple[float, float]  # (min, max) heat output
    typical_electrical_range_kw: tuple[float, float]  # Typical electrical consumption
    modulation_range: tuple[int, int]  # Hz or %
    modulation_type: str  # "inverter", "on_off", "staged"

    # Efficiency characteristics
    typical_cop_range: tuple[float, float]  # (min, max) COP
    optimal_flow_delta: float  # °C above outdoor for max efficiency
    cop_curve: dict[float, float]  # outdoor_temp → COP mapping

    # System capabilities
    supports_aux_heating: bool
    supports_modulation: bool
    supports_weather_compensation: bool
    max_flow_temp: float
    min_flow_temp: float

    # The simulator reads this so the plant model tracks what the integration believes.
    # It cannot do that while the profile restates the number, so it references it (F-076).
    dm_threshold_aux_swedish: float = DM_THRESHOLD_AUX_LIMIT

    # The DM at which THE PUMP ITSELF engages its additive heat, at factory settings - NIBE menu
    # 4.9.3 "start addition" (F-series) or "start diff additional heat" summed with the compressor
    # start (S-series/F11xx). A HARDWARE fact, distinct from DM_THRESHOLD_AUX_LIMIT (EffektGuard's
    # emergency floor, F-112): a real pump's elpatron fires HERE and works DM back up, so a plant
    # model that waits for -1500 misreports both aux energy and overshoot. Overridden per model.
    aux_start_dm: float = -700.0

    # Exhaust air heat pump features
    # Only EAHP models (F730, F750) support airflow optimization
    supports_exhaust_airflow: bool = False
    standard_airflow_m3h: float = 0.0  # Normal ventilation rate
    enhanced_airflow_m3h: float = 0.0  # Maximum ventilation rate

    # THE MANUFACTURER'S OWN MEASUREMENTS. See RatingPoint. Everything the simulator believes about
    # this machine's efficiency and its capacity is derived from these and from nothing else.
    datasheet_points: tuple[RatingPoint, ...] = field(default_factory=tuple)
    datasheet_source: str = ""  # the document these came out of. No source, no number.

    # The datasheet's own "Heating capacity (PH)" row, e.g. "3 - 12 kW". THIS IS NOT THE SAME THING
    # as the output at a rating point: NIBE's EN 14511 figures for the inverter machines are taken
    # at NOMINAL compressor frequency (50 Hz for the ground-source pumps), while this row is the
    # modulation envelope. An F1155-12 makes 5.06 kW at its 0/35 rating point and can reach 12.
    heating_capacity_range_kw: tuple[float, float] = (0.0, 0.0)

    # The immersion heater's DELIVERY SETTING, from the datasheet. 0.0 means the machine has none.
    # NIBE ships the F750/F730 with a 6.5 kW heater set to 3.5 kW at delivery, the F1155-12/S1155-12
    # with a 7 kW heater in seven automatic steps, and the F2040 with none (outdoor monobloc - its
    # electric addition lives in the paired indoor module, which this package does not model).
    immersion_heater_kw: float = 0.0

    # Pdesignh - the DESIGN HEAT LOAD this machine is certified for, from its own ErP declaration.
    # The only sourced way to size a simulated building: an oversized pump (e.g. a 12 kW GSHP on a
    # 6 kW house) cannot saturate, and a simulation that cannot saturate cannot test emergency
    # behaviour.
    design_heat_load_kw: float = 0.0

    # Pdesignh for the EN 14825 AVERAGE climate (design temperature -10 C). Tbiv and Psup below are
    # declared FOR THAT climate, so any statement that combines them must use this figure, not the
    # cold-climate Pdesignh.
    design_heat_load_average_kw: float = 0.0

    # Tbiv - the BIVALENT TEMPERATURE, from the ErP declaration. Below this outdoor temperature the
    # pump cannot meet the design heat load alone and supplementary heat is REQUIRED. This is the
    # design, not a defect: a correctly-sized Swedish ASHP is bivalent (F2040-8: Tbiv -9 C, 1.1 kW
    # supplementary). 0.0 = not declared (exhaust-air and ground-source machines are not bivalent in
    # the same sense - their heat source does not weaken with the weather).
    bivalent_temp_c: float = 0.0

    # Psup - the supplementary heat the ErP AVERAGE-climate declaration says this machine needs at
    # that climate's design point (-10 C). For the F2040-8: Pdesignh(avg) 8.2 kW with Psup 1.1 kW,
    # so the COMPRESSOR delivers 7.1 kW at -10 C - the only published statement about its capacity
    # below -7 C, and one COMPLETE declaration, not a splice of two.
    supplementary_heat_kw: float = 0.0

    @property
    def max_heat_output_kw(self) -> float:
        """The most heat this machine can make, from its own datasheet.

        NOT `rated_power_kw[1]`: the F750 carried 8.0 kW against a published maximum of 4.994. An
        exhaust-air pump's output is bounded by the ventilation air it breathes.
        """
        if self.heating_capacity_range_kw[1] > 0.0:
            return self.heating_capacity_range_kw[1]

        # The ErP declaration is also a published statement about the maximum. For the F2040-8, the
        # AVERAGE declaration (Pdesignh 8.2 kW, Psup 1.1 kW at -10 C) puts the COMPRESSOR at 7.1 kW -
        # above its coldest tabulated rating point (6.60 kW at -7 C), because inverter capacity rises
        # as the weather cools. Tbiv and Psup belong to the average climate, so it is the only
        # Pdesignh they may be combined with.
        published = max(point.heat_output_kw for point in self.datasheet_points)
        if self.design_heat_load_average_kw > 0.0 and self.supplementary_heat_kw > 0.0:
            published = max(
                published, self.design_heat_load_average_kw - self.supplementary_heat_kw
            )
        return published

    def rating_point_at(self, flow_temp_c: float) -> RatingPoint:
        """The published point closest to this flow temperature, at the highest output."""
        candidates = sorted(
            self.datasheet_points,
            key=lambda p: (abs(p.flow_temp_c - flow_temp_c), -p.heat_output_kw),
        )
        return candidates[0]

    @abstractmethod
    def validate_power_consumption(
        self,
        current_power_kw: float,
        outdoor_temp: float,
        flow_temp: float,
    ) -> ValidationResult:
        """Validate if current power consumption is normal.

        Args:
            current_power_kw: Current electrical consumption (kW)
            outdoor_temp: Outdoor temperature (°C)
            flow_temp: Flow temperature (°C)

        Returns:
            ValidationResult with status and recommendations
        """
        raise NotImplementedError

    def get_cop_at_temperature(self, outdoor_temp: float) -> float:
        """Get COP for given outdoor temperature using interpolation.

        Args:
            outdoor_temp: Outdoor temperature (°C)

        Returns:
            Estimated COP
        """
        temps = sorted(self.cop_curve.keys())

        if outdoor_temp >= temps[-1]:
            return self.cop_curve[temps[-1]]
        if outdoor_temp <= temps[0]:
            return self.cop_curve[temps[0]]

        # Linear interpolation
        for i in range(len(temps) - 1):
            if temps[i] <= outdoor_temp <= temps[i + 1]:
                t1, t2 = temps[i], temps[i + 1]
                cop1, cop2 = self.cop_curve[t1], self.cop_curve[t2]

                ratio = (outdoor_temp - t1) / (t2 - t1)
                return cop1 + (cop2 - cop1) * ratio

        return 3.0  # Conservative fallback

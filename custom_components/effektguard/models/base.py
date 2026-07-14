"""Base classes for heat pump model profiles.

Defines the abstract interface that all heat pump models must implement.
Model profiles contain manufacturer-specific characteristics for optimal
performance and validation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from ..const import DM_THRESHOLD_AUX_LIMIT


@dataclass
class ValidationResult:
    """Result of power consumption validation."""

    valid: bool
    severity: str  # "info", "warning", "error"
    message: str
    suggestions: list[str]


@dataclass(frozen=True)
class RatingPoint:
    """One EN 14511 rating point, exactly as the manufacturer publishes it.

    THIS EXISTS BECAUSE THE PERFORMANCE NUMBERS IN THIS PACKAGE WERE INVENTED.

    Every profile carried an outdoor-keyed `cop_curve` described in its own docstring as "Real-world
    COP curve (tested and validated)" and sourced to "NIBE F750 datasheet". It was neither. The F750
    and the F730 shipped byte-identical curves (5.0/4.5/4.0/3.5/3.0/2.7/2.3/2.0/1.8) despite being
    different machines, and the number 5.0 - labelled "Best COP" - appears nowhere in either
    datasheet. They were a template with the digits nudged, and the simulator computed a month of
    kWh and SEK from them.

    A rating point is not a curve. It is a measurement, taken at a stated condition, published by
    the people who built the machine. Carrying them verbatim means the fiction cannot be re-entered
    silently: `condition` is the datasheet's own string, and a test checks the model reproduces the
    COP at every one of them.

    NOTE `source_temp_c`: the temperature of the HEAT SOURCE, which is not the outdoor air for four
    of the five machines here. A20(12) is 20 C extract air (an exhaust-air pump breathes the house).
    B0 is 0 C brine (a ground-source pump does not care what the weather is doing). Only an
    air/water pump like the F2040 has outdoor air as its source, and only for it is an
    outdoor-keyed curve meaningful at all.
    """

    condition: str  # verbatim from the datasheet, e.g. "A20(12)W35, 252 m3/h, min compressor freq"
    source_temp_c: float  # the HEAT SOURCE temperature at this point (A20 -> 20, B0 -> 0, A7 -> 7)
    flow_temp_c: float  # W35 -> 35.0
    heat_output_kw: float  # PH, the specified heating output
    cop: float
    # For an exhaust-air pump the VENTILATION RATE is part of the source condition, not a detail.
    # The F750's two minimum-frequency points differ only in airflow (108 vs 252 m3/h): more air,
    # more source heat, higher output AND higher COP. Treating them as a load pair made efficiency
    # appear to RISE with compressor load, which is backwards, and the resulting fit extrapolated
    # to COP 9.86 at full load. They have to be told apart, so the airflow is carried.
    airflow_m3h: float | None = None


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

    # Optimization parameters (Swedish NIBE research)
    dm_threshold_start: float = -60  # Normal compressor start
    dm_threshold_extended: float = -240  # Extended runs acceptable
    dm_threshold_warning: float = -400  # Approaching danger
    dm_threshold_critical: float = -500  # Emergency recovery
    # The simulator reads this so the plant model tracks what the integration believes.
    # It cannot do that while the profile restates the number, so it references it (F-076).
    dm_threshold_aux_swedish: float = DM_THRESHOLD_AUX_LIMIT

    # Cycling protection
    min_runtime_minutes: int = 30
    min_rest_minutes: int = 10

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

    @property
    def max_heat_output_kw(self) -> float:
        """The most heat this machine can make, from its own datasheet.

        NOT `rated_power_kw[1]`, which was invented: the F750 carried 8.0 kW against a published
        maximum of 4.994, and the simulator used it as the compressor's capacity ceiling. An
        exhaust-air pump's output is bounded by the ventilation air it breathes, and no amount of
        naming a model "8 kW" changes that.
        """
        if self.heating_capacity_range_kw[1] > 0.0:
            return self.heating_capacity_range_kw[1]
        return max(point.heat_output_kw for point in self.datasheet_points)

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

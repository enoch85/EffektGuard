"""
Thermal Airflow Optimizer for Exhaust Air Heat Pumps

Original work based on thermodynamic first principles.
Calculates optimal airflow rates for exhaust air heat pump systems.

Physics basis:
- Heat extraction:  Q = ṁ × cp × ΔT
- Ventilation cost: the building must reheat every extra cubic metre from outdoor to indoor
- Energy balance:   Q_cond = P_el + Q_evap, so d(Q_cond) = d(Q_evap) = P_el × d(COP) at constant
                    electrical input. The extra extraction and the "COP improvement" are the same
                    joules; only one of them may be counted.

Net gain = (extra heat extracted) - (extra air to reheat), and it turns negative below an outdoor
temperature of (indoor - AIRFLOW_EVAPORATOR_TEMP_DROP):

| Outdoor °C | Net gain |
|------------|----------|
| +15        | +0.22 kW |
| +12        | +0.12 kW |
| +9         |   0.00   |  <- break-even
| +5         | -0.14 kW |
| 0          | -0.29 kW |
| -10        | -0.65 kW |

Enhancement therefore pays only in mild weather, and is a net thermal LOSS across the Swedish
heating season. The evaporator recovers only AIRFLOW_EVAPORATOR_TEMP_DROP from the extra air while
the building has to warm all of it the whole way.

Author: Original work
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

from homeassistant.util import dt as dt_util

from ..const import (
    AIRFLOW_AIR_DENSITY,
    AIRFLOW_COMPRESSOR_BASE_THRESHOLD,
    AIRFLOW_COMPRESSOR_SLOPE,
    AIRFLOW_DEFAULT_ENHANCED,
    AIRFLOW_DEFAULT_STANDARD,
    AIRFLOW_DEFICIT_LARGE_THRESHOLD,
    AIRFLOW_DEFICIT_MODERATE_THRESHOLD,
    AIRFLOW_DEFICIT_SMALL_THRESHOLD,
    AIRFLOW_DURATION_COLD_CAP,
    AIRFLOW_DURATION_COOL_CAP,
    AIRFLOW_DURATION_EXTREME_DEFICIT,
    AIRFLOW_DURATION_LARGE_DEFICIT,
    AIRFLOW_DURATION_MODERATE_DEFICIT,
    AIRFLOW_DURATION_SMALL_DEFICIT,
    AIRFLOW_EVAPORATOR_TEMP_DROP,
    AIRFLOW_INDOOR_DEFICIT_MIN,
    AIRFLOW_OUTDOOR_TEMP_MIN,
    AIRFLOW_SPECIFIC_HEAT,
    AIRFLOW_TEMP_COLD_THRESHOLD,
    AIRFLOW_TEMP_COOL_THRESHOLD,
    AIRFLOW_TREND_COOLING_THRESHOLD,
    AIRFLOW_TREND_WARMING_THRESHOLD,
    COMPRESSOR_HZ_MAX,
    COMPRESSOR_HZ_MIN,
)

if TYPE_CHECKING:
    from .prediction_layer import ThermalTrendDict


class FlowMode(Enum):
    """Airflow operating modes."""

    STANDARD = "standard"
    ENHANCED = "enhanced"


class ThermalState(NamedTuple):
    """Current thermal conditions for airflow evaluation."""

    temp_outdoor: float  # °C
    temp_indoor: float  # °C
    temp_target: float  # °C
    compressor_pct: float  # 0-100 (percentage of max frequency)
    trend_indoor: float  # °C/hour (negative = cooling down)


@dataclass
class FlowDecision:
    """Airflow recommendation with metadata.

    Attributes:
        mode: Standard or enhanced airflow mode
        duration_minutes: Recommended duration for enhanced mode (0 if standard)
        expected_gain_kw: Expected thermal gain from enhancement (kW)
        reason: Human-readable explanation of decision
        timestamp: When decision was made
    """

    mode: FlowMode
    duration_minutes: int
    expected_gain_kw: float
    reason: str
    timestamp: datetime | None = None

    def __post_init__(self):
        """Set timestamp if not provided.

        `dt_util.utcnow()`, never `datetime.now()`. Home Assistant works in aware UTC; a naive
        datetime cannot be compared with an aware one at all (TypeError), and if it could, the box
        runs UTC while `datetime.now()` returns local time - so the arithmetic would be wrong by the
        UTC offset, which in a Swedish summer is two hours.
        """
        if self.timestamp is None:
            self.timestamp = dt_util.utcnow()

    @property
    def should_enhance(self) -> bool:
        """Return True if enhanced airflow is recommended."""
        return self.mode == FlowMode.ENHANCED


def mass_flow_rate(volume_m3_per_hour: float) -> float:
    """Convert volumetric flow to mass flow rate (kg/s).

    Args:
        volume_m3_per_hour: Volumetric flow rate in m³/h

    Returns:
        Mass flow rate in kg/s
    """
    return (volume_m3_per_hour / 3600) * AIRFLOW_AIR_DENSITY


def ventilation_heat_loss(flow_m3h: float, temp_indoor: float, temp_outdoor: float) -> float:
    """Calculate heat lost through ventilation (kW).

    Cold outdoor air entering must be heated to indoor temp.

    Args:
        flow_m3h: Airflow rate in m³/h
        temp_indoor: Indoor temperature in °C
        temp_outdoor: Outdoor temperature in °C

    Returns:
        Heat loss in kW
    """
    m_dot = mass_flow_rate(flow_m3h)
    delta_t = temp_indoor - temp_outdoor
    return m_dot * AIRFLOW_SPECIFIC_HEAT * max(0, delta_t)


def evaporator_heat_extraction(
    flow_m3h: float, temp_drop: float = AIRFLOW_EVAPORATOR_TEMP_DROP
) -> float:
    """Calculate heat extracted from exhaust air (kW).

    Args:
        flow_m3h: Airflow rate in m³/h
        temp_drop: Temperature drop through evaporator in °C

    Returns:
        Heat extraction in kW
    """
    m_dot = mass_flow_rate(flow_m3h)
    return m_dot * AIRFLOW_SPECIFIC_HEAT * temp_drop


def calculate_net_thermal_gain(
    flow_standard: float,
    flow_enhanced: float,
    temp_indoor: float,
    temp_outdoor: float,
) -> float:
    """Calculate net thermal gain from enhanced airflow (kW).

    Net gain = (extra heat extracted at the evaporator) - (extra air the building must reheat)

    There is no third term. Extracting more heat from more air and "improving the COP" are not two
    benefits; they are the same joules described twice. The first law, in steady state, gives

        Q_cond = P_el + Q_evap

    and differentiating at constant electrical input gives

        d(Q_cond) = d(Q_evap) = P_el * d(COP)

    - an identity. Adding `P_el * d(COP)` to `d(Q_evap)` counts the same heat a second time.

    NIBE's S735 manual publishes four points at identical conditions (A20(12)W35, minimum
    compressor frequency) with exhaust airflow as the only variable. Over the 90 -> 252 m³/h step
    the measured heat-output rise is +0.410 kW, P_el*dCOP is +0.387 kW, and dQ_evap is +0.404 kW.
    One number, three ways.

    Consequence: enhancement pays only above an outdoor temperature of
    (indoor - AIRFLOW_EVAPORATOR_TEMP_DROP), around +9 °C. The evaporator recovers only
    AIRFLOW_EVAPORATOR_TEMP_DROP from the extra air, while the building must reheat every cubic
    metre of it all the way from outdoor to indoor. Below break-even - which is the whole Swedish
    heating season - enhancing is a net thermal LOSS, and this returns negative accordingly.

    Args:
        flow_standard: Standard airflow rate in m³/h
        flow_enhanced: Enhanced airflow rate in m³/h
        temp_indoor: Indoor temperature in °C
        temp_outdoor: Outdoor temperature in °C

    Returns:
        Net thermal gain in kW (positive = beneficial to enhance)
    """
    q_extract_std = evaporator_heat_extraction(flow_standard)
    q_extract_enh = evaporator_heat_extraction(flow_enhanced)
    delta_extraction = q_extract_enh - q_extract_std

    loss_std = ventilation_heat_loss(flow_standard, temp_indoor, temp_outdoor)
    loss_enh = ventilation_heat_loss(flow_enhanced, temp_indoor, temp_outdoor)
    delta_penalty = loss_enh - loss_std

    return delta_extraction - delta_penalty


def minimum_compressor_threshold(temp_outdoor: float) -> float:
    """Calculate minimum compressor % needed for enhanced flow to be beneficial.

    Colder outside → need higher compressor output to justify extra ventilation.

    Derived from break-even analysis:
    - At 0°C: ~50% compressor needed
    - At -10°C: ~75% compressor needed
    - Linear relationship between

    Args:
        temp_outdoor: Outdoor temperature in °C

    Returns:
        Minimum compressor percentage (0-100)
    """
    return max(
        AIRFLOW_COMPRESSOR_BASE_THRESHOLD,
        AIRFLOW_COMPRESSOR_BASE_THRESHOLD + AIRFLOW_COMPRESSOR_SLOPE * temp_outdoor,
    )


def calculate_duration(deficit: float, temp_outdoor: float) -> int:
    """Determine how long to run enhanced airflow (minutes).

    Based on:
    - Larger deficit → longer duration
    - Colder outside → shorter duration (diminishing returns)

    Args:
        deficit: Temperature deficit (target - current) in °C
        temp_outdoor: Outdoor temperature in °C

    Returns:
        Duration in minutes
    """
    if deficit < AIRFLOW_DEFICIT_SMALL_THRESHOLD:
        base_duration = AIRFLOW_DURATION_SMALL_DEFICIT
    elif deficit < AIRFLOW_DEFICIT_MODERATE_THRESHOLD:
        base_duration = AIRFLOW_DURATION_MODERATE_DEFICIT
    elif deficit < AIRFLOW_DEFICIT_LARGE_THRESHOLD:
        base_duration = AIRFLOW_DURATION_LARGE_DEFICIT
    else:
        base_duration = AIRFLOW_DURATION_EXTREME_DEFICIT

    # Reduce duration in cold conditions
    if temp_outdoor < AIRFLOW_TEMP_COLD_THRESHOLD:
        return min(base_duration, AIRFLOW_DURATION_COLD_CAP)
    if temp_outdoor < AIRFLOW_TEMP_COOL_THRESHOLD:
        return min(base_duration, AIRFLOW_DURATION_COOL_CAP)

    return base_duration


def evaluate_airflow(
    state: ThermalState,
    flow_standard: float = AIRFLOW_DEFAULT_STANDARD,
    flow_enhanced: float = AIRFLOW_DEFAULT_ENHANCED,
) -> FlowDecision:
    """Determine optimal airflow mode based on current thermal conditions.

    Args:
        state: Current thermal state
        flow_standard: Normal airflow (m³/h)
        flow_enhanced: Maximum airflow (m³/h)

    Returns:
        FlowDecision with recommended mode and duration
    """
    deficit = state.temp_target - state.temp_indoor

    # Hard limits - never enhance flow in these conditions
    if state.temp_outdoor < AIRFLOW_OUTDOOR_TEMP_MIN:
        return FlowDecision(
            FlowMode.STANDARD,
            0,
            0.0,
            f"Outdoor temp {state.temp_outdoor:.1f}°C too low - ventilation penalty exceeds gains",
        )

    if deficit < AIRFLOW_INDOOR_DEFICIT_MIN:
        # Negative deficit means indoor is above target, positive means below
        if deficit <= 0:
            reason = f"Already {abs(deficit):.1f}°C above target - no enhancement needed"
        else:
            reason = f"Only {deficit:.1f}°C below target - no enhancement needed"
        return FlowDecision(
            FlowMode.STANDARD,
            0,
            0.0,
            reason,
        )

    if state.trend_indoor > AIRFLOW_TREND_WARMING_THRESHOLD:
        return FlowDecision(
            FlowMode.STANDARD,
            0,
            0.0,
            f"Already warming (+{state.trend_indoor:.2f}°C/h) - let system stabilize",
        )

    # Check for cooling trend - if temperature is dropping, extra airflow isn't helping
    # This catches the case where compressor is maxed and enhanced ventilation
    # only brings in more cold air without additional heat production
    if state.trend_indoor < AIRFLOW_TREND_COOLING_THRESHOLD:
        return FlowDecision(
            FlowMode.STANDARD,
            0,
            0.0,
            f"Cooling ({state.trend_indoor:.2f}°C/h) - extra airflow making it worse",
        )

    # Check compressor threshold - minimum required
    min_compressor = minimum_compressor_threshold(state.temp_outdoor)
    if state.compressor_pct < min_compressor:
        return FlowDecision(
            FlowMode.STANDARD,
            0,
            0.0,
            f"Compressor {state.compressor_pct:.0f}% below {min_compressor:.0f}% threshold",
        )

    # Calculate expected benefit
    net_gain = calculate_net_thermal_gain(
        flow_standard, flow_enhanced, state.temp_indoor, state.temp_outdoor
    )

    if net_gain <= 0:
        return FlowDecision(
            FlowMode.STANDARD,
            0,
            net_gain,
            f"No thermal benefit from enhanced flow (gain: {net_gain:.2f} kW)",
        )

    # Enhanced flow is beneficial
    duration = calculate_duration(deficit, state.temp_outdoor)

    return FlowDecision(
        FlowMode.ENHANCED,
        duration,
        net_gain,
        f"Enhanced flow beneficial: +{net_gain:.2f} kW gain for {duration} min",
    )


def should_enhance_airflow(
    temp_outdoor: float,
    temp_indoor: float,
    temp_target: float,
    compressor_pct: float,
    trend_indoor: float = 0.0,
) -> tuple[bool, int]:
    """Simple interface: should we increase airflow?

    Convenience function for yes/no decisions.

    Args:
        temp_outdoor: Outdoor temperature in °C
        temp_indoor: Current indoor temperature in °C
        temp_target: Target indoor temperature in °C
        compressor_pct: Compressor frequency as percentage (0-100)
        trend_indoor: Indoor temperature trend in °C/hour (negative = cooling)

    Returns:
        Tuple of (should_enhance: bool, duration_minutes: int)

    Example:
        >>> enhance, duration = should_enhance_airflow(
        ...     temp_outdoor=0.0,
        ...     temp_indoor=20.5,
        ...     temp_target=21.0,
        ...     compressor_pct=80.0,
        ...     trend_indoor=-0.2
        ... )
        >>> if enhance:
        ...     print(f"Enhance airflow for {duration} minutes")
    """
    state = ThermalState(
        temp_outdoor=temp_outdoor,
        temp_indoor=temp_indoor,
        temp_target=temp_target,
        compressor_pct=compressor_pct,
        trend_indoor=trend_indoor,
    )
    decision = evaluate_airflow(state)
    return (decision.should_enhance, decision.duration_minutes)


class AirflowOptimizer:
    """Stateful airflow optimizer for integration with EffektGuard.

    Maintains enhancement state and provides decision history.
    Integrates with NIBE heat pump data from coordinator.

    Attributes:
        flow_standard: Standard airflow rate in m³/h
        flow_enhanced: Enhanced airflow rate in m³/h
        current_decision: Most recent flow decision
    """

    def __init__(
        self,
        flow_standard: float = AIRFLOW_DEFAULT_STANDARD,
        flow_enhanced: float = AIRFLOW_DEFAULT_ENHANCED,
    ):
        """Initialize airflow optimizer.

        Args:
            flow_standard: Normal airflow rate in m³/h (default: 150 for F750)
            flow_enhanced: Maximum airflow rate in m³/h (default: 252 for F750)
        """
        self.flow_standard = flow_standard
        self.flow_enhanced = flow_enhanced
        self.current_decision: FlowDecision | None = None

    def evaluate(
        self,
        temp_outdoor: float,
        temp_indoor: float,
        temp_target: float,
        compressor_pct: float,
        trend_indoor: float = 0.0,
    ) -> FlowDecision:
        """Evaluate current conditions and update decision.

        Args:
            temp_outdoor: Outdoor temperature in °C
            temp_indoor: Current indoor temperature in °C
            temp_target: Target indoor temperature in °C
            compressor_pct: Compressor frequency as percentage (0-100)
            trend_indoor: Indoor temperature trend in °C/hour

        Returns:
            FlowDecision with recommended mode and duration
        """
        state = ThermalState(
            temp_outdoor=temp_outdoor,
            temp_indoor=temp_indoor,
            temp_target=temp_target,
            compressor_pct=compressor_pct,
            trend_indoor=trend_indoor,
        )

        decision = evaluate_airflow(state, self.flow_standard, self.flow_enhanced)

        # Update state
        self.current_decision = decision

        return decision

    def evaluate_from_nibe(
        self,
        nibe_data,
        target_temp: float,
        thermal_trend: ThermalTrendDict | dict[str, float] | None = None,
    ) -> FlowDecision:
        """Evaluate using NIBE adapter data directly.

        Convenience method for coordinator integration.

        Args:
            nibe_data: NibeData from NIBE adapter
            target_temp: Target indoor temperature
            thermal_trend: Thermal trend dict from predictor (optional)

        Returns:
            FlowDecision with recommended mode and duration
        """
        # Extract compressor percentage from Hz
        # NIBE inverter compressors run 20-120 Hz
        # Convert to 0-100% capacity for threshold calculations
        compressor_hz = getattr(nibe_data, "compressor_hz", 0) or 0
        hz_range = COMPRESSOR_HZ_MAX - COMPRESSOR_HZ_MIN  # 120 - 20 = 100 Hz operating range
        compressor_pct = min(
            100.0, max(0.0, (compressor_hz - COMPRESSOR_HZ_MIN) / hz_range * 100.0)
        )

        # Get indoor trend rate if available
        trend_indoor = 0.0
        if thermal_trend and isinstance(thermal_trend, dict):
            trend_indoor = thermal_trend.get("rate_per_hour", 0.0)

        return self.evaluate(
            temp_outdoor=nibe_data.outdoor_temp,
            temp_indoor=nibe_data.indoor_temp,
            temp_target=target_temp,
            compressor_pct=compressor_pct,
            trend_indoor=trend_indoor,
        )

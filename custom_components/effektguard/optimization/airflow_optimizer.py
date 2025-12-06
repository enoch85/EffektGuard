"""
Thermal Airflow Optimizer for Exhaust Air Heat Pumps

Original work based on thermodynamic first principles.
Calculates optimal airflow rates for exhaust air heat pump systems.

Physics basis:
- Heat extraction: Q = ṁ × cp × ΔT
- COP relationship: COP ∝ T_evap / (T_cond - T_evap)
- Energy balance: Net gain = Heat gain - Ventilation penalty

When Enhanced Airflow Helps:
| Outdoor °C | Min Compressor % | Expected Gain |
|------------|-----------------|---------------|
| +10        | 50%             | +1.3 kW       |
| 0          | 50%             | +0.9 kW       |
| -5         | 62%             | +0.7 kW       |
| -10        | 75%             | +0.4 kW       |
| < -15      | Don't enhance   | Negative      |

Author: Original work
License: MIT
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import NamedTuple

from ..const import (
    AIRFLOW_AIR_DENSITY,
    AIRFLOW_BASE_COP,
    AIRFLOW_COMPRESSOR_BASE_THRESHOLD,
    AIRFLOW_COMPRESSOR_INPUT_KW,
    AIRFLOW_COMPRESSOR_SLOPE,
    AIRFLOW_COP_IMPROVEMENT_FACTOR,
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
    AIRFLOW_TREND_WARMING_THRESHOLD,
)


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
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()

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


def estimate_cop_improvement(base_cop: float = AIRFLOW_BASE_COP) -> float:
    """Estimate COP with enhanced airflow.

    More air → warmer evaporator → better COP.
    Empirically ~20% improvement at enhanced flow.

    Args:
        base_cop: Base COP without enhancement

    Returns:
        Enhanced COP
    """
    return base_cop * AIRFLOW_COP_IMPROVEMENT_FACTOR


def calculate_net_thermal_gain(
    flow_standard: float,
    flow_enhanced: float,
    temp_indoor: float,
    temp_outdoor: float,
    compressor_input_kw: float = AIRFLOW_COMPRESSOR_INPUT_KW,
) -> float:
    """Calculate net thermal gain from enhanced airflow (kW).

    Net Benefit = (Extra heat extracted) + (COP improvement) - (Ventilation penalty)

    Args:
        flow_standard: Standard airflow rate in m³/h
        flow_enhanced: Enhanced airflow rate in m³/h
        temp_indoor: Indoor temperature in °C
        temp_outdoor: Outdoor temperature in °C
        compressor_input_kw: Compressor electrical input in kW

    Returns:
        Net thermal gain in kW (positive = beneficial to enhance)
    """
    # Additional heat extraction
    q_extract_std = evaporator_heat_extraction(flow_standard)
    q_extract_enh = evaporator_heat_extraction(flow_enhanced)
    delta_extraction = q_extract_enh - q_extract_std

    # COP improvement benefit
    cop_std = AIRFLOW_BASE_COP
    cop_enh = estimate_cop_improvement(cop_std)
    heat_output_std = compressor_input_kw * cop_std
    heat_output_enh = compressor_input_kw * cop_enh
    delta_cop_benefit = heat_output_enh - heat_output_std

    # Ventilation penalty
    loss_std = ventilation_heat_loss(flow_standard, temp_indoor, temp_outdoor)
    loss_enh = ventilation_heat_loss(flow_enhanced, temp_indoor, temp_outdoor)
    delta_penalty = loss_enh - loss_std

    return delta_extraction + delta_cop_benefit - delta_penalty


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
    elif temp_outdoor < AIRFLOW_TEMP_COOL_THRESHOLD:
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

    # Check compressor threshold
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
        enhancement_active: Whether enhanced mode is currently active
        enhancement_end_time: When current enhancement should end
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
        self.enhancement_active = False
        self.enhancement_end_time: datetime | None = None
        self._decision_history: list[FlowDecision] = []

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

        # Maintain history (keep last 24 hours worth at 5-min intervals = 288 entries)
        self._decision_history.append(decision)
        if len(self._decision_history) > 288:
            self._decision_history = self._decision_history[-288:]

        return decision

    def evaluate_from_nibe(
        self,
        nibe_data,
        target_temp: float,
        thermal_trend: dict | None = None,
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
        # NIBE F750 typically runs 20-100 Hz, normalize to 0-100%
        compressor_hz = getattr(nibe_data, "compressor_hz", 0) or 0
        compressor_pct = min(100.0, max(0.0, compressor_hz))

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

    def get_enhancement_stats(self) -> dict:
        """Get statistics about enhancement recommendations.

        Returns:
            Dictionary with enhancement statistics
        """
        if not self._decision_history:
            return {
                "total_decisions": 0,
                "enhance_recommendations": 0,
                "enhance_percentage": 0.0,
                "average_gain_kw": 0.0,
            }

        enhance_decisions = [d for d in self._decision_history if d.should_enhance]
        enhance_count = len(enhance_decisions)
        total = len(self._decision_history)

        return {
            "total_decisions": total,
            "enhance_recommendations": enhance_count,
            "enhance_percentage": (enhance_count / total) * 100 if total > 0 else 0.0,
            "average_gain_kw": (
                sum(d.expected_gain_kw for d in enhance_decisions) / enhance_count
                if enhance_count > 0
                else 0.0
            ),
        }


if __name__ == "__main__":
    # Example: Your scenario
    state = ThermalState(
        temp_outdoor=0.0,
        temp_indoor=20.5,
        temp_target=21.0,
        compressor_pct=80.0,  # 80 Hz ≈ 80% of ~100 Hz max
        trend_indoor=-0.2,
    )

    decision = evaluate_airflow(state)
    print(f"Mode: {decision.mode.value}")
    print(f"Duration: {decision.duration_minutes} minutes")
    print(f"Expected gain: {decision.expected_gain_kw:.2f} kW")
    print(f"Reason: {decision.reason}")

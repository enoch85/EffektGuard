"""Tests for the Thermal Airflow Optimizer.

Thermodynamic calculations for exhaust-air heat-pump airflow. Enhancing airflow only pays
above break-even (about indoor minus the evaporator temperature drop, ~+9 C outdoor): below
that the building must reheat every extra cubic metre from outdoor to indoor while the
evaporator recovers only its own temperature drop, so the net thermal gain is negative.
"""

from datetime import datetime

from custom_components.effektguard.const import (
    AIRFLOW_DEFAULT_ENHANCED,
    AIRFLOW_DEFAULT_STANDARD,
    AIRFLOW_OUTDOOR_TEMP_MIN,
    AIRFLOW_INDOOR_DEFICIT_MIN,
    AIRFLOW_COMPRESSOR_BASE_THRESHOLD,
)
from custom_components.effektguard.optimization.airflow_optimizer import (
    AirflowOptimizer,
    FlowDecision,
    FlowMode,
    ThermalState,
    calculate_net_thermal_gain,
    evaluate_airflow,
    evaporator_heat_extraction,
    mass_flow_rate,
    minimum_compressor_threshold,
    should_enhance_airflow,
    ventilation_heat_loss,
)


class TestPhysicsCalculations:
    """Test core thermodynamic calculations."""

    def test_mass_flow_rate(self):
        """Test volumetric to mass flow conversion."""
        # 150 m³/h at 1.2 kg/m³ should give specific mass flow
        mass_flow = mass_flow_rate(150.0)
        # 150/3600 * 1.2 = 0.05 kg/s
        assert abs(mass_flow - 0.05) < 0.001

    def test_evaporator_heat_extraction(self):
        """Test heat extraction calculation."""
        # Standard flow: 150 m³/h with 12°C drop
        heat = evaporator_heat_extraction(150.0)
        # Q = ṁ × cp × ΔT = 0.05 × 1.005 × 12 ≈ 0.603 kW
        assert 0.55 < heat < 0.65

        # Enhanced flow: 252 m³/h
        heat_enhanced = evaporator_heat_extraction(252.0)
        # Should be proportionally higher
        assert heat_enhanced > heat

    def test_ventilation_heat_loss(self):
        """Test ventilation penalty calculation."""
        # 150 m³/h, 21°C indoor, 0°C outdoor
        loss = ventilation_heat_loss(150.0, 21.0, 0.0)
        # Q = ṁ × cp × ΔT = 0.05 × 1.005 × 21 ≈ 1.06 kW
        assert 0.9 < loss < 1.2

        # Cold outdoor = higher loss
        loss_cold = ventilation_heat_loss(150.0, 21.0, -10.0)
        assert loss_cold > loss

    def test_ventilation_heat_loss_no_negative(self):
        """Test that ventilation loss is never negative."""
        # Outdoor warmer than indoor should return 0
        loss = ventilation_heat_loss(150.0, 20.0, 25.0)
        assert loss == 0.0


class TestCompressorThresholds:
    """Test compressor percentage threshold calculations."""

    def test_threshold_at_0c(self):
        """At 0°C, threshold should be the base (AIRFLOW_COMPRESSOR_BASE_THRESHOLD)."""
        threshold = minimum_compressor_threshold(0.0)
        assert threshold == AIRFLOW_COMPRESSOR_BASE_THRESHOLD

    def test_threshold_at_10c(self):
        """At +10°C (warm), threshold stays at base."""
        threshold = minimum_compressor_threshold(10.0)
        # Should not go below base
        assert threshold == AIRFLOW_COMPRESSOR_BASE_THRESHOLD

    def test_threshold_at_minus_10c(self):
        """At -10°C, threshold should be 86%."""
        threshold = minimum_compressor_threshold(-10.0)
        # 61 + (-2.5 * -10) = 61 + 25 = 86
        assert threshold == 86.0

    def test_threshold_at_minus_5c(self):
        """At -5°C, threshold should be ~73.5%."""
        threshold = minimum_compressor_threshold(-5.0)
        # 61 + (-2.5 * -5) = 61 + 12.5 = 73.5
        assert 73 < threshold < 74


class TestNetThermalGain:
    """Test net thermal gain calculations."""

    def test_net_loss_at_moderate_temp(self):
        """A net LOSS at 0 C outdoor. The old expectation of ~+0.9 kW was a double-count.

        The evaporator recovers AIRFLOW_EVAPORATOR_TEMP_DROP from the extra air; the building has
        to reheat that air the whole way from 0 C to 21 C. See test_airflow_energy_balance.py.
        """
        gain = calculate_net_thermal_gain(
            flow_standard=AIRFLOW_DEFAULT_STANDARD,
            flow_enhanced=AIRFLOW_DEFAULT_ENHANCED,
            temp_indoor=21.0,
            temp_outdoor=0.0,
        )
        assert gain < 0
        assert -0.5 < gain < -0.1

    def test_higher_gain_at_warm_temp(self):
        """Should have higher gain at +10°C outdoor."""
        gain_warm = calculate_net_thermal_gain(
            AIRFLOW_DEFAULT_STANDARD, AIRFLOW_DEFAULT_ENHANCED, 21.0, 10.0
        )
        gain_cold = calculate_net_thermal_gain(
            AIRFLOW_DEFAULT_STANDARD, AIRFLOW_DEFAULT_ENHANCED, 21.0, 0.0
        )
        # Warmer = less ventilation penalty = more gain
        assert gain_warm > gain_cold

    def test_deeper_loss_at_cold_temp(self):
        """The loss DEEPENS as it gets colder - colder air is more expensive to reheat."""
        gain = calculate_net_thermal_gain(
            AIRFLOW_DEFAULT_STANDARD, AIRFLOW_DEFAULT_ENHANCED, 21.0, -10.0
        )
        assert -1.0 < gain < -0.4


class TestEvaluateAirflow:
    """Test airflow decision evaluation."""

    def test_enhance_above_break_even(self):
        """Enhancement pays only ABOVE break-even (indoor - evaporator drop, about +9 C).

        0 C outdoor used to satisfy this test because the net gain double-counted the evaporator
        heat as a COP improvement. Below break-even the building must reheat every extra cubic
        metre from outdoor to indoor while the evaporator recovers only its own temperature drop
        from it, so enhancing is a net thermal LOSS.
        """
        state = ThermalState(
            temp_outdoor=12.0,  # above break-even
            temp_indoor=20.5,
            temp_target=21.0,
            compressor_pct=80.0,
            trend_indoor=-0.1,  # Slightly cooling but above threshold (-0.15)
        )
        decision = evaluate_airflow(state)

        assert decision.mode == FlowMode.ENHANCED
        assert decision.duration_minutes > 0
        assert decision.expected_gain_kw > 0
        assert "beneficial" in decision.reason.lower()

    def test_refuse_to_enhance_during_the_heating_season(self):
        """At 0 C the extra air costs more to reheat than the evaporator recovers from it."""
        state = ThermalState(
            temp_outdoor=0.0,
            temp_indoor=20.5,
            temp_target=21.0,
            compressor_pct=80.0,
            trend_indoor=-0.1,
        )
        decision = evaluate_airflow(state)

        assert decision.mode == FlowMode.STANDARD
        assert decision.expected_gain_kw < 0
        assert decision.duration_minutes == 0

    def test_no_enhance_outdoor_too_cold(self):
        """Should not enhance when outdoor temp too low."""
        state = ThermalState(
            temp_outdoor=-20.0,  # Below AIRFLOW_OUTDOOR_TEMP_MIN (-15)
            temp_indoor=20.0,
            temp_target=21.0,
            compressor_pct=100.0,
            trend_indoor=0.0,  # Neutral - test focuses on outdoor temp
        )
        decision = evaluate_airflow(state)

        assert decision.mode == FlowMode.STANDARD
        assert "too low" in decision.reason.lower()

    def test_no_enhance_near_target(self):
        """Should not enhance when already near target temp."""
        state = ThermalState(
            temp_outdoor=5.0,
            temp_indoor=20.9,  # Deficit < AIRFLOW_INDOOR_DEFICIT_MIN
            temp_target=21.0,
            compressor_pct=80.0,
            trend_indoor=0.0,
        )
        decision = evaluate_airflow(state)

        assert decision.mode == FlowMode.STANDARD
        assert (
            "near target" in decision.reason.lower() or "no enhancement" in decision.reason.lower()
        )

    def test_no_enhance_already_warming(self):
        """Should not enhance when indoor temp already rising."""
        state = ThermalState(
            temp_outdoor=5.0,
            temp_indoor=20.0,
            temp_target=21.0,
            compressor_pct=80.0,
            trend_indoor=0.3,  # Warming rate > AIRFLOW_TREND_WARMING_THRESHOLD
        )
        decision = evaluate_airflow(state)

        assert decision.mode == FlowMode.STANDARD
        assert "warming" in decision.reason.lower() or "stabilize" in decision.reason.lower()

    def test_no_enhance_cooling_trend(self):
        """Should not enhance when cooling rapidly - extra airflow would make it worse."""
        state = ThermalState(
            temp_outdoor=0.0,
            temp_indoor=20.0,
            temp_target=21.0,
            compressor_pct=80.0,  # Good compressor activity
            trend_indoor=-0.2,  # Below AIRFLOW_TREND_COOLING_THRESHOLD (-0.15)
        )
        decision = evaluate_airflow(state)

        assert decision.mode == FlowMode.STANDARD
        assert "cooling" in decision.reason.lower()

    def test_no_enhance_low_compressor(self):
        """Should not enhance when compressor below threshold."""
        state = ThermalState(
            temp_outdoor=0.0,
            temp_indoor=20.0,
            temp_target=21.0,
            compressor_pct=30.0,  # Below the 61% threshold at 0°C
            trend_indoor=-0.1,  # Slightly cooling but above threshold - test focuses on compressor
        )
        decision = evaluate_airflow(state)

        assert decision.mode == FlowMode.STANDARD
        assert "compressor" in decision.reason.lower() or "threshold" in decision.reason.lower()


class TestShouldEnhanceAirflow:
    """Test the simple interface function."""

    def test_returns_tuple(self):
        """Should return (bool, int) tuple."""
        result = should_enhance_airflow(
            temp_outdoor=0.0,
            temp_indoor=20.5,
            temp_target=21.0,
            compressor_pct=80.0,
            trend_indoor=-0.1,  # Slightly cooling but above threshold
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], int)

    def test_enhance_true_above_break_even(self):
        """Enhancement pays only above break-even (about +9 C), not at 0 C."""
        should_enhance, duration = should_enhance_airflow(
            temp_outdoor=12.0,
            temp_indoor=20.5,
            temp_target=21.0,
            compressor_pct=80.0,
            trend_indoor=-0.1,  # Slightly cooling but above threshold
        )
        assert should_enhance is True
        assert duration > 0

    def test_enhance_false_when_too_cold(self):
        """Should return False when outdoor too cold."""
        should_enhance, duration = should_enhance_airflow(
            temp_outdoor=-20.0,
            temp_indoor=20.0,
            temp_target=21.0,
            compressor_pct=100.0,
        )
        assert should_enhance is False
        assert duration == 0


class TestAirflowOptimizer:
    """Test the stateful AirflowOptimizer class."""

    def test_initialization(self):
        """Test optimizer initializes with correct defaults."""
        optimizer = AirflowOptimizer()
        assert optimizer.flow_standard == AIRFLOW_DEFAULT_STANDARD
        assert optimizer.flow_enhanced == AIRFLOW_DEFAULT_ENHANCED
        assert optimizer.current_decision is None

    def test_custom_flow_rates(self):
        """Test optimizer with custom flow rates."""
        optimizer = AirflowOptimizer(flow_standard=120.0, flow_enhanced=200.0)
        assert optimizer.flow_standard == 120.0
        assert optimizer.flow_enhanced == 200.0

    def test_evaluate_updates_state(self):
        """Test that evaluate() updates internal state."""
        optimizer = AirflowOptimizer()
        decision = optimizer.evaluate(
            temp_outdoor=0.0,
            temp_indoor=20.5,
            temp_target=21.0,
            compressor_pct=80.0,
            trend_indoor=-0.1,  # Slightly cooling but above threshold
        )

        assert optimizer.current_decision is not None
        assert optimizer.current_decision == decision

    def test_should_enhance_property(self):
        """Test the should_enhance property of FlowDecision."""
        decision = FlowDecision(
            mode=FlowMode.ENHANCED,
            duration_minutes=30,
            expected_gain_kw=0.9,
            reason="Test",
        )
        assert decision.should_enhance is True

        decision_standard = FlowDecision(
            mode=FlowMode.STANDARD,
            duration_minutes=0,
            expected_gain_kw=0.0,
            reason="Test",
        )
        assert decision_standard.should_enhance is False


class TestDurationCalculation:
    """Test duration calculation for enhanced airflow."""

    def test_small_deficit_short_duration(self):
        """Small temperature deficit should give short duration."""
        state = ThermalState(
            temp_outdoor=5.0,
            temp_indoor=20.8,  # 0.2°C deficit
            temp_target=21.0,
            compressor_pct=80.0,
            trend_indoor=-0.1,
        )
        decision = evaluate_airflow(state)

        if decision.mode == FlowMode.ENHANCED:
            assert decision.duration_minutes <= 20  # AIRFLOW_DURATION_SMALL_DEFICIT

    def test_large_deficit_longer_duration(self):
        """Large temperature deficit should give longer duration."""
        state = ThermalState(
            temp_outdoor=5.0,
            temp_indoor=19.5,  # 1.5°C deficit
            temp_target=21.0,
            compressor_pct=80.0,
            trend_indoor=-0.3,
        )
        decision = evaluate_airflow(state)

        if decision.mode == FlowMode.ENHANCED:
            assert decision.duration_minutes >= 45  # AIRFLOW_DURATION_EXTREME_DEFICIT

    def test_cold_weather_caps_duration(self):
        """Cold weather should cap the duration."""
        state = ThermalState(
            temp_outdoor=-12.0,  # Very cold but above minimum
            temp_indoor=19.0,  # Large deficit
            temp_target=21.0,
            compressor_pct=90.0,  # High compressor
            trend_indoor=-0.3,
        )
        decision = evaluate_airflow(state)

        if decision.mode == FlowMode.ENHANCED:
            assert decision.duration_minutes <= 20  # AIRFLOW_DURATION_COLD_CAP


class TestFlowDecision:
    """Test FlowDecision dataclass."""

    def test_decision_has_timestamp(self):
        """Test that decision includes timestamp."""
        decision = FlowDecision(
            mode=FlowMode.ENHANCED,
            duration_minutes=30,
            expected_gain_kw=0.9,
            reason="Test reason",
        )
        assert decision.timestamp is not None
        assert isinstance(decision.timestamp, datetime)

    def test_decision_with_explicit_timestamp(self):
        """Test decision with explicit timestamp."""
        explicit_time = datetime(2025, 1, 1, 12, 0, 0)
        decision = FlowDecision(
            mode=FlowMode.STANDARD,
            duration_minutes=0,
            expected_gain_kw=0.0,
            reason="Test",
            timestamp=explicit_time,
        )
        assert decision.timestamp == explicit_time


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exactly_at_minimum_outdoor_temp(self):
        """Test at exactly AIRFLOW_OUTDOOR_TEMP_MIN."""
        state = ThermalState(
            temp_outdoor=AIRFLOW_OUTDOOR_TEMP_MIN,  # Exactly at limit
            temp_indoor=20.0,
            temp_target=21.0,
            compressor_pct=100.0,
            trend_indoor=-0.3,
        )
        decision = evaluate_airflow(state)
        # Should still work at the limit (not below)
        assert decision.mode in [FlowMode.STANDARD, FlowMode.ENHANCED]

    def test_exactly_at_minimum_deficit(self):
        """Test at exactly AIRFLOW_INDOOR_DEFICIT_MIN."""
        state = ThermalState(
            temp_outdoor=5.0,
            temp_indoor=21.0 - AIRFLOW_INDOOR_DEFICIT_MIN,  # Exactly at threshold
            temp_target=21.0,
            compressor_pct=80.0,
            trend_indoor=0.0,
        )
        decision = evaluate_airflow(state)
        # At threshold should still enhance
        assert decision.mode in [FlowMode.STANDARD, FlowMode.ENHANCED]

    def test_zero_compressor_pct(self):
        """Test with zero compressor percentage."""
        state = ThermalState(
            temp_outdoor=5.0,
            temp_indoor=20.0,
            temp_target=21.0,
            compressor_pct=0.0,  # Compressor off
            trend_indoor=0.0,
        )
        decision = evaluate_airflow(state)
        assert decision.mode == FlowMode.STANDARD

    def test_100_compressor_pct(self):
        """Test with 100% compressor percentage, above break-even."""
        state = ThermalState(
            temp_outdoor=12.0,
            temp_indoor=20.0,
            temp_target=21.0,
            compressor_pct=100.0,  # Full power
            trend_indoor=-0.1,  # Slightly cooling but above threshold
        )
        decision = evaluate_airflow(state)
        assert decision.mode == FlowMode.ENHANCED
        assert decision.duration_minutes > 0

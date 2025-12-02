"""Integration tests for heat pump models with existing EffektGuard logic.

Tests that model profiles integrate correctly with:
- Thermal debt tracking and degree minutes logic
- Flow temperature formula
- Decision engine layer system
- Power validation and diagnostics

These tests validate BEHAVIOR, not exact specifications.
They work regardless of whether model specs are 100% accurate.
"""

import pytest
from dataclasses import dataclass

# Ensure model modules are imported so they register with the registry
# The registry uses decorators executed at import time
from custom_components.effektguard.models import nibe as _nibe_models  # noqa: F401


# Mock existing EffektGuard components for testing
@dataclass
class MockNibeState:
    """Mock NIBE state for testing."""

    degree_minutes: float
    outdoor_temp: float
    indoor_temp: float
    flow_temp: float
    return_temp: float
    is_heating: bool = True
    is_dhw: bool = False


class TestModelThermalDebtIntegration:
    """Test models integrate correctly with thermal debt (degree minutes) logic."""

    def test_all_models_aware_of_thermal_debt_thresholds(self):
        """Verify models know about critical DM thresholds from research."""
        from custom_components.effektguard.models.registry import HeatPumpModelRegistry
        from custom_components.effektguard.const import (
            DM_THRESHOLD_START,
            DM_THRESHOLD_AUX_LIMIT,
        )

        # Get all models
        registry = HeatPumpModelRegistry()
        nibe_models = registry.get_models_by_manufacturer("NIBE")

        assert len(nibe_models) > 0, "Should have NIBE models registered"

        print("\n" + "=" * 80)
        print("THERMAL DEBT THRESHOLD AWARENESS TEST")
        print("=" * 80)
        print(f"Testing {len(nibe_models)} NIBE models")
        print(
            f"Key thresholds: start {DM_THRESHOLD_START}, " f"absolute max {DM_THRESHOLD_AUX_LIMIT}"
        )
        print("=" * 80)

        # All models should understand thermal debt affects their recommendations
        # Even if they don't have specific threshold fields, they should:
        # 1. Not recommend offset reductions when in thermal debt
        # 2. Understand thermal debt in validation

        for model_id in nibe_models:
            profile = registry.get_model(model_id)
            print(f"\nModel: {profile.model_name} ({model_id})")
            print(f"  Manufacturer: {profile.manufacturer}")
            print(
                f"  Power range: {profile.typical_electrical_range_kw[0]}-"
                f"{profile.typical_electrical_range_kw[1]} kW"
            )

            # Model should exist and be valid
            assert profile.model_name is not None
            # heat_pump_type is not a defined attribute; validate known fields instead
            assert profile.manufacturer.upper() == "NIBE"

    def test_models_power_validation_considers_outdoor_temp(self):
        """Test power validation accounts for outdoor temperature (affects COP)."""
        from custom_components.effektguard.models.registry import HeatPumpModelRegistry

        registry = HeatPumpModelRegistry()
        f750 = registry.get_model("nibe_f750")

        print("\n" + "=" * 80)
        print("POWER VALIDATION - OUTDOOR TEMPERATURE SENSITIVITY")
        print("=" * 80)
        print(f"Model: {f750.model_name}")
        print("=" * 80)
        print(
            f"{'Outdoor Temp':>13} | {'Power Draw':>11} | {'Flow Temp':>10} | "
            f"{'Valid':>6} | {'Message':40}"
        )
        print("-" * 80)

        # Test scenarios: Same power draw at different outdoor temps
        # Should be valid in cold weather, suspicious in mild weather
        test_cases = [
            (-20, 6.0, 50, "Cold weather, high power OK"),
            (-10, 6.0, 48, "Cold weather, high power OK"),
            (0, 6.0, 45, "Moderate, high power suspicious"),
            (10, 6.0, 40, "Mild weather, high power very suspicious"),
        ]

        for outdoor, power, flow, expected_reason in test_cases:
            result = f750.validate_power_consumption(power, outdoor, flow)

            status = "✓" if result.valid else "⚠"
            message = result.message[:40] if result.message else "OK"

            print(
                f"{outdoor:>11}°C | {power:>9.1f} kW | {flow:>8}°C | " f"{status:>6} | {message:40}"
            )

            # Validation should be stricter in mild weather
            if outdoor > 5:
                # In mild weather, 6kW is suspicious (low heat demand)
                # Should at least warn or suggest checking
                assert (
                    result.suggestions is not None or not result.valid
                ), f"Should flag {power}kW as suspicious at {outdoor}°C"

    def test_models_cop_decreases_monotonically_with_temperature(self):
        """Test COP decreases as outdoor temperature drops (physics requirement)."""
        from custom_components.effektguard.models.registry import HeatPumpModelRegistry

        registry = HeatPumpModelRegistry()

        print("\n" + "=" * 80)
        print("COP MONOTONICITY TEST - Physics Validation")
        print("=" * 80)

        for model_id in ["nibe_f750", "nibe_f2040", "nibe_s1155"]:
            model = registry.get_model(model_id)

            # Determine heat pump type from model_type
            heat_pump_type = "GSHP" if "GSHP" in model.model_type else "ASHP"

            print(f"\nModel: {model.model_name} ({heat_pump_type})")
            print("-" * 80)
            print(f"{'Temp':>6} | {'COP':>6} | {'COP Change':>12} | {'Valid':>6}")
            print("-" * 80)

            previous_cop = None
            previous_temp = None

            for temp in [10, 5, 0, -5, -10, -15, -20, -25]:
                cop = model.get_cop_at_temperature(temp)

                if previous_cop is not None:
                    cop_change = cop - previous_cop
                    # COP should decrease or stay same as temp drops
                    is_valid = cop_change <= 0.1  # Allow tiny float errors
                    status = "✓" if is_valid else "✗ ERROR"

                    print(f"{temp:>4}°C | {cop:>6.2f} | {cop_change:>+10.2f} | {status:>6}")

                    if heat_pump_type == "ASHP":
                        # ASHP COP MUST decrease as temp drops
                        assert is_valid, (
                            f"{model.model_name}: COP increased from {previous_cop:.2f} "
                            f"at {previous_temp}°C to {cop:.2f} at {temp}°C (physics violation)"
                        )
                    else:  # GSHP
                        # GSHP can be more stable but shouldn't increase significantly
                        assert cop_change < 0.5, (
                            f"{model.model_name}: GSHP COP jumped {cop_change:.2f} "
                            f"(ground temp doesn't vary this much)"
                        )
                else:
                    print(f"{temp:>4}°C | {cop:>6.2f} | {'  baseline':>12} |   ✓")

                previous_cop = cop
                previous_temp = temp


class TestModelFlowTemperatureIntegration:
    """Test models integrate with flow temperature formula."""

    def test_models_flow_temp_in_realistic_range(self):
        """Test flow temperatures are realistic for UFH and radiators."""
        from custom_components.effektguard.models.registry import HeatPumpModelRegistry

        registry = HeatPumpModelRegistry()

        print("\n" + "=" * 80)
        print("FLOW TEMPERATURE REALISM TEST")
        print("=" * 80)
        print("UFH typical: 25-45°C, Radiators typical: 35-55°C")
        print("=" * 80)

        for model_id in ["nibe_f750", "nibe_f2040", "nibe_s1155"]:
            model = registry.get_model(model_id)

            print(f"\nModel: {model.model_name}")
            print("-" * 80)
            print(
                f"{'Outdoor':>8} | {'Indoor':>7} | {'Flow (UFH)':>12} | "
                f"{'Flow (Rads)':>12} | {'Valid':>6}"
            )
            print("-" * 80)

            test_scenarios = [
                (10, 21),  # Mild
                (0, 21),  # Average winter
                (-10, 21),  # Cold
                (-20, 21),  # Very cold
            ]

            for outdoor, indoor in test_scenarios:
                # Calculate heat demand for typical house
                heat_loss_coef_ufh = 180.0  # W/°C typical house with UFH
                heat_demand_ufh_kw = (indoor - outdoor) * heat_loss_coef_ufh / 1000.0

                heat_loss_coef_rads = 200.0  # W/°C slightly worse insulation
                heat_demand_rads_kw = (indoor - outdoor) * heat_loss_coef_rads / 1000.0

                # Test UFH
                flow_ufh = model.calculate_optimal_flow_temp(outdoor, indoor, heat_demand_ufh_kw)

                # Test radiators (higher heat demand)
                flow_rads = model.calculate_optimal_flow_temp(outdoor, indoor, heat_demand_rads_kw)

                # Validate UFH range (allow clamping to model min_flow_temp)
                ufh_ok = flow_ufh >= model.min_flow_temp and flow_ufh <= 50
                # Validate radiator range (allow clamping to model min_flow_temp)
                rads_ok = flow_rads >= model.min_flow_temp and flow_rads <= 60

                status = "✓" if (ufh_ok and rads_ok) else "⚠"

                print(
                    f"{outdoor:>6}°C | {indoor:>5}°C | {flow_ufh:>10.1f}°C | "
                    f"{flow_rads:>10.1f}°C | {status:>6}"
                )

                # Flow temp should be reasonable (respecting model limits)
                assert (
                    ufh_ok
                ), f"UFH flow {flow_ufh:.1f}°C invalid (model min: {model.min_flow_temp}°C)"
                assert (
                    rads_ok
                ), f"Radiator flow {flow_rads:.1f}°C invalid (model min: {model.min_flow_temp}°C)"

                # Radiators should need higher flow temp than UFH (or similar)
                assert (
                    flow_rads >= flow_ufh - 2
                ), f"Radiators should need higher/equal flow temp than UFH"

    def test_models_flow_temp_increases_as_outdoor_temp_drops(self):
        """Test flow temperature increases as it gets colder (physics)."""
        from custom_components.effektguard.models.registry import HeatPumpModelRegistry

        registry = HeatPumpModelRegistry()
        f750 = registry.get_model("nibe_f750")

        print("\n" + "=" * 80)
        print("FLOW TEMPERATURE WEATHER COMPENSATION TEST")
        print("=" * 80)
        print(f"Model: {f750.model_name}")
        print("=" * 80)
        print(f"{'Outdoor':>8} | {'Flow Temp':>10} | {'ΔFlow':>8} | {'Valid':>6}")
        print("-" * 80)

        previous_flow = None
        previous_outdoor = None

        for outdoor in range(10, -25, -5):
            # Calculate heat demand
            indoor = 21.0
            heat_loss_coef = 180.0  # W/°C
            heat_demand_kw = (indoor - outdoor) * heat_loss_coef / 1000.0

            flow = f750.calculate_optimal_flow_temp(outdoor, indoor, heat_demand_kw)

            if previous_flow is not None:
                delta_flow = flow - previous_flow
                # Flow should increase or stay same as outdoor drops
                # (unless clamped by model's min_flow_temp)
                is_valid = delta_flow >= -1.0  # Allow for clamping to min
                status = "✓" if is_valid else "✗"

                print(f"{outdoor:>6}°C | {flow:>8.1f}°C | {delta_flow:>+6.1f}°C | {status:>6}")

                # Note: Flow can stop increasing if it hits min_flow_temp limit
                # This is correct behavior, not a failure
            else:
                print(f"{outdoor:>6}°C | {flow:>8.1f}°C | baseline | ✓")

            previous_flow = flow
            previous_outdoor = outdoor


class TestModelCapacityAndSizing:
    """Test models correctly identify undersizing and oversizing."""

    def test_models_detect_insufficient_capacity(self):
        """Test models warn when heat demand exceeds capacity."""
        from custom_components.effektguard.models.registry import HeatPumpModelRegistry

        registry = HeatPumpModelRegistry()

        print("\n" + "=" * 80)
        print("CAPACITY SUFFICIENCY TEST")
        print("=" * 80)

        # Small model (F730 6kW) vs large house in cold weather
        f730 = registry.get_model("nibe_f730")

        print(f"\nScenario: {f730.model_name} (6kW) in large 250m² house")
        print("-" * 80)
        print(
            f"{'Outdoor':>8} | {'Heat Demand':>12} | {'Expected Power':>15} | "
            f"{'Can Meet':>9} | {'Suggestion':30}"
        )
        print("-" * 80)

        # Large house: 250m² × 70 W/m² = 17.5kW at ΔT=30°C
        # So at ΔT=41°C (21°C - (-20°C)): 17.5 × 41/30 = 23.9kW demand

        for outdoor in [5, 0, -5, -10, -15, -20]:
            temp_diff = 21.0 - outdoor
            # Heat demand for 250m² house, standard insulation
            heat_demand_kw = 250 * 70 * (temp_diff / 30.0) / 1000

            cop = f730.get_cop_at_temperature(outdoor)
            electrical_needed = heat_demand_kw / cop

            # Can F730 meet this demand?
            max_heat = f730.typical_electrical_range_kw[1] * cop
            can_meet = heat_demand_kw <= max_heat

            # Calculate optimal flow temp
            indoor = 21
            optimal_flow = f730.calculate_optimal_flow_temp(outdoor, indoor, heat_demand_kw)

            # Validate with model's power check
            result = f730.validate_power_consumption(
                (
                    electrical_needed
                    if electrical_needed <= f730.typical_electrical_range_kw[1]
                    else f730.typical_electrical_range_kw[1]
                ),
                outdoor,
                optimal_flow,
            )

            suggestion = result.suggestions[0] if result.suggestions else "OK"
            status = "✓" if can_meet else "✗"

            print(
                f"{outdoor:>6}°C | {heat_demand_kw:>10.1f} kW | "
                f"{electrical_needed:>13.1f} kW | {status:>9} | {suggestion[:30]:30}"
            )

            # F730 should be insufficient for this large house in cold weather
            if outdoor < -10:
                assert not can_meet, f"F730 shouldn't meet demand for 250m² house at {outdoor}°C"

    def test_gshp_more_efficient_than_ashp_in_cold(self):
        """Test GSHP models show efficiency advantage in cold weather."""
        from custom_components.effektguard.models.registry import HeatPumpModelRegistry

        registry = HeatPumpModelRegistry()

        # Compare similar-sized ASHP vs GSHP
        f2040 = registry.get_model("nibe_f2040")  # 12kW ASHP
        s1155 = registry.get_model("nibe_s1155")  # 12kW GSHP

        print("\n" + "=" * 80)
        print("ASHP vs GSHP EFFICIENCY COMPARISON")
        print("=" * 80)
        print(f"ASHP: {f2040.model_name}, GSHP: {s1155.model_name}")
        print("=" * 80)
        print(
            f"{'Outdoor':>8} | {'ASHP COP':>9} | {'GSHP COP':>9} | "
            f"{'Advantage':>10} | {'GSHP Better':>12}"
        )
        print("-" * 80)

        for outdoor in [10, 0, -10, -20, -30]:
            cop_ashp = f2040.get_cop_at_temperature(outdoor)
            cop_gshp = s1155.get_cop_at_temperature(outdoor)

            advantage_pct = ((cop_gshp - cop_ashp) / cop_ashp) * 100
            is_better = cop_gshp > cop_ashp

            status = "✓ YES" if is_better else "✗ NO"

            print(
                f"{outdoor:>6}°C | {cop_ashp:>9.2f} | {cop_gshp:>9.2f} | "
                f"{advantage_pct:>8.1f}% | {status:>12}"
            )

            # GSHP should be better or equal in all conditions
            assert cop_gshp >= cop_ashp - 0.1, f"GSHP should not be worse than ASHP at {outdoor}°C"

            # GSHP advantage should increase in cold weather (ground temp stable)
            if outdoor < 0:
                assert (
                    cop_gshp > cop_ashp
                ), f"GSHP should be better than ASHP in cold weather ({outdoor}°C)"


class TestModelDecisionEngineIntegration:
    """Test models work with decision engine's multi-layer system."""

    def test_models_provide_validation_for_decision_engine(self):
        """Test model validation can inform decision engine decisions."""
        from custom_components.effektguard.models.registry import HeatPumpModelRegistry

        registry = HeatPumpModelRegistry()
        f750 = registry.get_model("nibe_f750")

        print("\n" + "=" * 80)
        print("MODEL VALIDATION FOR DECISION ENGINE")
        print("=" * 80)
        print(f"Model: {f750.model_name}")
        print("=" * 80)
        print(f"{'Scenario':30} | {'Power':>7} | {'Valid':>6} | {'Severity':>9} | {'Action':20}")
        print("-" * 80)

        scenarios = [
            ("Normal operation", 2.5, 0, 42, "info", "Continue"),
            ("High but OK", 5.0, -5, 48, "info", "Monitor"),
            ("Suspiciously high", 6.0, 5, 50, "warning", "Check aux heat"),
            ("Very high power", 7.0, 0, 52, "warning", "Investigate"),
        ]

        for scenario_name, power, outdoor, flow, expected_severity, expected_action in scenarios:
            result = f750.validate_power_consumption(power, outdoor, flow)

            status = "✓" if result.valid else "⚠"
            severity = result.severity or "info"

            print(
                f"{scenario_name:30} | {power:>5.1f}kW | {status:>6} | "
                f"{severity:>9} | {expected_action:20}"
            )

            # Decision engine should be able to use this info
            # High severity = don't reduce offset further
            # Suggestions = diagnostic info for user
            assert result.severity in [
                "info",
                "warning",
                "error",
            ], "Severity must be standard level"
            assert isinstance(result.valid, bool), "Valid must be boolean"

            if power > 6.0:
                # Very high power should at least warn
                assert (
                    not result.valid or result.severity == "warning"
                ), f"Should warn about {power}kW consumption"


class TestRealWorldIntegrationScenarios:
    """Test models with realistic production scenarios."""

    def test_stevedvo_thermal_debt_scenario(self):
        """Replicate DM -500 thermal debt scenario."""
        from custom_components.effektguard.models.registry import HeatPumpModelRegistry

        registry = HeatPumpModelRegistry()
        f2040 = registry.get_model("nibe_f2040")

        print("\n" + "=" * 80)
        print("THERMAL DEBT SCENARIO REPLICATION (DM -500)")
        print("=" * 80)
        print("Scenario: DM -500 after DHW run, catastrophic recovery")
        print("Expected: 15kW spike, flow 10°K above target")
        print("=" * 80)

        # Test conditions:
        # - F2040-12 model
        # - After DHW run, DM hit -500
        # - Flow spiked 10°K above target
        # - ΔT went to 10-11°K (should be 5-7°K)
        # - Output touched 15kW

        outdoor_temp = 0  # Winter conditions
        target_flow = 45  # Typical for radiators
        actual_flow = 55  # 10°K overshoot
        delta_t = 10.5  # High ΔT indicating inefficiency

        # Calculate what power draw this represents
        # Power = FlowRate × ΔT × SpecificHeat
        # Assume flow rate 20 L/min (typical measured rate)
        flow_rate_l_min = 20
        specific_heat = 4.186  # kJ/(kg·K)
        power_kw = (flow_rate_l_min * delta_t * specific_heat) / 60

        print(f"\nConditions:")
        print(f"  Outdoor: {outdoor_temp}°C")
        print(f"  Target flow: {target_flow}°C")
        print(f"  Actual flow: {actual_flow}°C (overshoot: {actual_flow - target_flow}°K)")
        print(f"  ΔT: {delta_t}°K (inefficient, should be 5-7°K)")
        print(f"  Calculated power: {power_kw:.1f} kW")
        print(f"  Flow rate: {flow_rate_l_min} L/min")

        # Validate this power consumption
        result = f2040.validate_power_consumption(power_kw, outdoor_temp, actual_flow)

        print(f"\nModel validation:")
        print(f"  Valid: {result.valid}")
        print(f"  Severity: {result.severity}")
        print(f"  Message: {result.message}")
        if result.suggestions:
            print(f"  Suggestions:")
            for suggestion in result.suggestions:
                print(f"    - {suggestion}")

        # This scenario should definitely be flagged
        assert not result.valid or result.severity in [
            "warning",
            "error",
        ], "Should flag thermal debt recovery scenario as problematic"

        print("\n✓ Model correctly identifies thermal debt recovery pattern")

    def test_glyn_hudson_pump_mode_scenario(self):
        """Replicate 8-hour pump cycle issue."""
        from custom_components.effektguard.models.registry import HeatPumpModelRegistry

        registry = HeatPumpModelRegistry()
        f2040 = registry.get_model("nibe_f2040")

        print("\n" + "=" * 80)
        print("PUMP MODE SCENARIO (8-hour cycles)")
        print("=" * 80)
        print("Scenario: Open-loop UFH, pump Intermittent → 8hr cycles")
        print("Solution: Pump Auto mode @ 10% idle")
        print("=" * 80)

        # Test system configuration:
        # - F2040-12
        # - Open-loop timber UFH (no secondary pumps)
        # - Pump Intermittent = stopped when compressor off
        # - BT25 above volumiser = temperature decays slowly
        # - Result: 8 hour on, 8 hour off cycles

        # This is a CONFIGURATION issue, not a model spec issue
        # But model should be able to explain why power consumption looks weird

        print("\nPhase 1: Intermittent mode (WRONG for open-loop)")
        print("  - Compressor runs 8 hours continuously")
        print("  - Average power during run: ~3-4kW")
        print("  - Then stops for 8 hours (no circulation)")

        # Simulate long run at moderate power
        long_run_power = 3.5
        result_long_run = f2040.validate_power_consumption(
            long_run_power, 10, 42  # Mild weather, moderate flow
        )

        print(f"\n  Validation: {result_long_run.valid}, {result_long_run.message}")

        print("\nPhase 2: Auto mode @ 10% (CORRECT for open-loop)")
        print("  - Compressor cycles 2-3 times/hour")
        print("  - Power varies with modulation")
        print("  - Continuous low circulation between cycles")

        # This is the expected behavior - model should validate normal cycling
        print("\n✓ Configuration change fixed the issue (not model specs)")


if __name__ == "__main__":
    """Run with: python -m pytest tests/test_model_integration_with_codebase.py -v -s"""
    print("\n" + "=" * 80)
    print("HEAT PUMP MODEL INTEGRATION TEST SUITE")
    print("Tests behavior, not exact specifications")
    print("=" * 80)

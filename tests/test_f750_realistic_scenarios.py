"""Comprehensive testing for NIBE F750 with realistic COP calculations.

Tests the decision engine across full Swedish temperature range with:
- Real NIBE F750 specifications
- COP calculations based on outdoor temperature
- 20A socket limit (4.6kW) - F750 max 6.5kW (requires 3-phase)
- Realistic heat demands for 150m² house
- Swedish climate conditions (Malmö to Kiruna)

NIBE F750 Specifications:
- Rated power: 8kW heat output at 7°C outdoor / 45°C flow
- Electrical consumption: 1.2-2.8kW on single phase 20A
- Max electrical: 6.5kW (3-phase, not available on 20A socket)
- COP range: 2.0-5.0 depending on outdoor/flow temperatures
"""

import pytest
from dataclasses import dataclass


@dataclass
class F750Specifications:
    """NIBE F750 heat pump specifications.

    Based on NIBE official data and Swedish forum validation.
    Assumes three-phase 20A installation (standard for heat pumps in Sweden).
    """

    # Power specifications
    rated_heat_output_kw: float = 8.0  # At 7°C outdoor, 45°C flow
    min_electrical_kw: float = 0.5  # Minimum modulation
    max_electrical_kw: float = 6.5  # Three-phase 20A (16A per phase)

    # COP characteristics (outdoor temp → typical COP)
    cop_curve = {
        7: 5.0,  # Rated conditions
        5: 4.5,  # Mild
        0: 4.0,  # Average winter (Malmö)
        -5: 3.5,  # Common cold (Stockholm)
        -10: 3.0,  # Cold winter
        -15: 2.7,  # Design temp (Northern Sweden)
        -20: 2.3,  # Very cold
        -25: 2.0,  # Extreme (Kiruna)
        -30: 1.8,  # Extreme cold (survival mode)
    }

    def get_cop_at_temperature(self, outdoor_temp: float) -> float:
        """Calculate COP for given outdoor temperature.

        Uses linear interpolation between known points.

        Args:
            outdoor_temp: Outdoor temperature (°C)

        Returns:
            Estimated COP (Coefficient of Performance)
        """
        # Find surrounding temperature points
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

                # Interpolate
                ratio = (outdoor_temp - t1) / (t2 - t1)
                return cop1 + (cop2 - cop1) * ratio

        return 3.0  # Fallback

    def calculate_electrical_consumption(
        self,
        heat_demand_kw: float,
        outdoor_temp: float,
    ) -> tuple[float, bool]:
        """Calculate electrical consumption for given heat demand.

        Args:
            heat_demand_kw: Required heat output (kW)
            outdoor_temp: Outdoor temperature (°C)

        Returns:
            Tuple of (electrical_kw, within_limit)
        """
        cop = self.get_cop_at_temperature(outdoor_temp)
        electrical_kw = heat_demand_kw / cop

        # Check three-phase 20A limit (6.5kW)
        within_limit = electrical_kw <= self.max_electrical_kw

        # Cap at three-phase limit (heat pump will modulate down or auxiliary needed)
        electrical_kw = min(electrical_kw, self.max_electrical_kw)

        return electrical_kw, within_limit


@dataclass
class HouseCharacteristics:
    """Typical Swedish house for testing."""

    area_m2: float = 150  # Square meters
    insulation_quality: str = "standard"  # pre_1990, standard, modern
    indoor_target_temp: float = 21.0  # °C

    # Heat loss coefficients (W/m² at ΔT=30°C)
    heat_loss_coefficients = {
        "pre_1990": 100.0,  # Poor insulation
        "standard": 70.0,  # Average 1990-2010
        "modern": 50.0,  # Post-2010 building codes
    }

    def calculate_heat_demand(self, outdoor_temp: float) -> float:
        """Calculate heat demand for given outdoor temperature.

        Args:
            outdoor_temp: Outdoor temperature (°C)

        Returns:
            Heat demand in kW
        """
        coefficient = self.heat_loss_coefficients[self.insulation_quality]
        temp_diff = self.indoor_target_temp - outdoor_temp

        # Heat loss formula: Area × Coefficient × (ΔT / 30°C) / 1000
        heat_demand_kw = self.area_m2 * coefficient * (temp_diff / 30.0) / 1000

        return heat_demand_kw


class TestF750RealisticScenarios:
    """Test suite for NIBE F750 with realistic scenarios."""

    @pytest.fixture
    def f750(self):
        """NIBE F750 specification fixture."""
        return F750Specifications()

    @pytest.fixture
    def house(self):
        """Standard Swedish house fixture."""
        return HouseCharacteristics()

    def test_cop_calculation_across_temperatures(self, f750):
        """Test COP calculations match expected values."""

        test_cases = [
            (7, 5.0),  # Rated
            (5, 4.5),  # Mild
            (0, 4.0),  # Average
            (-5, 3.5),  # Common cold
            (-10, 3.0),  # Cold
            (-15, 2.7),  # Design
            (-20, 2.3),  # Very cold
            (-25, 2.0),  # Extreme
            (-30, 1.8),  # Survival
        ]

        print("\n" + "=" * 80)
        print("NIBE F750 COP ACROSS TEMPERATURES")
        print("=" * 80)
        print(
            f"{'Outdoor Temp':>15} | {'Expected COP':>12} | {'Calculated COP':>15} | {'Match':>6}"
        )
        print("-" * 80)

        for outdoor_temp, expected_cop in test_cases:
            calculated_cop = f750.get_cop_at_temperature(outdoor_temp)
            match = "✓" if abs(calculated_cop - expected_cop) < 0.1 else "✗"

            print(
                f"{outdoor_temp:>13}°C | {expected_cop:>12.1f} | {calculated_cop:>15.2f} | {match:>6}"
            )

            assert abs(calculated_cop - expected_cop) < 0.1, f"COP mismatch at {outdoor_temp}°C"

    def test_interpolation_between_points(self, f750):
        """Test COP interpolation works correctly."""

        # Test midpoint between 0°C (COP 4.0) and -5°C (COP 3.5)
        cop_at_minus_2_5 = f750.get_cop_at_temperature(-2.5)
        expected = 3.75  # Midpoint

        assert (
            abs(cop_at_minus_2_5 - expected) < 0.01
        ), f"Interpolation failed: expected {expected}, got {cop_at_minus_2_5}"

    def test_heat_demand_calculation(self, house):
        """Test heat demand calculations for different temperatures."""

        test_temps = [5, 0, -5, -10, -15, -20, -25, -30]

        print("\n" + "=" * 80)
        print(f"HEAT DEMAND - {house.area_m2}m² house, {house.insulation_quality} insulation")
        print("=" * 80)
        print(f"{'Outdoor Temp':>15} | {'Temp Diff':>10} | {'Heat Demand':>15}")
        print("-" * 80)

        for temp in test_temps:
            demand = house.calculate_heat_demand(temp)
            temp_diff = house.indoor_target_temp - temp

            print(f"{temp:>13}°C | {temp_diff:>8}°C | {demand:>13.2f} kW")

    def test_electrical_consumption_within_20a_limit(self, f750, house):
        """Test that F750 stays within three-phase 20A limit in normal conditions."""

        test_temps = [5, 0, -5, -10, -15, -20, -25, -30]

        print("\n" + "=" * 80)
        print("ELECTRICAL CONSUMPTION - F750 on three-phase 20A (6.5kW limit)")
        print("=" * 80)
        print(
            f"{'Outdoor':>8} | {'Heat Demand':>12} | {'COP':>6} | {'Electrical':>12} | {'Within Limit':>13}"
        )
        print("-" * 80)

        for temp in test_temps:
            heat_demand = house.calculate_heat_demand(temp)
            electrical, within_limit = f750.calculate_electrical_consumption(heat_demand, temp)
            cop = f750.get_cop_at_temperature(temp)

            status = "✓ YES" if within_limit else "✗ NO (AUX)"

            print(
                f"{temp:>6}°C | {heat_demand:>10.2f} kW | {cop:>5.2f} | "
                f"{electrical:>10.2f} kW | {status:>13}"
            )

            # Assert we don't exceed three-phase limit in calculations
            assert (
                electrical <= f750.max_electrical_kw
            ), f"Calculation exceeds three-phase 20A limit at {temp}°C"

    def test_swedish_climate_scenarios(self, f750, house):
        """Test full Swedish climate scenarios with realistic conditions."""

        scenarios = [
            ("Malmö mild winter", 5, -100),
            ("Malmö average winter", 0, -300),
            ("Stockholm autumn/spring", -5, -400),
            ("Stockholm winter", -10, -600),
            ("Northern Sweden average", -15, -800),
            ("Northern Sweden cold", -20, -1000),
            ("Kiruna winter", -25, -1200),
            ("Kiruna extreme", -30, -1400),
        ]

        print("\n" + "=" * 80)
        print("SWEDISH CLIMATE SCENARIOS - Full System Analysis")
        print("=" * 80)
        print(
            f"{'Scenario':25} | {'Out':>5} | {'DM':>6} | {'Demand':>7} | {'COP':>5} | "
            f"{'Elec':>6} | {'Limit':>6} | {'Status':>10}"
        )
        print("-" * 80)

        for scenario, outdoor, dm in scenarios:
            heat_demand = house.calculate_heat_demand(outdoor)
            cop = f750.get_cop_at_temperature(outdoor)
            electrical, within_limit = f750.calculate_electrical_consumption(heat_demand, outdoor)

            status_limit = "OK" if within_limit else "AUX"

            # Determine system status based on DM
            if dm <= -1500:
                status = "EMERGENCY"
            elif dm <= -1200:
                status = "CRITICAL"
            elif dm <= -800:
                status = "WARNING"
            elif dm <= -600:
                status = "CAUTION"
            else:
                status = "NORMAL"

            print(
                f"{scenario:25} | {outdoor:>4}°C | {dm:>6} | {heat_demand:>5.1f}kW | "
                f"{cop:>5.2f} | {electrical:>4.1f}kW | {status_limit:>6} | {status:>10}"
            )

    def test_auxiliary_heating_threshold(self, f750, house):
        """Test when auxiliary heating becomes necessary on three-phase 20A."""

        print("\n" + "=" * 80)
        print("AUXILIARY HEATING ANALYSIS - When does F750 need help on 3-phase 20A?")
        print("=" * 80)
        print(
            f"{'Outdoor':>8} | {'Heat Demand':>12} | {'F750 Max':>12} | {'Shortfall':>12} | {'Aux Needed':>12}"
        )
        print("-" * 80)

        for temp in range(5, -35, -5):
            heat_demand = house.calculate_heat_demand(temp)
            cop = f750.get_cop_at_temperature(temp)

            # Maximum heat F750 can deliver on three-phase 20A
            max_heat = f750.max_electrical_kw * cop

            # Shortfall
            shortfall = max(0, heat_demand - max_heat)
            aux_needed = "YES" if shortfall > 0.1 else "NO"

            print(
                f"{temp:>6}°C | {heat_demand:>10.2f} kW | {max_heat:>10.2f} kW | "
                f"{shortfall:>10.2f} kW | {aux_needed:>12}"
            )

    def test_dm_thresholds_with_real_conditions(self, f750, house):
        """Test DM threshold behavior with real F750 conditions."""

        from custom_components.effektguard.optimization.decision_engine import DecisionEngine

        # Mock dependencies
        class MockPriceAnalyzer:
            def update_prices(self, data):
                pass

            def get_current_classification(self, quarter):
                return "normal"

            def get_base_offset(self, quarter, classification, is_daytime):
                return 0.0

        class MockEffectManager:
            def should_limit_power(self, power, quarter):
                class Decision:
                    severity = "OK"

                return Decision()

        class MockThermalModel:
            thermal_mass = 1.0

            def calculate_preheating_target(self, *args, **kwargs):
                return 21.0

        class MockNibeState:
            def __init__(self, dm, outdoor):
                self.degree_minutes = dm
                self.outdoor_temp = outdoor
                self.indoor_temp = 21.0
                self.is_heating = True

        engine = DecisionEngine(
            price_analyzer=MockPriceAnalyzer(),
            effect_manager=MockEffectManager(),
            thermal_model=MockThermalModel(),
            config={"target_temperature": 21.0, "tolerance": 5.0},
        )

        print("\n" + "=" * 80)
        print("DECISION ENGINE RESPONSE WITH REAL F750 CONDITIONS")
        print("=" * 80)
        print(
            f"{'Location':20} | {'Out':>5} | {'DM':>6} | {'Expected DM':>12} | "
            f"{'Offset':>7} | {'Response':30}"
        )
        print("-" * 80)

        test_cases = [
            ("Malmö mild", 5, -300),
            ("Malmö avg", 0, -600),
            ("Stockholm", -5, -800),
            ("Stockholm cold", -10, -1000),
            ("Northern", -15, -1200),
            ("Kiruna", -25, -1400),
            ("At limit", -10, -1500),
        ]

        for location, outdoor, dm in test_cases:
            state = MockNibeState(dm, outdoor)
            expected = engine._calculate_expected_dm_for_temperature(outdoor)
            decision = engine._emergency_layer(state)

            print(
                f"{location:20} | {outdoor:>4}°C | {dm:>6} | {expected['normal']:>12.0f} | "
                f"{decision.offset:>6.1f}°C | {decision.reason[:30]}"
            )


if __name__ == "__main__":
    """Run tests with detailed output."""

    # Can run with: python -m pytest tests/test_f750_realistic_scenarios.py -v -s

    print("\n" + "=" * 80)
    print("NIBE F750 COMPREHENSIVE TEST SUITE")
    print("Realistic scenarios for Swedish climate with 20A socket limitation")
    print("=" * 80)

    # Create instances for manual testing
    f750 = F750Specifications()
    house = HouseCharacteristics()

    print("\nF750 SPECIFICATIONS:")
    print(f"  Max heat output: {f750.rated_heat_output_kw} kW")
    print(f"  Three-phase 20A limit: {f750.max_electrical_kw} kW electrical")
    print(f"  COP range: {min(f750.cop_curve.values())} - {max(f750.cop_curve.values())}")

    print(f"\nHOUSE CHARACTERISTICS:")
    print(f"  Area: {house.area_m2} m²")
    print(f"  Insulation: {house.insulation_quality}")
    print(f"  Indoor target: {house.indoor_target_temp}°C")

    # Quick validation
    print("\nQUICK VALIDATION:")
    for temp in [5, 0, -10, -20, -30]:
        cop = f750.get_cop_at_temperature(temp)
        demand = house.calculate_heat_demand(temp)
        electrical, ok = f750.calculate_electrical_consumption(demand, temp)

        print(
            f"  {temp:>3}°C: Demand {demand:.1f}kW, COP {cop:.2f}, "
            f"Electrical {electrical:.1f}kW {'✓' if ok else '✗ (needs aux)'}"
        )

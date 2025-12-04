"""Tests for savings calculator.

Tests savings estimation logic including:
- Monthly savings calculation from peak reduction
- Spot price savings estimation
- Baseline peak tracking
- Savings per optimization cycle
"""

import pytest

from custom_components.effektguard.const import (
    BASELINE_EMA_WEIGHT_NEW,
    BASELINE_EMA_WEIGHT_OLD,
    BASELINE_PEAK_MULTIPLIER,
    DAYS_PER_MONTH,
    ORE_TO_SEK_CONVERSION,
    SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH,
)
from custom_components.effektguard.optimization.savings_calculator import (
    SavingsCalculator,
    SavingsEstimate,
)


class TestSavingsEstimate:
    """Test SavingsEstimate dataclass."""

    def test_savings_estimate_creation(self):
        """Test creating a SavingsEstimate."""
        estimate = SavingsEstimate(
            monthly_estimate=150.0,
            effect_savings=100.0,
            spot_savings=50.0,
            baseline_cost=500.0,
            optimized_cost=350.0,
        )

        assert estimate.monthly_estimate == 150.0
        assert estimate.effect_savings == 100.0
        assert estimate.spot_savings == 50.0
        assert estimate.baseline_cost == 500.0
        assert estimate.optimized_cost == 350.0


class TestSavingsCalculatorInit:
    """Test SavingsCalculator initialization."""

    def test_init_default_values(self):
        """Test calculator initializes with correct defaults from const.py."""
        calc = SavingsCalculator()

        assert calc.effect_tariff_sek_per_kw_month == SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH
        assert calc._baseline_monthly_peak is None
        assert calc._total_spot_savings == 0.0
        assert calc._optimization_days == 0

    def test_baseline_peak_property(self):
        """Test baseline_monthly_peak property."""
        calc = SavingsCalculator()
        assert calc.baseline_monthly_peak is None

        calc._baseline_monthly_peak = 10.5
        assert calc.baseline_monthly_peak == 10.5


class TestMonthlySavingsEstimation:
    """Test monthly savings estimation."""

    def test_estimate_with_known_baseline(self):
        """Test savings calculation with known baseline peak."""
        calc = SavingsCalculator()

        # Current peak: 8 kW, Baseline: 10 kW = 2 kW reduction
        # Expected savings: 2 kW × 50 SEK/kW = 100 SEK from effect tariff
        # Spot savings: 5 SEK/day × 30 days = 150 SEK
        # Total: 250 SEK
        estimate = calc.estimate_monthly_savings(
            current_peak_kw=8.0,
            baseline_peak_kw=10.0,
            average_spot_savings_per_day=5.0,
        )

        assert estimate.monthly_estimate == 250.0
        assert estimate.effect_savings == 100.0
        assert estimate.spot_savings == 150.0
        assert estimate.baseline_cost == 500.0  # 10 kW × 50 SEK
        assert estimate.optimized_cost == 400.0  # 8 kW × 50 SEK

    def test_estimate_without_baseline_uses_multiplier(self):
        """Test savings estimation without baseline uses BASELINE_PEAK_MULTIPLIER."""
        calc = SavingsCalculator()

        # Current peak: 8.5 kW
        # Baseline estimate: 8.5 × 1.176 = 10.0 kW (assumes 15% reduction)
        # Peak reduction: 10.0 - 8.5 = 1.5 kW
        # Effect savings: 1.5 × 50 = 75 SEK
        estimate = calc.estimate_monthly_savings(
            current_peak_kw=8.5,
            baseline_peak_kw=None,
            average_spot_savings_per_day=0.0,
        )

        expected_baseline = 8.5 * BASELINE_PEAK_MULTIPLIER
        expected_reduction = expected_baseline - 8.5
        expected_effect_savings = expected_reduction * SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH

        assert estimate.effect_savings == pytest.approx(expected_effect_savings, rel=1e-2)
        assert estimate.spot_savings == 0.0
        assert estimate.baseline_cost == pytest.approx(
            expected_baseline * SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH, rel=1e-2
        )

    def test_estimate_no_reduction_no_savings(self):
        """Test that no peak reduction means no effect savings."""
        calc = SavingsCalculator()

        estimate = calc.estimate_monthly_savings(
            current_peak_kw=10.0,
            baseline_peak_kw=10.0,
            average_spot_savings_per_day=0.0,
        )

        assert estimate.effect_savings == 0.0
        assert estimate.spot_savings == 0.0
        assert estimate.monthly_estimate == 0.0

    def test_estimate_higher_current_than_baseline_no_negative_savings(self):
        """Test that higher current peak than baseline doesn't give negative savings."""
        calc = SavingsCalculator()

        # Current peak higher than baseline (optimization made things worse?)
        estimate = calc.estimate_monthly_savings(
            current_peak_kw=12.0,
            baseline_peak_kw=10.0,
            average_spot_savings_per_day=0.0,
        )

        # Should not have negative effect savings (max with 0)
        assert estimate.effect_savings == 0.0

    def test_estimate_spot_savings_calculation(self):
        """Test spot savings calculation uses DAYS_PER_MONTH."""
        calc = SavingsCalculator()

        estimate = calc.estimate_monthly_savings(
            current_peak_kw=8.0,
            baseline_peak_kw=8.0,  # No effect savings
            average_spot_savings_per_day=7.5,
        )

        expected_spot = 7.5 * DAYS_PER_MONTH
        assert estimate.spot_savings == expected_spot
        assert estimate.effect_savings == 0.0
        assert estimate.monthly_estimate == expected_spot

    def test_estimate_combines_effect_and_spot_savings(self):
        """Test that total estimate combines both effect and spot savings."""
        calc = SavingsCalculator()

        estimate = calc.estimate_monthly_savings(
            current_peak_kw=7.0,
            baseline_peak_kw=10.0,  # 3 kW reduction = 150 SEK
            average_spot_savings_per_day=4.0,  # 120 SEK/month
        )

        assert estimate.effect_savings == 150.0
        assert estimate.spot_savings == 120.0
        assert estimate.monthly_estimate == 270.0


class TestCycleSavingsEstimation:
    """Test per-cycle savings estimation using actual power data."""

    def test_cycle_savings_same_price_no_savings(self):
        """Test that same price as average yields no savings."""
        calc = SavingsCalculator()

        # 4 kW power, current price = average price
        savings = calc.calculate_spot_savings_per_cycle(
            actual_power_kw=4.0,
            current_price=100.0,  # öre/kWh
            average_price_today=100.0,  # öre/kWh
            cycle_minutes=5.0,
        )

        # Same price = no savings
        assert savings == pytest.approx(0.0, abs=0.01)

    def test_cycle_savings_during_cheap_period(self):
        """Test savings when using power during cheap period."""
        calc = SavingsCalculator()

        # 4 kW power for 5 minutes during cheap period (50 öre vs 100 öre average)
        # Energy: 4 kW × (5/60) h = 0.333 kWh
        # Cost at cheap: 0.333 × 50 / 100 = 0.167 SEK
        # Cost at average: 0.333 × 100 / 100 = 0.333 SEK
        # Savings: 0.333 - 0.167 = 0.167 SEK
        savings = calc.calculate_spot_savings_per_cycle(
            actual_power_kw=4.0,
            current_price=50.0,
            average_price_today=100.0,
            cycle_minutes=5.0,
        )

        energy_kwh = 4.0 * (5.0 / 60.0)
        cost_cheap = (energy_kwh * 50.0) / ORE_TO_SEK_CONVERSION
        cost_average = (energy_kwh * 100.0) / ORE_TO_SEK_CONVERSION
        expected_savings = cost_average - cost_cheap

        assert savings == pytest.approx(expected_savings, rel=1e-2)

    def test_cycle_savings_during_expensive_period(self):
        """Test negative savings when using power during expensive period."""
        calc = SavingsCalculator()

        # 4 kW power for 5 minutes during expensive period (150 öre vs 100 öre)
        # Energy: 4 kW × (5/60) h = 0.333 kWh
        # Cost at expensive: 0.333 × 150 / 100 = 0.5 SEK
        # Cost at average: 0.333 × 100 / 100 = 0.333 SEK
        # Savings: 0.333 - 0.5 = -0.167 SEK (negative = cost more)
        savings = calc.calculate_spot_savings_per_cycle(
            actual_power_kw=4.0,
            current_price=150.0,
            average_price_today=100.0,
            cycle_minutes=5.0,
        )

        energy_kwh = 4.0 * (5.0 / 60.0)
        cost_expensive = (energy_kwh * 150.0) / ORE_TO_SEK_CONVERSION
        cost_average = (energy_kwh * 100.0) / ORE_TO_SEK_CONVERSION
        expected_savings = cost_average - cost_expensive

        assert savings == pytest.approx(expected_savings, rel=1e-2)
        assert savings < 0  # Negative savings = cost more than average

    def test_cycle_savings_very_cheap_period(self):
        """Test larger savings during very cheap period."""
        calc = SavingsCalculator()

        # Very cheap: 20 öre vs 100 öre average
        savings = calc.calculate_spot_savings_per_cycle(
            actual_power_kw=4.0,
            current_price=20.0,
            average_price_today=100.0,
            cycle_minutes=5.0,
        )

        energy_kwh = 4.0 * (5.0 / 60.0)
        cost_cheap = (energy_kwh * 20.0) / ORE_TO_SEK_CONVERSION
        cost_average = (energy_kwh * 100.0) / ORE_TO_SEK_CONVERSION
        expected_savings = cost_average - cost_cheap

        assert savings == pytest.approx(expected_savings, rel=1e-2)
        assert savings > 0.2  # Significant savings when price is 80% below average

    def test_cycle_savings_longer_cycle(self):
        """Test savings scale with cycle duration."""
        calc = SavingsCalculator()

        savings_5min = calc.calculate_spot_savings_per_cycle(
            actual_power_kw=4.0,
            current_price=50.0,
            average_price_today=100.0,
            cycle_minutes=5.0,
        )

        savings_10min = calc.calculate_spot_savings_per_cycle(
            actual_power_kw=4.0,
            current_price=50.0,
            average_price_today=100.0,
            cycle_minutes=10.0,
        )

        # 10 minutes should save 2x more than 5 minutes
        assert savings_10min == pytest.approx(savings_5min * 2.0, rel=1e-2)

    def test_cycle_savings_with_different_power(self):
        """Test cycle savings scale with power consumption."""
        calc = SavingsCalculator()

        # Higher power consumption = more savings potential
        savings_4kw = calc.calculate_spot_savings_per_cycle(
            actual_power_kw=4.0,
            current_price=50.0,
            average_price_today=100.0,
            cycle_minutes=5.0,
        )

        savings_6kw = calc.calculate_spot_savings_per_cycle(
            actual_power_kw=6.0,
            current_price=50.0,
            average_price_today=100.0,
            cycle_minutes=5.0,
        )

        # 6 kW should save 1.5x more than 4 kW
        assert savings_6kw == pytest.approx(savings_4kw * 1.5, rel=1e-2)


class TestBaselinePeakTracking:
    """Test baseline peak tracking and updating."""

    def test_update_baseline_first_time(self):
        """Test setting baseline peak for first time."""
        calc = SavingsCalculator()

        calc.update_baseline_peak(12.5)

        assert calc.baseline_monthly_peak == 12.5

    def test_update_baseline_with_ema(self):
        """Test baseline updates using exponential moving average."""
        calc = SavingsCalculator()

        # Set initial baseline
        calc.update_baseline_peak(10.0)
        assert calc.baseline_monthly_peak == 10.0

        # Update with new observation
        # EMA: 0.8 × 10.0 + 0.2 × 12.0 = 8.0 + 2.4 = 10.4
        calc.update_baseline_peak(12.0)
        expected = BASELINE_EMA_WEIGHT_OLD * 10.0 + BASELINE_EMA_WEIGHT_NEW * 12.0
        assert calc.baseline_monthly_peak == pytest.approx(expected, rel=1e-2)

    def test_update_baseline_multiple_times(self):
        """Test baseline converges with multiple updates."""
        calc = SavingsCalculator()

        calc.update_baseline_peak(10.0)
        calc.update_baseline_peak(11.0)
        calc.update_baseline_peak(11.5)
        calc.update_baseline_peak(12.0)

        # After multiple updates, should be closer to recent values
        # but still influenced by initial value
        assert calc.baseline_monthly_peak > 10.0
        assert calc.baseline_monthly_peak < 12.0


class TestSpotSavingsTracking:
    """Test spot savings recording and averaging."""

    def test_record_spot_savings_first_time(self):
        """Test recording spot savings for first time."""
        calc = SavingsCalculator()

        calc.record_spot_savings(5.5)

        assert calc._total_spot_savings == 5.5
        assert calc._optimization_days == 1
        assert calc.average_daily_spot_savings == 5.5

    def test_record_spot_savings_multiple_days(self):
        """Test recording savings over multiple days."""
        calc = SavingsCalculator()

        calc.record_spot_savings(5.0)
        calc.record_spot_savings(7.0)
        calc.record_spot_savings(6.0)

        assert calc._total_spot_savings == 18.0
        assert calc._optimization_days == 3
        assert calc.average_daily_spot_savings == pytest.approx(6.0, rel=1e-2)

    def test_average_daily_savings_no_data(self):
        """Test average daily savings returns 0 with no data."""
        calc = SavingsCalculator()

        assert calc.average_daily_spot_savings == 0.0

    def test_record_negative_savings(self):
        """Test that negative savings (cost increase) can be recorded."""
        calc = SavingsCalculator()

        calc.record_spot_savings(-2.5)

        assert calc._total_spot_savings == -2.5
        assert calc.average_daily_spot_savings == -2.5


class TestConstantsUsage:
    """Test that all constants from const.py are used correctly."""

    def test_effect_tariff_from_const(self):
        """Test effect tariff uses constant from const.py."""
        calc = SavingsCalculator()
        assert calc.effect_tariff_sek_per_kw_month == SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH

    def test_baseline_multiplier_from_const(self):
        """Test baseline multiplier uses constant."""
        calc = SavingsCalculator()
        estimate = calc.estimate_monthly_savings(current_peak_kw=10.0, baseline_peak_kw=None)
        expected_baseline = 10.0 * BASELINE_PEAK_MULTIPLIER
        peak_reduction = expected_baseline - 10.0
        expected_savings = peak_reduction * SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH
        assert estimate.effect_savings == pytest.approx(expected_savings, rel=1e-2)

    def test_ore_to_sek_conversion_in_cycle_savings(self):
        """Test öre to SEK conversion uses constant."""
        calc = SavingsCalculator()
        # 4 kW for 60 minutes = 4 kWh, price diff of 50 öre
        savings = calc.calculate_spot_savings_per_cycle(
            actual_power_kw=4.0,
            current_price=50.0,
            average_price_today=100.0,
            cycle_minutes=60.0,
        )
        # Energy = 4 kWh, price diff = 50 öre
        # Savings = 4 * 50 / 100 = 2 SEK
        expected_savings = (4.0 * 50.0) / ORE_TO_SEK_CONVERSION
        assert savings == pytest.approx(expected_savings, rel=1e-2)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_peak_reduction(self):
        """Test handling of zero peak reduction."""
        calc = SavingsCalculator()
        estimate = calc.estimate_monthly_savings(current_peak_kw=0.0, baseline_peak_kw=0.0)
        assert estimate.effect_savings == 0.0
        assert estimate.baseline_cost == 0.0
        assert estimate.optimized_cost == 0.0

    def test_very_large_peak_reduction(self):
        """Test handling of very large peak reduction."""
        calc = SavingsCalculator()
        estimate = calc.estimate_monthly_savings(current_peak_kw=5.0, baseline_peak_kw=20.0)
        # 15 kW reduction × 50 SEK = 750 SEK
        assert estimate.effect_savings == 750.0

    def test_zero_power_no_savings(self):
        """Test that zero power consumption yields no savings."""
        calc = SavingsCalculator()
        savings = calc.calculate_spot_savings_per_cycle(
            actual_power_kw=0.0,
            current_price=50.0,
            average_price_today=100.0,
            cycle_minutes=5.0,
        )
        # No power = no savings
        assert savings == 0.0

    def test_negative_power_treated_as_zero(self):
        """Test that negative power is treated as zero (no consumption)."""
        calc = SavingsCalculator()
        savings = calc.calculate_spot_savings_per_cycle(
            actual_power_kw=-2.0,
            current_price=50.0,
            average_price_today=100.0,
            cycle_minutes=5.0,
        )
        # Negative power = no savings (edge case protection)
        assert savings == 0.0

    def test_none_power_treated_as_zero(self):
        """Test that None power is treated as zero."""
        calc = SavingsCalculator()
        savings = calc.calculate_spot_savings_per_cycle(
            actual_power_kw=None,
            current_price=50.0,
            average_price_today=100.0,
            cycle_minutes=5.0,
        )
        # None power = no savings
        assert savings == 0.0


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_realistic_monthly_scenario(self):
        """Test realistic monthly scenario with actual usage patterns."""
        calc = SavingsCalculator()

        # Simulate a month of usage
        calc.update_baseline_peak(10.0)  # kW

        # Record daily spot savings (simulate 30 days of optimization)
        for _ in range(30):
            calc.record_spot_savings(5.0)  # 5 SEK per day

        # Effect tariff savings: 10 kW * 85% = 8.5 kW reduced
        current_peak = 10.0 * 0.85  # 1.5 kW reduction

        # Monthly estimate - pass the average daily spot savings
        estimate = calc.estimate_monthly_savings(
            current_peak_kw=current_peak,
            baseline_peak_kw=10.0,
            average_spot_savings_per_day=calc.average_daily_spot_savings,
        )

        # Should have both effect and spot savings
        assert estimate.monthly_estimate > 0
        assert estimate.effect_savings > 0
        assert estimate.spot_savings > 0
        assert estimate.baseline_cost > estimate.optimized_cost

    def test_no_optimization_benefit_scenario(self):
        """Test scenario where optimization provides no benefit."""
        calc = SavingsCalculator()

        # Peak unchanged
        estimate = calc.estimate_monthly_savings(
            current_peak_kw=10.0,
            baseline_peak_kw=10.0,
            average_spot_savings_per_day=0.0,
        )

        assert estimate.monthly_estimate == 0.0
        assert estimate.effect_savings == 0.0
        assert estimate.spot_savings == 0.0

    def test_effect_tariff_only_scenario(self):
        """Test scenario with only effect tariff savings."""
        calc = SavingsCalculator()

        # Peak reduced but no spot optimization
        estimate = calc.estimate_monthly_savings(
            current_peak_kw=8.0,
            baseline_peak_kw=10.0,
            average_spot_savings_per_day=0.0,
        )

        assert estimate.effect_savings == 100.0  # 2 kW × 50 SEK
        assert estimate.spot_savings == 0.0
        assert estimate.monthly_estimate == 100.0

    def test_spot_only_scenario(self):
        """Test scenario with only spot price savings."""
        calc = SavingsCalculator()

        # No peak reduction but spot optimization
        estimate = calc.estimate_monthly_savings(
            current_peak_kw=10.0,
            baseline_peak_kw=10.0,
            average_spot_savings_per_day=8.0,
        )

        assert estimate.effect_savings == 0.0
        assert estimate.spot_savings == 240.0  # 8 × 30
        assert estimate.monthly_estimate == 240.0

"""Savings math must honor the configured GE-Spot price unit.

Regression: costs always divided by 100 (öre/kWh assumption), so a
1.00 SEK/kWh price was treated as 0.01 SEK/kWh. Ranking is unit-invariant;
reported savings are not.
"""

import pytest

from custom_components.effektguard.optimization.savings_calculator import SavingsCalculator


@pytest.mark.parametrize(
    "unit,factor",
    [
        ("öre/kWh", 0.01),
        ("ore/kWh", 0.01),
        ("cent/kWh", 0.01),
        ("SEK/kWh", 1.0),
        ("EUR/kWh", 1.0),
        (None, 0.01),  # unknown: legacy öre assumption, logged once
    ],
)
def test_price_unit_factor(unit, factor):
    calc = SavingsCalculator()
    calc.price_unit = unit
    assert calc.price_to_main_unit_factor() == pytest.approx(factor)


def test_sek_per_kwh_not_divided_by_100():
    """1 kWh at 1.00 SEK/kWh vs 2.00 SEK/kWh average = 1.00 SEK saved."""
    calc = SavingsCalculator()
    calc.price_unit = "SEK/kWh"
    savings = calc.calculate_spot_savings_per_cycle(
        actual_power_kw=12.0,  # 12 kW for 5 min = 1 kWh
        cycle_minutes=5.0,
        current_price=1.0,
        average_price_today=2.0,
    )
    assert savings == pytest.approx(1.0)


def test_ore_per_kwh_unchanged():
    """100 öre at 1 kWh vs 200 öre average = 1.00 SEK saved (legacy path)."""
    calc = SavingsCalculator()
    calc.price_unit = "öre/kWh"
    savings = calc.calculate_spot_savings_per_cycle(
        actual_power_kw=12.0,
        cycle_minutes=5.0,
        current_price=100.0,
        average_price_today=200.0,
    )
    assert savings == pytest.approx(1.0)


def test_eur_spot_savings_are_not_aggregated_as_sek():
    calc = SavingsCalculator()
    calc.price_unit = "EUR/kWh"

    assert calc.is_sek_price_unit is False

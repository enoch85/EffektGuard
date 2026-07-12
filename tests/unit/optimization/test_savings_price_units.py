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
    ],
)
def test_price_unit_factor(unit, factor):
    calc = SavingsCalculator()
    calc.price_unit = unit
    assert calc.price_to_main_unit_factor() == pytest.approx(factor)


@pytest.mark.parametrize("unit", [None, "", "widgets/kWh"])
def test_unknown_unit_refuses_to_guess(unit):
    """An unrecognised unit must yield None, not the legacy öre assumption.

    Every price integration publishes `<currency>/kWh` by DEFAULT - Nord Pool (HA core) has
    no cents option at all, and both custom-components/nordpool and GE-Spot emit SEK/kWh
    unless the user opts into a subunit display. So the old öre fallback was 100x wrong
    against all three, and it fired whenever `price_unit` was None - which it is until the
    first successful price read.
    """
    calc = SavingsCalculator()
    calc.price_unit = unit

    assert calc.price_to_main_unit_factor() is None, (
        "An unknown price unit was guessed as öre/kWh. Against a SEK/kWh feed that "
        "overstates savings by 100x. Skip the figure instead of fabricating one."
    )


def test_unknown_unit_reports_no_savings_rather_than_a_wrong_number():
    calc = SavingsCalculator()
    calc.price_unit = None

    savings = calc.calculate_spot_savings_per_cycle(
        actual_power_kw=12.0,  # 12 kW for 5 min = 1 kWh
        cycle_minutes=5.0,
        current_price=1.0,
        average_price_today=2.0,
    )

    assert savings == 0.0, (
        "With an unknown unit this used to return 0.01 SEK (the öre assumption) for what "
        "is really a 1.00 SEK saving - or 100x too much the other way. Report nothing."
    )


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

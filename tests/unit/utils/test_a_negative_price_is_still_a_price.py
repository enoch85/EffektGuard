"""Nordic spot prices go to zero and below, and the DHW optimizer's arithmetic broke on both.

Exactly-zero quarters occur roughly a hundred hours a year per SE bidding zone, and negative
prices - where the grid PAYS you to take the power - are routine on windy days.

The DHW optimizer decides whether to heat hot water NOW or defer to a cheaper window. It did that
with:

    if current_quarter_price and optimal_window.avg_price < current_quarter_price:
        price_savings_pct = (current - optimal) / current

**TRUTHINESS.** `if current_quarter_price` is False when the price is exactly 0.00, so the whole
branch is skipped - and the water is heated now rather than deferred to a window where the grid
would have paid for it.

**A SIGNED DIVISOR.** Dividing by the price rather than its magnitude inverts the fraction whenever
the current price is negative:

    current -10 ore, window -60 ore  ->  (-10 - -60) / -10  =  -5.00
    current -50 ore, window -60 ore  ->  (-50 - -60) / -50  =  -0.20

Both windows are genuinely cheaper - the grid pays MORE in them - and both come out negative, fail
the "at least 15 % cheaper" test, and are declined.

AND THE FILE HAD TWO OF THESE COMPARISONS. One of them had already been fixed, comment and all, and
the other had not - because the logic was COPIED rather than shared. Both now call one function.
"""

from __future__ import annotations

import pytest

from custom_components.effektguard.const import DHW_OPTIMAL_WINDOW_MIN_SAVINGS
from custom_components.effektguard.utils.price_math import price_savings_fraction


class TestPricesAtExactlyZero:
    """0.00 ore is a real Nordic price, and `if price:` says it is not a price at all."""

    def test_a_zero_price_is_not_the_same_as_no_price(self):
        savings = price_savings_fraction(current=0.0, candidate=-40.0)

        assert savings is not None, (
            "A current price of exactly 0.00 ore was treated as 'no price' - the truthiness test "
            "`if current_quarter_price` is False on 0.0 - so the optimizer never even considered "
            "deferring the hot water to a window where the grid PAYS 40 ore/kWh to take it. "
            "Exactly-zero prices occur about a hundred hours a year per SE bidding zone."
        )
        assert savings >= DHW_OPTIMAL_WINDOW_MIN_SAVINGS

    def test_free_now_and_paid_later_is_a_total_saving(self):
        """Nothing to divide by. It is still unambiguously worth waiting."""
        assert price_savings_fraction(current=0.0, candidate=-1.0) == 1.0

    def test_free_now_and_dearer_later_is_no_saving(self):
        assert price_savings_fraction(current=0.0, candidate=10.0) is None

    def test_absent_is_not_zero(self):
        """`None` means we do not have a price. It must not be read as 'free'."""
        assert price_savings_fraction(current=None, candidate=-40.0) is None


class TestNegativePrices:
    """The grid pays you. A window that pays MORE is cheaper, and the sign must not flip."""

    @pytest.mark.parametrize(
        ("current", "candidate"),
        [
            (-10.0, -60.0),  # gave -5.00
            (-50.0, -60.0),  # gave -0.20
            (-1.0, -100.0),
        ],
    )
    def test_a_window_that_pays_more_is_a_saving_not_a_loss(self, current, candidate):
        savings = price_savings_fraction(current, candidate)

        assert savings is not None and savings > 0.0, (
            f"With the price at {current} ore and a window at {candidate} ore - where the grid pays "
            f"MORE to take the power - the saving came out as {savings}. Dividing by the SIGNED "
            f"price inverts the fraction, so a genuinely better window fails the 15 % test and the "
            f"hot water is heated now instead."
        )

    def test_the_deeper_negative_window_wins(self):
        assert price_savings_fraction(-10.0, -60.0) > price_savings_fraction(-50.0, -60.0)

    def test_a_shallower_negative_window_is_not_a_saving(self):
        """current -50, window -10: the grid pays LESS there. Do not defer to it."""
        assert price_savings_fraction(current=-50.0, candidate=-10.0) is None

    def test_crossing_zero_downwards_is_a_saving(self):
        assert price_savings_fraction(current=5.0, candidate=-20.0) > 0.0


class TestOrdinaryPositivePrices:
    """The regression guard. None of this may change the common case."""

    def test_a_cheaper_window_is_the_fraction_it_always_was(self):
        assert price_savings_fraction(current=50.0, candidate=30.0) == pytest.approx(0.4)

    def test_a_dearer_window_is_never_a_saving(self):
        assert price_savings_fraction(current=30.0, candidate=50.0) is None

    def test_an_identical_window_is_never_a_saving(self):
        assert price_savings_fraction(current=30.0, candidate=30.0) is None


def test_the_sign_of_the_result_only_ever_reflects_which_price_is_lower():
    """The property the signed divisor destroyed, stated once."""
    prices = [-100.0, -50.0, -10.0, 0.0, 10.0, 50.0, 100.0]

    for current in prices:
        for candidate in prices:
            savings = price_savings_fraction(current, candidate)
            if candidate < current:
                assert (
                    savings is not None and savings > 0.0
                ), f"{candidate} is cheaper than {current} and the saving came out {savings}."
            else:
                assert savings is None, (
                    f"{candidate} is not cheaper than {current}, yet a saving of {savings} was "
                    f"reported."
                )

"""`price_savings_fraction` must handle Nordic prices at zero and below.

The DHW optimizer decides whether to heat now or defer to a cheaper window, and the old arithmetic
broke on both edge cases: `if current_quarter_price` is False at exactly 0.00 (a real price, ~100
hours/year per SE zone), skipping the whole branch; and dividing by the SIGNED price inverts the
fraction when the current price is negative, so a genuinely cheaper (deeper-negative) window comes
out negative and is declined. The fix divides by the MAGNITUDE, returns 1.0 when current is zero and
a cheaper window exists, and returns None (not 0) when there is no current price - shared by both
call sites that had drifted apart.
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

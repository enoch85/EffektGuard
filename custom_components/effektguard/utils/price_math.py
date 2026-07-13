"""How much cheaper is one price than another, when either of them may be zero or negative.

Nordic spot prices go to zero and below. Exactly-zero quarters occur roughly a hundred hours a year
per SE bidding zone, and negative prices - where the grid PAYS you to take the power - are routine
on windy days. Both break the obvious arithmetic, and both broke it here.

    if current_quarter_price and optimal_window.avg_price < current_quarter_price:
        price_savings_pct = (current - optimal) / current

Two failures, in three lines:

**TRUTHINESS.** `if current_quarter_price` is False when the price is exactly 0.00. A real Nordic
price, and the whole branch is skipped - so the hot water is heated NOW rather than deferred to a
window where the grid would have paid for it.

**A SIGNED DIVISOR.** Dividing by the price rather than its magnitude inverts the fraction whenever
the current price is negative:

    current  -10 ore, optimal  -60 ore  ->  (-10 - -60) / -10  =  -5.00
    current  -50 ore, optimal  -60 ore  ->  (-50 - -60) / -50  =  -0.20

Both are genuinely cheaper windows - the grid pays MORE in them - and both come out negative, fail
the "at least 15 % cheaper" test, and are declined.

The DHW optimizer had TWO of these comparisons. One had been fixed, comment and all. The other had
not, because the logic was copied rather than shared. It lives here now, so there is one of it.
"""


def price_savings_fraction(current: float | None, candidate: float) -> float | None:
    """How much cheaper `candidate` is than `current`, as a fraction of what `current` costs.

    Args:
        current: The price right now. `None` means we do not have one - which is NOT the same as
            zero, and the caller must not conflate them.
        candidate: The price of the window being considered.

    Returns:
        The saving as a fraction in [0.0, 1.0+], or None when there is no current price, or when
        `candidate` is not actually cheaper. A window that is not cheaper is never a saving,
        however the arithmetic is arranged.

        The denominator is the MAGNITUDE of the current price, so the sign of the result reflects
        which price is lower and nothing else. When the current price is exactly zero any cheaper
        (i.e. negative) window is a total saving, and 1.0 is returned rather than dividing by zero.
    """
    if current is None:
        return None

    if candidate >= current:
        return None

    reference = abs(current)
    if reference == 0.0:
        # Free now, and being PAID in the candidate window. That is as good as it gets.
        return 1.0

    return (current - candidate) / reference

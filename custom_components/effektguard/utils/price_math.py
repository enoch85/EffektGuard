"""How much cheaper one price is than another, when either may be zero or negative.

Nordic spot prices reach exactly 0.00 (~100 h/year per SE zone) and go negative on windy days,
which breaks the naive `(current - optimal) / current` two ways: truthiness treats a real 0.00
price as "no price" and skips the cheaper window, and a signed divisor inverts the fraction when
the current price is negative, so genuinely cheaper windows score negative and are declined.
Shared so the DHW optimizer's two comparisons cannot drift apart again.

tests/unit/utils/test_a_negative_price_is_still_a_price.py
"""


def price_savings_fraction(current: float | None, candidate: float) -> float | None:
    """How much cheaper `candidate` is than `current`, as a fraction of `current`'s magnitude.

    `current is None` means no current price (NOT zero) and returns None. A candidate that is not
    cheaper returns None. The denominator is |current|, so the sign tracks which price is lower and
    nothing else; when current is exactly 0.00 any cheaper (negative) window returns 1.0 rather than
    dividing by zero.
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

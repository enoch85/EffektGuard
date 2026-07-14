"""The median guard I added to stop the optimiser buying at the day's highest price also
stopped it buying at the day's lowest.

THE PROBLEM IT WAS SOLVING IS REAL. On a high-wind Nordic day the price distribution is not a
curve, it is a step: 83 quarters at 120 ore and 13 at MINUS 10, where the grid pays you to take
the power. The middle of that distribution is a plateau, so p25 == p75 == p90 == 120, and the
83 quarters at the day's HIGHEST price all satisfy `price <= p25`. Without a guard they classify
CHEAP and the optimiser commands +4.0 C of extra heat at the most expensive moment of the day.

The guard was to require each band to sit on the correct SIDE of the median:

    if price <= p10 and price < median:   VERY_CHEAP
    if price <= p25 and price < median:   CHEAP

which works, because 120 is not < 120.

AND IT BREAKS ON THE MIRROR IMAGE, WHICH IS THE MORE COMMON ONE. Turn the step upside down - a
long free stretch and a short expensive one, which is what a windy night into a calm evening
actually looks like - and the plateau IS the median:

    14 hours at exactly 0.00 ore, 10 hours at 80 ore

    p10 = 0.0   p25 = 0.0   median = 0.0   p75 = 80.0   p90 = 80.0

    the 56 free quarters   ->  NORMAL      <- `0.0 < 0.0` is False
    the 40 costly quarters ->  NORMAL      <- `80.0 > 80.0` is False

EVERY QUARTER OF THE DAY IS NORMAL. The price layer goes completely blind on a day with an 80 ore
spread: it will not pre-heat on free electricity, and it will not back off at 80 ore. The one thing
this integration exists to do, and it declines to do it.

Exactly-zero and negative prices are not exotic. price_math's own docstring puts them at "roughly a
hundred hours a year per SE bidding zone", and they arrive in long contiguous runs - which is
precisely the shape that makes the plateau the median.

THE FIX IS TO ASK THE QUESTION THE GUARD WAS STANDING IN FOR: is there anything meaningfully dearer
today? That is `price < p90`, and it belongs on exactly ONE band.

I had put the median on all four, and mutating them one at a time shows three of those guards were
doing nothing at all - they only ever broke the free day. The spread check upstream already
guarantees p90 > p10, so:

    VERY_CHEAP   `price <= p10` already implies `price < p90`.               Redundant.
    PEAK         `price > p90`, and p90 >= p10, implies `price > p10`.       Redundant.
    EXPENSIVE    `price > p75`, and p75 >= p10, implies `price > p10`.       Redundant.
    CHEAP        p25 CAN equal p90 - that IS the dear plateau - so
                 `price <= p25` does NOT imply `price < p90`.                Earns its place.

So one guard, on one band, and it is precisely the one that stops the 120 ore plateau being
classified cheap. Everything else was noise that broke the mirror case.

The dear side deliberately keeps its strict `>`. Loosening it to `>=` would make all 83 quarters of
the high-wind day PEAK, telling the house to coast for twenty hours with only three hours of cheap
power to charge in. A plateau you cannot escape is not a peak; it is just the price of the day.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest

from custom_components.effektguard.optimization.price_layer import (
    PriceAnalyzer,
    QuarterClassification,
)


class _Period:
    """A quarter-hour period, as the price adapter hands them over."""

    def __init__(self, index: int, price: float):
        self.price = price
        self.start = datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc) + timedelta(
            minutes=15 * index
        )
        self.end = self.start + timedelta(minutes=15)


def _classify(prices: list[float]) -> list[QuarterClassification]:
    periods = [_Period(index, price) for index, price in enumerate(prices)]
    result = PriceAnalyzer().classify_quarterly_periods(periods)
    return [result[index] for index in range(len(prices))]


# A windy night into a calm evening. Fourteen hours of free power, ten hours at 80 ore.
FREE_HOURS = 14
COSTLY_HOURS = 10
A_FREE_DAY = [0.0] * (FREE_HOURS * 4) + [80.0] * (COSTLY_HOURS * 4)

# The high-wind day the median guard was written for: a short negative run, a long dear plateau.
A_NEGATIVE_PRICE_DAY = [-10.0] * 13 + [120.0] * 83


class TestFreeElectricityIsBought:
    """The bug. Fourteen hours of free power, and the optimiser would not touch it."""

    def test_the_free_quarters_are_not_called_normal(self):
        classifications = _classify(A_FREE_DAY)
        free = classifications[: FREE_HOURS * 4]

        assert all(c == QuarterClassification.VERY_CHEAP for c in free), (
            f"{FREE_HOURS} hours at exactly 0.00 ore classified as {Counter(c.name for c in free)}. "
            f"The electricity is FREE. It is more than half the day, so it is also the median - and "
            f"the cheap bands demanded `price < median`, which 0.0 is not. The one thing this "
            f"integration exists to do is move heat into hours like these."
        )

    def test_the_expensive_quarters_are_not_called_cheap(self):
        """The other half. Fixing the floor must not tell the house to heat at 80 ore."""
        costly = _classify(A_FREE_DAY)[FREE_HOURS * 4 :]

        assert not any(
            c in (QuarterClassification.VERY_CHEAP, QuarterClassification.CHEAP) for c in costly
        ), (
            f"The 80 ore quarters classified as {Counter(c.name for c in costly)}. They are the "
            f"most expensive power available today and must never be a reason to add heat."
        )

    def test_the_day_is_not_uniformly_normal(self):
        """The symptom, stated plainly: an 80 ore spread produced no signal whatsoever."""
        classifications = _classify(A_FREE_DAY)

        assert len(set(classifications)) > 1, (
            "Every quarter of a day with an 80 ore spread classified NORMAL. The price layer is "
            "blind: it will not pre-heat on free power and it will not coast at 80 ore."
        )


class TestTheCaseTheGuardWasWrittenFor:
    """The regression guard, and it is the more dangerous of the two failures."""

    def test_negative_prices_are_still_very_cheap(self):
        negative = _classify(A_NEGATIVE_PRICE_DAY)[:13]

        assert all(c == QuarterClassification.VERY_CHEAP for c in negative), (
            f"Quarters at MINUS 10 ore - the grid is paying the house to take the power - "
            f"classified as {Counter(c.name for c in negative)}."
        )

    def test_the_dear_plateau_is_never_called_cheap(self):
        """THE bug the median guard exists to prevent: +4.0 C at the day's highest price."""
        plateau = _classify(A_NEGATIVE_PRICE_DAY)[13:]

        assert not any(
            c in (QuarterClassification.VERY_CHEAP, QuarterClassification.CHEAP) for c in plateau
        ), (
            f"The 83 quarters at the day's HIGHEST price (120 ore) classified as "
            f"{Counter(c.name for c in plateau)}. They satisfy `price <= p25` because the plateau "
            f"IS the 25th percentile, and classifying them cheap commands +4.0 C of extra heat at "
            f"the most expensive moment of the day."
        )

    def test_an_inescapable_plateau_is_not_a_peak_either(self):
        """A plateau you cannot escape is not a peak, it is just the price of the day.

        Loosening the dear side to `>=` would fix nothing and would make 83 of the day's 96
        quarters PEAK - telling the house to coast for twenty hours, with three hours of cheap
        power to charge in. The strict `>` stays.
        """
        plateau = _classify(A_NEGATIVE_PRICE_DAY)[13:]

        assert not any(c == QuarterClassification.PEAK for c in plateau), (
            f"{sum(c == QuarterClassification.PEAK for c in plateau)} of the day's 96 quarters "
            f"classified PEAK. There is nowhere to shift the load to."
        )


class TestAnOrdinaryDayIsUntouched:
    """The bands only move where the median IS the plateau. Everywhere else, nothing changes."""

    def test_a_normal_price_curve_still_classifies_every_band(self):
        """A textbook Nordic day: cheap at night, a morning peak, an evening peak."""
        prices = [20.0 + 60.0 * ((index % 48) / 48.0) for index in range(96)]

        classifications = _classify(prices)
        seen = Counter(c.name for c in classifications)

        for band in ("VERY_CHEAP", "CHEAP", "NORMAL", "EXPENSIVE", "PEAK"):
            assert seen[band] > 0, (
                f"An ordinary day with a 60 ore range produced no {band} quarters at all: {seen}. "
                f"The fix was meant to be inert on days where the median is not a plateau."
            )

    def test_the_cheapest_quarters_of_an_ordinary_day_are_the_cheap_ones(self):
        prices = [20.0 + 60.0 * ((index % 48) / 48.0) for index in range(96)]
        classifications = _classify(prices)

        cheapest = min(range(96), key=lambda i: prices[i])
        dearest = max(range(96), key=lambda i: prices[i])

        assert classifications[cheapest] == QuarterClassification.VERY_CHEAP
        assert classifications[dearest] == QuarterClassification.PEAK


@pytest.mark.parametrize("free_fraction", [0.55, 0.60, 0.75, 0.90])
def test_free_power_is_bought_however_much_of_the_day_it_covers(free_fraction):
    """The plateau only has to exceed half the day to become the median. Beyond that it is worse.

    price_math's own docstring puts exactly-zero prices at "roughly a hundred hours a year per SE
    bidding zone", and they arrive in long contiguous runs - which is exactly the shape that makes
    the plateau the median.
    """
    free_quarters = int(96 * free_fraction)
    prices = [0.0] * free_quarters + [80.0] * (96 - free_quarters)

    classifications = _classify(prices)[:free_quarters]

    assert all(c == QuarterClassification.VERY_CHEAP for c in classifications), (
        f"With {free_fraction:.0%} of the day at exactly 0.00 ore, the free quarters classified as "
        f"{Counter(c.name for c in classifications)}."
    )


class TestTheOneGuardThatEarnsItsPlace:
    """`price < p90` on the CHEAP band. Everything else was redundant, and mutation proves it."""

    # Three levels, with the DEAR one spanning p25 through p90. This is the shape that needs the
    # guard: without it the 60 ore quarters - which are p25, p75 AND p90 - classify CHEAP.
    A_DEAR_PLATEAU_AT_THE_QUARTILE = [5.0] * 20 + [60.0] * 76

    def test_a_dear_plateau_sitting_on_p25_is_not_cheap(self):
        prices = self.A_DEAR_PLATEAU_AT_THE_QUARTILE
        plateau = _classify(prices)[20:]

        assert not any(
            c in (QuarterClassification.VERY_CHEAP, QuarterClassification.CHEAP) for c in plateau
        ), (
            f"76 quarters at the day's HIGHEST price classified {Counter(c.name for c in plateau)}. "
            f"They are p25, p75 and p90 all at once, so rank alone calls them cheap. This is the "
            f"one case the guard exists for."
        )

    def test_the_cheap_quarters_of_that_day_are_still_found(self):
        cheap = _classify(self.A_DEAR_PLATEAU_AT_THE_QUARTILE)[:20]

        assert all(
            c in (QuarterClassification.VERY_CHEAP, QuarterClassification.CHEAP) for c in cheap
        ), f"The 5 ore quarters classified {Counter(c.name for c in cheap)}."


class TestAMidLevelPlateauThatIsAlsoTheMedian:
    """The third shape, and the one that proves `p90` is the right question and `median` is not.

    p25 is never above the median, so on the CHEAP band `price <= p25` already implies
    `price <= median`. The two spellings can therefore only disagree when p25 IS the median - a
    plateau covering the whole lower half of the day - and that plateau is still meaningfully
    cheaper than the evening:

        12 quarters at 0 ore, 40 at 30 ore, 44 at 90 ore
        p10 = 0    p25 = 30    median = 30    p75 = 90    p90 = 90

    Heating at 30 rather than at 90 is a third of the price. The band exists to say so. Asking
    `price < median` says 30 is not below 30 and calls twenty hours of cheap power NORMAL.
    """

    A_MID_PLATEAU_DAY = [0.0] * 12 + [30.0] * 40 + [90.0] * 44

    def test_the_mid_plateau_is_cheap_because_it_is_cheaper_than_the_evening(self):
        plateau = _classify(self.A_MID_PLATEAU_DAY)[12:52]

        assert all(c == QuarterClassification.CHEAP for c in plateau), (
            f"40 quarters at 30 ore - against an evening at 90 - classified "
            f"{Counter(c.name for c in plateau)}. They are the 25th percentile AND the median, so "
            f"`price < median` rejects them. They are a third of the evening price."
        )

    def test_the_free_quarters_are_still_the_very_cheap_ones(self):
        assert all(
            c == QuarterClassification.VERY_CHEAP for c in _classify(self.A_MID_PLATEAU_DAY)[:12]
        )

    def test_the_evening_is_still_the_expensive_one(self):
        evening = _classify(self.A_MID_PLATEAU_DAY)[52:]

        assert not any(
            c in (QuarterClassification.VERY_CHEAP, QuarterClassification.CHEAP) for c in evening
        ), f"The 90 ore evening classified {Counter(c.name for c in evening)}."

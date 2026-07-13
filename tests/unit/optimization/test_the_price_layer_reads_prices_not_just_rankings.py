"""Percentile RANK is scale-invariant, so on its own it cannot see a price at all.

The price layer banded every quarter by where it ranked in the day. That is all it did, and it has
two consequences that a ranking can never notice.

**A flat day earned the full banding.** A day that ran from 39.80 to 40.20 ore - a spread of four
tenths of an ore - was classified VERY_CHEAP through PEAK, commanding offsets from +4.0 C to
-10.0 C. Fourteen degrees of swing on a heat pump, to chase four tenths of an ore.

**Free electricity was classified NORMAL.** On a high-wind day - 83 quarters at 120 ore and 13 at
MINUS 10, where the grid pays you to take the power - the MIDDLE of the distribution is a plateau,
so p25 == p75 == p90 == 120. The old guard tested exactly that (`if p25 == p90`) and gave up,
marking the whole day NORMAL. The cheapest power of the year went unbought.

AND THE OBVIOUS FIX IS WORSE THAN THE BUG. Simply deleting that guard makes the 83 quarters at the
day's HIGHEST price satisfy `price <= p25`, so they are classified CHEAP - commanding +4.0 C of
extra heat at the most expensive moment of the day. That trap is why an earlier attempt at this was
reverted, and it is pinned below.

The fix is two rules, and neither of them needs to know what a price is worth:

  * a band must sit on the correct SIDE of the median, which resolves the plateau;
  * the day's spread must be material against the day's own price SCALE, which resolves the flat
    day - and being relative, it survives the fact that NOTHING HERE KNOWS ITS UNIT. `PriceData`
    carries none, and GE-Spot publishes whatever the owner configured. An absolute threshold in ore
    would be a hundred times wrong for anyone reporting SEK/kWh, and it is precisely because
    ranking is scale-invariant that nobody has ever noticed.
"""

from __future__ import annotations

import collections
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from custom_components.effektguard.adapters.gespot_adapter import QuarterPeriod
from custom_components.effektguard.const import (
    PRICE_FLAT_DAY_SPREAD_FRACTION,
    QuarterClassification,
)
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer

MIDNIGHT = datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)


def _day(prices: list[float]) -> list[QuarterPeriod]:
    return [
        QuarterPeriod(start_time=MIDNIGHT + timedelta(minutes=15 * i), price=float(p))
        for i, p in enumerate(prices)
    ]


def _bands(prices: list[float]) -> collections.Counter:
    result = PriceAnalyzer().classify_quarterly_periods(_day(prices))
    return collections.Counter(c.value if hasattr(c, "value") else c for c in result.values())


HIGH_WIND = [120.0] * 83 + [-10.0] * 13
FLAT = list(np.linspace(39.8, 40.2, 96))
ORDINARY = [28.0] * 32 + [40.0] * 32 + [52.0] * 32
VOLATILE = list(np.linspace(20.0, 250.0, 96))


class TestFreeElectricityIsBought:
    """The grid is PAYING you. This is the single cheapest power of the year."""

    def test_the_negative_quarters_are_classified_very_cheap(self):
        bands = _bands(HIGH_WIND)

        assert bands[QuarterClassification.VERY_CHEAP] == 13, (
            f"On a day with 13 quarters at MINUS 10 ore - the grid paying you to take the power - "
            f"the classification came out {dict(bands)}. The middle of the distribution is a "
            f"plateau (p25 == p75 == p90 == 120), and the old guard tested exactly that and gave "
            f"up, marking the whole day NORMAL."
        )

    def test_the_expensive_plateau_is_not_classified_cheap(self):
        """THE TRAP. Deleting the plateau guard naively is WORSE than leaving the bug in."""
        bands = _bands(HIGH_WIND)

        assert bands[QuarterClassification.CHEAP] == 0, (
            f"The 83 quarters at the day's HIGHEST price (120 ore) were classified CHEAP - which "
            f"commands +4.0 C of EXTRA HEAT at the most expensive moment of the day. They satisfy "
            f"`price <= p25` because p25 sits on the plateau. Got {dict(bands)}."
        )
        assert bands[QuarterClassification.NORMAL] == 83


class TestAFlatDayIsNotOptimised:
    """Ranking noise is not a price signal."""

    def test_four_tenths_of_an_ore_does_not_earn_a_fourteen_degree_swing(self):
        bands = _bands(FLAT)

        assert set(bands) == {QuarterClassification.NORMAL}, (
            f"A day spanning 39.80 to 40.20 ore - a spread of 0.4 ore - was classified "
            f"{dict(bands)}. VERY_CHEAP commands +4.0 C and PEAK commands -10.0 C, so this is a "
            f"14 C swing in commanded offset, and a heat pump thrown around all day, to chase four "
            f"tenths of an ore."
        )

    def test_the_test_is_relative_because_nothing_here_knows_its_unit(self):
        """The same flat day in SEK/kWh instead of ore. An absolute threshold would be 100x wrong.

        PriceData carries no unit. GE-Spot publishes whatever the owner configured. A threshold
        expressed in ore would silently misbehave for every user reporting SEK/kWh - and because
        percentile ranking is scale-invariant, nothing would ever have flagged it.
        """
        in_sek = [p / 100.0 for p in FLAT]

        assert _bands(in_sek) == _bands(FLAT), (
            "The same day, priced in SEK/kWh rather than ore/kWh, classified differently. The "
            "flat-day test must be scale-invariant - the layer does not know its own unit."
        )

    def test_a_genuinely_volatile_day_is_still_optimised(self):
        """The regression guard on the guard: do not switch the product off."""
        bands = _bands(VOLATILE)

        assert bands[QuarterClassification.PEAK] > 0
        assert bands[QuarterClassification.VERY_CHEAP] > 0

    def test_the_threshold_is_a_fraction_of_the_days_own_scale(self):
        assert 0.0 < PRICE_FLAT_DAY_SPREAD_FRACTION < 0.5


class TestTheRegressionThatGotTheLastAttemptReverted:
    """An ordinary day must not suddenly sprout critical PEAK quarters."""

    def test_an_ordinary_day_produces_no_peak_quarters(self):
        """A previous attempt flipped `> p90` to `>= p90` and turned a THIRD of an ordinary day
        into PEAK quarters at weight 1.0 and PRICE_OFFSET_PEAK (-10.0). It had to be reverted.
        """
        bands = _bands(ORDINARY)

        assert bands[QuarterClassification.PEAK] == 0, (
            f"An ordinary 28/40/52 ore day produced {bands[QuarterClassification.PEAK]} PEAK "
            f"quarters. PEAK commands -10.0 C at critical weight. A third of an ordinary day "
            f"spent at maximum heat reduction is how the last attempt at this was reverted."
        )

    def test_an_ordinary_day_is_unchanged_by_this_fix(self):
        bands = _bands(ORDINARY)
        assert bands[QuarterClassification.VERY_CHEAP] == 32
        assert bands[QuarterClassification.NORMAL] == 64


class TestTheOldBehaviourThatWasCorrect:
    def test_the_uniform_fallback_day_is_still_all_normal(self):
        assert set(_bands([1.0] * 96)) == {QuarterClassification.NORMAL}

    def test_an_all_negative_day_is_still_ranked(self):
        """Prices below zero happen routinely in SE1-SE4. Relative differences still matter."""
        bands = _bands(list(np.linspace(-50.0, -5.0, 96)))

        assert bands[QuarterClassification.VERY_CHEAP] > 0
        assert bands[QuarterClassification.PEAK] > 0

    def test_an_empty_day_does_not_raise(self):
        assert PriceAnalyzer().classify_quarterly_periods([]) == {}

"""A day with a flat plateau is not a day without a price signal.

The classifier short-circuits to "everything is NORMAL, no optimization" when it decides prices
are uniform. That guard exists for fallback mode, where the adapter has no data and invents 96
identical quarters - classifying those would be inventing a signal that does not exist.

It tested `p25 == p90`, which is true whenever the middle 65% of the day sits at ONE price. It
does not mean the day is flat; it means the day has a PLATEAU. A Nordic day with many hours of
near-zero prices - high wind, low demand, the exact day worth optimising - has precisely that
shape, and so does any day with a long block at the same clearing price.

Measured on a day of 83 quarters at 120 ore and 13 quarters at MINUS 10 ore: p25 = p90 = 120, the
day is declared uniform, and all 96 quarters - including the ones where the grid is PAYING to be
consumed from - are classified NORMAL. The price layer then bids +0.00. Optimisation switches
itself off on the most profitable day of the year.

Uniform means uniform: no spread between the cheapest and dearest quarter at all.
"""

from datetime import datetime, timedelta

import pytest

from custom_components.effektguard.adapters.gespot_adapter import QuarterPeriod
from custom_components.effektguard.const import QuarterClassification
from custom_components.effektguard.optimization.price_layer import (
    PriceAnalyzer,
    get_fallback_prices,
)

DAY = datetime(2026, 1, 15, 0, 0)


def _day(prices: list[float]) -> list[QuarterPeriod]:
    return [
        QuarterPeriod(start_time=DAY + timedelta(minutes=15 * q), price=price)
        for q, price in enumerate(prices)
    ]


@pytest.fixture
def analyzer() -> PriceAnalyzer:
    return PriceAnalyzer()


def test_a_negative_price_is_never_normal(analyzer):
    """When the grid pays you to consume, that is not a NORMAL quarter."""
    prices = [-10.0 if 44 <= q <= 56 else 120.0 for q in range(96)]

    classes = analyzer.classify_quarterly_periods(_day(prices))

    free_money = {classes[q] for q in range(44, 57)}
    assert free_money == {QuarterClassification.VERY_CHEAP}, (
        f"13 quarters at -10 ore/kWh - the grid paying to be consumed from - were classified "
        f"{free_money}. The day has 83 quarters at 120 ore, so p25 == p90 == 120 and the day is "
        f"declared 'uniform'. Optimisation switches itself off on the most profitable day there is."
    )


def test_a_plateau_day_still_has_a_dear_end(analyzer):
    """The same day's expensive quarters must still be recognised as expensive."""
    prices = [-10.0 if 44 <= q <= 56 else 120.0 for q in range(96)]

    classes = analyzer.classify_quarterly_periods(_day(prices))

    assert QuarterClassification.NORMAL not in {classes[q] for q in range(44, 57)}
    assert len({c for c in classes.values()}) > 1, "a day with a 130 ore spread has a signal"


def test_a_long_cheap_block_day_still_coasts_the_dear_evening(analyzer):
    """High wind, low demand: most of the day near zero, a short dear evening.

    The plateau guard swallowed this day whole - every quarter NORMAL, no signal, no action.

    Note what the RIGHT answer is here, because it is not "call 72 quarters VERY_CHEAP". The
    fabric fills in two or three hours; there is nothing to charge for eighteen. Commanding +4 °C
    of pre-heat across three quarters of a day would not arbitrage anything, it would just cook
    the house. On a day that is mostly free, being free IS the normal state - and the whole of the
    arbitrage is to COAST through the expensive evening. That is what must be recognised.
    """
    prices = [0.5 if q < 72 else 90.0 for q in range(96)]

    classes = analyzer.classify_quarterly_periods(_day(prices))

    dear_evening = {classes[q] for q in range(72, 96)}
    assert dear_evening <= {QuarterClassification.EXPENSIVE, QuarterClassification.PEAK}, (
        f"An evening at 90 ore against a day of 0.5 ore must be recognised as dear so the house "
        f"coasts through it. Got {dear_evening}."
    )
    assert len(set(classes.values())) > 1, "a day with a 90 ore spread has a signal to trade on"


def test_genuinely_uniform_prices_are_still_refused(analyzer):
    """The guard must keep doing its real job: fallback data carries no signal to trade on."""
    classes = analyzer.classify_quarterly_periods(get_fallback_prices().today)

    assert set(classes.values()) == {QuarterClassification.NORMAL}, (
        "Fallback prices are 96 identical invented values. Classifying them would manufacture a "
        "price signal out of the absence of one."
    )


def test_a_hair_of_variance_is_not_a_signal(analyzer):
    """Floating-point noise on a flat tariff must not become a trading signal either."""
    prices = [1.0 for _ in range(96)]
    classes = analyzer.classify_quarterly_periods(_day(prices))

    assert set(classes.values()) == {QuarterClassification.NORMAL}

"""Heat for what is coming - a dearer period, a colder day - is bought in a cheap or very cheap
quarter. It takes hours to reach a slab, so a slab buys it early, never at a normal price. A lighter
house would lose heat bought that early, so it may still buy heat for a price jump in a normal one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    WEATHER_GENTLE_OFFSET,
    THERMAL_MASS_CONCRETE_UFH_THRESHOLD,
    QuarterClassification,
)
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.weather_layer import WeatherPredictionLayer

MIDNIGHT = datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)
DEAR_START = 56  # two dear hours from 14:00
NORMAL_QUARTER = 45  # 59 öre, in a run of four normal quarters, eleven quarters before them
CHEAP_QUARTER = 43  # 53 öre


def _prices() -> PriceData:
    prices = [50.0 + (q % 7) * 3.0 for q in range(96)]
    for q in range(DEAR_START, DEAR_START + 8):
        prices[q] = 400.0
    return PriceData(
        today=[
            QuarterPeriod(MIDNIGHT + timedelta(minutes=15 * q), p) for q, p in enumerate(prices)
        ],
        tomorrow=[],
        has_tomorrow=False,
    )


def _price_vote(freezer, quarter: int, thermal_mass: float = THERMAL_MASS_CONCRETE_UFH_THRESHOLD):
    now = MIDNIGHT + timedelta(minutes=15 * quarter + 5)
    freezer.move_to(now)
    analyzer = PriceAnalyzer()
    prices = _prices()
    analyzer.update_prices(prices)
    state = NibeState(
        outdoor_temp=-5.0,
        indoor_temp=21.0,
        supply_temp=35.0,
        return_temp=30.0,
        degree_minutes=-100.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=now,
    )
    return analyzer.evaluate_layer(
        nibe_state=state,
        price_data=prices,
        thermal_mass=thermal_mass,
        target_temp=21.0,
        tolerance=1.2,
        mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],
    )


def test_a_slab_does_not_pre_heat_in_a_normal_quarter_for_a_dearer_one_ahead(freezer):
    vote = _price_vote(freezer, NORMAL_QUARTER)

    assert vote.classification == QuarterClassification.NORMAL, vote.reason
    assert "pre-heat now" not in vote.reason, vote.reason


def test_a_lighter_house_still_does(freezer):
    vote = _price_vote(freezer, NORMAL_QUARTER, thermal_mass=1.0)

    assert "pre-heat now" in vote.reason, vote.reason


def test_a_cheap_quarter_does(freezer):
    vote = _price_vote(freezer, CHEAP_QUARTER)

    assert vote.classification == QuarterClassification.CHEAP, vote.reason
    assert "pre-heat now" in vote.reason, vote.reason


@pytest.mark.parametrize(
    ("classification", "pre_heats"),
    [
        (QuarterClassification.VERY_CHEAP, True),
        (QuarterClassification.CHEAP, True),
        (QuarterClassification.NORMAL, False),
        (QuarterClassification.EXPENSIVE, False),
        (QuarterClassification.PEAK, False),
        (None, True),  # no prices: nothing to wait for
    ],
)
def test_a_colder_day_is_pre_heated_in_a_cheap_quarter(classification, pre_heats):
    state = MagicMock(outdoor_temp=0.0, timestamp=MIDNIGHT)
    weather = MagicMock()
    weather.forecast_hours = [MagicMock(temperature=t) for t in [0.0] * 12 + [-10.0] * 36]

    vote = WeatherPredictionLayer(thermal_mass=1.0).evaluate_layer(
        nibe_state=state,
        weather_data=weather,
        thermal_trend={"rate_per_hour": 0.0, "confidence": 0.8},
        past_day_mean=0.0,
        price_classification=classification,
    )

    assert (vote.offset == WEATHER_GENTLE_OFFSET) is pre_heats, vote.reason

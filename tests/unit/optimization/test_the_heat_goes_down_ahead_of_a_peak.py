"""The price layer turns the heat down PRICE_PRE_PEAK_QUARTERS ahead of a PEAK, not one quarter.

One quarter was too late: a pump deep in degree minutes kept running fast into the PEAK, because its
degree minutes had no time to recover before the PEAK arrived.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    PRICE_PRE_PEAK_QUARTERS,
)
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer

MIDNIGHT = datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)
PEAK_START = 64  # the day's dearest two hours start at 16:00


def _prices() -> PriceData:
    prices = [50.0 + (q % 7) * 3.0 for q in range(96)]
    for q in range(PEAK_START, PEAK_START + 8):
        prices[q] = 400.0
    return PriceData(
        today=[
            QuarterPeriod(MIDNIGHT + timedelta(minutes=15 * q), p) for q, p in enumerate(prices)
        ],
        tomorrow=[],
        has_tomorrow=False,
    )


def _vote(freezer, quarters_ahead: int):
    now = MIDNIGHT + timedelta(minutes=15 * (PEAK_START - quarters_ahead) + 5)
    freezer.move_to(now)
    analyzer = PriceAnalyzer()
    prices = _prices()
    analyzer.update_prices(prices)
    state = NibeState(
        outdoor_temp=-5.0,
        indoor_temp=21.0,
        supply_temp=35.0,
        return_temp=30.0,
        degree_minutes=-300.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=now,
    )
    return analyzer.evaluate_layer(
        nibe_state=state,
        price_data=prices,
        thermal_mass=1.0,
        target_temp=21.0,
        tolerance=1.2,
        mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],
    )


@pytest.mark.parametrize("quarters_ahead", range(1, PRICE_PRE_PEAK_QUARTERS + 1))
def test_a_peak_within_reach_turns_the_heat_down(freezer, quarters_ahead):
    vote = _vote(freezer, quarters_ahead)

    assert vote.offset < 0 and "peak within" in vote.reason, vote.reason


def test_a_peak_further_away_does_not(freezer):
    vote = _vote(freezer, PRICE_PRE_PEAK_QUARTERS + 1)

    assert "peak within" not in vote.reason, vote.reason

"""A pre-heat for a dearer period ahead keeps its weight while the current quarter's class flickers.

A quarter whose price class runs for less than 45 minutes is "volatile", and the price layer then
votes at 30 % of its weight - meant for a class boost too short to be worth a write. It also cut the
look-ahead's pre-heat, which reads the hours ahead, not the current class. On 15-minute SE4 prices
(14-27 Oct 2026, the house held at target) that cut 116 of the 529 quarters whose vote was a
pre-heat; on the 26th it fell on 00:00-00:45, with 101.8 öre/kWh at 08:00 after 2.9-20.9 öre from
23:00 to 05:00.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
from custom_components.effektguard.const import (
    LAYER_WEIGHT_PRICE,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
)
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer

MIDNIGHT = datetime(2026, 10, 26, tzinfo=ZoneInfo("Europe/Stockholm"))
# A night whose quarters alternate between two cheap prices, a dear morning, a middling day.
NIGHT = [6.0, 12.0] * 12  # 00:00-06:00
MORNING = [100.0] * 16  # 06:00-10:00
DAY = [50.0] * 56  # 10:00-24:00
PRICES = NIGHT + MORNING + DAY


def _vote(quarter: int):
    today = [
        QuarterPeriod(start_time=MIDNIGHT + timedelta(minutes=15 * i), price=p)
        for i, p in enumerate(PRICES)
    ]
    prices = PriceData(today=today, tomorrow=[], has_tomorrow=False)
    analyzer = PriceAnalyzer()
    analyzer.update_prices(prices)
    return analyzer.evaluate_layer(
        nibe_state=SimpleNamespace(indoor_temp=21.0, degree_minutes=0.0),
        price_data=prices,
        thermal_mass=2.0,
        target_temp=21.0,
        tolerance=1.2,
        mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],
    )


@pytest.mark.parametrize("quarter", [4, 5], ids=["01:00 at 6 öre", "01:15 at 12 öre"])
def test_a_flickering_night_quarter_pre_heats_at_full_weight(freezer, quarter):
    freezer.move_to(MIDNIGHT + timedelta(minutes=15 * quarter, seconds=10))

    vote = _vote(quarter)

    assert "Price volatile" in vote.reason, f"the night must flicker for this test: {vote.reason}"
    assert vote.offset > 0, vote.reason
    assert vote.weight == pytest.approx(LAYER_WEIGHT_PRICE), (
        f"a pre-heat before a morning at 100 öre voted {vote.offset:+.2f} at weight "
        f"{vote.weight:.2f}: {vote.reason}"
    )

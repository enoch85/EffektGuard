"""A GE-Spot entry whose value is NaN or infinite is dropped, like one with no value.

`float()` accepts "nan" and "inf". A NaN in the price layer's look-ahead made `min()` return NaN,
which no price equals, and the search for the cheapest quarter raised StopIteration - an exception
the decision's handler does not catch - so the cycle failed, the entities went unavailable and
nothing was written, every cycle until the NaN left the window (control review F-7).
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.gespot_adapter import GESpotAdapter


def _raw_day(odd_value) -> list[dict[str, object]]:
    base = dt_util.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return [
        {
            "time": (base + timedelta(minutes=15 * i)).isoformat(),
            "value": odd_value if i == 50 else 40.0 + i % 24,
        }
        for i in range(96)
    ]


@pytest.mark.parametrize("odd_value", [float("nan"), "nan", float("inf"), "-inf"])
def test_a_price_that_is_not_a_finite_number_is_dropped(odd_value):
    periods = GESpotAdapter(MagicMock(), {"gespot_entity": "sensor.gespot"})._parse_periods(
        _raw_day(odd_value)
    )

    assert len(periods) == 95, f"{odd_value!r} became a price period"
    assert all(p.price == p.price and abs(p.price) != float("inf") for p in periods)

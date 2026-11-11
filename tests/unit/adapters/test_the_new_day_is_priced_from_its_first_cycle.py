"""The first cycle after midnight prices the new day, before GE-Spot has written it.

GE-Spot dates each quarter when it writes its state (ge_spot/sensor/base.py dates its HH:MM keys
with the current date) and writes the new day a random 0-30 s after midnight. The cycle at 00:00:10
could read the state written before midnight: "today" was the day that had ended, no quarter
covered the time, and the price layer abstained with "No price interval covers the current time".
The day that has begun was there all along, under "tomorrow".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.gespot_adapter import GESpotAdapter

STOCKHOLM = ZoneInfo("Europe/Stockholm")
ENDED = datetime(2026, 11, 10, tzinfo=STOCKHOLM)
BEGUN = datetime(2026, 11, 11, tzinfo=STOCKHOLM)
FIRST_CYCLE = BEGUN + timedelta(seconds=10)


def _day(midnight: datetime) -> list[dict[str, object]]:
    return [
        {"time": midnight + timedelta(minutes=15 * quarter), "value": 40.0 + quarter}
        for quarter in range(96)
    ]


@pytest.mark.parametrize(
    ("tomorrow", "first_quarter"),
    [(_day(BEGUN), BEGUN), ([], None)],
    ids=["the new day is there under tomorrow", "no prices for the new day"],
)
async def test_the_first_cycle_after_midnight_finds_its_quarter(
    monkeypatch, tomorrow, first_quarter
):
    """With the new day under tomorrow, it is today. Without it, nothing is made up."""
    monkeypatch.setattr(dt_util, "utcnow", lambda: FIRST_CYCLE.astimezone(timezone.utc))
    state = MagicMock()
    state.state = "40"
    state.attributes = {
        "unit_of_measurement": "öre/kWh",
        "today_interval_prices": _day(ENDED),
        "tomorrow_interval_prices": tomorrow,
    }
    hass = MagicMock()
    hass.states.get.return_value = state

    prices = await GESpotAdapter(hass, {"gespot_entity": "sensor.gespot"}).get_prices()

    index = prices.get_period_index(FIRST_CYCLE)
    assert (prices.today[index].start_time if index is not None else None) == first_quarter, (
        f"At {FIRST_CYCLE.isoformat()} today runs from {prices.today[0].start_time.isoformat()}: "
        "the price layer finds no quarter for the time and abstains for the cycle."
    )
    assert not prices.has_tomorrow

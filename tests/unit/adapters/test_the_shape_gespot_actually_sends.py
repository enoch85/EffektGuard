"""Pin the price parser to the shape GE-Spot actually publishes: datetime objects.

`_parse_periods` accepts `time` as either an ISO string or a timezone-aware datetime.
GE-Spot sends the datetime-object form (its sensor/base.py builds `{"time": dt, ...}`),
but the other tests all build fixtures with `.isoformat()`, exercising only the string
branch. This file exercises the datetime branch: a full day parses and stays tz-aware and
time-ordered; `value` (the billed price) is used, not `raw_value` (pre-VAT/tariff); a
missing `value` is still dropped, not defaulted to 0.0; and a naive datetime does not
crash the parser but resolves to None at the timestamp-containment lookup.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from custom_components.effektguard.adapters.gespot_adapter import GESpotAdapter

STOCKHOLM = ZoneInfo("Europe/Stockholm")


def _adapter() -> GESpotAdapter:
    return GESpotAdapter(MagicMock(), {"gespot_entity": "sensor.gespot"})


def _live_day(broken_at: int | None = None) -> list[dict[str, object]]:
    """A day exactly as GE-Spot builds it: datetime objects, and a pre-VAT `raw_value`.

    Mirrors ge_spot/sensor/base.py - `datetime(y, m, d, hour, minute, 0, tzinfo=target_tz)`,
    `value` rounded to 4 places, `raw_value` added only when GE-Spot has it.
    """
    midnight = datetime(2026, 1, 15, 0, 0, tzinfo=STOCKHOLM)
    day: list[dict[str, object]] = []

    for quarter in range(96):
        item: dict[str, object] = {"time": midnight + timedelta(minutes=15 * quarter)}
        if quarter != broken_at:
            item["value"] = round(40.0 + quarter * 0.5, 4)
            item["raw_value"] = round((40.0 + quarter * 0.5) * 0.6, 4)  # before VAT and tariffs
        day.append(item)

    return day


def test_the_shape_gespot_actually_publishes_parses():
    """A datetime object in `time`, not an ISO string. The production path, finally exercised."""
    periods = _adapter()._parse_periods(_live_day())

    assert len(periods) == 96, (
        "GE-Spot's real output - datetime objects in `time` - did not parse into a full day. "
        "This is the shape the adapter receives in production."
    )
    assert all(period.start_time.tzinfo is not None for period in periods), (
        "A timezone-aware datetime from GE-Spot came back naive. Every downstream comparison is "
        "against dt_util.now(), which is aware; mixing the two raises TypeError, and PriceData "
        "swallows it and returns None - silently pricing every quarter as unknown."
    )


def test_the_instant_gespot_sent_is_the_instant_we_store():
    """No round-trip through a string, so no chance to lose the offset."""
    day = _live_day()
    periods = _adapter()._parse_periods(day)

    assert periods[0].start_time == day[0]["time"]
    assert periods[40].start_time == day[40]["time"]


def test_the_pre_vat_price_is_not_mistaken_for_the_price_the_owner_pays():
    """`raw_value` is the market price before VAT and tariffs. It is not what anything costs.

    GE-Spot publishes both. `value` is what the owner is billed; `raw_value` is roughly 60 % of
    it. They differ by enough that optimising against the wrong one would rank quarters by a
    number nobody pays - and, worse, would look entirely plausible in every log and every chart.
    """
    periods = _adapter()._parse_periods(_live_day())

    assert periods[0].price == 40.0, (
        f"The parser took {periods[0].price} for the first quarter. `value` (40.0) is the price "
        f"the owner pays; `raw_value` (24.0) is the market price before VAT and tariffs. "
        f"Optimising against the pre-tax price ranks quarters by a number nobody is billed for."
    )


def test_a_missing_price_is_still_dropped_on_the_path_that_actually_runs():
    """A missing `value` is dropped on the datetime path too, not defaulted to 0.0."""
    periods = _adapter()._parse_periods(_live_day(broken_at=50))

    assert len(periods) == 95, (
        "A GE-Spot entry with a real datetime but no `value` key was still turned into a price "
        "period. On this path - the production path - the missing price is invented as 0.0, the "
        "cheapest possible price, and that quarter is ranked the best of the day and answered "
        "with the most aggressive pre-heating the price layer can command."
    )
    assert all(period.price >= 40.0 for period in periods), "a fabricated 0.0 survived"


def test_a_live_day_is_ordered_by_instant_without_ever_seeing_a_string():
    """The sort key is `.timestamp()`, which needs the datetime path to be right."""
    shuffled = _live_day()
    shuffled.reverse()

    periods = _adapter()._parse_periods(shuffled)

    instants = [period.start_time.timestamp() for period in periods]
    assert instants == sorted(instants), "GE-Spot's intervals did not come back in time order"
    assert periods[0].start_time.hour == 0
    assert periods[-1].start_time.hour == 23


def test_a_naive_datetime_from_a_foreign_price_integration_is_not_silently_accepted():
    """A naive datetime parses but must resolve to None at the containment lookup.

    A naive timestamp compared against an aware dt_util.now() raises TypeError, which
    _index_containing catches and answers with None - it must never raise into pump control.
    """
    from custom_components.effektguard.adapters.gespot_adapter import PriceData

    naive = [
        {"time": datetime(2026, 1, 15, 0, 0) + timedelta(minutes=15 * q), "value": 40.0 + q}
        for q in range(4)
    ]

    periods = _adapter()._parse_periods(naive)
    price_data = PriceData(today=periods, tomorrow=[], has_tomorrow=False)

    # The lookup refuses rather than raising into pump control.
    assert price_data.get_period_index(datetime(2026, 1, 15, 0, 7, tzinfo=timezone.utc)) is None

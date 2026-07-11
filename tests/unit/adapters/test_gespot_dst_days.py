"""Regression tests for DST/gap handling in the GE-Spot adapter.

The coordinator indexes today[current_quarter] positionally, so parsed days
must stay dense at 96 entries. DST days (92 or 100 periods) and mid-day data
gaps previously got synthetic day-average prices (skewing classification) and
filled entries stamped with today's date even in tomorrow's list.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from custom_components.effektguard.adapters.gespot_adapter import GESpotAdapter


def make_adapter() -> GESpotAdapter:
    return GESpotAdapter(hass=MagicMock(), config={"gespot_entity": "sensor.gespot"})


def make_raw_day(base: datetime, skip_hours: tuple[int, ...] = ()) -> list[dict]:
    """Build a raw GE-Spot day; price = quarter index for traceability."""
    raw = []
    for quarter in range(96):
        hour = quarter // 4
        if hour in skip_hours:
            continue
        raw.append(
            {
                "time": (base + timedelta(minutes=15 * quarter)).isoformat(),
                "value": float(quarter),
            }
        )
    return raw


class TestSpringDstDay:
    def test_92_period_day_preserves_only_real_intervals(self):
        adapter = make_adapter()
        base = datetime(2026, 3, 29, 0, 0)  # EU spring DST: 02:00 skipped
        periods = adapter._parse_periods(make_raw_day(base, skip_hours=(2,)))

        assert len(periods) == 92
        assert all(period.start_time.hour != 2 for period in periods)
        assert periods[8].start_time.hour == 3


class TestAutumnDstDay:
    def test_100_period_day_preserves_both_delivery_hours(self):
        adapter = make_adapter()
        raw = [
            {"time": "2026-10-25T02:00:00+02:00", "value": 10.0},
            {"time": "2026-10-25T02:15:00+02:00", "value": 11.0},
            {"time": "2026-10-25T02:00:00+01:00", "value": 90.0},
            {"time": "2026-10-25T02:15:00+01:00", "value": 91.0},
        ]
        periods = adapter._parse_periods(raw)

        assert len(periods) == 4
        first_hour = datetime(2026, 10, 25, 0, 0, tzinfo=timezone.utc)
        second_hour = datetime(2026, 10, 25, 1, 0, tzinfo=timezone.utc)
        from custom_components.effektguard.adapters.gespot_adapter import PriceData

        price_data = PriceData(today=periods, tomorrow=[], has_tomorrow=False)
        assert price_data.get_period(first_hour).price == 10.0
        assert price_data.get_period(second_hour).price == 90.0


class TestDegenerateDays:
    def test_empty_day_returns_empty_not_fabricated(self):
        adapter = make_adapter()
        assert adapter._parse_periods([]) == []

    def test_leading_gap_backfills_from_first_real_price(self):
        adapter = make_adapter()
        base = datetime(2026, 1, 7, 0, 0)
        periods = adapter._parse_periods(make_raw_day(base, skip_hours=(0,)))
        assert len(periods) == 92
        assert periods[0].start_time.hour == 1
        assert periods[0].price == 4.0

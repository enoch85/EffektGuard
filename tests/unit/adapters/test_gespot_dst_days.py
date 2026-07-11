"""Regression tests for DST/gap handling in the GE-Spot adapter.

The coordinator indexes today[current_quarter] positionally, so parsed days
must stay dense at 96 entries. DST days (92 or 100 periods) and mid-day data
gaps previously got synthetic day-average prices (skewing classification) and
filled entries stamped with today's date even in tomorrow's list.
"""

from datetime import datetime, timedelta
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
    def test_92_period_day_stays_dense_and_forward_fills(self):
        adapter = make_adapter()
        base = datetime(2026, 3, 29, 0, 0)  # EU spring DST: 02:00 skipped
        periods = adapter._parse_periods(make_raw_day(base, skip_hours=(2,)))

        assert len(periods) == 96
        # Positional alignment preserved: quarter 12 (03:00) is still 03:00.
        assert periods[12].price == 12.0
        # The skipped hour forward-fills from 01:45 (quarter 7), not a
        # day-average (which would be ~47.6 and skew classification).
        for quarter in range(8, 12):
            assert periods[quarter].price == 7.0

    def test_filled_entries_carry_the_days_date(self):
        adapter = make_adapter()
        tomorrow = datetime(2026, 3, 29, 0, 0)
        periods = adapter._parse_periods(make_raw_day(tomorrow, skip_hours=(2,)))
        # Previously stamped with dt_util.now()'s date (today), corrupting
        # tomorrow's timeline.
        assert periods[8].start_time.date() == tomorrow.date()


class TestAutumnDstDay:
    def test_100_period_day_dedupes_first_occurrence_wins(self):
        adapter = make_adapter()
        base = datetime(2026, 10, 25, 0, 0)
        raw = make_raw_day(base)
        # Repeated 02:00 hour: 4 extra periods with duplicate wall-clock
        # quarters but different prices (second Nord Pool delivery hour).
        for quarter in range(8, 12):
            raw.append(
                {
                    "time": (base + timedelta(minutes=15 * quarter)).isoformat(),
                    "value": 999.0,
                }
            )
        periods = adapter._parse_periods(raw)

        assert len(periods) == 96
        for quarter in range(8, 12):
            assert periods[quarter].price == float(quarter)  # first pass wins


class TestDegenerateDays:
    def test_empty_day_returns_empty_not_fabricated(self):
        adapter = make_adapter()
        assert adapter._parse_periods([]) == []

    def test_leading_gap_backfills_from_first_real_price(self):
        adapter = make_adapter()
        base = datetime(2026, 1, 7, 0, 0)
        periods = adapter._parse_periods(make_raw_day(base, skip_hours=(0,)))
        assert len(periods) == 96
        for quarter in range(4):
            assert periods[quarter].price == 4.0  # first real quarter (01:00)

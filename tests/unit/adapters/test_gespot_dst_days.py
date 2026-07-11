"""Regression tests for DST/gap handling in the GE-Spot adapter.

Days keep their native interval count (92 on spring DST, 96 normally, 100 on
autumn DST); nothing is fabricated for gaps. Consumers locate intervals by
timestamp containment (get_period_index) or list position - wall-clock
quarter numbers are ambiguous during the repeated autumn hour.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from custom_components.effektguard.adapters.gespot_adapter import (
    GESpotAdapter,
    PriceData,
    QuarterPeriod,
)
from custom_components.effektguard.const import QUARTER_INTERVAL_MINUTES, QUARTERS_PER_DAY
from custom_components.effektguard.utils.time_utils import resolve_period_index


def make_adapter() -> GESpotAdapter:
    return GESpotAdapter(hass=MagicMock(), config={"gespot_entity": "sensor.gespot"})


def make_raw_day(base: datetime, skip_hours: tuple[int, ...] = ()) -> list[dict]:
    """Build a raw GE-Spot day; price = quarter index for traceability."""
    raw = []
    for quarter in range(QUARTERS_PER_DAY):
        hour = quarter // 4
        if hour in skip_hours:
            continue
        raw.append(
            {
                "time": (base + timedelta(minutes=QUARTER_INTERVAL_MINUTES * quarter)).isoformat(),
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

    def test_period_index_resolves_by_timestamp_not_wall_clock(self):
        adapter = make_adapter()
        base = datetime(2026, 3, 29, 0, 0)
        periods = adapter._parse_periods(make_raw_day(base, skip_hours=(2,)))
        price_data = PriceData(today=periods, tomorrow=[], has_tomorrow=False)

        # 03:07 is wall-clock quarter 12 but list position 8 after the gap
        index = price_data.get_period_index(base.replace(hour=3, minute=7))
        assert index == 8
        assert price_data.today[index].start_time.hour == 3
        # Inside the skipped hour nothing matches
        assert price_data.get_period_index(base.replace(hour=2, minute=30)) is None


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

        price_data = PriceData(today=periods, tomorrow=[], has_tomorrow=False)
        assert price_data.get_period(first_hour).price == 10.0
        assert price_data.get_period(second_hour).price == 90.0


class TestDegenerateDays:
    def test_empty_day_returns_empty_not_fabricated(self):
        adapter = make_adapter()
        assert adapter._parse_periods([]) == []

    def test_leading_gap_preserves_only_real_intervals(self):
        adapter = make_adapter()
        base = datetime(2026, 1, 7, 0, 0)
        periods = adapter._parse_periods(make_raw_day(base, skip_hours=(0,)))
        assert len(periods) == 92
        assert periods[0].start_time.hour == 1
        assert periods[0].price == 4.0


class TestTomorrowIndex:
    def test_tomorrow_period_index_searches_tomorrow_list(self):
        today = datetime(2026, 1, 7, 0, 0, tzinfo=timezone.utc)
        tomorrow = today + timedelta(days=1)
        make = lambda base: [  # noqa: E731 - tiny local builder
            QuarterPeriod(start_time=base + timedelta(minutes=15 * q), price=float(q))
            for q in range(QUARTERS_PER_DAY)
        ]
        price_data = PriceData(today=make(today), tomorrow=make(tomorrow), has_tomorrow=True)

        when = tomorrow.replace(hour=6, minute=20)
        assert price_data.get_period_index(when) is None
        index = price_data.get_tomorrow_period_index(when)
        assert index == 25
        assert price_data.tomorrow[index].start_time == tomorrow.replace(hour=6, minute=15)


class TestResolvePeriodIndex:
    """Shared index resolution used by comfort/thermal/volatile layers."""

    def make_price_data(self, skip_hours: tuple[int, ...] = ()) -> PriceData:
        base = datetime(2026, 3, 29, 0, 0, tzinfo=timezone.utc)
        adapter = make_adapter()
        periods = adapter._parse_periods(make_raw_day(base, skip_hours=skip_hours))
        return PriceData(today=periods, tomorrow=[], has_tomorrow=False)

    def test_prefers_timestamp_containment_on_dst_day(self):
        price_data = self.make_price_data(skip_hours=(2,))
        when = datetime(2026, 3, 29, 3, 7, tzinfo=timezone.utc)
        assert resolve_period_index(price_data, when) == 8

    def test_wall_clock_fallback_only_on_dense_days(self):
        # A stand-in without get_period_index: dense day falls back to the
        # wall-clock quarter, a short (DST) day refuses to guess
        class Plain:
            def __init__(self, count):
                self.today = list(range(count))

        when = datetime(2026, 3, 29, 3, 7, tzinfo=timezone.utc)
        assert resolve_period_index(Plain(QUARTERS_PER_DAY), when) == 12
        assert resolve_period_index(Plain(92), when) is None

    def test_none_price_data_resolves_to_none(self):
        assert resolve_period_index(None) is None
        assert resolve_period_index(PriceData(today=[], tomorrow=[], has_tomorrow=False)) is None

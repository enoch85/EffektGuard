"""A cheap window is a run of REAL quarter-hours, found on the absolute timeline.

`find_cheapest_window` measured everything on the local clock: the horizon as `current_time +
timedelta`, continuity as the difference of two local datetimes, `hours_until` likewise. GE-Spot's
intervals and `dt_util.now()` share one ZoneInfo, and two aware datetimes with the same tzinfo do
that arithmetic on the wall clock (PEP 495 ignores `fold`). So on the night the clocks go back the
quarter after 02:45 CEST (02:00 CET) looked like an hour BEFORE it and the cheapest half hour of
the night was rejected as a gap; on the night they go forward a two-hour horizon reached one real
hour and the cheap quarter inside it was never seen.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from custom_components.effektguard.adapters.gespot_adapter import QuarterPeriod
from custom_components.effektguard.const import MINUTES_PER_QUARTER
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer

STOCKHOLM = ZoneInfo("Europe/Stockholm")
UTC = timezone.utc


def _periods(first_utc: datetime, prices: list[float]) -> list[QuarterPeriod]:
    """Quarter-hours from `first_utc`, published as GE-Spot does: in the area's own zone."""
    return [
        QuarterPeriod(
            start_time=(first_utc + timedelta(minutes=MINUTES_PER_QUARTER * q)).astimezone(
                STOCKHOLM
            ),
            price=price,
        )
        for q, price in enumerate(prices)
    ]


def test_the_half_hour_across_the_clocks_going_back_is_one_continuous_window():
    # 00:00Z is 02:00 CEST; the clocks go back at 01:00Z. Eight quarters: 02:00-02:45 CEST, then
    # 02:00-02:45 CET. The two cheap ones are 02:45 CEST and 02:00 CET - adjacent in real time.
    periods = _periods(datetime(2026, 10, 25, 0, 0, tzinfo=UTC), [9, 9, 9, 1, 1, 9, 9, 9])
    now = datetime(2026, 10, 25, 2, 40, tzinfo=STOCKHOLM, fold=0)  # 00:40Z, still CEST

    window = PriceAnalyzer().find_cheapest_window(
        current_time=now, price_periods=periods, duration_minutes=30, lookahead_hours=2.0
    )

    assert window is not None, (
        "the cheapest half hour of the night, 02:45 CEST to 02:15 CET, was rejected as "
        "discontinuous: on the wall clock the quarter after 02:45 CEST starts at 02:00, an hour "
        "earlier"
    )
    assert window.start_time.astimezone(UTC) == datetime(2026, 10, 25, 0, 45, tzinfo=UTC)
    assert window.end_time.astimezone(UTC) == datetime(2026, 10, 25, 1, 15, tzinfo=UTC), (
        f"the window ends at {window.end_time.isoformat()}: 02:00 CET plus a quarter is 02:15 "
        f"CET, not 02:15 CEST an hour before it"
    )
    assert window.hours_until == 5 / 60
    assert window.avg_price == 1.0


def test_the_horizon_is_real_hours_across_the_clocks_going_forward():
    # 2027-03-28: 01:00Z is 02:00 CET -> 03:00 CEST. Quarters from 01:30 CET; the cheap one is
    # 04:00 CEST, 1.5 real hours after 01:30 CET - inside a two-hour horizon.
    periods = _periods(datetime(2027, 3, 28, 0, 30, tzinfo=UTC), [9, 9, 9, 9, 9, 9, 1, 9, 9, 9])
    now = datetime(2027, 3, 28, 1, 30, tzinfo=STOCKHOLM)

    window = PriceAnalyzer().find_cheapest_window(
        current_time=now, price_periods=periods, duration_minutes=15, lookahead_hours=2.0
    )

    assert window is not None
    assert window.start_time.astimezone(UTC) == datetime(2027, 3, 28, 2, 0, tzinfo=UTC), (
        f"the cheapest quarter is 04:00 CEST, 1.5 real hours away, but the window found starts "
        f"at {window.start_time.isoformat()}: a two-hour horizon from 01:30 CET ended at 03:30 "
        f"CEST on the wall clock, one real hour out"
    )
    assert window.hours_until == 1.5

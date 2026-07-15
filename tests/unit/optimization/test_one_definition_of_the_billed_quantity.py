"""One definition of the billed quantity: the time-weighted mean power over a billing hour.

That number decides whether the pump is throttled for the rest of the month, so `BillingPeriodAccumulator`
must compute it exactly. These tests pin the arithmetic the tariff pays for:
  * the time-weighted mean, which is NOT the arithmetic sample mean when Home Assistant's update
    cycle jitters or a restart drops samples;
  * the hour counted on the absolute time line, so the repeated DST fall-back hour is two hours;
  * the local hour label and local start stamp, because the night discount and the calendar month a
    peak belongs to are both wall-clock facts;
  * an hour begun before observation, or cut short by shutdown, is not billed.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from custom_components.effektguard.const import (
    BILLING_PERIOD_MINUTES,
    POWER_SOURCE_EXTERNAL_METER,
)
from custom_components.effektguard.optimization.billing_period import BillingPeriodAccumulator

STOCKHOLM = ZoneInfo("Europe/Stockholm")
UTC = ZoneInfo("UTC")


def _local(*args) -> datetime:
    return datetime(*args, tzinfo=STOCKHOLM)


def test_a_flat_hour_is_billed_at_its_flat_power():
    """The simplest case, and the one everything else is measured against."""
    accumulator = BillingPeriodAccumulator()
    completed = None

    for minute in range(0, 60, 5):
        completed = (
            accumulator.add(_local(2026, 1, 15, 10, minute), 6.0, POWER_SOURCE_EXTERNAL_METER)
            or completed
        )
    # The first sample of the NEXT hour is what closes this one.
    completed = (
        accumulator.add(_local(2026, 1, 15, 11, 0), 6.0, POWER_SOURCE_EXTERNAL_METER) or completed
    )

    assert completed is not None, "a whole hour went by and no billing period completed"
    assert completed.mean_power_kw == pytest.approx(6.0)
    assert completed.billing_hour == 10
    assert completed.started_at == _local(2026, 1, 15, 10, 0)


def test_the_mean_is_time_weighted_not_sample_counted():
    """The time-weighted mean is not the arithmetic sample mean when samples are unevenly spaced.

        readings   1 kW at :00, :15, :30, then 9 kW at :45 and :55
        time-weighted (what the grid bills):  (1*45 + 9*15) / 60  = 3.0 kW
        arithmetic mean of the samples:       (1+1+1+9+9) / 5     = 4.2 kW  (40% high)

    Home Assistant's update cycle jitters, so the samples in an hour are not evenly spaced. The gaps
    here stay within MAX_BILLING_OBSERVATION_GAP_MINUTES, so the hour is actually observed and billed.
    """
    accumulator = BillingPeriodAccumulator()

    for minute, power in ((0, 1.0), (15, 1.0), (30, 1.0), (45, 9.0), (55, 9.0)):
        accumulator.add(_local(2026, 1, 15, 10, minute), power, POWER_SOURCE_EXTERNAL_METER)
    completed = accumulator.add(_local(2026, 1, 15, 11, 0), 1.0, POWER_SOURCE_EXTERNAL_METER)

    assert completed is not None
    assert completed.mean_power_kw == pytest.approx((1.0 * 45 + 9.0 * 15) / 60), (
        f"the hour was billed at {completed.mean_power_kw:.2f} kW. 1 kW stood for 45 minutes and "
        f"9 kW for fifteen; the grid bills the time-weighted mean, 3.0 kW. Counting samples instead "
        f"gives 4.2 kW - 40% high, persisted as the month's peak."
    )


def test_the_hour_is_counted_on_the_absolute_time_line():
    """The DST fall-back: wall-clock 02:00 happens twice, and both hours are billable.

    PEP 495 - for two aware datetimes with the same tzinfo, `fold` is IGNORED in comparisons - is why
    the naive version of this merged them and deleted a peak.
    """
    accumulator = BillingPeriodAccumulator()
    completed = []

    # Step REAL time across the transition; the tz database does the rest.
    start = datetime(2026, 10, 25, 0, 0, tzinfo=UTC)  # 02:00 CEST
    for step in range(0, 150, 5):
        instant = (start + timedelta(minutes=step)).astimezone(STOCKHOLM)
        power = 9.0 if step < 60 else 1.0  # 9 kW through the FIRST 02:00, 1 kW through the second
        event = accumulator.add(instant, power, POWER_SOURCE_EXTERNAL_METER)
        if event is not None:
            completed.append(event)

    means = [round(event.mean_power_kw, 2) for event in completed]
    hours = [event.billing_hour for event in completed]

    assert hours == [
        2,
        2,
    ], f"two separately-metered hours both labelled 02 must both complete. Got hours {hours}."
    assert means == [9.0, 1.0], (
        f"the two 02:00 hours billed {means}. They are an hour apart and both real. Merging them "
        f"deletes the 9 kW hour - which is what the coordinator did until 37f2fef."
    )


def test_the_start_stamp_is_local_so_the_month_is_right():
    """The effect layer buckets peaks by calendar month, and that is a wall-clock fact.

    The billing hour 00:00-01:00 on 1 November IS 23:00-00:00 on 31 October in UTC. Stamping it in
    UTC files a November peak against a month that is already billed.
    """
    accumulator = BillingPeriodAccumulator()
    completed = None

    start = datetime(2026, 10, 31, 23, 0, tzinfo=UTC)  # 00:00 local, 1 November
    for step in range(0, 65, 5):
        instant = (start + timedelta(minutes=step)).astimezone(STOCKHOLM)
        completed = accumulator.add(instant, 7.0, POWER_SOURCE_EXTERNAL_METER) or completed

    assert completed is not None
    assert (completed.started_at.year, completed.started_at.month) == (2026, 11), (
        f"the hour was stamped {completed.started_at.isoformat()} - month "
        f"{completed.started_at.month}. It is the first hour of November."
    )
    assert completed.billing_hour == 0


def test_an_hour_that_began_before_observation_is_not_billed():
    """Home Assistant starts mid-hour. That hour was never fully measured, so it is not a bill."""
    accumulator = BillingPeriodAccumulator()

    accumulator.add(
        _local(2026, 1, 15, 10, 23), 5.0, POWER_SOURCE_EXTERNAL_METER
    )  # first ever sample: mid-hour
    accumulator.add(_local(2026, 1, 15, 10, 55), 5.0, POWER_SOURCE_EXTERNAL_METER)
    completed = accumulator.add(_local(2026, 1, 15, 11, 0), 5.0, POWER_SOURCE_EXTERNAL_METER)

    assert completed is None, (
        f"the 10:00 hour was billed at {completed.mean_power_kw if completed else None} kW, but it "
        f"was only observed from 10:23. A partial hour is not a measurement of an hour."
    )

    # ...and the NEXT, fully-observed hour is billed normally.
    for minute in range(5, 60, 5):
        accumulator.add(_local(2026, 1, 15, 11, minute), 5.0, POWER_SOURCE_EXTERNAL_METER)
    completed = accumulator.add(_local(2026, 1, 15, 12, 0), 5.0, POWER_SOURCE_EXTERNAL_METER)

    assert completed is not None and completed.mean_power_kw == pytest.approx(5.0)
    assert completed.billing_hour == 11


def test_flush_closes_the_hour_in_progress():
    """The simulator's run ends. The hour it ends on is complete in sim-time and must be billed.

    Production never calls this - Home Assistant keeps running, and an hour cut short by a shutdown
    is not a bill. It exists so the harness does not silently drop its final hour.
    """
    accumulator = BillingPeriodAccumulator()
    for minute in range(0, 60, 5):
        accumulator.add(_local(2026, 1, 15, 10, minute), 4.0, POWER_SOURCE_EXTERNAL_METER)

    completed = accumulator.flush()

    assert completed is not None and completed.mean_power_kw == pytest.approx(4.0)
    assert completed.billing_hour == 10
    assert accumulator.flush() is None, "flushing twice must not bill the same hour twice"


def test_the_billing_period_is_the_hour_the_tariff_actually_uses():
    """The accumulator must not carry its own private idea of how long an hour is."""
    assert BILLING_PERIOD_MINUTES == 60

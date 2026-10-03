"""One definition of the billed quantity: the time-weighted mean power over a billing period.

The owner's effect tariff measures a 15-minute period (BILLING_PERIOD_MINUTES; operator models vary
- F-107 - and this pins the OWNER'S). That number decides whether the pump is throttled for the rest
of the month, so `BillingPeriodAccumulator` must compute it exactly. These tests pin the arithmetic
the tariff pays for:
  * the time-weighted mean, which is NOT the arithmetic sample mean when Home Assistant's update
    cycle jitters or a restart drops samples;
  * the period counted on the absolute time line, so the repeated DST fall-back quarters are
    separate periods;
  * the local period label and local start stamp, because the night discount and the calendar month
    a peak belongs to are both wall-clock facts;
  * a period begun before observation, or cut short by shutdown, is not billed.
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


def test_a_flat_period_is_billed_at_its_flat_power():
    """The simplest case, and the one everything else is measured against."""
    accumulator = BillingPeriodAccumulator()
    completed = None

    for minute in range(0, 15, 5):
        completed = (
            accumulator.add(_local(2026, 1, 15, 10, minute), 6.0, POWER_SOURCE_EXTERNAL_METER)
            or completed
        )
    # The first sample of the NEXT period is what closes this one.
    completed = (
        accumulator.add(_local(2026, 1, 15, 10, 15), 6.0, POWER_SOURCE_EXTERNAL_METER) or completed
    )

    assert completed is not None, "a whole period went by and no billing period completed"
    assert completed.mean_power_kw == pytest.approx(6.0)
    assert completed.billing_period == 10 * 4  # 10:00-10:15 is quarter 40
    assert completed.started_at == _local(2026, 1, 15, 10, 0)


def test_the_mean_is_time_weighted_not_sample_counted():
    """The time-weighted mean is not the arithmetic sample mean when samples are unevenly spaced.

        readings   1 kW at :00, then 9 kW at :10 and :12
        time-weighted (what the grid bills):  (1*10 + 9*5) / 15   = 3.67 kW
        arithmetic mean of the samples:       (1+9+9) / 3         = 6.33 kW  (73% high)

    Home Assistant's update cycle jitters, so the samples in a period are not evenly spaced. The
    gaps here stay within MAX_BILLING_OBSERVATION_GAP_MINUTES, so the period is observed and billed.
    """
    accumulator = BillingPeriodAccumulator()

    for minute, power in ((0, 1.0), (10, 9.0), (12, 9.0)):
        accumulator.add(_local(2026, 1, 15, 10, minute), power, POWER_SOURCE_EXTERNAL_METER)
    completed = accumulator.add(_local(2026, 1, 15, 10, 15), 1.0, POWER_SOURCE_EXTERNAL_METER)

    assert completed is not None
    assert completed.mean_power_kw == pytest.approx((1.0 * 10 + 9.0 * 5) / 15), (
        f"the period was billed at {completed.mean_power_kw:.2f} kW. 1 kW stood for ten minutes and "
        f"9 kW for five; the grid bills the time-weighted mean, 3.67 kW. Counting samples instead "
        f"gives 6.33 kW - 73% high, persisted as the month's peak."
    )


def test_the_period_is_counted_on_the_absolute_time_line():
    """The DST fall-back: wall-clock 02:00-03:00 happens twice, and all eight quarters are billable.

    PEP 495 - for two aware datetimes with the same tzinfo, `fold` is IGNORED in comparisons - is why
    the naive version of this merged the repeated quarters and deleted a peak.
    """
    accumulator = BillingPeriodAccumulator()
    completed = []

    # Step REAL time across the transition; the tz database does the rest.
    # 02:00 CEST through 03:00 CET: two real wall-clock 02:xx hours, then one closing sample.
    start = datetime(2026, 10, 25, 0, 0, tzinfo=UTC)  # 02:00 CEST
    for step in range(0, 125, 5):
        instant = (start + timedelta(minutes=step)).astimezone(STOCKHOLM)
        power = 9.0 if step < 60 else 1.0  # 9 kW through the FIRST 02:xx hour, 1 kW the second
        event = accumulator.add(instant, power, POWER_SOURCE_EXTERNAL_METER)
        if event is not None:
            completed.append(event)

    labels = [event.billing_period for event in completed]
    means = [round(event.mean_power_kw, 2) for event in completed]

    # Quarters 8..11 are 02:00-03:00. Both wall-clock passes must complete, separately.
    assert labels == [
        8,
        9,
        10,
        11,
        8,
        9,
        10,
        11,
    ], f"the repeated 02:xx hour must yield its four quarters TWICE. Got periods {labels}."
    assert means == [9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0], (
        f"the two passes billed {means}. They are an hour apart and both real. Merging them deletes "
        f"the 9 kW peaks - which is what the coordinator did until 37f2fef."
    )


def test_the_start_stamp_is_local_so_the_month_is_right():
    """The effect layer buckets peaks by calendar month, and that is a wall-clock fact.

    The first billing period of 1 November IS 23:00-23:15 on 31 October in UTC. Stamping it in UTC
    files a November peak against a month that is already billed.
    """
    accumulator = BillingPeriodAccumulator()
    completed = None

    start = datetime(2026, 10, 31, 23, 0, tzinfo=UTC)  # 00:00 local, 1 November
    for step in range(0, 20, 5):
        instant = (start + timedelta(minutes=step)).astimezone(STOCKHOLM)
        completed = accumulator.add(instant, 7.0, POWER_SOURCE_EXTERNAL_METER) or completed

    assert completed is not None
    assert (completed.started_at.year, completed.started_at.month) == (2026, 11), (
        f"the period was stamped {completed.started_at.isoformat()} - month "
        f"{completed.started_at.month}. It is the first period of November."
    )
    assert completed.billing_period == 0


def test_a_period_that_began_before_observation_is_not_billed():
    """Home Assistant starts mid-period. That period was never fully measured, so it is not a bill."""
    accumulator = BillingPeriodAccumulator()

    accumulator.add(
        _local(2026, 1, 15, 10, 8), 5.0, POWER_SOURCE_EXTERNAL_METER
    )  # first ever sample: mid-period
    accumulator.add(_local(2026, 1, 15, 10, 13), 5.0, POWER_SOURCE_EXTERNAL_METER)
    completed = accumulator.add(_local(2026, 1, 15, 10, 15), 5.0, POWER_SOURCE_EXTERNAL_METER)

    assert completed is None, (
        f"the 10:00 period was billed at {completed.mean_power_kw if completed else None} kW, but "
        f"it was only observed from 10:08. A partial period is not a measurement of a period."
    )

    # ...and the NEXT, fully-observed period is billed normally.
    for minute in (20, 25):
        accumulator.add(_local(2026, 1, 15, 10, minute), 5.0, POWER_SOURCE_EXTERNAL_METER)
    completed = accumulator.add(_local(2026, 1, 15, 10, 30), 5.0, POWER_SOURCE_EXTERNAL_METER)

    assert completed is not None and completed.mean_power_kw == pytest.approx(5.0)
    assert completed.billing_period == 10 * 4 + 1  # 10:15-10:30


def test_flush_closes_the_period_in_progress():
    """The simulator's run ends. The period it ends on is complete in sim-time and must be billed.

    Production never calls this - Home Assistant keeps running, and a period cut short by a shutdown
    is not a bill. It exists so the harness does not silently drop its final period.
    """
    accumulator = BillingPeriodAccumulator()
    for minute in range(0, 15, 5):
        accumulator.add(_local(2026, 1, 15, 10, minute), 4.0, POWER_SOURCE_EXTERNAL_METER)

    completed = accumulator.flush()

    assert completed is not None and completed.mean_power_kw == pytest.approx(4.0)
    assert completed.billing_period == 10 * 4
    assert accumulator.flush() is None, "flushing twice must not bill the same period twice"


def test_the_billing_period_is_the_one_the_owners_tariff_uses():
    """The accumulator must not carry its own private idea of how long a period is.

    15 minutes is the OWNER'S tariff cadence (operator models vary - F-107). Changing this constant
    changes what every monthly peak means, so it is changed deliberately or not at all.
    """
    assert BILLING_PERIOD_MINUTES == 15

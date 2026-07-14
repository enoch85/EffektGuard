"""The billed quantity had two definitions, and the simulator was validating the wrong one.

The effect tariff bills the MEAN POWER OVER A BILLING HOUR. That number decides whether the heat pump
is throttled for the rest of the month, so it is the single most consequential figure the integration
computes. It was computed twice, by two different pieces of code, using two different formulas:

    coordinator.py      a TIME-WEIGHTED mean: each sample weighted by how long it stood, the last one
                        extrapolated to the hour boundary, divided by 3600 seconds.

    sim_harness.py      `sum(period_samples) / len(period_samples)` - a plain ARITHMETIC mean.

They agree when the samples are evenly spaced, and the simulator steps a uniform 5 minutes, so its
numbers were never WRONG. They were something worse: they were produced by code that ships to nobody.
Every tariff figure the harness has ever printed - every SEK, every kW of peak, every claim about the
feature this integration is NAMED for - was computed by an implementation no user runs.

AND THAT IS NOT A THEORETICAL COMPLAINT. The daylight-saving defect (`37f2fef`) lived in the
coordinator's accumulator: on the night the clocks go back it merged the repeated hour and deleted a
9 kW billing peak, recording it as 1 kW. The simulator had the SAME BUG, INDEPENDENTLY, in its own
copy - and so it could not see it. Two implementations of one quantity, both broken, each blind to the
other. An instrument that re-implements the thing it is measuring cannot measure it.

So there is now ONE definition, here, and both the coordinator and the harness call it. Break it and
the simulator fails - which is the property that was missing, and is verified by mutation.

These tests pin the arithmetic that the tariff actually pays for:
  * the time-weighted mean, which is NOT the arithmetic mean when Home Assistant's update cycle
    jitters or a restart drops samples - and it does, and they do;
  * the hour counted on the absolute time line, so a repeated DST hour is two hours;
  * the local hour label and local start stamp, because the night discount and the calendar month a
    peak belongs to are both wall-clock facts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from custom_components.effektguard.const import BILLING_PERIOD_MINUTES
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
        completed = accumulator.add(_local(2026, 1, 15, 10, minute), 6.0) or completed
    # The first sample of the NEXT hour is what closes this one.
    completed = accumulator.add(_local(2026, 1, 15, 11, 0), 6.0) or completed

    assert completed is not None, "a whole hour went by and no billing period completed"
    assert completed.mean_power_kw == pytest.approx(6.0)
    assert completed.billing_hour == 10
    assert completed.started_at == _local(2026, 1, 15, 10, 0)


def test_the_mean_is_time_weighted_not_sample_counted():
    """THE DIVERGENCE. This is the test the simulator's own formula could not pass.

    Home Assistant's update cycle is not a metronome: it jitters and it is delayed under load, so the
    samples in an hour are not evenly spaced and their arithmetic mean is not the hour's mean power.

        readings   1 kW at :00, :15, :30, then 9 kW at :45 and :55
        spans      15, 15, 15, 10, and 5 minutes to the boundary

        time-weighted (what the grid bills):  (1*45 + 9*15) / 60  = 3.0 kW
        arithmetic mean of the samples:       (1+1+1+9+9) / 5     = 4.2 kW

    The second number is 40% high, and it would be persisted as the month's peak and defended for
    weeks. The harness computed the second number. It only ever agreed with the first because the
    harness's clock ticks a perfectly uniform five minutes - which Home Assistant's does not.

    NOTE the gaps here are all within MAX_BILLING_OBSERVATION_GAP_MINUTES. An earlier version of this
    test made the point with a single 55-minute gap, which is a far more vivid illustration and also
    an hour the meter slept through - the accumulator now refuses to bill those at all, and rightly.
    The arithmetic has to be demonstrable on an hour that was actually observed.
    """
    accumulator = BillingPeriodAccumulator()

    for minute, power in ((0, 1.0), (15, 1.0), (30, 1.0), (45, 9.0), (55, 9.0)):
        accumulator.add(_local(2026, 1, 15, 10, minute), power)
    completed = accumulator.add(_local(2026, 1, 15, 11, 0), 1.0)

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
        event = accumulator.add(instant, power)
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
        completed = accumulator.add(instant, 7.0) or completed

    assert completed is not None
    assert (completed.started_at.year, completed.started_at.month) == (2026, 11), (
        f"the hour was stamped {completed.started_at.isoformat()} - month "
        f"{completed.started_at.month}. It is the first hour of November."
    )
    assert completed.billing_hour == 0


def test_an_hour_that_began_before_observation_is_not_billed():
    """Home Assistant starts mid-hour. That hour was never fully measured, so it is not a bill."""
    accumulator = BillingPeriodAccumulator()

    accumulator.add(_local(2026, 1, 15, 10, 23), 5.0)  # first ever sample: mid-hour
    accumulator.add(_local(2026, 1, 15, 10, 55), 5.0)
    completed = accumulator.add(_local(2026, 1, 15, 11, 0), 5.0)

    assert completed is None, (
        f"the 10:00 hour was billed at {completed.mean_power_kw if completed else None} kW, but it "
        f"was only observed from 10:23. A partial hour is not a measurement of an hour."
    )

    # ...and the NEXT, fully-observed hour is billed normally.
    for minute in range(5, 60, 5):
        accumulator.add(_local(2026, 1, 15, 11, minute), 5.0)
    completed = accumulator.add(_local(2026, 1, 15, 12, 0), 5.0)

    assert completed is not None and completed.mean_power_kw == pytest.approx(5.0)
    assert completed.billing_hour == 11


def test_flush_closes_the_hour_in_progress():
    """The simulator's run ends. The hour it ends on is complete in sim-time and must be billed.

    Production never calls this - Home Assistant keeps running, and an hour cut short by a shutdown
    is not a bill. It exists so the harness does not silently drop its final hour.
    """
    accumulator = BillingPeriodAccumulator()
    for minute in range(0, 60, 5):
        accumulator.add(_local(2026, 1, 15, 10, minute), 4.0)

    completed = accumulator.flush()

    assert completed is not None and completed.mean_power_kw == pytest.approx(4.0)
    assert completed.billing_hour == 10
    assert accumulator.flush() is None, "flushing twice must not bill the same hour twice"


def test_the_billing_period_is_the_hour_the_tariff_actually_uses():
    """The accumulator must not carry its own private idea of how long an hour is."""
    assert BILLING_PERIOD_MINUTES == 60

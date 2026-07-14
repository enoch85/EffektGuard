"""Peak protection must compare an HOURLY MEAN against an hourly-mean record.

The monthly record is the mean power of a whole billing hour - that is what Ellevio bills.
The effect layer was handed the instantaneous reading of the last cycle and compared it
against that record: a five-minute oven spike read as if it were a whole hour of it, and
the pump was throttled to defend a peak the meter would have averaged away.

The like-for-like quantity is the PROJECTED hour mean: what this billing hour becomes if
the current draw persists to the boundary. Early in the hour a spike projects to almost
nothing; the closer the boundary, the more the accumulated hour dominates and the less
anyone can pretend the spike away.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from custom_components.effektguard.const import POWER_SOURCE_EXTERNAL_METER
from custom_components.effektguard.optimization.billing_period import BillingPeriodAccumulator

STOCKHOLM = ZoneInfo("Europe/Stockholm")


def _t(minute: int, hour: int = 10) -> datetime:
    return datetime(2026, 1, 15, hour, minute, tzinfo=STOCKHOLM)


def test_half_an_hour_of_low_draw_halves_a_spike():
    acc = BillingPeriodAccumulator()
    for minute in range(0, 35, 5):
        acc.add(_t(minute), 2.0, POWER_SOURCE_EXTERNAL_METER)

    # 9 kW starting at 10:30: the hour's mean, if it persists, is (2*30 + 9*30)/60.
    projected = acc.projected_hour_mean(_t(30), 9.0)

    assert projected == (2.0 * 30 + 9.0 * 30) / 60


def test_an_empty_hour_projects_the_draw_itself():
    acc = BillingPeriodAccumulator()

    assert acc.projected_hour_mean(_t(0), 9.0) == 9.0


def test_a_spike_in_the_last_five_minutes_barely_moves_the_hour():
    acc = BillingPeriodAccumulator()
    for minute in range(0, 60, 5):
        acc.add(_t(minute), 1.0, POWER_SOURCE_EXTERNAL_METER)

    projected = acc.projected_hour_mean(_t(55), 9.0)

    assert projected == (1.0 * 55 + 9.0 * 5) / 60


def test_the_coordinator_feeds_the_projection_to_the_engine():
    """The wiring contract: the decision path consumes the like-for-like quantity."""
    import inspect

    from custom_components.effektguard.coordinator import EffektGuardCoordinator

    src = inspect.getsource(EffektGuardCoordinator._read_and_decide)
    assert "projected_hour_mean" in src, (
        "The decision path no longer projects the billing hour. Handing the effect layer an "
        "instantaneous reading compares a five-minute spike against an HOURLY-MEAN record - "
        "the layer throttles the pump to defend a peak the meter would average away."
    )

"""Time-related utility functions for EffektGuard.

Provides shared time calculations used across the integration.

Two clocks run here. Wall-clock labels (the quarter of the day, the 07:00 hot-water schedule,
the calendar month a peak belongs to) are local-time facts. Elapsed durations and the order of
two instants are not: on the night the clocks go back the local hour 02:xx happens twice, and PEP
495 ignores ``fold`` when two aware datetimes share a tzinfo, so ``later - earlier`` and
``moment + delta`` are wall-clock arithmetic - an hour short across the repeated hour, an hour
long across the spring gap. ``seconds_between`` and ``after`` do that arithmetic on the absolute
timeline.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from homeassistant.util import dt as dt_util

from ..const import MINUTES_PER_QUARTER, QUARTERS_PER_DAY

# Minutes per quarter (15 min = Swedish Effektavgift measurement period)
QUARTERS_PER_HOUR = 60 // MINUTES_PER_QUARTER  # 4


def seconds_between(earlier: datetime, later: datetime) -> float:
    """Real seconds from ``earlier`` to ``later``.

    Epoch seconds honour ``fold``, so 02:50 CEST to 02:40 CET is 50 minutes, where subtraction
    says -10. A naive pair subtracts as it always did, and a naive/aware mix raises TypeError as
    subtraction does - a naive timestamp from a foreign source must not be silently placed.
    """
    if earlier.tzinfo is None or later.tzinfo is None:
        return (later - earlier).total_seconds()
    return later.timestamp() - earlier.timestamp()


def after(moment: datetime, delta: timedelta) -> datetime:
    """``moment`` plus ``delta`` real seconds, in ``moment``'s own zone.

    ``moment + delta`` is wall-clock arithmetic and resets ``fold`` to 0: 02:30 CEST plus 60
    minutes is 03:30 CET, 120 real minutes later. Going through UTC adds real seconds and lands on
    the right side of a repeated hour. A naive datetime has no absolute timeline and is shifted as
    it is.
    """
    if moment.tzinfo is None:
        return moment + delta
    return (moment.astimezone(timezone.utc) + delta).astimezone(moment.tzinfo)


def get_current_quarter(now: Optional[datetime] = None) -> int:
    """Get the current quarter of the day (0-95).

    Each day has 96 quarters (24 hours × 4 quarters per hour).
    Quarter 0 = 00:00-00:14, Quarter 95 = 23:45-23:59.

    Args:
        now: Optional datetime to use (defaults to current time)

    Returns:
        Quarter index (0-95)
    """
    if now is None:
        now = dt_util.now()
    return (now.hour * QUARTERS_PER_HOUR) + (now.minute // MINUTES_PER_QUARTER)


def resolve_period_index(price_data: object, now: Optional[datetime] = None) -> Optional[int]:
    """Resolve the index into price_data.today for the interval containing now.

    Classifications and period lists are position-keyed, while wall-clock
    quarter numbers are ambiguous on DST days (92 or 100 native intervals).
    PriceData.get_period_index (timestamp containment) is authoritative:
    when it finds no interval - a data gap, a stale day - the honest answer
    is None, never a positional guess. Only stand-ins WITHOUT the method
    (simple test doubles) fall back to the wall-clock quarter, and only on
    dense 96-interval days where position and wall-clock coincide.

    Args:
        price_data: PriceData or a stand-in with a ``today`` list
        now: Timestamp to resolve (defaults to current time)

    Returns:
        Index valid for price_data.today, or None when now is outside it
    """
    periods = getattr(price_data, "today", None)
    if not periods:
        return None

    if now is None:
        now = dt_util.now()

    finder = getattr(price_data, "get_period_index", None)
    if callable(finder):
        index = finder(now)
        return index if isinstance(index, int) else None

    quarter = get_current_quarter(now)
    if len(periods) == QUARTERS_PER_DAY and quarter < len(periods):
        return quarter
    return None


def get_current_billing_period(now: Optional[datetime] = None) -> int:
    """The owner's effect-tariff billing period: the 15-minute quarter of the day, 0-95.

    The owner runs 15-minute measurement intervals, so the billing period is the quarter-hour, the
    same cadence get_current_quarter returns. Operator models vary (F-107); this is the owner's.
    """
    return get_current_quarter(now)

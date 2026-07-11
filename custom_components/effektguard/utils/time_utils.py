"""Time-related utility functions for EffektGuard.

Provides shared time calculations used across the integration.
"""

from datetime import datetime
from typing import Optional

from homeassistant.util import dt as dt_util

from ..const import QUARTER_INTERVAL_MINUTES, QUARTERS_PER_DAY

# Minutes per quarter (15 min = Swedish Effektavgift measurement period)
QUARTERS_PER_HOUR = 60 // QUARTER_INTERVAL_MINUTES  # 4


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
    return (now.hour * QUARTERS_PER_HOUR) + (now.minute // QUARTER_INTERVAL_MINUTES)


def resolve_period_index(price_data: object, now: Optional[datetime] = None) -> Optional[int]:
    """Resolve the index into price_data.today for the interval containing now.

    Classifications and period lists are position-keyed, while wall-clock
    quarter numbers are ambiguous on DST days (92 or 100 native intervals).
    Prefer PriceData.get_period_index (timestamp containment); fall back to
    the wall-clock quarter only for plain 96-interval days, where position
    and wall-clock quarter coincide (keeps simple test/simulation stand-ins
    without the method working).

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
        if isinstance(index, int):
            return index

    quarter = get_current_quarter(now)
    if len(periods) == QUARTERS_PER_DAY and quarter < len(periods):
        return quarter
    return None

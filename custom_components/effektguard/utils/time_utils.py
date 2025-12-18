"""Time-related utility functions for EffektGuard.

Provides shared time calculations used across the integration.
"""

from datetime import datetime
from typing import Optional

from homeassistant.util import dt as dt_util

from ..const import QUARTER_INTERVAL_MINUTES

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

"""Degree minutes, as the anti-windup and the wear guard's saturation tracker both read them."""

from collections.abc import Iterable
from datetime import datetime, timedelta
from itertools import groupby
from operator import itemgetter


def dm_rate_per_hour(
    readings: Iterable[tuple[datetime, float, int]], min_span: timedelta
) -> float | None:
    """Degree minutes per hour over (time, DM, part) readings in time order.

    A part is a stretch without a pause - the saturation tracker starts a new one after hot water,
    whose degree minutes must not count - so each part adds its last reading less its first. None
    until the parts together cover min_span.
    """
    change, span = 0.0, timedelta()
    for _, group in groupby(readings, key=itemgetter(2)):
        part = list(group)
        change += part[-1][1] - part[0][1]
        span += part[-1][0] - part[0][0]
    if span < min_span:
        return None
    return change / (span.total_seconds() / 3600)

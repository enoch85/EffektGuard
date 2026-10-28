"""Battery mode (experimental): the house as a heat battery, on top of the layers' decision.

The nine layers vote as always. In battery mode the aggregate then moves up in the day's cheapest
hours (charge: the slab stores heat while it is cheap) and down in its dearest (coast: the slab
gives it back while it is dear). Nothing else changes: trend damping, the compressor wear cap and
the coordinator's volatility gate act on the result exactly as on any other decision.

The day is ranked by hour, from the mean of its quarters, so a quarter flickering between price
classes does not move it. Guards:

- nothing while the compressor is saturated, while the proactive layer is in Z3-Z5 (degree minutes
  past half their normal range), or while the emergency layer is in any tier: a battery stores
  only what the compressor can spare. Without these guards the closed loop showed the
  immersion heater paying for both the charge and the refill after a coast (F-124);
- charge only while the room is below its own past-day mean (the target when that is higher, or
  when no full day was read) + the mode's allowed pre-heat overshoot: the battery is full there.
  Measured from the target alone, a house the layers hold 0.8-0.9 C above it never charged
  (closed loop, owner's F750 + slab: 0.1 charging hours in 74 autumn days);
- coast only while the room is above the band's lower edge;
- nothing without a room reading, without today's prices, or on a flat day;
- a charge hour must be cheaper than the day's median hour, a coast hour dearer: on a day of a few
  very cheap hours, the 'cheapest N' would otherwise reach into the dear ones.
"""

from __future__ import annotations

from datetime import datetime
from statistics import median

import numpy as np
from homeassistant.util import dt as dt_util

from ..const import (
    BATTERY_MODE_CHARGE,
    BATTERY_MODE_COAST,
    BATTERY_MODE_HOURS,
    PRICE_FLAT_DAY_SPREAD_FRACTION,
    PRICE_PERCENTILE_EXPENSIVE,
    PRICE_PERCENTILE_MEDIAN,
    PRICE_PERCENTILE_VERY_CHEAP,
)

BATTERY_IDLE_TIERS: frozenset[str] = frozenset({"", "OK"})
# Z1 starts at 2 % of the zone's normal degree-minute range: nearly always while the compressor
# runs. Guarding on it, the battery never acted in the repository's scenarios. From Z3 (half the
# range) on, the pump is falling behind.
BATTERY_IDLE_ZONES: frozenset[str] = frozenset({"", "NONE", "Z1", "Z2"})


def battery_adjustment(
    now: datetime,
    price_data,
    nibe_state,
    target_temp: float,
    tolerance_range: float,
    overshoot_allowed: float,
    emergency_tier: str,
    proactive_zone: str,
    compressor_saturated: bool,
    room_reference: float | None = None,
) -> tuple[float, str]:
    """The battery's move for this cycle (°C) and a reason ('' when it does nothing)."""
    if compressor_saturated or emergency_tier not in BATTERY_IDLE_TIERS:
        return 0.0, ""
    if proactive_zone not in BATTERY_IDLE_ZONES:
        return 0.0, ""
    if not getattr(nibe_state, "indoor_temp_valid", True):
        return 0.0, ""
    if not price_data or not getattr(price_data, "today", None):
        return 0.0, ""

    day = dt_util.as_local(now).date()
    by_hour: dict[int, list[float]] = {}
    for period in price_data.today:
        start = dt_util.as_local(period.start_time)
        if start.date() == day:
            by_hour.setdefault(start.hour, []).append(period.price)
    if not by_hour:
        return 0.0, ""

    # The price layer's own flat-day rule (classify_quarterly_periods): nothing to shift on a day
    # whose spread is noise against its price scale.
    prices = [p for quarter in by_hour.values() for p in quarter]
    p10 = float(np.percentile(prices, PRICE_PERCENTILE_VERY_CHEAP))
    p90 = float(np.percentile(prices, PRICE_PERCENTILE_EXPENSIVE))
    scale = max(abs(float(np.percentile(prices, PRICE_PERCENTILE_MEDIAN))), abs(p10), abs(p90))
    if scale <= 0.0 or p90 - p10 < scale * PRICE_FLAT_DAY_SPREAD_FRACTION:
        return 0.0, ""

    mean = {hour: sum(q) / len(q) for hour, q in by_hour.items()}
    ranked = sorted(mean, key=lambda hour: mean[hour])
    middle = median(mean.values())
    hour = dt_util.as_local(now).hour
    if hour not in mean:
        return 0.0, ""
    room = nibe_state.indoor_temp

    if hour in ranked[:BATTERY_MODE_HOURS] and mean[hour] < middle:
        if room < max(target_temp, room_reference or target_temp) + overshoot_allowed:
            return (
                BATTERY_MODE_CHARGE,
                f"Battery: charging (+{BATTERY_MODE_CHARGE:.0f}°C, cheap hour)",
            )
        return 0.0, ""

    if BATTERY_MODE_COAST > 0.0 and hour in ranked[-BATTERY_MODE_HOURS:] and mean[hour] > middle:
        if room > target_temp - tolerance_range:
            return (
                -BATTERY_MODE_COAST,
                f"Battery: coasting (-{BATTERY_MODE_COAST:.0f}°C, dear hour)",
            )
        return 0.0, ""

    return 0.0, ""

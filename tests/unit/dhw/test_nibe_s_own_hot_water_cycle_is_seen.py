"""NIBE's own high-temperature hot-water cycle is seen when the tank reaches it.

The hygiene boost runs while no cycle has been seen for DHW_LEGIONELLA_MAX_DAYS. A cycle used to be
seen only once BT7 had ALSO cooled 3 C below its peak inside a 4-hour history. The owner's F750 holds
its tank at 55-56 C for two hours and then cools slowly: 5.5-17.7 h to lose 3 C in October 2026. None
of its cycles was seen, and the hygiene boost ran 13 times in ten days.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from custom_components.effektguard.const import DHW_LEGIONELLA_DETECT
from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler

START = datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc)
STEP = timedelta(minutes=5)


def _tank() -> list[float]:
    """The owner's 2026-10-17 cycle, in shape: 46 C, an hour's heating to 56.3 C, two hours at
    55 C or above, then 0.5 C an hour of cooling - read every 5 minutes for 12 hours."""
    heating = [46.0 + (56.3 - 46.0) * k / 12 for k in range(13)]
    holding = [56.3 - 0.05 * k for k in range(24)]
    cooling = [holding[-1] - 0.5 * k / 12 for k in range(1, 108)]
    return heating + holding + cooling


def test_the_cycle_is_seen_when_the_tank_reaches_it():
    scheduler = IntelligentDHWScheduler()
    readings = _tank()
    for k, temp in enumerate(readings):
        scheduler.update_bt7_temperature(temp, START + k * STEP)

    first = next(k for k, temp in enumerate(readings) if temp >= DHW_LEGIONELLA_DETECT)
    assert scheduler.last_legionella_boost == START + first * STEP


def test_a_cycle_is_seen_once_not_on_every_reading_above_the_threshold():
    """The time of a cycle is when it began: the hours held above the threshold do not move it."""
    scheduler = IntelligentDHWScheduler()
    seen = set()
    for k, temp in enumerate(_tank()):
        scheduler.update_bt7_temperature(temp, START + k * STEP)
        seen.add(scheduler.last_legionella_boost)

    assert len(seen - {None}) == 1

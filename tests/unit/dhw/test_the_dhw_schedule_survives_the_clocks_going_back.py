"""The hours until a DHW demand period are REAL hours, not wall-clock arithmetic.

`_check_upcoming_demand_period` measured the distance to the next scheduled shower with
naive datetime subtraction. On the night the clocks go back, wall-clock arithmetic loses the
repeated hour: 00:30 CEST to 06:00 CET is 5.5 wall-clock hours but 6.5 REAL hours - and the
planner heats water against that figure. The last DST-fragile site in production (F-041).
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from custom_components.effektguard.optimization.dhw_optimizer import (
    DHWDemandPeriod,
    IntelligentDHWScheduler,
)

STOCKHOLM = ZoneInfo("Europe/Stockholm")


def _scheduler_with_morning_period() -> IntelligentDHWScheduler:
    scheduler = IntelligentDHWScheduler.__new__(IntelligentDHWScheduler)
    scheduler.demand_periods = [
        DHWDemandPeriod(availability_hour=6, target_temp=50.0, duration_hours=2)
    ]
    return scheduler


def test_the_fall_back_night_counts_its_extra_hour():
    # 00:30 CEST on fall-back night: 06:00 CET is 6.5 REAL hours away (02:00 happens twice).
    current = datetime(2026, 10, 25, 0, 30, tzinfo=STOCKHOLM)

    info = _scheduler_with_morning_period()._check_upcoming_demand_period(current)

    assert info is not None
    assert info["hours_until"] == 6.5, (
        f"Reported {info['hours_until']} h to the 06:00 demand period. The clocks "
        f"go back at 03:00 CEST, so the pump has 6.5 real hours to heat water, not 5.5 - "
        f"wall-clock subtraction plans the heating an hour short."
    )


def test_an_ordinary_night_is_unchanged():
    current = datetime(2026, 1, 15, 0, 30, tzinfo=STOCKHOLM)

    info = _scheduler_with_morning_period()._check_upcoming_demand_period(current)

    assert info is not None
    assert info["hours_until"] == 5.5

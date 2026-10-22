"""A hot-water charge is learned once, when the pump's priority leaves hot water.

Through the control cycle: tests/unit/coordinator/test_one_hot_water_charge_is_one_observation.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler

START = datetime(2026, 1, 15, 6, 0, tzinfo=timezone.utc)
STEP = timedelta(minutes=5)
CHARGE = [(44.0 + k, True) for k in range(8)] + [(52.0, False)]


def _feed(scheduler: IntelligentDHWScheduler, readings, now: datetime) -> datetime:
    for temp, charging in readings:
        scheduler.update_bt7_temperature(temp, now, charging)
        now += STEP
    return now


def test_a_restart_does_not_learn_a_charge_twice():
    scheduler = IntelligentDHWScheduler()
    now = _feed(scheduler, CHARGE, START)

    restarted = IntelligentDHWScheduler()
    restarted.restore_from_persistence(scheduler.get_dhw_state_for_persistence())
    restarted.bt7_history.extend(scheduler.bt7_history)
    _feed(restarted, [(52.0, False)] * 12, now)

    assert scheduler.heating_rate_observations == restarted.heating_rate_observations == 1


def test_a_priority_that_cannot_say_is_not_timed_across():
    """Unavailable mid-charge ends the span; the rest of the charge is measured on its own."""
    scheduler = IntelligentDHWScheduler()
    _feed(scheduler, CHARGE[:4] + [(48.0, None)] + CHARGE[4:], START)

    assert scheduler.heating_rate_observations == 1
    assert scheduler.learned_heating_rate == pytest.approx(4.0 / (20 / 60))  # 48 -> 52 C

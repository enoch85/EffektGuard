"""The hygiene fallback (RULE 2.3) runs only when NIBE's own cycle is overdue, and once per stretch.

On the owner's F750, NIBE's periodic increase ran every 14-17 days, April-October 2026. The fallback
was due after 14 days and counted only a reading of 55 C on BT7, which a temporary-lux run reaches at
most just (54.3-55.3 C): it ran in ~120 cheap periods, ~90 kWh of electric heat. It now waits for
DHW_LEGIONELLA_OVERDUE_DAYS, and a run that has run its course counts, whatever it reached.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from custom_components.effektguard.const import DHW_LEGIONELLA_OVERDUE_DAYS
from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler

NOON = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
CYCLE = timedelta(minutes=5)


def _decide(scheduler: IntelligentDHWScheduler, now: datetime, price: str = "cheap") -> str:
    return scheduler.should_start_dhw(
        current_dhw_temp=46.0,
        space_heating_demand_kw=0.5,
        thermal_debt_dm=-100.0,
        indoor_temp=21.0,
        target_indoor_temp=21.0,
        outdoor_temp=15.0,
        price_classification=price,
        current_time=now,
    ).priority_reason


def _overdue() -> IntelligentDHWScheduler:
    scheduler = IntelligentDHWScheduler()
    scheduler.last_legionella_boost = NOON - timedelta(days=DHW_LEGIONELLA_OVERDUE_DAYS + 1)
    return scheduler


def _charge(scheduler, now: datetime, peak: float, *, price_while_charging: str = "cheap"):
    """A temporary-lux charge from 46 C to `peak`, the scheduler deciding at every cycle."""
    for k, temp in enumerate([46.0, 49.0, 52.0, peak]):
        scheduler.update_bt7_temperature(temp, now, True)
        _decide(scheduler, now, price_while_charging if k else "cheap")
        now += CYCLE
    scheduler.update_bt7_temperature(peak, now, False)  # the pump ends the charge
    return now


def test_the_fallback_leaves_the_pump_s_own_cycle_to_the_pump():
    scheduler = IntelligentDHWScheduler()
    scheduler.last_legionella_boost = NOON - timedelta(days=15)

    assert _decide(scheduler, NOON) != "DHW_HYGIENE_BOOST", (
        "15 days after the pump's own cycle - due on the owner's F750 in a day or two - the "
        "fallback ran an electric top-up of its own"
    )
    assert _decide(_overdue(), NOON) == "DHW_HYGIENE_BOOST"


def test_a_run_that_ran_its_course_is_not_run_again_the_next_day():
    """It peaked at 54.6 C on BT7, below the 55 C a cycle is seen at: as hot as lux goes."""
    scheduler = _overdue()
    assert _decide(scheduler, NOON) == "DHW_HYGIENE_BOOST"
    _charge(scheduler, NOON + CYCLE, 54.6)

    assert (
        _decide(scheduler, NOON + timedelta(days=1)) != "DHW_HYGIENE_BOOST"
    ), "the fallback ran its course yesterday and ran again in the next cheap period"


def test_a_run_cut_short_is_run_again():
    """The cheap period ended mid-charge: the run did not run its course."""
    scheduler = _overdue()
    _decide(scheduler, NOON)
    _charge(scheduler, NOON + CYCLE, 50.0, price_while_charging="expensive")

    assert _decide(scheduler, NOON + timedelta(days=1)) == "DHW_HYGIENE_BOOST"


def test_a_restart_remembers_the_run():
    scheduler = _overdue()
    _decide(scheduler, NOON)
    _charge(scheduler, NOON + CYCLE, 54.6)

    restarted = IntelligentDHWScheduler()
    restarted.restore_from_persistence(scheduler.get_dhw_state_for_persistence())

    assert _decide(restarted, NOON + timedelta(days=1)) != "DHW_HYGIENE_BOOST"

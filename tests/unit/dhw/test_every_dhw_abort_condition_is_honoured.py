"""An abort condition the decision advertises must be one the checker can act on.

`should_start_dhw` attaches `abort_conditions` to its decision, and the coordinator feeds
them to `check_abort_conditions` on every cycle while the lux window is open. Four call
sites emitted

    f"dhw_temp >= {self.user_target_temp}"

and `check_abort_conditions` had branches for `thermal_debt <` and `indoor_temp <` only.
The string fell through the if/elif chain and the function returned "no abort" - so the
tank's own target, published as an abort condition, did nothing at all. The DHW cycle ran
until NIBE stopped it or the hour's rate limit expired.

The structural half of the fix matters more than the branch: an UNRECOGNISED condition now
logs an ERROR instead of being skipped in silence, so the next condition someone adds
cannot repeat this.
"""

import logging

import pytest

from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler


@pytest.fixture
def scheduler():
    return IntelligentDHWScheduler()


def test_dhw_temp_target_reached_aborts(scheduler):
    """The condition four call sites emit must actually fire."""
    should_abort, reason = scheduler.check_abort_conditions(
        ["dhw_temp >= 50"],
        thermal_debt=-100.0,
        indoor_temp=21.0,
        target_indoor=21.0,
        current_dhw_temp=51.2,
    )
    assert should_abort is True, (
        "DHW reached 51.2 C against a 50 C target and the abort condition the decision "
        "itself attached was ignored."
    )
    assert "51.2" in reason and "50" in reason


def test_dhw_temp_below_target_does_not_abort(scheduler):
    should_abort, _ = scheduler.check_abort_conditions(
        ["dhw_temp >= 50"],
        thermal_debt=-100.0,
        indoor_temp=21.0,
        target_indoor=21.0,
        current_dhw_temp=46.0,
    )
    assert should_abort is False


def test_a_missing_dhw_sensor_is_not_a_reached_target(scheduler):
    """No reading is not the same as "not yet there" - and must not abort either."""
    should_abort, _ = scheduler.check_abort_conditions(
        ["dhw_temp >= 50"],
        thermal_debt=-100.0,
        indoor_temp=21.0,
        target_indoor=21.0,
        current_dhw_temp=None,
    )
    assert should_abort is False


def test_the_existing_conditions_still_work(scheduler):
    """Regression guard for the two branches that always worked."""
    debt, _ = scheduler.check_abort_conditions(
        ["thermal_debt < -500"], -600.0, 21.0, 21.0, 45.0
    )
    cold, _ = scheduler.check_abort_conditions(
        ["indoor_temp < 20.5"], -100.0, 20.1, 21.0, 45.0
    )
    assert debt is True and cold is True


def test_the_first_triggered_condition_wins_in_order(scheduler):
    """Space heating outranks the tank: thermal debt is checked before dhw_temp."""
    should_abort, reason = scheduler.check_abort_conditions(
        ["thermal_debt < -500", "dhw_temp >= 50"],
        thermal_debt=-900.0,
        indoor_temp=21.0,
        target_indoor=21.0,
        current_dhw_temp=55.0,
    )
    assert should_abort is True
    assert "Thermal debt" in reason


def test_an_unknown_condition_is_logged_as_an_error_not_swallowed(scheduler, caplog):
    """The guard that stops this class of bug recurring."""
    with caplog.at_level(logging.ERROR):
        should_abort, _ = scheduler.check_abort_conditions(
            ["flow_temp > 60"], -100.0, 21.0, 21.0, 45.0
        )
    assert should_abort is False
    assert any(
        "Unhandled DHW abort condition" in record.message for record in caplog.records
    ), (
        "An abort condition with no branch was ignored without a trace. That is exactly "
        "how 'dhw_temp >=' went unnoticed across four call sites."
    )


def test_every_condition_should_start_dhw_emits_is_understood(scheduler, caplog):
    """Sweep the decision's own vocabulary: no emitted condition may hit the error path.

    Reads the conditions off real decisions rather than a hand-written list, so a new
    condition added to should_start_dhw is covered the day it appears.
    """
    from datetime import datetime, timedelta, timezone

    tz = timezone(timedelta(hours=1))
    emitted: set[str] = set()
    # Sweep tank temperature and price class; collect whatever conditions come back.
    for dhw_temp in (15.0, 25.0, 35.0, 44.0, 48.0, 55.0):
        for price in ("very_cheap", "cheap", "normal", "expensive", "peak"):
            decision = scheduler.should_start_dhw(
                current_dhw_temp=dhw_temp,
                space_heating_demand_kw=1.0,
                thermal_debt_dm=-100.0,
                indoor_temp=21.0,
                target_indoor_temp=21.0,
                outdoor_temp=2.0,
                price_classification=price,
                current_time=datetime(2026, 1, 15, 12, 0, tzinfo=tz),
                hours_since_last_dhw=6.0,
            )
            emitted.update(decision.abort_conditions)

    assert emitted, "precondition: the decision emits at least one abort condition"
    with caplog.at_level(logging.ERROR):
        for condition in sorted(emitted):
            scheduler.check_abort_conditions([condition], -100.0, 21.0, 21.0, 45.0)

    unhandled = [r.message for r in caplog.records if "Unhandled DHW abort" in r.message]
    assert not unhandled, (
        "should_start_dhw emits conditions check_abort_conditions cannot evaluate: "
        f"{unhandled}"
    )

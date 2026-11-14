"""A meter must report recently; an unchanged value may still be freshly reported."""

from datetime import datetime, timedelta, timezone

import pytest
from homeassistant.core import State

from custom_components.effektguard.const import MAX_BILLING_OBSERVATION_GAP_MINUTES
from custom_components.effektguard.utils.power import power_kw_from_state

NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)
POWER_KW = 20.0


@pytest.mark.parametrize(
    "age,accepted",
    [(MAX_BILLING_OBSERVATION_GAP_MINUTES, True), (MAX_BILLING_OBSERVATION_GAP_MINUTES + 1, False)],
)
def test_a_silent_meter_is_refused_after_the_observation_gap(freezer, age, accepted):
    freezer.move_to(NOW)
    old = NOW - timedelta(minutes=age)
    state = State(
        "sensor.grid",
        str(POWER_KW),
        {"unit_of_measurement": "kW"},
        last_changed=old,
        last_updated=old,
        last_reported=old,
    )
    assert (power_kw_from_state(state, require_recent=True) == POWER_KW) == accepted


def test_a_fresh_report_of_an_unchanged_value_is_still_a_measurement(freezer):
    freezer.move_to(NOW)
    old = NOW - timedelta(days=1)
    state = State(
        "sensor.grid",
        str(POWER_KW),
        {"unit_of_measurement": "kW"},
        last_changed=old,
        last_updated=old,
        last_reported=NOW,
    )
    assert power_kw_from_state(state, require_recent=True) == POWER_KW

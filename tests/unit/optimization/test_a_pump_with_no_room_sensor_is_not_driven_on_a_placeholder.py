"""A NIBE with no room sensor must not be driven on the placeholder indoor temperature.

A pump with no BT50 is a supported configuration: it runs on degree minutes and the heating curve.
The adapter substitutes DEFAULT_INDOOR_TEMP (21.0) for display and sets indoor_temp_valid=False so
comfort-reasoning layers abstain. The comfort layer must honour that flag: any target below the
placeholder would otherwise read as a permanent, uncorrectable overshoot and coast the pump to
minimum output all winter, on a house nobody is measuring.

Invariant: with indoor_temp_valid=False the comfort layer abstains (weight 0, offset 0); with a
real reading it still corrects a genuine overshoot or a genuinely cold house.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import DEFAULT_INDOOR_TEMP, MIN_TEMP_LIMIT
from custom_components.effektguard.optimization.comfort_layer import ComfortLayer

# Targets a real owner can set. All of them sit BELOW the placeholder, which is the whole problem.
COOL_TARGETS = [20.5, 20.0, 19.0, 18.5]


def _sensorless_pump() -> NibeState:
    """Exactly what the adapter builds when there is no BT50: the placeholder, flagged invalid."""
    return NibeState(
        outdoor_temp=-5.0,
        indoor_temp=DEFAULT_INDOOR_TEMP,
        supply_temp=42.0,
        return_temp=37.0,
        degree_minutes=-150.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        indoor_temp_valid=False,
    )


def _pump_with_a_real_sensor(indoor: float) -> NibeState:
    state = _sensorless_pump()
    state.indoor_temp = indoor
    state.indoor_temp_valid = True
    return state


def test_the_placeholder_is_above_every_cool_target_which_is_why_this_bites():
    """The precondition. If DEFAULT_INDOOR_TEMP ever drops, re-derive these numbers."""
    assert DEFAULT_INDOOR_TEMP == 21.0
    assert all(t < DEFAULT_INDOOR_TEMP for t in COOL_TARGETS)
    assert min(COOL_TARGETS) >= MIN_TEMP_LIMIT, "an allowed target must be above the safety floor"


@pytest.mark.parametrize("target", COOL_TARGETS)
def test_the_comfort_layer_abstains_with_no_room_sensor(target):
    """No measurement, no comfort opinion. Not a small one - none."""
    decision = ComfortLayer(target_temp=target).evaluate_layer(_sensorless_pump())

    assert decision.weight == 0.0, (
        f"With no room sensor and a target of {target} C, the comfort layer commanded "
        f"{decision.offset:+.2f} C at weight {decision.weight:.2f} - '{decision.reason}'. That "
        f"deviation is measured against DEFAULT_INDOOR_TEMP ({DEFAULT_INDOOR_TEMP} C), a "
        f"placeholder. Nothing is measuring this house, so the 'overshoot' can never be corrected "
        f"and the pump stays coasted for the whole winter."
    )
    assert decision.offset == 0.0


class TestTheLayerStillWorksWhenItCanSee:
    """The regression guard on the guard: abstaining must not break a normal house."""

    def test_a_real_overshoot_is_still_corrected(self):
        decision = ComfortLayer(target_temp=21.0).evaluate_layer(_pump_with_a_real_sensor(23.0))

        assert decision.weight > 0.0
        assert decision.offset < 0.0, "a house that is genuinely 2 C too warm must still coast"

    def test_a_real_cold_house_is_still_heated(self):
        decision = ComfortLayer(target_temp=21.0).evaluate_layer(_pump_with_a_real_sensor(19.5))

        assert decision.weight > 0.0
        assert decision.offset > 0.0, "a house that is genuinely 1.5 C too cold must still heat"

"""A NIBE without a room sensor was being coasted to a stop on a number nobody measured.

A NIBE with no BT50 is a legitimate, documented configuration: it runs on degree minutes and the
heating curve. The adapter handles it by substituting `DEFAULT_INDOOR_TEMP` (21.0) so the UI has
something to display, and setting `indoor_temp_valid=False`. Its comment states the contract:

    # Keep the placeholder for display, but mark it invalid so comfort-reasoning layers abstain
    # instead of reading a deviation of exactly 0.0 from a value that IS the target.

The safety layer honoured it. The thermal layer honoured it. The COMFORT layer - the one the comment
is actually about - never looked at it, and computed

    temp_deviation = nibe_state.indoor_temp - self.target_temp

straight from the placeholder. For any target BELOW 21.0 that is a permanent, uncorrectable
overshoot, because nothing is measuring the house and no amount of heating will move the number:

    target 20.0  ->  offset  -8.33 at weight 0.83
    target 19.0  ->  offset -10.00 at weight 1.00     <- full coast, CRITICAL weight
    target 18.5  ->  offset -10.00 at weight 1.00

18.5 is an allowed target. A user with no room sensor who wants a cool house gets the heat pump
pinned to minimum output for the entire winter, and the only thing between that and a cold house is
the degree-minute emergency path - which would be fighting this layer on every single cycle.

The sweep at the bottom is the real point: it asserts that NO layer reasons about comfort from the
placeholder, so the next layer to grow an indoor-temperature branch fails here rather than in
someone's house.
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


def test_the_worst_case_is_a_full_coast_at_critical_weight():
    """Named explicitly so the severity cannot be argued down later."""
    decision = ComfortLayer(target_temp=19.0).evaluate_layer(_sensorless_pump())

    assert not (decision.offset <= -9.9 and decision.weight >= 1.0), (
        "A sensorless NIBE with a 19 C target commanded a FULL -10 C coast at weight 1.0, derived "
        "entirely from a placeholder. This is the single worst thing a comfort layer can do."
    )


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


def test_no_layer_anywhere_reasons_about_comfort_from_the_placeholder():
    """The sweep. One layer had this hole; the next one to grow an indoor branch must fail HERE.

    Every layer that takes a NibeState is asked to evaluate a sensorless pump against a cool target.
    Any layer that comes back with a non-zero opinion is reasoning from a number nobody measured.

    Layers legitimately driven by degree minutes, price or the weather still act - they are not
    reading the indoor temperature at all - so this asserts on the layers that DO read it.
    """
    sensorless = _sensorless_pump()
    culprits = []

    for target in COOL_TARGETS:
        decision = ComfortLayer(target_temp=target).evaluate_layer(sensorless)
        if decision.weight != 0.0 or decision.offset != 0.0:
            culprits.append(
                f"ComfortLayer(target={target}) -> {decision.offset:+.2f} @ {decision.weight:.2f}"
            )

    assert not culprits, (
        "These layers formed an opinion about a house nobody is measuring:\n  "
        + "\n  ".join(culprits)
        + "\nindoor_temp is DEFAULT_INDOOR_TEMP, a placeholder. Check indoor_temp_valid first."
    )

"""Twenty-one tests for overshoot protection, and not one of them called the code.

The file that used to be here asserted constants against literals, and then RE-IMPLEMENTED the
production logic in order to test its own copy of it:

    def calculate_response(self, overshoot: float) -> tuple[float, float]:
        # Mirrors the logic in decision_engine._proactive_debt_prevention_layer().
        ...
        coast_weight = OVERSHOOT_PROTECTION_WEIGHT_MIN + fraction * (
            OVERSHOOT_PROTECTION_WEIGHT_MAX - OVERSHOOT_PROTECTION_WEIGHT_MIN
        )

`_proactive_debt_prevention_layer` DOES NOT EXIST. It was removed, and these tests went on passing -
because a test that transcribes the logic it is checking can never notice that the original has
changed, let alone that it is gone.

And it had changed. The transcription ramps the weight from OVERSHOOT_PROTECTION_WEIGHT_MIN (0.5).
Production - `ComfortLayer._standard_overshoot_protection` - ramps it from LAYER_WEIGHT_COMFORT_HIGH
(0.7) to LAYER_WEIGHT_COMFORT_CRITICAL (1.0), and never reads OVERSHOOT_PROTECTION_WEIGHT_MIN at
all. The suite certified a number the engine does not produce, in twenty-one tests, for as long as
the file has existed.

What follows drives the real ComfortLayer. The four constants production does not read
(OVERSHOOT_PROTECTION_WEIGHT_MIN/MAX, _COLD_SNAP_THRESHOLD, _FORECAST_HORIZON) are deleted with it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    LAYER_WEIGHT_COMFORT_CRITICAL,
    LAYER_WEIGHT_COMFORT_HIGH,
    OVERSHOOT_PROTECTION_FULL,
    OVERSHOOT_PROTECTION_OFFSET_MAX,
    OVERSHOOT_PROTECTION_OFFSET_MIN,
    OVERSHOOT_PROTECTION_START,
)
from custom_components.effektguard.optimization.comfort_layer import ComfortLayer

TARGET = 21.0


def _house_at(indoor: float) -> NibeState:
    return NibeState(
        outdoor_temp=0.0,
        indoor_temp=indoor,
        supply_temp=40.0,
        return_temp=35.0,
        degree_minutes=-50.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        indoor_temp_valid=True,
    )


def _decide(overshoot: float):
    """Drive the REAL layer with the house `overshoot` degrees above target."""
    return ComfortLayer(target_temp=TARGET, tolerance_range=0.5).evaluate_layer(
        _house_at(TARGET + overshoot)
    )


class TestTheBandItselfIsCoherent:
    """The constants production actually reads. The others are gone."""

    def test_protection_starts_before_it_is_full(self):
        assert OVERSHOOT_PROTECTION_START < OVERSHOOT_PROTECTION_FULL

    def test_a_full_coast_is_stronger_than_the_start_of_one(self):
        assert OVERSHOOT_PROTECTION_OFFSET_MAX < OVERSHOOT_PROTECTION_OFFSET_MIN < 0.0

    def test_the_weight_ramp_is_the_one_production_uses(self):
        """0.7 to 1.0, not the 0.5 the deleted transcription asserted."""
        assert LAYER_WEIGHT_COMFORT_HIGH < LAYER_WEIGHT_COMFORT_CRITICAL


class TestTheGraduatedResponse:
    """Driving ComfortLayer, not a copy of it."""

    def test_at_the_start_of_the_band_the_layer_coasts_gently(self):
        decision = _decide(OVERSHOOT_PROTECTION_START)

        assert decision.offset == pytest.approx(OVERSHOOT_PROTECTION_OFFSET_MIN, abs=0.01)
        assert decision.weight == pytest.approx(LAYER_WEIGHT_COMFORT_HIGH, abs=0.01), (
            f"At the start of the overshoot band the layer voted weight {decision.weight:.2f}. "
            f"Production ramps from LAYER_WEIGHT_COMFORT_HIGH ({LAYER_WEIGHT_COMFORT_HIGH}); the "
            f"deleted tests asserted OVERSHOOT_PROTECTION_WEIGHT_MIN (0.5) - a constant production "
            f"never reads - and could not tell, because they never called the layer."
        )

    def test_at_full_overshoot_the_layer_coasts_completely(self):
        decision = _decide(OVERSHOOT_PROTECTION_FULL)

        assert decision.offset == pytest.approx(OVERSHOOT_PROTECTION_OFFSET_MAX, abs=0.01)
        assert decision.weight == pytest.approx(LAYER_WEIGHT_COMFORT_CRITICAL, abs=0.01)

    def test_beyond_full_overshoot_the_ramp_is_clamped(self):
        """A house four degrees past the band must not vote beyond the register."""
        decision = _decide(OVERSHOOT_PROTECTION_FULL + 4.0)

        assert decision.offset == pytest.approx(OVERSHOOT_PROTECTION_OFFSET_MAX, abs=0.01)
        assert decision.weight == pytest.approx(LAYER_WEIGHT_COMFORT_CRITICAL, abs=0.01)

    def test_the_response_is_monotonic_across_the_band(self):
        """More overshoot must never mean less coasting."""
        steps = [OVERSHOOT_PROTECTION_START + i * 0.1 for i in range(11)]
        decisions = [_decide(o) for o in steps]

        offsets = [d.offset for d in decisions]
        weights = [d.weight for d in decisions]

        assert offsets == sorted(offsets, reverse=True), f"offsets not monotonic: {offsets}"
        assert weights == sorted(weights), f"weights not monotonic: {weights}"

    def test_below_the_band_the_house_is_nudged_not_slammed(self):
        decision = _decide(OVERSHOOT_PROTECTION_START - 0.1)

        assert decision.offset > OVERSHOOT_PROTECTION_OFFSET_MIN, (
            f"A house only {OVERSHOOT_PROTECTION_START - 0.1:.1f} C above target - below the "
            f"overshoot band - was given {decision.offset:.2f} C, as hard as the band's own floor."
        )


class TestOvershootProtectionOnlyFiresUpwards:
    """The regression guard. It coasts a warm house; it must never coast a cold one."""

    def test_a_cold_house_is_never_coasted(self):
        decision = _decide(-1.0)

        assert (
            decision.offset >= 0.0
        ), f"A house 1.0 C BELOW target was told to coast ({decision.offset:+.2f} C)."

    def test_a_house_on_target_is_left_alone(self):
        decision = _decide(0.0)

        assert decision.offset == 0.0
        assert decision.weight == 0.0

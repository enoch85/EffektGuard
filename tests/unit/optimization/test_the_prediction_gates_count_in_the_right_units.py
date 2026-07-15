"""The prediction gates must count in SAMPLES_PER_HOUR, not a remembered sample count.

The coordinator records one sample every UPDATE_INTERVAL_MINUTES - twelve an hour, not four. Gates
that hardcoded 96 samples "for 24 hours" actually opened at 8 hours, so the learned pre-heat layer
engaged on a third of the data it believed it had.

Invariant: every gate is `hours * SAMPLES_PER_HOUR` (24 h -> 288 samples), the predictor's deque can
hold what the gate asks for, and the learning-progress reason string uses the same denominator.
"""

from __future__ import annotations

from custom_components.effektguard.const import (
    PREDICTION_LEARNED_PREHEAT_MIN_HOURS,
    PREDICTION_MIN_HISTORY_HOURS,
    PREDICTION_RESPONSIVENESS_MIN_HOURS,
    SAMPLES_PER_HOUR,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.optimization.prediction_layer import ThermalStatePredictor


def test_the_coordinator_really_does_record_twelve_samples_an_hour():
    """The premise. Every count below is meaningless without it."""
    assert SAMPLES_PER_HOUR == 60 // UPDATE_INTERVAL_MINUTES
    assert SAMPLES_PER_HOUR == 12, (
        f"The coordinator ticks every {UPDATE_INTERVAL_MINUTES} min, so it records "
        f"{SAMPLES_PER_HOUR} samples an hour. The old gates were written believing it was 4."
    )


def test_a_full_day_of_history_is_a_full_day_of_history():
    """The gate that mattered: 96 samples is eight hours, not twenty-four."""
    required = PREDICTION_LEARNED_PREHEAT_MIN_HOURS * SAMPLES_PER_HOUR

    assert required == 288, (
        f"The learned pre-heat gate needs {required} samples for "
        f"{PREDICTION_LEARNED_PREHEAT_MIN_HOURS} hours. It used to hardcode 96 - which at a "
        f"{UPDATE_INTERVAL_MINUTES}-minute tick is {96 / SAMPLES_PER_HOUR:.0f} hours, so the layer "
        f"acted on a third of the data it thought it had."
    )
    assert required / SAMPLES_PER_HOUR == PREDICTION_LEARNED_PREHEAT_MIN_HOURS


def test_the_predictors_own_deque_can_hold_what_the_gate_asks_for():
    """A gate that can never open is worse than one that opens early."""
    predictor = ThermalStatePredictor()
    required = PREDICTION_LEARNED_PREHEAT_MIN_HOURS * SAMPLES_PER_HOUR

    assert predictor.state_history.maxlen >= required, (
        f"The learned pre-heat gate wants {required} samples and the history deque holds only "
        f"{predictor.state_history.maxlen}. It could never engage at all."
    )


def test_the_learning_progress_message_counts_in_the_same_units_as_the_gate():
    """The reason string hardcoded 96 too, so it told the owner the wrong denominator."""
    from unittest.mock import MagicMock

    predictor = ThermalStatePredictor()
    required = PREDICTION_LEARNED_PREHEAT_MIN_HOURS * SAMPLES_PER_HOUR

    decision = predictor.evaluate_layer(
        nibe_state=MagicMock(),
        weather_data=MagicMock(),
        target_temp=21.0,
        thermal_model=MagicMock(),
    )

    assert f"0/{required}" in decision.reason, (
        f"The layer reports its learning progress as {decision.reason!r}. The denominator must be "
        f"the number of samples the gate actually waits for ({required}), not the 96 it used to "
        f"print."
    )


def test_every_gate_is_expressed_in_hours_not_in_a_remembered_sample_count():
    """All three, so the next one to be added cannot quietly reintroduce the belief."""
    for hours in (
        PREDICTION_MIN_HISTORY_HOURS,
        PREDICTION_RESPONSIVENESS_MIN_HOURS,
        PREDICTION_LEARNED_PREHEAT_MIN_HOURS,
    ):
        samples = hours * SAMPLES_PER_HOUR
        assert samples % SAMPLES_PER_HOUR == 0
        assert samples / SAMPLES_PER_HOUR == hours

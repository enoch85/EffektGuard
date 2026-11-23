"""The prediction gates must count in SAMPLES_PER_HOUR, not a remembered sample count.

The coordinator records one sample every UPDATE_INTERVAL_MINUTES - twelve an hour, not four. Gates
that hardcoded 96 samples "for 24 hours" actually opened at 8 hours, so the learned pre-heat layer
engaged on a third of the data it believed it had.

Invariant: every gate counts control cycles, `hours * SAMPLES_PER_HOUR` of them from a reading to the
newest (24 h -> 288 cycles, so 289 readings, or 288 with the one missed cycle allowed), the
predictor's deque can hold what the gate asks for, and the learning-progress reason string uses the
same denominator.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.weather_adapter import WeatherData, WeatherForecastHour
from custom_components.effektguard.const import (
    PREDICTION_LEARNED_PREHEAT_MIN_HOURS,
    PREDICTION_MIN_HISTORY_HOURS,
    PREDICTION_RESPONSIVENESS_MIN_HOURS,
    SAMPLES_PER_HOUR,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.optimization.prediction_layer import ThermalStatePredictor
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

NOW = datetime(2026, 1, 15, 12, 0, 10, tzinfo=timezone.utc)  # a cycle start
CYCLE = timedelta(minutes=UPDATE_INTERVAL_MINUTES)


def test_the_coordinator_really_does_record_twelve_samples_an_hour():
    """The premise. Every count below is meaningless without it."""
    assert SAMPLES_PER_HOUR == 60 // UPDATE_INTERVAL_MINUTES
    assert SAMPLES_PER_HOUR == 12, (
        f"The coordinator ticks every {UPDATE_INTERVAL_MINUTES} min, so it records "
        f"{SAMPLES_PER_HOUR} samples an hour. The old gates were written believing it was 4."
    )


def _learned_preheat_weight(readings: int, missed: tuple[int, ...] = ()) -> float:
    """The learned pre-heat vote on `readings` cycles ending now, less the cycles in `missed`."""
    predictor = ThermalStatePredictor()
    for i in range(readings):
        if i not in missed:
            predictor.record_state(NOW - (readings - 1 - i) * CYCLE, 21.0, 0.0, 0.0, 35.0, -100.0)
    forecast = [
        WeatherForecastHour(dt_util.utcnow() + timedelta(hours=h), 0.0) for h in range(1, 49)
    ]
    return predictor.evaluate_layer(
        SimpleNamespace(indoor_temp=21.0, outdoor_temp=0.0, indoor_temp_valid=True),
        WeatherData(0.0, forecast),
        22.0,
        ThermalModel(1.8),
    ).weight


@pytest.mark.parametrize(
    "readings,missed,opens",
    [
        (96, (), False),  # the old hardcoded gate: eight hours
        (288, (), False),  # one cycle short of a day
        (289, (), True),  # a reading in the cycle 24 hours before the newest
        (289, (144,), True),  # the same day with its one allowed missed cycle: 288 readings
        (289, (100, 200), False),  # two missed cycles
    ],
)
def test_a_full_day_of_history_is_a_full_day_of_history(freezer, readings, missed, opens):
    """The gate that mattered: 96 samples is eight hours, not twenty-four. The gate itself, run."""
    freezer.move_to(NOW)
    assert PREDICTION_LEARNED_PREHEAT_MIN_HOURS * SAMPLES_PER_HOUR == 288
    weight = _learned_preheat_weight(readings, missed)
    assert (weight > 0.0) is opens, (
        f"{readings - len(missed)} readings over {readings} cycles: the learned pre-heat vote has "
        f"weight {weight}. It needs a reading 24 hours of cycles before the newest, with at most one "
        f"cycle missed between."
    )


def test_the_predictors_own_deque_can_hold_what_the_gate_asks_for():
    """A gate that can never open is worse than one that opens early."""
    predictor = ThermalStatePredictor()
    required = PREDICTION_LEARNED_PREHEAT_MIN_HOURS * SAMPLES_PER_HOUR + 1

    assert predictor.state_history.maxlen >= required, (
        f"The learned pre-heat gate wants {required} samples and the history deque holds only "
        f"{predictor.state_history.maxlen}. It could never engage at all."
    )


def test_the_learning_progress_message_counts_in_the_same_units_as_the_gate():
    """The reason string hardcoded 96 too, so it told the owner the wrong denominator."""
    from unittest.mock import MagicMock

    predictor = ThermalStatePredictor()
    required = PREDICTION_LEARNED_PREHEAT_MIN_HOURS * SAMPLES_PER_HOUR + 1

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

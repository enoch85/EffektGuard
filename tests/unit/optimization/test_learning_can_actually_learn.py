"""Learning cannot engage on a real house, because it is asked to see through its own noise floor.

The indoor sensor (NIBE BT1) reports to 0.1 °C. The coordinator observes every 5 minutes. A house
warming at a brisk 0.6 °C/h moves 0.05 °C between two observations - **half a sensor tick** - so every
recorded rate is quantised to 0 °C/h or 1.2 °C/h, and nothing in between. The rate series is not a
measurement of the building; it is a measurement of the sampling interval.

The window is the second half of it. `LEARNING_OBSERVATION_WINDOW = 672` is commented "1 week of
15-minute observations", but the coordinator recorded every 5 minutes, so the deque spanned **56
hours**. It is a rolling window, so day 90 saw exactly what day 3 saw. The README's "Day 8-14: high
confidence, fully optimized" was unreachable by construction - there is no day 8 in a 56-hour memory.

Sampling a building's thermal response every five minutes is measuring noise. A house has a time
constant of hours; a concrete slab lags six. Hourly observation puts the signal above the sensor's
resolution AND gives the same 672-entry deque a 28-day memory, which is the timescale the learning was
always described in. That is what `LEARNING_OBSERVATION_INTERVAL_MINUTES` fixes, and the two tests
below hold it.

**IT IS NOT ENOUGH, AND AN EARLIER DRAFT OF THIS FILE CLAIMED IT WAS.** That draft carried a table
promising 0.707 at 30 minutes and 0.811 at 60. Measured against the real learner, on three house types
across five cadences, the answer is the same everywhere: **0.415, and it never engages.**

`consistency = 1 - std/mean` is taken over EVERY heating observation, and a house sitting at
equilibrium contributes a rate of exactly zero. Those zeros are averaged INTO the mean - 186 of 338
samples on a wooden house at hourly cadence - so the mean is dragged below the 0.1 °C/h floor by the
samples where the house was doing nothing, consistency pins to 0.0, and confidence caps at
obs(0.4) + time(0.2) = **0.600** against a 0.7 gate. Forever, on any house. **The better the control,
the stiller the house, the less it can be learned.**

And it cannot be repaired by tuning. Filtering down to the samples that DO move scores the 5-minute
cadence at a **perfect 1.000** - because at that cadence the only rates clearing the floor are exactly
one sensor quantum, so they are all identical, std collapses to zero, and the quantisation artefact
reads as certainty. That is the flatlined-sensor bug wearing a different hat. `std/mean` does not
measure knowledge; it rewards data for being degenerate, and a dead sensor is the most degenerate data
there is.

Confidence has to be measured by what it claims to measure: PREDICTION ERROR against held-out
observations. That is a redesign of a metric that gates the pre-heating layer at weight 0.65, so it is
the owner's call, and it is recorded as a strict xfail below rather than quietly left green.

Owner decision: *"Learning is one of the key stones here."* So it has to be able to learn - and the
cadence was necessary, but it was not the thing standing in the way.
"""

from __future__ import annotations

import inspect
import math
from datetime import datetime, timedelta

import pytest

from custom_components.effektguard.const import (
    LEARNING_CONFIDENCE_THRESHOLD,
    LEARNING_OBSERVATION_INTERVAL_MINUTES,
    LEARNING_OBSERVATION_WINDOW,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.optimization import decision_engine
from custom_components.effektguard.optimization.adaptive_learning import AdaptiveThermalModel

SENSOR_QUANTUM = 0.1  # °C - what a NIBE BT1 can actually report


def _observe_a_real_house(cadence_minutes: int, days: int = 30) -> AdaptiveThermalModel:
    """A house with an honest thermal response, watched at `cadence_minutes`.

    Indoor temperature follows the outdoor swing with lag and is nudged by the heating offset. The
    crucial detail is the last line: the sensor is READ THROUGH ITS QUANTUM, so what the model sees is
    what a NIBE actually reports, not the true continuous temperature.
    """
    model = AdaptiveThermalModel(initial_thermal_mass=1.0)

    start = datetime(2026, 1, 1, 0, 0)
    indoor_true = 21.0

    for step in range(int(days * 24 * 60 / cadence_minutes)):
        now = start + timedelta(minutes=cadence_minutes * step)
        hours = step * cadence_minutes / 60.0

        # Outdoor: a -5 °C winter mean with a 5 °C diurnal swing.
        outdoor = -5.0 + 5.0 * math.sin(2 * math.pi * hours / 24.0)

        # Heating: the curve pushes harder when it is colder.
        offset = 2.0 if outdoor < -5.0 else 0.0

        # First-order building response toward an equilibrium set by outdoor + heating.
        equilibrium = 21.0 + 0.15 * (outdoor + 5.0) + 0.8 * offset
        tau_hours = 12.0
        dt_hours = cadence_minutes / 60.0
        indoor_true += (equilibrium - indoor_true) * (dt_hours / tau_hours)

        # The sensor can only say what a sensor can say.
        model.record_observation(
            timestamp=now,
            indoor_temp=round(indoor_true / SENSOR_QUANTUM) * SENSOR_QUANTUM,
            outdoor_temp=outdoor,
            heating_offset=offset,
        )

    model.update_learned_parameters()
    return model


def test_the_observation_window_spans_the_timescale_a_building_is_learned_on():
    """672 observations at the recording cadence must be a MEMORY, not a weekend."""
    span_hours = LEARNING_OBSERVATION_WINDOW * LEARNING_OBSERVATION_INTERVAL_MINUTES / 60.0

    assert span_hours >= 7 * 24, (
        f"The observation deque holds {LEARNING_OBSERVATION_WINDOW} entries recorded every "
        f"{LEARNING_OBSERVATION_INTERVAL_MINUTES} minutes, so it remembers {span_hours:.0f} hours - "
        f"{span_hours / 24:.1f} days. It is a ROLLING window, so the model on day 90 sees exactly "
        f"what it saw on day {span_hours / 24:.1f}. A building cannot be learned from a memory "
        f"shorter than the promise made about it."
    )


def test_the_observation_cadence_is_slower_than_the_control_cadence():
    """Learning and control are different questions on different timescales.

    Control runs every 5 minutes because the pump needs steering. Learning must not: a 0.1 °C sensor
    sampled every 5 minutes reports the quantisation, not the house.
    """
    assert LEARNING_OBSERVATION_INTERVAL_MINUTES > UPDATE_INTERVAL_MINUTES, (
        f"Learning observes every {LEARNING_OBSERVATION_INTERVAL_MINUTES} min, the same as the "
        f"control loop ({UPDATE_INTERVAL_MINUTES} min). A house warming at 0.6 °C/h moves 0.05 °C in "
        f"five minutes - half a sensor tick - so every rate quantises to 0.0 or 1.2 °C/h and the "
        f"scatter is pure sampling artefact."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "F-132b: learning cannot engage on ANY house, at ANY cadence, and the cadence was only half "
        "the story. `consistency = 1 - std/mean` is computed over EVERY heating observation, and a "
        "house at equilibrium contributes a rate of exactly zero - so the mean is dragged under the "
        "0.1 C/h floor by the samples where the house was doing nothing, and consistency is pinned "
        "to 0.0. Confidence then caps at obs(0.4) + time(0.2) = 0.600, under a 0.7 gate, forever. "
        "Measured: wooden 0.415, brick 0.415, concrete 0.415 - every house, every cadence. "
        "The metric is not repairable by tuning: filtering to the samples that DO move makes the "
        "5-minute cadence score a PERFECT 1.000, because at that cadence the only rates above the "
        "floor are exactly one sensor quantum and therefore all identical - std collapses to zero "
        "and the quantisation artefact reads as certainty. std/mean rewards data for being "
        "degenerate. Confidence has to be measured by PREDICTION ERROR against held-out "
        "observations, which is a redesign of a control-path metric at weight 0.65. OWNER DECISION."
    ),
)
def test_learning_engages_on_a_house_that_behaves_like_a_house():
    """The whole point. A real building, watched properly, must become knowable."""
    model = _observe_a_real_house(LEARNING_OBSERVATION_INTERVAL_MINUTES, days=30)
    params = model.get_parameters()

    assert params is not None, "no parameters were learned at all"
    assert params.confidence >= LEARNING_CONFIDENCE_THRESHOLD, (
        f"After 30 days of hourly observation of a house with an entirely ordinary thermal response, "
        f"confidence reached {params.confidence:.3f} against a gate of {LEARNING_CONFIDENCE_THRESHOLD}. "
        f"Learning never engages, so the adaptive model is decoration."
    )
    assert model.should_use_learned_parameters()


def test_the_production_cadence_could_not_learn_this_same_house():
    """The control, so nobody has to take the docstring on trust.

    Identical house, identical physics, identical sensor - only the sampling interval differs.
    """
    model = _observe_a_real_house(UPDATE_INTERVAL_MINUTES, days=30)
    params = model.get_parameters()

    confidence = params.confidence if params else 0.0
    assert confidence < LEARNING_CONFIDENCE_THRESHOLD, (
        "precondition failed: the 5-minute cadence now DOES learn this house, which means the "
        "premise of this change is wrong and it should be revisited rather than kept."
    )


def test_a_flatlined_sensor_still_teaches_us_nothing():
    """The F-132 regression guard, and the reason the metric cannot simply be loosened.

    A dead indoor sensor - one value, forever - used to score PERFECT consistency, because std/mean
    reads zero scatter as certainty. It earned 0.867 confidence and engaged, while a house that was
    genuinely heating scored 0.467 and did not. Whatever replaces the metric must keep this at zero.
    """
    model = AdaptiveThermalModel(initial_thermal_mass=1.0)
    start = datetime(2026, 1, 1, 0, 0)

    for step in range(LEARNING_OBSERVATION_WINDOW):
        model.record_observation(
            timestamp=start + timedelta(minutes=LEARNING_OBSERVATION_INTERVAL_MINUTES * step),
            indoor_temp=21.0,  # the sensor died; it says 21.0 and will say 21.0 forever
            outdoor_temp=-5.0 + 5.0 * math.sin(2 * math.pi * step / 24.0),
            heating_offset=2.0,
        )

    model.update_learned_parameters()
    params = model.get_parameters()
    confidence = params.confidence if params else 0.0

    assert confidence < LEARNING_CONFIDENCE_THRESHOLD, (
        f"A flatlined indoor sensor scored {confidence:.3f} confidence and would drive the pump "
        f"through the pre-heating layer at weight 0.65 on parameters derived from a dead sensor."
    )
    assert not model.should_use_learned_parameters()


def test_the_heat_loss_coefficient_is_never_used_as_a_control_input():
    """It is quarantined at the source, and it must stay that way.

    `_calculate_heat_loss_coefficient` says so itself: indoor temperature decay ALONE cannot yield a
    W/K coefficient - that needs thermal capacitance or measured heat input, and neither is available.
    The `* 3600 * 50` in it is, in its own words, "a heuristic mapping into a plausible-looking
    100-300 range, nothing more". It comes out clamped at 300.0 on the houses above: a ceiling, not a
    measurement.

    The decision engine takes heat_loss_coefficient from the user's configuration. This test exists so
    that stays true - the number LOOKS like physics, and that is exactly what makes it dangerous.
    """
    source = inspect.getsource(decision_engine)

    assert (
        "learned" not in source or "heat_loss_coefficient" not in source.split("learned")[1][:200]
    )

    model = _observe_a_real_house(LEARNING_OBSERVATION_INTERVAL_MINUTES, days=30)
    params = model.get_parameters()
    assert params is not None

    # It is pinned to its clamp, which is the tell: this is not a measurement of anything.
    assert (
        params.heat_loss_coefficient in (100.0, 180.0, 300.0)
        or 100.0 <= params.heat_loss_coefficient <= 300.0
    )


class TestTwoDefectsWereCancellingEachOther:
    """The xfail above blames the confidence metric. That is ONE of two reasons learning is dead.

    THE SECOND: `should_use_learned_parameters()` reads `learned_parameters["confidence"]`, and
    `update_learned_parameters()` returns the confidence on a dataclass and NEVER stores it in that
    dict. The only production writer of `learned_parameters` is the `insulation_quality` setter,
    and it writes one key: `heat_loss_coefficient`. So the gate returns False forever, however well
    the model learns and whatever the owner does to the confidence metric.

    AND THE DEAD GATE WAS THE ONLY THING KEEPING THE CODE SAFE. `calculate_preheating_target` had:

        if params and self.should_use_learned_parameters():
            heat_loss_coef = params.heat_loss_coefficient      # a RELATIVE, DIMENSIONLESS INDEX
        else:
            heat_loss_coef = 180.0  # W/°C typical house       # a PHYSICAL COEFFICIENT
        heat_loss_rate = (heat_loss_coef / 1000.0) * decay_rate * temp_diff / 10

    Two different UNITS on the two branches of one variable, divided by 1000 as if it were watts.
    And `_calculate_heat_loss_coefficient`'s own docstring forbids exactly this:

        "It MUST NOT be used as an absolute W/°C coefficient anywhere in the control path"

    So repairing the gate - which looks like an obvious one-line bug - would have silently armed a
    unit error in the pre-heat path. Two defects cancelling is not a working system; it is a trap
    for whoever fixes the first one.

    The trap is disarmed: the coefficient comes from configuration, never from learning, exactly as
    the estimator instructs. ENABLING learning is still F-132b and still the owner's call. Making
    it safe to enable was not.
    """

    def test_the_gate_can_never_open_however_much_the_model_learns(self):
        """The second, independent reason. Recorded, not fixed - opening it is F-132b."""
        model = _observe_a_real_house(LEARNING_OBSERVATION_INTERVAL_MINUTES, days=30)
        params = model.update_learned_parameters()

        assert params is not None, "precondition: the model must have learned something at all"
        assert "confidence" not in model.learned_parameters, (
            f"`learned_parameters` now holds {sorted(model.learned_parameters)}. If a confidence "
            f"has appeared there, someone has repaired the gate - check first that "
            f"calculate_preheating_target still takes its heat-loss coefficient from CONFIGURATION "
            f"and not from the quarantined relative index, or the pre-heat is now sized with a "
            f"dimensionless number divided by 1000 as if it were watts."
        )
        assert not model.should_use_learned_parameters(), (
            "the gate reads a confidence that nothing writes, so it is False forever - a SECOND "
            "reason learning never engages, independent of the confidence metric that the xfail "
            "above records"
        )

    def test_the_preheat_never_sizes_itself_with_the_quarantined_index(self):
        """The trap, disarmed. The control path must not touch the learned coefficient at all.

        The estimator's own docstring says its value is a relative cooling index and must never be
        used as W/°C in the control path. So the pre-heat must size identically whether that index
        reads 180 or 3000 - because production does not read it.

        The decay rate is pinned to a realistic POSITIVE value here. Left to itself the model
        learns a decay of -0.10 (it believes the house warms as it cools, which is its own
        symptom of F-132b), and a negative decay zeroes the deficit for any coefficient at all -
        so the first version of this test compared 21.0 against 21.0 and passed against a plant
        with the trap fully re-armed. A test has to be in a regime where the thing it guards could
        actually move the answer.
        """
        model = _observe_a_real_house(LEARNING_OBSERVATION_INTERVAL_MINUTES, days=30)
        model.learned_parameters = {"confidence": 1.0}  # force the gate wide open
        model._calculate_thermal_decay_rate = lambda: 0.15  # a house that actually cools

        call = dict(
            current_temp=21.0,
            desired_temp=21.0,
            hours_until_peak=6,
            outdoor_temp=-5.0,
            forecast_min_temp=-10.0,
        )

        model._calculate_heat_loss_coefficient = lambda: 180.0
        baseline = model.calculate_preheating_target(**call)

        assert baseline > call["desired_temp"], (
            f"PRECONDITION: the pre-heat must actually be sizing something ({baseline:.2f} C vs a "
            f"target of {call['desired_temp']:.2f}), or a change in the coefficient could not move "
            f"it and this test would prove nothing."
        )

        model._calculate_heat_loss_coefficient = lambda: 3000.0  # an absurd relative index
        with_absurd_index = model.calculate_preheating_target(**call)

        assert with_absurd_index == pytest.approx(baseline), (
            f"Multiplying the QUARANTINED relative cooling index by 17 moved the pre-heat target "
            f"from {baseline:.2f} C to {with_absurd_index:.2f} C. That index is dimensionless - "
            f"its own estimator says it 'MUST NOT be used as an absolute W/°C coefficient anywhere "
            f"in the control path' - and this is the control path, dividing it by 1000 as if it "
            f"were watts."
        )

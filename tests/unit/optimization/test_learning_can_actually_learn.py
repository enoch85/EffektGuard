"""Learning must be able to engage on a real house - and today it cannot, for two reasons.

Learning observes hourly (LEARNING_OBSERVATION_INTERVAL_MINUTES), not at the 5-minute control
cadence: a 0.1 C sensor sampled every five minutes reports quantisation, not the house. The
672-entry deque is therefore a 28-day memory. The window/cadence tests hold that.

Even so, learning never engages (F-132b, the strict xfail below): the confidence metric caps at
0.600 on any real house, and - independently - the confidence gate reads a dict key that nothing
writes, so it is False forever. The remaining tests hold that the pre-heat never sizes itself from
the quarantined heat-loss index. Enabling learning is the owner's call.
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
    """A dead indoor sensor - one value forever - must score below the gate.

    std/mean reads zero scatter as certainty, so a flat line could earn perfect consistency.
    Whatever replaces the confidence metric must keep this case at zero.
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
    """The learned heat-loss coefficient is a relative index, not W/K, and must never reach control.

    `_calculate_heat_loss_coefficient` cannot yield a physical W/K value from decay alone; it
    lands clamped in a plausible 100-300 range. The decision engine takes the coefficient from
    configuration instead, and this test holds that the learned index stays out of the source.
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
    """A second, independent reason learning is inert - and the trap disarming it revealed.

    The gate `should_use_learned_parameters()` reads `learned_parameters["confidence"]`, a key
    only the `insulation_quality` setter ever writes (and it writes `heat_loss_coefficient`, not
    confidence), so the gate is False forever. That dead gate once masked a unit error:
    `calculate_preheating_target` would have fed the learned relative index into
    `heat_loss_coef / 1000.0` as if it were W/K. Production now always uses the configured
    coefficient; these two tests hold the gate closed and the pre-heat off the learned index.
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
        """The control path must not touch the learned coefficient at all.

        The pre-heat must size identically whether the learned index reads 180 or 3000, because
        production takes the coefficient from configuration. Decay is pinned to a positive value
        so the deficit is non-zero and a re-armed trap could actually move the answer.
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

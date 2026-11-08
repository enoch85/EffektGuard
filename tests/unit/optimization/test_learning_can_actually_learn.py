"""Learning is earned by held-out forecast error, with exact-horizon control gates.

Hourly observations can identify a useful model without allowing a flat sensor, a burst of
five-minute samples, or a configured confidence number to authorize heating. Estimates of
physical heat loss remain quarantined from temperature prediction and control.
"""

from __future__ import annotations

import inspect
import math
from datetime import datetime, timedelta, timezone

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


def _observe_a_real_house(
    cadence_minutes: int, days: int = 30, start: datetime | None = None
) -> AdaptiveThermalModel:
    """A house with an honest thermal response, watched at `cadence_minutes`.

    Indoor temperature follows the outdoor swing with lag and is nudged by the heating offset. The
    crucial detail is the last line: the sensor is READ THROUGH ITS QUANTUM, so what the model sees is
    what a NIBE actually reports, not the true continuous temperature.
    """
    model = AdaptiveThermalModel(initial_thermal_mass=1.0)

    start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
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


def test_the_gate_reads_validated_forecasts_instead_of_the_config_dictionary():
    model = _observe_a_real_house(LEARNING_OBSERVATION_INTERVAL_MINUTES, days=30)
    model.learned_parameters["confidence"] = 0.0
    assert model.should_use_learned_parameters()
    model.observations.clear()
    model._forecast_cache = None
    model.learned_parameters["confidence"] = 1.0
    assert not model.should_use_learned_parameters()


def test_the_room_forecast_never_uses_the_quarantined_heat_loss_index():
    from dataclasses import replace

    model = _observe_a_real_house(LEARNING_OBSERVATION_INTERVAL_MINUTES, days=30)
    params = model.get_parameters()
    forecast = params.forecasts[6]
    room = model.observations[-1].indoor_temp
    baseline = forecast.predict(room, [-5.0] * 6, [0.0] * 6)
    assert baseline is not None
    changed = replace(params, heat_loss_coefficient=3000.0, heating_efficiency=100.0)
    assert changed.forecasts[6].predict(room, [-5.0] * 6, [0.0] * 6) == baseline


def test_validation_targets_cannot_change_the_fitted_coefficients():
    from dataclasses import replace
    from custom_components.effektguard.optimization.thermal_forecast import fit_forecast

    model = _observe_a_real_house(LEARNING_OBSERVATION_INTERVAL_MINUTES, days=30)
    history = list(model.observations)
    original = fit_forecast(history, 6)
    assert original is not None and original.skill >= LEARNING_CONFIDENCE_THRESHOLD
    split = len(history) // 2
    altered = history[:split] + [
        replace(o, indoor_temp=o.indoor_temp + 2.0) for o in history[split:]
    ]
    failed = fit_forecast(altered, 6)
    assert failed is not None
    assert failed.coefficients == pytest.approx(original.coefficients)
    assert failed.skill < LEARNING_CONFIDENCE_THRESHOLD


def test_validated_forecasts_refuse_inputs_outside_their_training_domain():
    model = _observe_a_real_house(LEARNING_OBSERVATION_INTERVAL_MINUTES, days=30)
    forecast = model.get_parameters().forecasts[6]
    room = model.observations[-1].indoor_temp
    assert forecast.predict(room, [-5.0] * 6, [0.0] * 6) is not None
    assert forecast.predict(room, [-50.0] * 6, [0.0] * 6) is None
    assert forecast.predict(room, [-5.0] * 6, [10.0] * 6) is None
    assert forecast.predict(room, [-5.0] * 6, [0.0]) is None


@pytest.mark.parametrize("model_age_hours", [0, 2])
def test_a_validated_horizon_can_recommend_heat_without_trusting_other_horizons(
    freezer, model_age_hours
):
    from types import SimpleNamespace
    from custom_components.effektguard.optimization.prediction_layer import ThermalStatePredictor

    model = _observe_a_real_house(LEARNING_OBSERVATION_INTERVAL_MINUTES, days=30)
    model.thermal_mass = 1.3  # The configured timber horizon looks six hours ahead.
    end = model.observations[-1].timestamp + timedelta(hours=model_age_hours)
    freezer.move_to(end)
    room = model.observations[-1].indoor_temp
    predictor = ThermalStatePredictor()
    for cycle in reversed(range(24 * 60 // UPDATE_INTERVAL_MINUTES + 1)):
        predictor.record_state(
            end - timedelta(minutes=cycle * UPDATE_INTERVAL_MINUTES), room, -5.0, 0.0, 35.0, -100.0
        )
    weather = SimpleNamespace(forecast_hours=[SimpleNamespace(temperature=-5.0)] * 12)
    state = SimpleNamespace(indoor_temp_valid=True, indoor_temp=room, outdoor_temp=-5.0)
    proposal = predictor.evaluate_layer(state, weather, room + 1.0, model)
    assert (proposal.weight > 0.0) == (model_age_hours == 0)
    assert 0.0 <= proposal.offset <= 2.0
    params = model.get_parameters()
    projection = predictor.predict_temperature(6, [-5.0] * 6, [0.0] * 6, learned_params=params)
    assert (projection.hourly_confidence[-1] >= LEARNING_CONFIDENCE_THRESHOLD) == (
        model_age_hours == 0
    )
    assert projection.hourly_confidence[0] == 0.0
    if model_age_hours == 0:
        current_room = round(room - SENSOR_QUANTUM, 1)
        changed = predictor.predict_temperature(
            6, [-5.0] * 6, [0.0] * 6, learned_params=params, current_indoor_temp=current_room
        )
        expected = params.forecasts[6].predict(current_room, [-5.0] * 6, [0.0] * 6)
        assert expected is not None
        assert changed.predicted_temps[-1] == pytest.approx(expected)
        assert changed.predicted_temps[-1] != pytest.approx(projection.predicted_temps[-1])
        from dataclasses import replace

        below = replace(
            params,
            forecasts={
                6: replace(
                    params.forecasts[6], skill=LEARNING_CONFIDENCE_THRESHOLD - SENSOR_QUANTUM
                )
            },
        )
        untrusted = predictor.predict_temperature(6, [-5.0] * 6, [0.0] * 6, learned_params=below)
        assert untrusted.predicted_temps == [room] * 6
        assert untrusted.hourly_confidence == [0.0] * 6


def test_a_new_hourly_observation_refreshes_the_cached_forecast_state():
    model = _observe_a_real_house(LEARNING_OBSERVATION_INTERVAL_MINUTES, days=30)
    previous = model.get_parameters().observations_end
    model.record_observation(previous + timedelta(hours=1), 21.9, -5.0, 0.0)
    assert model.get_parameters().observations_end == previous + timedelta(hours=1)

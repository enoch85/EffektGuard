"""Forecast skill must use the same future-weather and applied-offset intervals as control."""

from datetime import datetime, timedelta, timezone
from random import Random
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from custom_components.effektguard.adapters.weather_adapter import WeatherData, WeatherForecastHour
from custom_components.effektguard.const import (
    LEARNING_CONFIDENCE_THRESHOLD,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.optimization.adaptive_learning import AdaptiveThermalModel
from custom_components.effektguard.optimization.prediction_layer import ThermalStatePredictor
from tests.unit.optimization.test_learning_can_actually_learn import _observe_a_real_house


def test_weather_is_from_the_future_but_the_applied_offset_starts_now():
    rng = Random(47)
    outside = [rng.uniform(-12.0, 2.0) for _ in range(481)]
    offsets = [rng.choice([-2.0, 0.0, 2.0]) for _ in range(480)]
    model = AdaptiveThermalModel()
    room = 21.0
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # One-hour energy-balance steps: outside[i+1] is that interval's right-end weather;
    # the register written at i acts during [i,i+1), before the next write.
    for i, offset in enumerate(offsets):
        model.record_observation(start + timedelta(hours=i), round(room, 1), outside[i], offset)
        equilibrium = 21.0 + 0.6 * (outside[i + 1] + 5.0) + 0.8 * offset
        room += (equilibrium - room) / 4.0
    forecast = model.get_parameters().forecasts[1]
    assert forecast.skill > 0.9
    assert forecast.validation_mae < 0.1


def _trusted_predictor(now):
    model = _observe_a_real_house(60, start=now - timedelta(hours=30 * 24 - 1))
    model.thermal_mass = 1.3
    assert model.get_parameters().forecasts[6].skill >= LEARNING_CONFIDENCE_THRESHOLD
    room = model.observations[-1].indoor_temp
    predictor = ThermalStatePredictor()
    for cycle in reversed(range(24 * 60 // UPDATE_INTERVAL_MINUTES + 1)):
        predictor.record_state(
            now - timedelta(minutes=cycle * UPDATE_INTERVAL_MINUTES), room, -5.0, 0.0, 35.0, -100.0
        )
    state = SimpleNamespace(indoor_temp_valid=True, indoor_temp=room, outdoor_temp=-5.0)
    return predictor, model, state


@pytest.mark.parametrize(
    "case", ["valid", "past", "gap", "duplicate", "naive", "short", "too_late"]
)
def test_preheating_requires_weather_covering_the_actual_requested_hours(freezer, case):
    now = datetime(2026, 1, 15, 10, 5, tzinfo=timezone.utc)
    freezer.move_to(now)
    predictor, model, state = _trusted_predictor(now)
    hours = [WeatherForecastHour(now + timedelta(hours=h), -5.0) for h in range(1, 13)]
    if case == "past":
        hours = [WeatherForecastHour(h.datetime - timedelta(days=2), h.temperature) for h in hours]
    elif case == "gap":
        del hours[2]
    elif case == "duplicate":
        hours.insert(2, hours[2])
    elif case == "naive":
        hours = [WeatherForecastHour(h.datetime.replace(tzinfo=None), h.temperature) for h in hours]
    elif case == "short":
        hours = hours[:5]
    elif case == "too_late":
        hours = [WeatherForecastHour(h.datetime + timedelta(days=2), h.temperature) for h in hours]
    vote = predictor.evaluate_layer(state, WeatherData(-5.0, hours), state.indoor_temp + 1.0, model)
    assert (vote.weight > 0.0) == (case == "valid")


@pytest.mark.parametrize("instant", ["2026-03-29T00:30:00+00:00", "2026-10-25T00:30:00+00:00"])
def test_clock_changes_keep_the_same_real_hours_as_the_explicit_trajectory(freezer, instant):
    now = datetime.fromisoformat(instant)
    freezer.move_to(now)
    predictor, model, state = _trusted_predictor(now)
    source_temperatures = [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0]
    temperatures = [(a + b) / 2 for a, b in zip(source_temperatures, source_temperatures[1:])]
    weather = WeatherData(
        -5.0,
        [
            WeatherForecastHour(
                (now + timedelta(minutes=30, hours=i)).astimezone(ZoneInfo("Europe/Stockholm")), t
            )
            for i, t in enumerate(source_temperatures)
        ],
    )
    # Straddle the actual control threshold: a shifted real-hour window must change one vote.
    previous = None
    boundary = None
    for step in range(-200, 201):
        target = state.indoor_temp + step / 100
        expected = predictor.should_pre_heat(
            target, 6, temperatures, -5.0, state.indoor_temp, learned_params=model.get_parameters()
        )
        if expected.should_preheat and previous is not None and not previous[1].should_preheat:
            boundary = [previous, (target, expected)]
            break
        previous = (target, expected)
    assert boundary is not None
    for target, expected in boundary:
        vote = predictor.evaluate_layer(state, weather, target, model)
        assert (vote.weight > 0.0) == expected.should_preheat
        assert vote.offset == pytest.approx(expected.recommended_offset)

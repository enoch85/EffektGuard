"""The weather vote Home Assistant builds reads the weather history restored at setup."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import _create_coordinator
from custom_components.effektguard.adapters.weather_adapter import WeatherData, WeatherForecastHour
from custom_components.effektguard.const import (
    DOMAIN,
    LEARNING_STORAGE_VERSION,
    STORAGE_KEY_LEARNING,
)
from custom_components.effektguard.optimization.weather_learning import WeatherPatternLearner

FORECAST_HOURS = 24
STOCKHOLM_LATITUDE = 59.33
FLOW_ABOVE_REQUIRED = 4.0
DEGREE_MINUTES = -50.0
MID_JANUARY = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
EARLY_JANUARY = datetime(2026, 1, 5, 12, tzinfo=timezone.utc)
# 23:30 UTC on 7 January is 00:30 on 8 January in Stockholm; the coordinator files the forecast under
# the UTC date, the first week.
LAST_UTC_HALF_HOUR_OF_THE_FIRST_WEEK = datetime(2026, 1, 7, 23, 30, tzinfo=timezone.utc)
STOCKHOLM = "Europe/Stockholm"
FIRST_WEEK_LAST_YEAR = datetime(2025, 1, 2, tzinfo=timezone.utc)
SECOND_WEEK_LAST_YEAR = datetime(2025, 1, 9, tzinfo=timezone.utc)
LAST_YEARS = (2023, 2024, 2025)
NORMAL = 0.0
COLD = -15.0
COLDER = -30.0
WARM = 15.0


def _store_history(hass_storage, patterns):
    """Persist weather patterns the way the coordinator saves them."""
    learner = WeatherPatternLearner()
    for taken, temperature in patterns:
        learner.record_weather_pattern(taken, [temperature] * FORECAST_HOURS)
    hass_storage[STORAGE_KEY_LEARNING] = {
        "version": LEARNING_STORAGE_VERSION,
        "minor_version": 1,
        "key": STORAGE_KEY_LEARNING,
        "data": {"version": LEARNING_STORAGE_VERSION, "weather_patterns": learner.to_dict()},
    }


async def _engine(hass):
    hass.config.latitude = STOCKHOLM_LATITUDE
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    return (await _create_coordinator(hass, entry)).engine


def _vote(engine, now, forecast_temperature, outdoor=NORMAL):
    required = engine.weather_comp.calculate_optimal_flow_temp(engine.target_temp, outdoor)
    state = SimpleNamespace(
        outdoor_temp=outdoor,
        flow_temp=required.flow_temp + FLOW_ABOVE_REQUIRED,
        degree_minutes=DEGREE_MINUTES,
    )
    weather = WeatherData(
        outdoor,
        [
            WeatherForecastHour(now + timedelta(hours=hour), forecast_temperature)
            for hour in range(1, FORECAST_HOURS + 1)
        ],
    )
    return engine.weather_comp_layer.evaluate_layer(state, weather, engine.target_temp)


async def test_the_live_vote_judges_the_forecast_by_the_history_restored_at_setup(
    hass, hass_storage, freezer
):
    freezer.move_to(MID_JANUARY)
    _store_history(hass_storage, [(MID_JANUARY.replace(year=y), COLD) for y in LAST_YEARS])
    engine = await _engine(hass)
    without_history = WeatherPatternLearner(engine.weather_comp_layer.weather_learner.climate_zone)
    assert without_history.detect_unusual_weather(MID_JANUARY, [COLD] * FORECAST_HOURS).is_unusual

    usual = _vote(engine, MID_JANUARY, COLD, outdoor=COLD)
    colder = _vote(engine, MID_JANUARY, COLDER, outdoor=COLD)

    assert not usual.unusual_weather, "this house's own Januaries are this cold"
    assert colder.unusual_weather
    assert colder.safety_margin > usual.safety_margin
    assert colder.weight > usual.weight
    assert colder.offset > usual.offset


async def test_the_forecast_is_compared_with_the_week_it_was_taken_in(hass, hass_storage, freezer):
    freezer.move_to(EARLY_JANUARY)
    _store_history(hass_storage, [(FIRST_WEEK_LAST_YEAR, NORMAL), (SECOND_WEEK_LAST_YEAR, COLD)])
    engine = await _engine(hass)
    learner = engine.weather_comp_layer.weather_learner

    assert not learner.detect_unusual_weather(EARLY_JANUARY, [NORMAL] * FORECAST_HOURS).is_unusual
    assert learner.detect_unusual_weather(EARLY_JANUARY, [COLD] * FORECAST_HOURS).is_unusual
    assert _vote(engine, EARLY_JANUARY, COLD).unusual_weather


@pytest.mark.parametrize(
    ("first_week", "second_week", "unusual"),
    [(NORMAL, COLD, True), (COLD, NORMAL, False)],
    ids=["colder than its own week", "usual for its own week"],
)
async def test_after_local_midnight_the_vote_reads_the_week_the_forecast_is_filed_under(
    hass, hass_storage, freezer, first_week, second_week, unusual
):
    await hass.config.async_set_time_zone(STOCKHOLM)
    freezer.move_to(LAST_UTC_HALF_HOUR_OF_THE_FIRST_WEEK)
    _store_history(
        hass_storage, [(FIRST_WEEK_LAST_YEAR, first_week), (SECOND_WEEK_LAST_YEAR, second_week)]
    )
    engine = await _engine(hass)

    vote = _vote(engine, LAST_UTC_HALF_HOUR_OF_THE_FIRST_WEEK, COLD)

    assert vote.unusual_weather is unusual


async def test_a_forecast_warmer_than_usual_does_not_hold_back_the_cut(hass, hass_storage, freezer):
    freezer.move_to(MID_JANUARY)
    _store_history(hass_storage, [(MID_JANUARY.replace(year=y), NORMAL) for y in LAST_YEARS])
    engine = await _engine(hass)
    assert engine.weather_comp_layer.weather_learner.detect_unusual_weather(
        MID_JANUARY, [WARM] * FORECAST_HOURS
    ).is_unusual

    usual = _vote(engine, MID_JANUARY, NORMAL)
    warmer = _vote(engine, MID_JANUARY, WARM)

    assert not warmer.unusual_weather
    assert (warmer.safety_margin, warmer.weight, warmer.offset) == (
        usual.safety_margin,
        usual.weight,
        usual.offset,
    )

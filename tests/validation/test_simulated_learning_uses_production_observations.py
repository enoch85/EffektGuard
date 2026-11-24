"""Simulation learning must change the same votes as the coordinator's observations."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pytest_homeassistant.common import MockConfigEntry

from scripts.simulation import sim_harness as sim
from custom_components.effektguard import _create_coordinator
from custom_components.effektguard.adapters.weather_adapter import WeatherData, WeatherForecastHour
from custom_components.effektguard.const import (
    DOMAIN,
    LEARNING_STORAGE_VERSION,
    STORAGE_KEY_LEARNING,
)
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.weather_learning import WeatherPatternLearner

FALL_BACK = datetime(2026, 10, 25, 1, tzinfo=timezone.utc)


def test_learned_unusual_weather_changes_the_simulated_weather_vote(freezer):
    now = datetime(2026, 1, 15, tzinfo=timezone.utc)
    freezer.move_to(now)
    engine, _ = sim.build_engine(sim.HOUSES[0])
    for year in (2023, 2024, 2025):
        engine.weather_learner.record_weather_pattern(now.replace(year=year), [0.0] * 24)
    required = engine.weather_comp.calculate_optimal_flow_temp(engine.target_temp, 0.0).flow_temp
    state = SimpleNamespace(outdoor_temp=0.0, flow_temp=required + 2.0, degree_minutes=-50.0)

    def vote(temperature):
        weather = WeatherData(
            0.0, [WeatherForecastHour(now + timedelta(hours=h), temperature) for h in range(1, 25)]
        )
        return engine.weather_comp_layer.evaluate_layer(state, weather, engine.target_temp)

    normal, unusual = vote(0.0), vote(-30.0)
    assert unusual.unusual_weather
    assert unusual.safety_margin > normal.safety_margin
    assert unusual.weight > normal.weight
    assert unusual.offset > normal.offset


def test_simulation_records_forecast_patterns_and_hourly_observations_across_dst(monkeypatch):
    engines = []
    original = sim.build_engine

    def capture(*args, **kwargs):
        engine, effect = original(*args, **kwargs)
        engines.append(engine)
        return engine, effect

    monkeypatch.setattr(sim, "build_engine", capture)
    start = datetime(2026, 10, 24, tzinfo=sim.TZ)
    times, temperatures, raw, unit = sim._synthetic_days(start, 3)
    stats, violations, _ = sim.simulate(
        sim.HOUSES[0], times, temperatures, sim.PriceSource(raw, unit), 3
    )
    assert stats["exceptions"] == 0
    observations = list(engines[0].thermal.observations)
    assert len(observations) > 48
    elapsed = [
        b.timestamp.timestamp() - a.timestamp.timestamp()
        for a, b in zip(observations, observations[1:])
    ]
    assert set(elapsed) == {3600.0}
    patterns = [p for group in engines[0].weather_learner.pattern_db.values() for p in group]
    assert len(patterns) >= 3
    dates = [p.date.astimezone(timezone.utc).date() for p in patterns]
    assert len(dates) == len(set(dates))
    assert all(p.date.utcoffset() == timedelta(0) for p in patterns)


def _fall_back_run(monkeypatch, days, temperatures=None):
    """A simulated run from 24 October 2026 in Stockholm, over the night the clocks go back."""
    engines = []
    original = sim.build_engine

    def capture(*args, **kwargs):
        engine, effect = original(*args, **kwargs)
        engines.append(engine)
        return engine, effect

    monkeypatch.setattr(sim, "build_engine", capture)
    times, temps, raw, unit = sim._synthetic_days(datetime(2026, 10, 24, tzinfo=sim.TZ), days)
    temps = temperatures(times) if temperatures else temps
    stats, _, _ = sim.simulate(sim.HOUSES[0], times, temps, sim.PriceSource(raw, unit), days)
    assert stats["exceptions"] == 0
    return engines[0], times


def test_the_pattern_learned_across_the_fall_back_is_the_next_24_real_hours(monkeypatch):
    # Rising 0.1 C each real hour, so a window an hour off shows in every figure of the pattern.
    engine, times = _fall_back_run(
        monkeypatch, 3, lambda times: [-5.0 + 0.1 * hour for hour in range(len(times))]
    )
    recorded = datetime(2026, 10, 25, tzinfo=timezone.utc)  # the first cycle of that UTC date
    hour_now = (recorded - times[0]).total_seconds() / 3600
    expected = [-5.0 + 0.1 * (hour_now + ahead) for ahead in range(1, 25)]
    patterns = [p for group in engine.weather_learner.pattern_db.values() for p in group]
    pattern = next(p for p in patterns if p.date == recorded)
    assert pattern.avg_temp == pytest.approx(sum(expected) / len(expected))
    assert (pattern.min_temp, pattern.max_temp) == pytest.approx((expected[0], expected[-1]))


def test_the_predictor_records_on_the_real_time_line_across_the_fall_back(monkeypatch):
    engine, _ = _fall_back_run(monkeypatch, 2)
    history = list(engine.predictor.state_history)
    assert history[0].timestamp < FALL_BACK < history[-1].timestamp
    steps = {(b.timestamp - a.timestamp).total_seconds() for a, b in zip(history, history[1:])}
    assert steps == {sim.STEP_MIN * 60.0}


def test_the_engine_reads_the_pump_on_the_real_time_line_across_the_fall_back(monkeypatch):
    """Production's adapter dates the pump's state in UTC, and the guards that subtract those dates
    (saturation, anti-windup, capacity shortfall, the immersion start) count real time. Two local
    times of one zone subtract on the wall clock: an hour backwards in the repeated hour."""
    stamps = []
    decide = DecisionEngine.calculate_decision

    def capture(self, *args, **kwargs):
        stamps.append(kwargs["nibe_state"].timestamp)
        return decide(self, *args, **kwargs)

    monkeypatch.setattr(DecisionEngine, "calculate_decision", capture)
    _fall_back_run(monkeypatch, 2)
    assert stamps[0] < FALL_BACK < stamps[-1]
    assert {stamp.utcoffset() for stamp in stamps} == {timedelta(0)}
    steps = {(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])}
    assert steps == {sim.STEP_MIN * 60.0}


async def test_the_simulated_weather_vote_is_the_one_production_sets_up(
    hass, hass_storage, freezer
):
    now = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
    freezer.move_to(now)
    history = WeatherPatternLearner()
    for year in (2023, 2024, 2025):
        history.record_weather_pattern(now.replace(year=year), [0.0] * 24)
    hass_storage[STORAGE_KEY_LEARNING] = {
        "version": LEARNING_STORAGE_VERSION,
        "minor_version": 1,
        "key": STORAGE_KEY_LEARNING,
        "data": {"version": LEARNING_STORAGE_VERSION, "weather_patterns": history.to_dict()},
    }
    hass.config.latitude = sim.STOCKHOLM_LATITUDE
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    production = (await _create_coordinator(hass, entry)).engine
    simulated, _ = sim.build_engine(sim.HOUSES[0])
    simulated.weather_learner.load_from_dict(history.to_dict())

    def vote(engine, forecast):
        required = engine.weather_comp.calculate_optimal_flow_temp(engine.target_temp, 0.0)
        state = SimpleNamespace(
            outdoor_temp=0.0, flow_temp=required.flow_temp + 2.0, degree_minutes=-50.0
        )
        weather = WeatherData(
            0.0, [WeatherForecastHour(now + timedelta(hours=h), forecast) for h in range(1, 25)]
        )
        decision = engine.weather_comp_layer.evaluate_layer(state, weather, engine.target_temp)
        return decision.unusual_weather, decision.safety_margin

    for forecast in (-10.0, 15.0):  # colder and warmer than this house's own Januaries
        assert vote(simulated, forecast) == vote(production, forecast)
    blank, _ = sim.build_engine(sim.HOUSES[0])  # the same engine with no history
    assert vote(simulated, -10.0)[0] and not vote(blank, -10.0)[0], "only the history marks -10 C"

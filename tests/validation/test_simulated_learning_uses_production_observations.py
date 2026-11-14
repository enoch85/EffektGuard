"""Simulation learning must change the same votes as the coordinator's observations."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from scripts.simulation import sim_harness as sim
from custom_components.effektguard.adapters.weather_adapter import WeatherData, WeatherForecastHour


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

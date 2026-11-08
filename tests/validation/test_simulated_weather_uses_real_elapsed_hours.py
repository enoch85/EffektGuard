"""The real engine must receive each real future hour, including both DST transitions."""

from datetime import datetime, timedelta, timezone

import pytest

from scripts.simulation import sim_harness as sim


@pytest.mark.parametrize("month,day", [(3, 28), (10, 24)])
def test_hourly_weather_interpolation_uses_elapsed_time(month, day):
    start = datetime(2026, month, day, tzinfo=sim.TZ)
    times, _, _, _ = sim._synthetic_days(start, 5)
    temperatures = [-5.0 + i / 10 for i in range(len(times))]
    instant = datetime(2026, month, day + 1, 1, 30, tzinfo=timezone.utc)
    elapsed = (instant - times[0].astimezone(timezone.utc)).total_seconds() / 3600
    assert sim.outdoor_at(times, temperatures, instant.astimezone(sim.TZ)) == pytest.approx(
        -5.0 + elapsed / 10
    )


@pytest.mark.parametrize("month,day", [(3, 28), (10, 24)])
def test_the_engine_receives_real_hourly_forecast_instants(monkeypatch, month, day):
    start = datetime(2026, month, day, tzinfo=sim.TZ)
    times, _, prices, unit = sim._synthetic_days(start, 5)
    temperatures = [-5.0 + i / 10 for i in range(len(times))]
    real_build = sim.build_engine
    bad = []
    crossed_change = []

    def build(*args, **kwargs):
        engine, effect = real_build(*args, **kwargs)
        real_decide = engine.calculate_decision

        def decide(*args, **kwargs):
            state, weather = kwargs["nibe_state"], kwargs["weather_data"]
            origin = state.timestamp.astimezone(timezone.utc)
            for index, point in enumerate(weather.forecast_hours, start=1):
                future = point.datetime.astimezone(timezone.utc)
                if (future - origin).total_seconds() != index * 3600:
                    if len(bad) < 3:
                        bad.append((origin.isoformat(), point.datetime.isoformat(), index))
                elapsed = (future - times[0].astimezone(timezone.utc)).total_seconds() / 3600
                if abs(point.temperature - (-5.0 + elapsed / 10)) > 1e-9:
                    if len(bad) < 3:
                        bad.append(("temperature", point.datetime.isoformat(), point.temperature))
                if point.datetime.utcoffset() != state.timestamp.utcoffset():
                    crossed_change.append(True)
            return real_decide(*args, **kwargs)

        engine.calculate_decision = decide
        return engine, effect

    monkeypatch.setattr(sim, "build_engine", build)
    stats, _, _ = sim.simulate(sim.HOUSES[0], times, temperatures, sim.PriceSource(prices, unit), 2)
    assert stats["exceptions"] == 0
    assert crossed_change, "the forecast must actually span the clock change"
    assert not bad, bad

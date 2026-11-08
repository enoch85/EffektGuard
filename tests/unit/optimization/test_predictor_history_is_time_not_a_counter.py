"""Only a fresh, continuous day of measurements may authorize learned heat."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from custom_components.effektguard.adapters.weather_adapter import WeatherData, WeatherForecastHour
from custom_components.effektguard.const import (
    PREDICTION_LEARNED_PREHEAT_MIN_HOURS,
    SAMPLES_PER_HOUR,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.optimization.prediction_layer import ThermalStatePredictor
from tests.unit.optimization.test_learning_can_actually_learn import _observe_a_real_house

NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)
SAMPLES = PREDICTION_LEARNED_PREHEAT_MIN_HOURS * SAMPLES_PER_HOUR + 1
CADENCE = timedelta(minutes=UPDATE_INTERVAL_MINUTES)


def _record(predictor, timestamp, indoor=21.8):
    predictor.record_state(timestamp, indoor, 0.0, 0.0, 35.0, -100.0)


def _vote(predictor):
    weather = WeatherData(
        0.0, [WeatherForecastHour(NOW + timedelta(hours=h), 0.0) for h in range(1, 49)]
    )
    model = _observe_a_real_house(60, start=NOW - timedelta(hours=30 * 24 - 1))
    model.thermal_mass = 1.3
    return predictor.evaluate_layer(
        SimpleNamespace(indoor_temp=21.8, outdoor_temp=0.0, indoor_temp_valid=True),
        weather,
        22.8,
        model,
    )


@pytest.mark.parametrize("kind", ["duplicates", "sparse", "gap", "stale", "future", "compressed"])
def test_unusable_history_does_not_authorize_preheat(freezer, kind):
    freezer.move_to(NOW)
    predictor = ThermalStatePredictor()
    for i in range(SAMPLES):
        timestamp = NOW - (SAMPLES - i - 1) * CADENCE
        if kind == "duplicates":
            timestamp = NOW
        elif kind == "sparse":
            timestamp = NOW - (SAMPLES - i - 1) * CADENCE * 2
        elif kind == "gap":
            timestamp = (
                NOW - timedelta(hours=24) + timedelta(minutes=2 * i)
                if i < 144
                else NOW - timedelta(hours=8) + timedelta(hours=8 * (i - 144) / 144)
            )
        elif kind == "stale":
            timestamp -= timedelta(days=2)
        elif kind == "future":
            timestamp += timedelta(hours=1)
        elif kind == "compressed":
            timestamp = NOW - (SAMPLES - i - 1) * CADENCE * (SAMPLES - 2) / (SAMPLES - 1)
        _record(predictor, timestamp)
    predictor = ThermalStatePredictor.from_dict(predictor.to_dict())
    assert _vote(predictor).weight == 0.0


def test_a_complete_fresh_day_still_can_authorize_preheat(freezer):
    freezer.move_to(NOW)
    predictor = ThermalStatePredictor()
    for i in range(SAMPLES):
        _record(predictor, NOW - (SAMPLES - i - 1) * CADENCE)
    assert _vote(predictor).weight > 0.0


def test_the_day_gate_waits_for_the_last_elapsed_interval(freezer):
    freezer.move_to(NOW)
    predictor = ThermalStatePredictor()
    for i in range(SAMPLES - 1):
        _record(predictor, NOW - (SAMPLES - i - 2) * CADENCE)
    assert _vote(predictor).weight == 0.0


def test_repeated_and_out_of_order_reads_do_not_invent_time(freezer):
    freezer.move_to(NOW)
    predictor = ThermalStatePredictor()
    _record(predictor, NOW)
    _record(predictor, NOW, 22.0)
    _record(predictor, NOW - CADENCE, 19.0)
    assert len(predictor.state_history) == 1
    assert predictor.state_history[-1].indoor_temp == 22.0
    assert predictor._calculate_prediction_confidence() == 0.0


def test_room_trend_uses_real_elapsed_time_across_the_repeated_hour(freezer):
    end = datetime(2026, 10, 25, 3, 30, tzinfo=ZoneInfo("Europe/Stockholm"))
    freezer.move_to(end)
    predictor = ThermalStatePredictor()
    for i in range(SAMPLES):
        t = (end.astimezone(timezone.utc) - (SAMPLES - i - 1) * CADENCE).astimezone(end.tzinfo)
        _record(predictor, t, 21.0 + i * 0.01)
    assert predictor.get_current_trend()["rate_per_hour"] == pytest.approx(0.01 * SAMPLES_PER_HOUR)

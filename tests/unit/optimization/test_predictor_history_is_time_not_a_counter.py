"""Only a fresh, continuous day of measurements may authorize learned heat."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from custom_components.effektguard.adapters.weather_adapter import WeatherData, WeatherForecastHour
from custom_components.effektguard.const import (
    PREDICTION_LEARNED_PREHEAT_MIN_HOURS,
    PREDICTION_RESPONSIVENESS_MIN_HOURS,
    SAMPLES_PER_HOUR,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.optimization.prediction_layer import ThermalStatePredictor
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

# A control cycle's start: the coordinator runs at :00:10, :05:10, ...
NOW = datetime(2026, 1, 15, 0, 0, 10, tzinfo=timezone.utc)
SAMPLES = PREDICTION_LEARNED_PREHEAT_MIN_HOURS * SAMPLES_PER_HOUR + 1
CADENCE = timedelta(minutes=UPDATE_INTERVAL_MINUTES)


def _record(predictor, timestamp, indoor=21.0):
    predictor.record_state(timestamp, indoor, 0.0, 0.0, 35.0, -100.0)


def _vote(predictor):
    weather = WeatherData(
        0.0, [WeatherForecastHour(NOW + timedelta(hours=h), 0.0) for h in range(1, 49)]
    )
    return predictor.evaluate_layer(
        SimpleNamespace(indoor_temp=21.0, outdoor_temp=0.0, indoor_temp_valid=True),
        weather,
        22.0,
        ThermalModel(1.8),
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


def test_outdoor_trend_uses_real_elapsed_time_across_the_repeated_hour(freezer):
    end = datetime(2026, 10, 25, 3, 30, tzinfo=ZoneInfo("Europe/Stockholm"))
    freezer.move_to(end)
    predictor = ThermalStatePredictor()
    for i in range(SAMPLES):
        t = (end.astimezone(timezone.utc) - (SAMPLES - i - 1) * CADENCE).astimezone(end.tzinfo)
        predictor.record_state(t, 21.0, i * 0.02, 0.0, 35.0, -100.0)
    assert predictor.get_outdoor_trend()["rate_per_hour"] == pytest.approx(0.02 * SAMPLES_PER_HOUR)


def test_a_day_whose_newest_reading_is_dated_after_now_is_not_used(freezer):
    ahead = NOW + CADENCE
    freezer.move_to(ahead)
    predictor = ThermalStatePredictor()
    for i in range(SAMPLES):
        _record(predictor, ahead - (SAMPLES - i - 1) * CADENCE)
    assert _vote(predictor).weight > 0.0, "the same day, read when it was measured"

    freezer.move_to(NOW)
    assert _vote(predictor).weight == 0.0


def test_a_restored_snapshot_dated_in_the_future_does_not_keep_the_present_out(freezer):
    freezer.move_to(NOW)
    predictor = ThermalStatePredictor()
    for i in range(SAMPLES):
        _record(predictor, NOW - (SAMPLES - i - 1) * CADENCE)
    stored = predictor.to_dict()
    ahead = dict(stored["state_history"][-1], timestamp=(NOW + timedelta(days=2)).isoformat())
    stored["state_history"].append(ahead)

    predictor = ThermalStatePredictor.from_dict(stored)
    assert _vote(predictor).weight > 0.0, "the real day behind the future snapshot is kept"

    later = NOW + CADENCE
    freezer.move_to(later)
    _record(predictor, later)
    assert predictor.state_history[-1].timestamp == later


def test_a_clock_that_ran_ahead_does_not_keep_the_present_out(freezer):
    freezer.move_to(NOW)
    predictor = ThermalStatePredictor()
    _record(predictor, NOW + timedelta(days=2))
    _record(predictor, NOW)
    assert [snapshot.timestamp for snapshot in predictor.state_history] == [NOW]


# A reading dated late in its cycle, up to the start of the next one, still belongs to it.
LATE = [timedelta(0), timedelta(seconds=1), CADENCE - timedelta(seconds=1)]


def test_a_reading_belongs_to_the_cycle_it_is_dated_in(freezer):
    freezer.move_to(NOW + 2 * CADENCE)
    predictor = ThermalStatePredictor()
    _record(predictor, NOW, 21.0)
    _record(predictor, NOW + CADENCE - timedelta(seconds=1), 21.5)
    assert [s.indoor_temp for s in predictor.state_history] == [21.5], "one reading per cycle"
    _record(predictor, NOW + CADENCE, 22.0)
    assert [s.indoor_temp for s in predictor.state_history] == [21.5, 22.0]


def _day_with_missed_cycles(missed, late=timedelta(0), which="after"):
    """A day of readings ending now, `missed` consecutive cycles left out halfway. The reading just
    "before" the gap, just "after" it, or the "last" one lands `late` after its cycle."""
    predictor = ThermalStatePredictor()
    gap = range(SAMPLES // 2, SAMPLES // 2 + missed)
    late_at = {"before": gap.start - 1, "after": gap.stop, "last": SAMPLES + missed - 1}[which]
    for i in range(SAMPLES + missed):
        if i in gap:
            continue
        timestamp = NOW - (SAMPLES + missed - i - 1) * CADENCE
        if i == late_at:
            timestamp += late
        _record(predictor, timestamp)
    return predictor


@pytest.mark.parametrize("late", LATE, ids=str)
@pytest.mark.parametrize("which", ["after", "last"])
def test_one_missed_cycle_keeps_the_day_when_a_reading_runs_late(freezer, which, late):
    freezer.move_to(NOW + CADENCE)  # the next cycle decides before it records
    assert _vote(_day_with_missed_cycles(1, late, which)).weight > 0.0


@pytest.mark.parametrize("late", LATE, ids=str)
def test_two_missed_cycles_break_the_history_after_a_late_reading(freezer, late):
    freezer.move_to(NOW + CADENCE)
    predictor = _day_with_missed_cycles(2, late, "before")
    assert predictor.get_outdoor_trend()["samples"] == SAMPLES - SAMPLES // 2, "after the gap only"
    assert _vote(predictor).weight == 0.0


@pytest.mark.parametrize("late", LATE, ids=str)
@pytest.mark.parametrize(("missed", "used"), [(1, True), (2, False)])
def test_two_missed_cycles_at_the_end_leave_the_day_stale(freezer, missed, used, late):
    freezer.move_to(NOW + late)
    predictor = _day_with_missed_cycles(0, late, "last")
    freezer.move_to(NOW + (missed + 1) * CADENCE)  # the next cycle that runs, before it records
    assert (_vote(predictor).weight > 0.0) is used


@pytest.mark.parametrize("late", LATE, ids=str)
def test_two_separate_missed_cycles_in_a_day_do_not_make_a_day(freezer, late):
    freezer.move_to(NOW + CADENCE)
    predictor = ThermalStatePredictor()
    cycles = SAMPLES + 2
    for i in range(cycles):
        if i in (cycles // 3, 2 * cycles // 3):
            continue
        _record(predictor, NOW - (cycles - i - 1) * CADENCE + (late if i == 0 else timedelta(0)))
    assert len(predictor.state_history) >= SAMPLES - 1
    assert _vote(predictor).weight == 0.0


def test_responsiveness_uses_real_elapsed_time_across_the_repeated_hour(freezer):
    end = datetime(2026, 10, 25, 2, 40, tzinfo=timezone.utc)
    freezer.move_to(end)
    predictor = ThermalStatePredictor()
    readings = PREDICTION_RESPONSIVENESS_MIN_HOURS * SAMPLES_PER_HOUR + 1
    rise = 0.01
    for i in range(readings):
        utc = end - (readings - i - 1) * CADENCE
        # An offset only in the 20 minutes before 01:00 UTC, when Stockholm's clocks go back: every
        # pair the estimate can use straddles the change.
        offset = 1.0 if utc.hour == 0 and utc.minute >= 45 else 0.0
        local = utc.astimezone(ZoneInfo("Europe/Stockholm"))
        predictor.record_state(local, 21.0 + i * rise, 25.0, offset, 35.0, -100.0)
    predicted = predictor.predict_temperature(1, [25.0], planned_offsets=[1.0]).predicted_temps
    assert predicted == pytest.approx([21.0 + (readings - 1) * rise + rise * SAMPLES_PER_HOUR])

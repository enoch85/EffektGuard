"""A decision nothing acts on decides as a control cycle would, and the engine keeps nothing of it.

Home Assistant's read-only refresh runs the decision engine too, and the engine keeps state between
decisions: the emergency layer's degree-minute history and anti-windup, the immersion guard's
readings, where the immersion heater starts and where the pump runs short, the forecast's bias, the
last ask and whether the room waited. A read kept all of it as a control cycle does, and outside the
control lock it could run at the same moment as one and record a second degree-minute sample (the
HACS review of v0.9.0). A read now decides on a throwaway copy.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import partial
from types import BuiltinFunctionType, FunctionType, MethodType
from unittest.mock import MagicMock, Mock

import pytest

from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.adapters.weather_adapter import (
    WeatherData,
    WeatherForecastHour,
)
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.prediction_layer import ThermalStatePredictor
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

MIDNIGHT = datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)
NOW = MIDNIGHT + timedelta(hours=12, minutes=5, seconds=10)  # inside the dearest two hours
CYCLE = timedelta(minutes=5)
TARGET = 21.0
OUTDOOR = -2.0

# What a decision records, each a path from the engine.
RECORDS = (
    "emergency_layer._dm_history",
    "emergency_layer._anti_windup_cooldown_until",
    "emergency_layer._last_offset_value",
    "emergency_layer._last_offset_raise_time",
    "_immersion_readings",
    "immersion_start",
    "capacity_shortfall",
    "weather_prediction._forecast_bias",
    "_waiting_for_cheaper",
)


def _prices() -> PriceData:
    """Noon's two hours at 400 ore, the rest of the day 50-68 ore."""
    prices = [50.0 + (q % 7) * 3.0 for q in range(96)]
    for q in range(46, 54):
        prices[q] = 400.0
    return PriceData(
        today=[
            QuarterPeriod(MIDNIGHT + timedelta(minutes=15 * q), p) for q, p in enumerate(prices)
        ],
        tomorrow=[],
        has_tomorrow=False,
    )


def _weather(at: datetime) -> WeatherData:
    """A day warming 3.9 C over the next six hours, as on 5 Nov 2026."""
    hour = at.replace(minute=0, second=0, microsecond=0)
    return WeatherData(
        current_temp=OUTDOOR,
        forecast_hours=[
            WeatherForecastHour(
                datetime=hour + timedelta(hours=n), temperature=OUTDOOR + 3.9 * min(n, 6) / 6
            )
            for n in range(24)
        ],
    )


def _state(at: datetime, indoor: float, degree_minutes: float, register: float) -> NibeState:
    return NibeState(
        outdoor_temp=OUTDOOR,
        indoor_temp=indoor,
        supply_temp=35.0,
        return_temp=30.0,
        degree_minutes=degree_minutes,
        current_offset=register,
        is_heating=True,
        is_hot_water=False,
        timestamp=at,
        compressor_hz=58,
        power_kw=2.5,
        immersion_power_kw=0.0,
    )


def _engine_after_an_hour_of_cycles() -> DecisionEngine:
    """An F750 after a day of history and an hour of control cycles, the room at its target.

    The register stays at or below 0 and the degree minutes fall 20 a cycle, so the read's raise
    to +1 starts the anti-windup; the room is warm only at the read, so only the read waits.
    """
    predictor = ThermalStatePredictor()
    for step in range(24 * 12, 0, -1):
        predictor.record_state(NOW - step * CYCLE, TARGET, OUTDOOR, 0.0, 35.0, -200.0)
    engine = DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(0.7, 1.0),
        config={
            "target_indoor_temp": TARGET,
            "tolerance": 1.5,
            "optimization_mode": "balanced",
            "latitude": 59.33,
            "heating_type": "radiator",
        },
        thermal_predictor=predictor,
        heat_pump_model=NibeF750Profile(),
    )
    for step in range(12, 0, -1):
        at = NOW - step * CYCLE
        register = -1.0 if step > 6 else 0.0
        engine.calculate_decision(
            _state(at, TARGET, -80.0 - 20.0 * (12 - step), register),
            _prices(),
            _weather(at),
            0.0,
            0.0,
        )
    return engine


def _read_inputs() -> tuple:
    """The moment of the read: the room warm, the register raised to +1, DM still falling."""
    return (_state(NOW, TARGET + 0.2, -320.0, 1.0), _prices(), _weather(NOW), 0.0, 0.0)


def _kept(value, seen: set[int] | None = None):
    """Everything reachable from `value`, as plain data: what a decision could leave behind."""
    seen = set() if seen is None else seen
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if value is None or isinstance(value, (bool, int, float, str, datetime, timedelta, Enum)):
        return value
    if isinstance(value, Mock):
        return "mock"
    if isinstance(value, (FunctionType, MethodType, BuiltinFunctionType, partial)):
        return getattr(value, "__qualname__", type(value).__name__)
    if id(value) in seen:
        return "seen"
    seen.add(id(value))
    if isinstance(value, dict):
        return {key: _kept(item, seen) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [type(value).__name__, *sorted(repr(_kept(item, seen)) for item in value)]
    if isinstance(value, (list, tuple, deque)):
        return [type(value).__name__, *(_kept(item, seen) for item in value)]
    fields = getattr(value, "__dict__", None)
    if fields is None:
        return repr(value)
    return [type(value).__name__, {name: _kept(item, seen) for name, item in fields.items()}]


def _at(engine: DecisionEngine, path: str):
    for name in path.split("."):
        engine = getattr(engine, name)
    return engine


@pytest.fixture(autouse=True)
def _noon(freezer):
    freezer.move_to(NOW)


def test_a_read_leaves_the_engine_as_it_was():
    engine = _engine_after_an_hour_of_cycles()
    records = {path: _kept(_at(engine, path)) for path in RECORDS}
    everything = _kept(engine)

    engine.calculate_decision(*_read_inputs(), record=False)

    changed = [path for path in RECORDS if _kept(_at(engine, path)) != records[path]]
    assert not changed, f"a read changed what the engine keeps between decisions: {changed}"
    assert _kept(engine) == everything, "a read changed something else the engine keeps"


def test_a_read_decides_as_the_control_cycle_would_at_that_moment():
    read = _engine_after_an_hour_of_cycles().calculate_decision(*_read_inputs(), record=False)
    cycle = _engine_after_an_hour_of_cycles().calculate_decision(*_read_inputs())

    assert _kept(read) == _kept(cycle)
    assert read.offset == -5.0 and read.anti_windup_active, read.reasoning


@pytest.mark.parametrize("path", RECORDS)
def test_the_control_cycle_still_records(path):
    """Each record the read leaves alone, the control cycle at the same moment does change."""
    read, cycle = _engine_after_an_hour_of_cycles(), _engine_after_an_hour_of_cycles()

    read.calculate_decision(*_read_inputs(), record=False)
    cycle.calculate_decision(*_read_inputs())

    assert _kept(_at(cycle, path)) != _kept(_at(read, path))


def test_a_read_and_a_cycle_at_one_instant_keep_one_degree_minute_sample():
    """The review's case: a read landing on a control cycle recorded the reading twice."""
    engine = _engine_after_an_hour_of_cycles()

    engine.calculate_decision(*_read_inputs(), record=False)
    engine.calculate_decision(*_read_inputs())

    assert [at for at, _ in engine.emergency_layer._dm_history].count(NOW) == 1

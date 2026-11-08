"""A heating-only house must stop adding heat before a forecast warm afternoon."""

from dataclasses import replace
from datetime import timedelta

import pytest

from custom_components.effektguard.const import (
    MIN_OFFSET,
    MIN_TARGET_TEMP,
    MIN_TEMP_LIMIT,
    SAFETY_EMERGENCY_OFFSET,
)
from tests.unit.optimization.test_a_warm_room_waits_for_cheaper_hours import (
    NOW,
    _engine,
    _state,
    _weather,
)


def _inputs():
    engine = _engine(target=22.0)
    state = replace(_state(21.9), outdoor_temp=12.0)
    # The room is one BT50 step above balanced mode's floor, before the afternoon warms.
    weather = _weather([12.0] * 6 + [20.0] * 18)
    return engine, state, weather


def _decide(engine, state, weather):
    return engine.calculate_decision(state, None, weather, 0.0, 0.0)


@pytest.mark.parametrize("mode,room", [("balanced", 21.9), ("comfort", 22.1), ("savings", 21.1)])
def test_a_room_above_its_mode_floor_coasts_without_needing_an_expensive_price(freezer, mode, room):
    freezer.move_to(NOW)
    engine, state, weather = _inputs()
    engine.config["optimization_mode"] = mode
    engine.update_mode_config()
    state = replace(state, indoor_temp=room)
    assert engine.predictor is None, "weather coasting does not need learned room history"
    decision = _decide(engine, state, weather)
    assert decision.offset == MIN_OFFSET, decision.reasoning


@pytest.mark.parametrize(
    "case",
    [
        "disabled",
        "concrete_floor_heating",
        "timber_floor_heating",
        "sensor_missing",
        "at_floor",
        "comfort_floor",
        "savings_floor",
        "cool_day",
        "missing",
        "past",
        "gap",
        "short",
        "naive",
    ],
)
def test_coasting_requires_a_measured_warm_room_and_complete_future_weather(freezer, case):
    freezer.move_to(NOW)
    engine, state, weather = _inputs()
    if case == "disabled":
        engine.config["enable_weather_prediction"] = False
    elif case in ("concrete_floor_heating", "timber_floor_heating"):
        kind = "concrete_ufh" if case == "concrete_floor_heating" else "timber_ufh"
        engine = type(engine)(
            engine.price,
            engine.effect,
            engine.thermal,
            engine.config | {"heating_type": kind},
            heat_pump_model=engine.heat_pump_model,
        )
    elif case == "sensor_missing":
        state = replace(state, indoor_temp_valid=False)
    elif case in ("at_floor", "comfort_floor", "savings_floor"):
        mode, room = {
            "at_floor": ("balanced", 21.8),
            "comfort_floor": ("comfort", 22.0),
            "savings_floor": ("savings", 21.0),
        }[case]
        engine.config["optimization_mode"] = mode
        engine.update_mode_config()
        state = replace(state, indoor_temp=room)
    elif case == "cool_day":
        weather = _weather([12.0] * 24)
    elif case == "missing":
        weather = None
    elif case == "past":
        weather.forecast_hours = [
            replace(h, datetime=h.datetime - timedelta(days=2)) for h in weather.forecast_hours
        ]
    elif case == "gap":
        del weather.forecast_hours[8]
    elif case == "short":
        weather.forecast_hours = weather.forecast_hours[:12]
    elif case == "naive":
        weather.forecast_hours = [
            replace(h, datetime=h.datetime.replace(tzinfo=None)) for h in weather.forecast_hours
        ]
    assert _decide(engine, state, weather).offset > MIN_OFFSET


@pytest.mark.parametrize(
    "case", ["manual", "cold_room", "safety_at_savings_floor", "capacity_shortfall", "emergency"]
)
def test_warm_weather_does_not_override_existing_control_guards(freezer, case):
    freezer.move_to(NOW)
    engine, state, weather = _inputs()
    if case == "manual":
        engine.set_manual_override(1.0)
    elif case == "cold_room":
        state = replace(state, indoor_temp=17.0)
    elif case == "safety_at_savings_floor":
        engine.target_temp = MIN_TARGET_TEMP
        engine.config["optimization_mode"] = "savings"
        engine.update_mode_config()
        state = replace(state, indoor_temp=MIN_TEMP_LIMIT - 0.1)
    elif case == "capacity_shortfall":
        engine.capacity_shortfall.is_short = lambda *_: True
        engine.capacity_shortfall._seen.append((NOW, 12.0))
    elif case == "emergency":
        state = replace(state, degree_minutes=-1600.0)
    decision = _decide(engine, state, weather)
    if case == "manual":
        assert decision.offset == 1.0
    elif case in ("cold_room", "safety_at_savings_floor"):
        assert decision.offset == SAFETY_EMERGENCY_OFFSET
    elif case == "emergency":
        control, _, _ = _inputs()
        control.config["enable_weather_prediction"] = False
        assert decision.offset == _decide(control, state, weather).offset
    else:
        assert decision.offset >= 0.0


@pytest.mark.parametrize("tier", ["T1", "T2", "T3", "EMERGENCY"])
def test_a_recovery_proposal_cannot_be_replaced_by_warm_weather(freezer, monkeypatch, tier):
    from custom_components.effektguard.optimization.thermal_layer import EmergencyLayerDecision

    freezer.move_to(NOW)
    engine, state, weather = _inputs()
    recovery = EmergencyLayerDecision("Thermal Debt", 1.5, 0.9, "Recovery", tier=tier)
    monkeypatch.setattr(engine.emergency_layer, "evaluate_layer", lambda **_: recovery)
    offset = _decide(engine, state, weather).offset
    if tier == "EMERGENCY":
        assert offset >= 0.0
    else:
        assert offset > 0.0

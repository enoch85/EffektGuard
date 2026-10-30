"""The pump's immersion heater runs only for a room that needs the heat.

While a measured room is at or above its mode's floor, the engine does not drive degree minutes to
where the pump starts its immersion heater (the model profile's aux_start_dm): near it, DM may
fall only toward a hold point short of it, and the cap follows the supply both ways in whole
steps. Below the floor the heat is the room's. Closed loop on the owner's F750 + slab: most of the
immersion heat it bought ran with the room at or above its target.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

START = datetime(2026, 1, 15, 2, 0, tzinfo=timezone.utc)
CYCLE = timedelta(minutes=UPDATE_INTERVAL_MINUTES)
TARGET = 21.0
FLOOR = TARGET - MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED].room_floor_below_target
ASKED = 2.0  # what the layers ask for: the guard follows them


def _engine(profile=NibeF750Profile) -> DecisionEngine:
    engine = DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(0.7, 1.0),
        config={
            "target_indoor_temp": TARGET,
            "tolerance": 0.5,
            "optimization_mode": OPTIMIZATION_MODE_BALANCED,
            "latitude": 59.33,
            "heating_type": "radiator",
        },
        heat_pump_model=profile() if profile else None,
    )
    engine._aggregate_layers = lambda layers, starvation=0.0: ASKED
    return engine


def _run(engine, degree_minutes, register: float = 0.0, **state):
    """The engine's decisions on readings five minutes apart, the register held where it is."""
    decisions = []
    for i, dm in enumerate(degree_minutes):
        decisions.append(
            engine.calculate_decision(
                nibe_state=NibeState(
                    outdoor_temp=-5.0,
                    indoor_temp=state.get("indoor_temp", FLOOR + 0.4),
                    supply_temp=35.0,
                    return_temp=30.0,
                    degree_minutes=dm,
                    current_offset=register,
                    is_heating=True,
                    is_hot_water=state.get("is_hot_water", False),
                    timestamp=START + CYCLE * i,
                    indoor_temp_valid=state.get("indoor_temp_valid", True),
                ),
                price_data=None,
                weather_data=None,
                current_peak=0.0,
                current_power=3.0,
            )
        )
    return decisions


FALLING = [-250.0 - 50.0 * i for i in range(5)]  # 600 DM an hour, toward the F750's -700


def test_a_fall_toward_the_heater_is_braked_for_a_room_above_its_floor():
    """The register holds until the readings span a quarter of an hour, then the fall is cut."""
    decisions = _run(_engine(), FALLING)

    assert [d.offset for d in decisions[:3]] == [0.0, 0.0, 0.0]
    assert decisions[-1].offset <= -1.0
    assert all(d.heater_capped for d in decisions)


def test_as_degree_minutes_recover_the_offset_comes_back_up():
    """A cut that overshot: DM rising 150 an hour lets the register back up a whole step."""
    rising = [-600.0 + 12.5 * i for i in range(5)]

    decision = _run(_engine(), rising, register=-3.0)[-1]

    assert decision.offset == -2.0


@pytest.mark.parametrize(
    ("engine", "state"),
    [
        pytest.param(_engine, {"indoor_temp": FLOOR - 0.1}, id="the-room-below-its-floor"),
        pytest.param(_engine, {"indoor_temp_valid": False}, id="no-room-sensor"),
        pytest.param(_engine, {"is_hot_water": True}, id="hot-water"),
        pytest.param(lambda: _engine(profile=None), {}, id="no-model-profile"),
    ],
)
def test_otherwise_the_layers_ask_stands(engine, state):
    decision = _run(engine(), FALLING, **state)[-1]

    assert decision.offset == ASKED
    assert not decision.heater_capped

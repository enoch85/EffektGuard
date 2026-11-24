"""A pump that has held its degree minutes at its immersion start still gets the layers' cut.

EffektGuard used to read three hours at the immersion start, at or above the curve, as a pump short
of capacity, and then cut nothing below the curve while it stayed as cold. On the owner's F750 that
cost more than it saved, so it was removed: the cut stands, and what an older version stored about
it is left behind on the next save.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from pytest_homeassistant.common import MockConfigEntry

from custom_components.effektguard import _create_coordinator
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    DOMAIN,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

START = datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)
CYCLE = timedelta(minutes=UPDATE_INTERVAL_MINUTES)
TARGET = 21.0
# Below the balanced floor: the immersion guard leaves the layers' ask alone.
ROOM = TARGET - MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED].room_floor_below_target - 0.1
AT_START = NibeF750Profile().immersion_start_dm + 30.0  # where an immersion heater holds DM
QUIET = -300.0  # DM well clear of the immersion heater
OUTDOOR = -12.0
CUT = -2.0
RAISE = 1.0


def _engine() -> DecisionEngine:
    return DecisionEngine(
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
        heat_pump_model=NibeF750Profile(),
    )


def test_the_cut_stands_after_hours_at_the_immersion_start():
    engine = _engine()
    ask = RAISE
    engine._aggregate_layers = lambda layers, starvation=0.0: ask
    now = START
    decision = None
    # Six quiet hours, then four at the immersion start at the curve, then the layers ask for a cut.
    readings = [(QUIET, RAISE)] * 72 + [(AT_START, RAISE)] * 48 + [(QUIET, CUT)]
    for degree_minutes, ask in readings:
        decision = engine.calculate_decision(
            nibe_state=NibeState(
                outdoor_temp=OUTDOOR,
                indoor_temp=ROOM,
                supply_temp=35.0,
                return_temp=30.0,
                degree_minutes=degree_minutes,
                current_offset=0.0,
                is_heating=True,
                is_hot_water=False,
                timestamp=now,
            ),
            price_data=None,
            weather_data=None,
            current_peak=0.0,
            current_power=3.0,
        )
        now += CYCLE

    assert decision.offset == CUT


async def test_a_stored_shortfall_from_an_older_version_is_left_behind(hass, hass_storage):
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    before = await _create_coordinator(hass, entry)
    before.engine.immersion_start.restore({"starts": [[START.isoformat(), -1456.0]]})
    await before._save_learned_data(
        before.adaptive_learning, before.thermal_predictor, before.weather_learner
    )
    saved = next(s["data"] for s in hass_storage.values() if "immersion_start" in s.get("data", {}))
    saved["capacity_shortfall"] = {"seen": [[START.isoformat(), 0.2]]}  # v0.9.1 stored this

    after = await _create_coordinator(hass, entry)  # restart on this version
    await after._save_learned_data(
        after.adaptive_learning, after.thermal_predictor, after.weather_learner
    )

    assert after.engine.immersion_start_in_force() == -1456.0
    saved = next(s["data"] for s in hass_storage.values() if "immersion_start" in s.get("data", {}))
    assert "capacity_shortfall" not in saved

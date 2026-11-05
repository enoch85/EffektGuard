"""The heater guard keeps off the heater start the pump has shown, not the one its model assumes.

A NIBE pump starts its immersion heater where its degree minutes cross its own setting, and an
installer can move that far from the model profile's factory value: the owner's F750 starts at
-1450, its profile says -700. The pump publishes the heater's power. Read while the compressor heats
the house, a reading with the heater off followed by one with it on puts the start at the second
reading's degree minutes; two with it off put it deeper than both. Hot water runs the same heater at
any degree minutes, so it teaches nothing, and nothing moves the start shallower than the factory
value.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import _create_coordinator
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    DM_RATE_WINDOW_MINUTES,
    DOMAIN,
    HEATER_GUARD_ENGAGE_DM,
    HEATER_START_MAX_GAP_MINUTES,
    HEATER_START_MEMORY,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    SHORTFALL_AFTER_CUT_HOURS,
    SHORTFALL_DWELL_MINUTES,
    SHORTFALL_NEAR_HEATER_START_DM,
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
ROOM = TARGET - MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED].room_floor_below_target + 0.4
ASKED = 2.0  # what the layers ask for: the guard caps it or lets it stand
CUT = -2.0
FACTORY = NibeF750Profile().aux_start_dm
OWN_START = -1450.0  # the owner's F750: start compressor -250, start diff additional heat 1200
HEATER_KW = 0.5  # the owner's F750's first heater step
DEEPER_THAN_FACTORY = FACTORY - 300.0
BLOCKED = -2000.0  # a heater that may not start (NIBE's "stop additional heat") stays off this deep


def _fall(start: float) -> list[float]:
    """Degree minutes falling 600 an hour from `start`, five readings."""
    return [start - 50.0 * i for i in range(5)]


# Past where the guard engages for the factory start, short of where it engages for the owner's
PAST_FACTORY = _fall(FACTORY + 300.0)
NEAR_OWN = _fall(OWN_START + HEATER_GUARD_ENGAGE_DM)
# 300 DM an hour, all of it short of where the guard engages for the factory start
SHALLOW = [FACTORY + HEATER_GUARD_ENGAGE_DM + 150.0 - 25.0 * i for i in range(5)]


class _Pump:
    """An engine fed readings five minutes apart, the layers' ask fixed."""

    def __init__(self):
        self.ask = ASKED
        self.engine = DecisionEngine(
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
        self.engine._aggregate_layers = lambda layers, starvation=0.0: self.ask
        self.now = START

    def read(self, degree_minutes, heater_kw=0.0, *, step=CYCLE, **state):
        decision = self.engine.calculate_decision(
            nibe_state=NibeState(
                outdoor_temp=-5.0,
                indoor_temp=ROOM,
                supply_temp=35.0,
                return_temp=30.0,
                degree_minutes=degree_minutes,
                current_offset=0.0,
                is_heating=state.get("is_heating", True),
                is_hot_water=state.get("is_hot_water", False),
                timestamp=self.now,
                heater_power_kw=heater_kw,
            ),
            price_data=None,
            weather_data=None,
            current_peak=0.0,
            current_power=3.0,
        )
        self.now += step
        return decision

    def heater_switches_on(self, at_dm, heater_kw=HEATER_KW, **state):
        """A reading with the heater off just short of `at_dm`, the next with it on at `at_dm`."""
        self.read(at_dm + 10.0, 0.0, **state)
        self.read(at_dm, heater_kw, **state)

    def falls(self, readings=PAST_FACTORY) -> bool:
        """Whether the guard caps a fall at 600 DM an hour, the heater off, on its last reading."""
        self.now += timedelta(minutes=DM_RATE_WINDOW_MINUTES) * 2  # the guard's readings forgotten
        return [self.read(dm) for dm in readings][-1].heater_capped


@pytest.mark.parametrize(
    ("readings", "shown"),
    [
        pytest.param([(OWN_START, 0.0)], False, id="nothing-shown"),
        pytest.param([(OWN_START + 10.0, 0.0), (OWN_START, HEATER_KW)], True, id="a-start"),
        pytest.param([(OWN_START + 10.0, 0.0), (OWN_START, 0.0)], True, id="held-off"),
    ],
)
def test_what_the_pump_shows_past_the_factory_value_moves_the_guard_there(readings, shown):
    """Shown its heater starting at -1450, or staying off there, the pump may fall past the
    factory -700 for the room's heat, and the guard holds it off -1450 instead."""
    pump = _Pump()
    for degree_minutes, heater_kw in readings:
        pump.read(degree_minutes, heater_kw)

    assert pump.falls() is not shown
    assert pump.falls(NEAR_OWN)


@pytest.mark.parametrize(
    "state",
    [
        pytest.param({"is_hot_water": True}, id="hot-water"),
        pytest.param({"is_hot_water": None}, id="priority-unknown"),
        pytest.param({"is_heating": False}, id="compressor-not-heating"),
    ],
)
def test_a_heater_run_for_anything_but_space_heating_teaches_nothing(state):
    pump = _Pump()
    pump.heater_switches_on(OWN_START, **state)

    assert pump.falls()


@pytest.mark.parametrize(
    ("before", "step"),
    [
        pytest.param(
            0.0, timedelta(minutes=HEATER_START_MAX_GAP_MINUTES) + CYCLE, id="too-far-apart"
        ),
        pytest.param(HEATER_KW, CYCLE, id="already-running"),
    ],
)
def test_readings_that_do_not_show_the_heater_switch_on_teach_nothing(before, step):
    pump = _Pump()
    pump.read(OWN_START + 10.0, before, step=step)
    pump.read(OWN_START, HEATER_KW)

    assert pump.falls()


def test_one_reading_deep_with_the_heater_off_is_not_a_depth_held():
    """A single reading far past the start between two shallow ones - a glitch, or a reset."""
    pump = _Pump()
    for degree_minutes in (FACTORY + 300.0, BLOCKED, FACTORY + 300.0):
        pump.read(degree_minutes)

    assert pump.falls(NEAR_OWN)


def test_a_start_outweighs_the_depth_held_with_the_heater_off():
    """A heater that may not start stays off far past its start; once it switches on, where it does
    is the start."""
    pump = _Pump()
    pump.read(BLOCKED + 10.0)
    pump.read(BLOCKED)
    assert not pump.falls(NEAR_OWN)

    pump.heater_switches_on(OWN_START)
    assert pump.falls(NEAR_OWN)


@pytest.mark.parametrize(
    "heater_kw",
    [pytest.param(HEATER_KW, id="a-start"), pytest.param(0.0, id="held-off")],
)
def test_nothing_shallower_than_the_factory_value_moves_the_start(heater_kw):
    """The heater is off at shallow degree minutes nearly all the time, and one switched on there
    for anything else says nothing of its start: the factory value stays the shallowest."""
    pump = _Pump()
    pump.read(SHALLOW[0] - 10.0)
    pump.read(SHALLOW[0] - 20.0, heater_kw)

    assert not pump.falls(SHALLOW)


def test_the_start_is_the_shallowest_of_the_last_starts_seen():
    """A start seen shallower is followed at once; one deeper, once the shallower ones are gone."""
    pump = _Pump()
    pump.heater_switches_on(DEEPER_THAN_FACTORY)
    for _ in range(HEATER_START_MEMORY - 1):
        pump.heater_switches_on(OWN_START)
    assert pump.falls()

    pump.heater_switches_on(OWN_START)
    assert not pump.falls()

    pump.heater_switches_on(DEEPER_THAN_FACTORY)
    assert pump.falls()


def test_a_pump_held_off_its_heater_is_short_of_capacity_only_once_the_heater_has_started():
    """Short of capacity, a pump's heater holds its degree minutes at its start. Held off past the
    factory value, the pump has shown only that its start is deeper: three hours just short of where
    it held off, without the heater, are the compressor's, not a shortfall. Once it has switched the
    heater on, the same three hours at its start are."""
    pump = _Pump()
    pump.read(OWN_START + 10.0)
    pump.read(OWN_START)  # held off at -1440
    near_start = OWN_START + SHORTFALL_NEAR_HEATER_START_DM - 20.0

    def short_after_three_hours_near_the_start() -> bool:
        for _ in range(int(timedelta(hours=SHORTFALL_AFTER_CUT_HOURS) / CYCLE)):
            pump.read(OWN_START + HEATER_GUARD_ENGAGE_DM * 2)  # quiet hours, well clear of it
        for _ in range(int(timedelta(minutes=SHORTFALL_DWELL_MINUTES) / CYCLE) + 1):
            pump.read(near_start)
        pump.ask = CUT
        short = pump.read(near_start).offset == 0.0
        pump.ask = ASKED
        return short

    assert not short_after_three_hours_near_the_start()

    pump.heater_switches_on(OWN_START)
    assert short_after_three_hours_near_the_start()


@pytest.mark.parametrize(
    ("shown", "in_force"),
    [
        pytest.param({"starts": [[START.isoformat(), OWN_START]]}, OWN_START, id="a-start"),
        pytest.param({"held_off": BLOCKED}, BLOCKED, id="held-off"),
    ],
)
async def test_the_coordinator_stores_it_across_a_restart(hass, hass_storage, shown, in_force):
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    before = await _create_coordinator(hass, entry)
    before.engine.heater_start.restore(shown)
    await before._save_learned_data(
        before.adaptive_learning, before.thermal_predictor, before.weather_learner
    )

    after = await _create_coordinator(hass, entry)  # restart

    assert after.engine.heater_start_dm() == in_force

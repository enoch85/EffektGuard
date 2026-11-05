"""A pump short of capacity keeps its own curve: no cut below it, and no heater guard.

A NIBE pump that cannot keep up with its curve holds its degree minutes at its immersion heater's
start. Once the engine has seen that for SHORTFALL_DWELL_MINUTES - at or above the curve, outside
hot water, at least SHORTFALL_AFTER_CUT_HOURS after its own last cut and after it started - it cuts
nothing below the curve while the coming hours are as cold: heat held back then comes back only
from the heater, or not at all. Repository simulator, undersized F2040 in the January 2024 cold
snap: 1450 -> 890 minutes below the comfort band.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import _create_coordinator
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.adapters.weather_adapter import WeatherData, WeatherForecastHour
from custom_components.effektguard.const import (
    DOMAIN,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    SHORTFALL_AFTER_CUT_HOURS,
    SHORTFALL_DWELL_MINUTES,
    SHORTFALL_FORECAST_HOURS,
    SHORTFALL_MEMORY_DAYS,
    SHORTFALL_NEAR_HEATER_START_DM,
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
# Below the balanced floor: the heater guard leaves the layers' ask alone, so what is tested is the
# shortfall alone.
ROOM = TARGET - MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED].room_floor_below_target - 0.1
HEATER_START = NibeF750Profile().aux_start_dm
AT_START = HEATER_START + SHORTFALL_NEAR_HEATER_START_DM  # the heater holds DM here
QUIET = -300.0  # DM well clear of the heater
OUTDOOR = -12.0
CUT = -2.0
RAISE = 1.0

QUIET_CYCLES = int(timedelta(hours=SHORTFALL_AFTER_CUT_HOURS) / CYCLE)
DWELL_CYCLES = int(timedelta(minutes=SHORTFALL_DWELL_MINUTES) / CYCLE)


class _Pump:
    """An engine driven cycle by cycle, the layers' vote set by the test."""

    def __init__(self):
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
        self.ask = RAISE
        self.engine._aggregate_layers = lambda layers, starvation=0.0: self.ask
        self.now = START

    def run(self, cycles, ask, degree_minutes, *, outdoor=OUTDOOR, forecast=None, **state):
        """`cycles` readings five minutes apart; the decision on the last one."""
        self.ask = ask
        decision = None
        for _ in range(cycles):
            weather = None
            if forecast is not None:
                weather = WeatherData(
                    current_temp=outdoor,
                    forecast_hours=[
                        WeatherForecastHour(
                            datetime=self.now + timedelta(hours=h + 1), temperature=t
                        )
                        for h, t in enumerate(forecast)
                    ],
                    source_entity="test",
                )
            decision = self.engine.calculate_decision(
                nibe_state=NibeState(
                    outdoor_temp=outdoor,
                    indoor_temp=ROOM,
                    supply_temp=35.0,
                    return_temp=30.0,
                    degree_minutes=degree_minutes,
                    current_offset=state.get("register", 0.0),
                    is_heating=True,
                    is_hot_water=state.get("is_hot_water", False),
                    timestamp=self.now,
                ),
                price_data=None,
                weather_data=weather,
                current_peak=0.0,
                current_power=3.0,
            )
            self.now += CYCLE
        return decision

    def seen_short(self):
        """Quiet hours, then the pump held at its heater start for the dwell."""
        self.run(QUIET_CYCLES, RAISE, QUIET)
        self.run(DWELL_CYCLES + 1, RAISE, AT_START)


def test_a_pump_held_at_its_heater_start_gets_no_cut_while_it_stays_as_cold():
    pump = _Pump()
    assert pump.run(1, CUT, QUIET).offset == CUT  # nothing seen yet: the cut stands

    pump.seen_short()
    decision = pump.run(1, CUT, QUIET)

    assert pump.engine.capacity_shortfall.temperature == OUTDOOR
    assert decision.offset == 0.0
    assert "short of capacity" in decision.reasoning
    assert pump.run(1, RAISE, QUIET).offset == RAISE  # a raise is the layers' own


@pytest.mark.parametrize(
    ("before", "stretch"),
    [
        pytest.param(
            lambda pump: (
                pump.run(QUIET_CYCLES, RAISE, QUIET),
                pump.run(1, CUT, QUIET),
                pump.run(QUIET_CYCLES - 12, RAISE, QUIET),
            ),
            {},
            id="five-hours-after-our-own-cut",
        ),
        pytest.param(lambda pump: None, {}, id="right-after-start-up"),
        pytest.param(
            lambda pump: pump.run(QUIET_CYCLES, RAISE, QUIET),
            {"is_hot_water": True},
            id="hot-water",
        ),
        pytest.param(
            lambda pump: pump.run(QUIET_CYCLES, RAISE, QUIET),
            {"register": -1.0},
            id="below-the-curve",
        ),
        pytest.param(
            lambda pump: pump.run(QUIET_CYCLES, RAISE, QUIET),
            {"degree_minutes": AT_START + 10.0},
            id="short-of-the-heater-start",
        ),
        pytest.param(
            lambda pump: pump.run(QUIET_CYCLES, RAISE, QUIET),
            {"cycles": DWELL_CYCLES - 1},
            id="not-for-long-enough",
        ),
    ],
)
def test_what_is_not_a_shortfall(before, stretch):
    stretch = dict(stretch)
    pump = _Pump()
    before(pump)
    pump.run(
        stretch.pop("cycles", DWELL_CYCLES + 1),
        RAISE,
        stretch.pop("degree_minutes", AT_START),
        **stretch,
    )

    assert pump.engine.capacity_shortfall.temperature is None
    assert pump.run(1, CUT, QUIET).offset == CUT


@pytest.mark.parametrize(
    ("outdoor", "forecast", "kept"),
    [
        pytest.param(OUTDOOR + 3.0, [OUTDOOR + 3.0] * SHORTFALL_FORECAST_HOURS, False, id="warmer"),
        pytest.param(
            OUTDOOR + 3.0,
            [OUTDOOR + 3.0] * (SHORTFALL_FORECAST_HOURS - 1) + [OUTDOOR - 1.0],
            True,
            id="as-cold-within-the-forecast",
        ),
        pytest.param(
            OUTDOOR + 3.0,
            [OUTDOOR + 3.0] * SHORTFALL_FORECAST_HOURS + [OUTDOOR - 1.0],
            False,
            id="as-cold-only-beyond-it",
        ),
    ],
)
def test_the_coming_hours_decide(outdoor, forecast, kept):
    pump = _Pump()
    pump.seen_short()

    decision = pump.run(1, CUT, QUIET, outdoor=outdoor, forecast=forecast)

    assert decision.offset == (0.0 if kept else CUT)


def test_a_manual_override_below_the_curve_is_a_cut_of_ours():
    """Five hours after the owner forced a cut, the pump at its heater start is still recovering."""
    pump = _Pump()
    pump.run(QUIET_CYCLES, RAISE, QUIET)
    pump.engine.set_manual_override(CUT)
    pump.run(1, RAISE, QUIET, register=CUT)
    pump.engine.clear_manual_override()
    pump.run(QUIET_CYCLES - 12, RAISE, QUIET)
    pump.run(DWELL_CYCLES + 1, RAISE, AT_START)

    assert pump.engine.capacity_shortfall.temperature is None


def test_a_shortfall_is_forgotten_after_the_memory():
    pump = _Pump()
    pump.seen_short()
    pump.now += timedelta(days=SHORTFALL_MEMORY_DAYS, hours=1)

    assert pump.run(1, CUT, QUIET).offset == CUT


def test_a_shortfall_survives_a_restart():
    pump = _Pump()
    pump.seen_short()
    restarted = _Pump()
    restarted.now = pump.now
    restarted.engine.capacity_shortfall.restore(pump.engine.capacity_shortfall.to_dict())

    assert restarted.run(1, CUT, QUIET).offset == 0.0


async def test_the_coordinator_stores_it_across_a_restart(hass, hass_storage):
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    before = await _create_coordinator(hass, entry)
    before.engine.capacity_shortfall.restore({"seen": [[START.isoformat(), OUTDOOR]]})
    await before._save_learned_data(
        before.adaptive_learning, before.thermal_predictor, before.weather_learner
    )

    after = await _create_coordinator(hass, entry)  # restart

    assert after.engine.capacity_shortfall.temperature == OUTDOOR


def test_the_heater_guard_stands_down_while_the_pump_is_short():
    """Above its floor the room would have the guard cap a fall toward the heater; short, it may not."""
    pump = _Pump()
    pump.engine.capacity_shortfall.restore(
        {"seen": [[(START - timedelta(days=1)).isoformat(), OUTDOOR]]}
    )
    falling = [-250.0 - 50.0 * i for i in range(5)]  # 600 DM an hour, toward the F750's -700

    pump.ask = RAISE
    decisions = []
    for dm in falling:
        decisions.append(
            pump.engine.calculate_decision(
                nibe_state=NibeState(
                    outdoor_temp=OUTDOOR,
                    indoor_temp=TARGET + 0.4,
                    supply_temp=35.0,
                    return_temp=30.0,
                    degree_minutes=dm,
                    current_offset=0.0,
                    is_heating=True,
                    is_hot_water=False,
                    timestamp=pump.now,
                ),
                price_data=None,
                weather_data=None,
                current_peak=0.0,
                current_power=3.0,
            )
        )
        pump.now += CYCLE

    assert [d.offset for d in decisions] == [RAISE] * len(falling)
    assert not any(d.heater_capped for d in decisions)

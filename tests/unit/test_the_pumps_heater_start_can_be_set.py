"""The owner can tell EffektGuard where the pump starts its electric heater.

The heater guard and the capacity shortfall both work from the pump's heater start. The model
profile only knows the factory value (F750 -700), and an installer can move it: the owner's F750
starts its heater at -1450. Told the true start, the closed loop on the owner's F750 + slab kept
savings mode's coldest moment at 19.6 C instead of 18.7 C (four plant fits, Oct 18 - Apr 30).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
import voluptuous as vol
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    CONF_HEAT_PUMP_MODEL,
    CONF_HEATER_START_DM,
    DOMAIN,
    HEATER_START_DM_MIN,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    SHORTFALL_AFTER_CUT_HOURS,
    SHORTFALL_DWELL_MINUTES,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel
from custom_components.effektguard.options import EffektGuardOptionsFlow

START = datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)
CYCLE = timedelta(minutes=UPDATE_INTERVAL_MINUTES)
TARGET = 21.0
FLOOR = TARGET - MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED].room_floor_below_target
OWNERS_START = -1450.0  # start compressor -250, start diff additional heat 1200
ASKED = 2.0


def _engine(**config) -> DecisionEngine:
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
            **config,
        },
        heat_pump_model=NibeF750Profile(),
    )
    engine._aggregate_layers = lambda layers, starvation=0.0: ASKED
    return engine


def _run(engine, degree_minutes, indoor_temp):
    decisions, now = [], START
    for dm in degree_minutes:
        decisions.append(
            engine.calculate_decision(
                nibe_state=NibeState(
                    outdoor_temp=-12.0,
                    indoor_temp=indoor_temp,
                    supply_temp=35.0,
                    return_temp=30.0,
                    degree_minutes=dm,
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
        )
        now += CYCLE
    return decisions


FALLING = [-250.0 - 50.0 * i for i in range(5)]  # toward the F750's factory -700


@pytest.mark.parametrize(
    ("config", "capped"),
    [
        pytest.param({}, True, id="factory-start"),
        pytest.param({CONF_HEATER_START_DM: OWNERS_START}, False, id="the-owners-start"),
    ],
)
def test_the_heater_guard_works_from_the_start_that_is_set(config, capped):
    """A fall to -450 is near the factory -700, and nowhere near the owner's -1450."""
    decision = _run(_engine(**config), FALLING, indoor_temp=FLOOR + 0.4)[-1]

    assert decision.heater_capped is capped
    assert (decision.offset < ASKED) is capped


@pytest.mark.parametrize(
    ("config", "short"),
    [
        pytest.param({}, True, id="factory-start"),
        pytest.param({CONF_HEATER_START_DM: OWNERS_START}, False, id="the-owners-start"),
    ],
)
def test_the_capacity_shortfall_works_from_the_start_that_is_set(config, short):
    """DM held at -900 is the factory F750 at its heater; the owner's pump has 550 DM to go."""
    engine = _engine(**config)
    quiet = int(timedelta(hours=SHORTFALL_AFTER_CUT_HOURS) / CYCLE)
    dwell = int(timedelta(minutes=SHORTFALL_DWELL_MINUTES) / CYCLE)
    _run(engine, [-300.0] * quiet + [-900.0] * (dwell + 1), indoor_temp=FLOOR - 0.1)

    assert (engine.capacity_shortfall.temperature is not None) is short


def _form_default(schema, section_name, key):
    section = schema.schema[next(k for k in schema.schema if str(k) == section_name)]
    marker = next(k for k in section.schema.schema if str(k) == key)
    return marker.default()


async def test_the_form_offers_the_models_factory_start_then_the_owners(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_HEAT_PUMP_MODEL: "nibe_f750"}, options={})
    entry.add_to_hass(hass)
    flow = EffektGuardOptionsFlow()
    flow.hass = hass
    flow.handler = entry.entry_id

    form = await flow.async_step_init()
    assert _form_default(form["data_schema"], "heat_pump", CONF_HEATER_START_DM) == (
        NibeF750Profile().aux_start_dm
    )

    hass.config_entries.async_update_entry(entry, options={CONF_HEATER_START_DM: OWNERS_START})
    form = await flow.async_step_init()
    assert _form_default(form["data_schema"], "heat_pump", CONF_HEATER_START_DM) == OWNERS_START


def test_the_form_keeps_the_value_and_refuses_one_the_pump_cannot_have():
    flow = EffektGuardOptionsFlow()

    kept = flow._validate_and_convert_dhw_config({"heat_pump": {CONF_HEATER_START_DM: -1450}})
    assert kept[CONF_HEATER_START_DM] == OWNERS_START

    with pytest.raises(vol.Invalid):
        flow._validate_and_convert_dhw_config(
            {"heat_pump": {CONF_HEATER_START_DM: HEATER_START_DM_MIN - 1}}
        )


async def test_a_changed_start_reaches_the_running_engine():
    from custom_components.effektguard.coordinator import EffektGuardCoordinator

    coordinator = MagicMock()
    coordinator.engine = _engine()
    coordinator.airflow_optimizer = None

    await EffektGuardCoordinator.async_update_config(
        coordinator, {CONF_HEATER_START_DM: OWNERS_START}
    )

    assert coordinator.engine._heater_start() == OWNERS_START

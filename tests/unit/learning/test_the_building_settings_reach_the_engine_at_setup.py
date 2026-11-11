"""The thermal mass and insulation the owner set are what the engine runs with from setup.

Setup built the engine from the options, then swapped in the AdaptiveThermalModel as `engine.thermal`
without them: the price layer's look-ahead and the learned pre-heat read 1.0, or the thermal mass
learning saved last session, after every restart - while the heating type and the weather pre-heat,
derived from the same options at engine construction, ran on the owner's values. Only a hot reload
(saving the options) wrote them onto the model.
"""

from __future__ import annotations

import pytest
from pytest_homeassistant.common import MockConfigEntry

from custom_components.effektguard import _create_coordinator
from custom_components.effektguard.const import (
    CONF_INSULATION_QUALITY,
    CONF_THERMAL_MASS,
    DOMAIN,
    LEARNING_STORAGE_VERSION,
    STORAGE_KEY_LEARNING,
)

CONCRETE_SLAB_MASS = 1.8
POOR_INSULATION = 0.7
LEARNED_LAST_SESSION = 0.5


@pytest.mark.parametrize(
    "stored_thermal_mass",
    [None, LEARNED_LAST_SESSION],
    ids=["first start", "restart after learning saved a thermal mass"],
)
async def test_the_engine_runs_on_the_owner_s_building_settings_from_setup(
    hass, hass_storage, stored_thermal_mass
):
    if stored_thermal_mass is not None:
        hass_storage[STORAGE_KEY_LEARNING] = {
            "version": LEARNING_STORAGE_VERSION,
            "minor_version": 1,
            "key": STORAGE_KEY_LEARNING,
            "data": {
                "version": LEARNING_STORAGE_VERSION,
                "thermal_model": {"thermal_mass": stored_thermal_mass, "ufh_type": "unknown"},
            },
        }
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={},
        options={CONF_THERMAL_MASS: CONCRETE_SLAB_MASS, CONF_INSULATION_QUALITY: POOR_INSULATION},
    )
    entry.add_to_hass(hass)

    engine = (await _create_coordinator(hass, entry)).engine

    assert (engine.thermal.thermal_mass, engine.thermal.insulation_quality) == pytest.approx(
        (CONCRETE_SLAB_MASS, POOR_INSULATION)
    ), (
        f"options say thermal mass {CONCRETE_SLAB_MASS}, insulation {POOR_INSULATION}; the price "
        f"layer and the learned pre-heat run on {engine.thermal.thermal_mass}, "
        f"{engine.thermal.insulation_quality}"
    )


def _derived(engine) -> dict:
    """Everything the engine builds from the thermal mass."""
    return {
        "thermal_mass": engine.thermal.thermal_mass,
        "ladder": engine.emergency_layer.heating_type,
        "dm_history": engine.emergency_layer._dm_history.maxlen,
        "zones": engine.proactive_layer.heating_type,
        "emitters": engine.weather_comp.heating_type,
        "emitter_exponent": engine.weather_comp.emitter_exponent,
        "design_flow": engine.weather_comp.design_flow_temp,
        "pre_heat_mass": engine.weather_prediction.thermal_mass,
        "pre_heat_horizon": engine.weather_prediction.forecast_horizon,
    }


async def test_a_thermal_mass_saved_in_the_options_builds_what_a_restart_builds(hass):
    """Saved in the options, a slab's 1.8 reached the thermal model alone: the ladder kept the
    radiator's buffer and the weather pre-heat its 12-hour horizon, until the next restart built the
    slab's, 48 hours. The options flow's save, as the update listener applies it."""
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    coordinator = await _create_coordinator(hass, entry)

    await coordinator.async_update_config({CONF_THERMAL_MASS: CONCRETE_SLAB_MASS})

    restarted = MockConfigEntry(
        domain=DOMAIN, data={}, options={CONF_THERMAL_MASS: CONCRETE_SLAB_MASS}
    )
    restarted.add_to_hass(hass)
    assert _derived(coordinator.engine) == _derived(
        (await _create_coordinator(hass, restarted)).engine
    )

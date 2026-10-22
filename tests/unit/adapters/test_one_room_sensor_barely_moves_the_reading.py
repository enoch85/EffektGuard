"""One room sensor moving must move the indoor reading by a fraction of its change, not all of it.

The median follows one sensor at a time. The owner's nine sensors (2026-10-20 02:55) sorted 19.8,
20.7, 20.95, 21.44, 22.0, 22, 22, 22.03, 22.66 with BT50 in the middle; BT50 alone fell 22.0 -> 21.7
and the reading fell with it, 0.3 °C, which the comfort layer turned from -5.17 at weight 0.61 into
-2.27 at 0.37 and the pump went -2 -> -1. In 3.6 days of v0.7.0 the median jumped by 0.2 °C or more
31 times. The interquartile mean of the same sensors moves by at most a fifth of one sensor's change.

Through the adapter and Home Assistant's own config-entry lifecycle; the combining functions
themselves are in tests/unit/utils/test_room_temperature.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter
from custom_components.effektguard.config_flow import EffektGuardConfigFlow
from custom_components.effektguard.const import CONF_INDOOR_TEMP_METHOD, CONF_NIBE_ENTITY, DOMAIN

# The owner's nine sensors at 02:50, BT50 first.
OWNER = [22.0, 19.8, 20.7, 20.95, 21.44, 22.0, 22.0, 22.03, 22.66]


def _adapter(rooms: list[float], config: dict) -> NibeAdapter:
    """The owner's pump and its eight extra room sensors, all fresh."""
    readings = {
        "sensor.bt1": "-3.0",
        "sensor.bt2": "35.0",
        "sensor.dm": "-150",
        "sensor.bt50": str(rooms[0]),
        **{f"sensor.room{i}": str(t) for i, t in enumerate(rooms[1:])},
    }

    def get(entity_id):
        if entity_id not in readings:
            return None
        state = MagicMock()
        state.state = readings[entity_id]
        state.attributes = {"unit_of_measurement": "°C"}
        state.last_reported = state.last_updated = dt_util.utcnow()
        return state

    hass = MagicMock()
    hass.states.get.side_effect = get
    adapter = NibeAdapter(hass, {"nibe_entity": "number.offset", **config})
    adapter._entity_cache = {
        "outdoor_temp": "sensor.bt1",
        "supply_temp": "sensor.bt2",
        "indoor_temp": "sensor.bt50",
        "degree_minutes": "sensor.dm",
    }
    adapter._additional_indoor_sensors = [f"sensor.room{i}" for i in range(len(rooms) - 1)]
    return adapter


async def _reading(rooms: list[float], config: dict | None = None) -> float:
    return (await _adapter(rooms, config or {}).get_current_state()).indoor_temp


@pytest.mark.asyncio
async def test_the_owners_bt50_moving_moves_the_reading_a_fifth_as_much():
    moved = await _reading(OWNER) - await _reading([21.7] + OWNER[1:])

    assert 0 < moved <= 0.3 / 5 + 1e-9, (
        f"BT50 fell 0.3 °C and the indoor reading fell {moved:.3f} °C. Five sensors are kept; one of "
        f"them moving should move their mean by at most a fifth of its change."
    )


@pytest.mark.asyncio
async def test_a_chosen_median_is_still_honoured():
    assert await _reading(OWNER, {CONF_INDOOR_TEMP_METHOD: "median"}) == pytest.approx(22.0)


@pytest.mark.parametrize(
    ("stored", "migrated"),
    [("median", "interquartile_mean"), ("average", "average"), (None, None)],
)
async def test_an_entry_from_an_older_version_moves_off_the_median(
    hass, enable_custom_integrations, stored, migrated
):
    """The median was the setup's default, so it is what nearly every entry holds."""
    data = {CONF_NIBE_ENTITY: "number.offset"}
    if stored is not None:
        data[CONF_INDOOR_TEMP_METHOD] = stored
    entry = MockConfigEntry(domain=DOMAIN, version=1, minor_version=1, data=data)
    entry.add_to_hass(hass)

    with (
        patch("custom_components.effektguard.async_setup_entry", AsyncMock(return_value=True)),
        patch("custom_components.effektguard.async_unload_entry", AsyncMock(return_value=True)),
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.config_entries.async_unload(entry.entry_id)

    assert entry.data.get(CONF_INDOOR_TEMP_METHOD) == migrated
    assert entry.data[CONF_NIBE_ENTITY] == "number.offset"


@pytest.mark.asyncio
async def test_reconfigure_offers_and_stores_the_method():
    """Setup was the only place to choose it; a migrated user must be able to choose again."""
    entry = MagicMock()
    entry.data = {CONF_NIBE_ENTITY: "number.offset", CONF_INDOOR_TEMP_METHOD: "interquartile_mean"}
    flow = EffektGuardConfigFlow()
    flow.hass = MagicMock()
    flow._get_reconfigure_entry = MagicMock(return_value=entry)
    flow.async_update_reload_and_abort = MagicMock(return_value={"type": "abort"})

    form = await flow.async_step_reconfigure()
    fields = [k for k in form["data_schema"].schema if k == CONF_INDOOR_TEMP_METHOD]
    assert fields, "Reconfigure does not offer how the room sensors are combined"
    assert fields[0].default() == "interquartile_mean"
    options = form["data_schema"].schema[fields[0]].config["options"]
    assert set(options) == {"interquartile_mean", "median", "average"}

    await flow.async_step_reconfigure({CONF_INDOOR_TEMP_METHOD: "median"})

    stored = flow.async_update_reload_and_abort.call_args.kwargs["data_updates"]
    assert stored[CONF_INDOOR_TEMP_METHOD] == "median"

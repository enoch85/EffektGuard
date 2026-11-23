"""Discovery keeps climate systems 2-8 out because EffektGuard writes climate system 1's offset.
When the offset picked in setup is another circuit's - nibe_heatpump's "Heat Offset S2" or "Heating
offset climate system 2" - that circuit's sensors stay findable, as before, and a warning says to
pick them by hand. The circuit comes from the offset's own registry name, so a renamed offset entity
keeps it.
"""

from unittest.mock import MagicMock

import pytest

from homeassistant.core import HomeAssistant

from conftest import make_mock_async_all
from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter
from custom_components.effektguard.const import CONF_NIBE_ENTITY

TEMP = {"device_class": "temperature", "unit_of_measurement": "°C"}


def make_hass(
    entity_ids: list[str], offset: str, offset_name: str | None, monkeypatch
) -> MagicMock:
    """These temperature sensors, in this order, each a nibe_heatpump entity in the registry; the
    offset is in the registry too when it has a name there."""
    by_id, entries = {}, {}
    if offset_name:
        entry = MagicMock()
        entry.entity_id, entry.platform, entry.original_name = offset, "nibe_heatpump", offset_name
        entry.domain, entry.disabled_by = "number", None
        entries[offset] = entry
    for entity_id in entity_ids:
        state = MagicMock()
        state.entity_id, state.state, state.attributes = entity_id, "1", TEMP
        by_id[entity_id] = state
        entry = MagicMock()
        entry.entity_id, entry.platform, entry.original_name = entity_id, "nibe_heatpump", None
        entry.domain, entry.disabled_by = "sensor", None
        entries[entity_id] = entry
    hass = MagicMock(spec=HomeAssistant)
    hass.states = MagicMock()
    hass.states.async_all = make_mock_async_all(by_id)
    hass.states.get = by_id.get
    registry = MagicMock()
    registry.entities.values.return_value = list(entries.values())
    registry.async_get = entries.get
    monkeypatch.setattr(
        "custom_components.effektguard.adapters.nibe_adapter.er.async_get", lambda h: registry
    )
    return hass


@pytest.mark.parametrize(
    ("offset", "offset_name", "supply", "room"),
    [
        pytest.param(
            "number.heat_offset_s2_47010",
            None,
            "sensor.ep21_bt2_supply_temp_s2_40007",
            "sensor.ep21_bt50_room_temp_s2_40032",
            id="F-series, SMO, VVM",
        ),
        pytest.param(
            "number.upstairs_heating_offset",
            "Heat Offset S2",
            "sensor.ep21_bt2_supply_temp_s2_40007",
            "sensor.ep21_bt50_room_temp_s2_40032",
            id="an offset entity the user renamed",
        ),
        pytest.param(
            "number.heating_offset_climate_system_2_40030",
            None,
            "sensor.supply_line_ep21_bt2_30005",
            "sensor.room_average_temp_clim_system_2_bt50_30116",
            id="S-series",
        ),
    ],
)
async def test_climate_system_2_s_offset_finds_its_sensors(
    offset, offset_name, supply, room, monkeypatch
):
    hass = make_hass([supply, room], offset, offset_name, monkeypatch)
    adapter = NibeAdapter(hass, {CONF_NIBE_ENTITY: offset})
    await adapter.discover_entities()

    assert (adapter.entity_cache.get("supply_temp"), adapter.entity_cache.get("indoor_temp")) == (
        supply,
        room,
    )

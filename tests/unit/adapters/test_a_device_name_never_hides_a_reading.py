"""What keeps another circuit's, unit's or cooling sensor out of discovery is the coil's own name,
and only on nibe_heatpump. A device name is part of every entity id it makes - a MyUplink system
called "Cool House" gives `sensor.cool_house_current_outd_temp_bt1` - so it must not decide; nor may
a nibe_heatpump entity id the user renamed, nor a generic Modbus name the user chose.
"""

from unittest.mock import MagicMock

import pytest

from homeassistant.core import HomeAssistant

from conftest import make_mock_async_all
from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter

TEMP = {"device_class": "temperature", "unit_of_measurement": "°C"}
DM = {"unit_of_measurement": "DM"}


def make_hass(
    states: list[tuple[str, str, dict, str | None]], platform: str | None, monkeypatch
) -> MagicMock:
    """These states, in this order, each in the registry under `platform` with its own (original)
    name; with no platform, none is in the registry (generic Modbus YAML)."""
    by_id, entries = {}, {}
    for entity_id, value, attributes, own_name in states:
        state = MagicMock()
        state.entity_id, state.state, state.attributes = entity_id, value, attributes
        by_id[entity_id] = state
        if platform is None:
            continue
        entry = MagicMock()
        entry.entity_id, entry.platform, entry.original_name = entity_id, platform, own_name
        entry.domain, entry.disabled_by = entity_id.split(".")[0], None
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
    ("platform", "states", "bound"),
    [
        pytest.param(
            "myuplink",
            [
                (
                    "sensor.cool_house_current_outd_temp_bt1",
                    "-3.2",
                    TEMP,
                    "Current outd temp (BT1)",
                ),
                ("sensor.cool_house_supply_line_bt2", "35.8", TEMP, "Supply line (BT2)"),
                ("number.cool_house_degree_minutes", "-150", DM, "Degree minutes"),
            ],
            {
                "outdoor_temp": "sensor.cool_house_current_outd_temp_bt1",
                "supply_temp": "sensor.cool_house_supply_line_bt2",
                "degree_minutes": "number.cool_house_degree_minutes",
            },
            id="a MyUplink system called Cool House",
        ),
        pytest.param(
            "nibe_heatpump",
            [
                ("sensor.cool_house_bt1_outdoor_40004", "-3.2", TEMP, "BT1 Outdoor Temperature"),
                ("sensor.supply_temp_upstairs", "30.1", TEMP, "EP21-BT2 Supply temp S2"),
                ("sensor.bt2_supply_temp_s1_40008", "35.8", TEMP, "BT2 Supply temp S1"),
                ("number.degree_minutes_16_bit_43005", "-150", DM, "Degree Minutes (16 bit)"),
            ],
            {
                "outdoor_temp": "sensor.cool_house_bt1_outdoor_40004",
                "supply_temp": "sensor.bt2_supply_temp_s1_40008",
                "degree_minutes": "number.degree_minutes_16_bit_43005",
            },
            id="nibe_heatpump entity ids the user renamed",
        ),
        pytest.param(
            None,
            [
                ("sensor.cool_house_outdoor_40004", "-3.2", TEMP, None),
                ("sensor.cool_house_supply_40008", "35.8", TEMP, None),
                ("number.cool_house_degree_minutes_43005", "-150", DM, None),
            ],
            {
                "outdoor_temp": "sensor.cool_house_outdoor_40004",
                "supply_temp": "sensor.cool_house_supply_40008",
                "degree_minutes": "number.cool_house_degree_minutes_43005",
            },
            id="a generic Modbus name",
        ),
    ],
)
async def test_the_coil_s_own_name_decides(platform, states, bound, monkeypatch):
    hass = make_hass(states, platform, monkeypatch)
    adapter = NibeAdapter(hass, {})
    await adapter.discover_entities()

    assert {key: adapter.entity_cache.get(key) for key in bound} == bound
    state = await adapter.get_current_state()
    assert (state.outdoor_temp, state.supply_temp, state.degree_minutes) == (-3.2, 35.8, -150)

"""A sensor state of "nan" or "inf" is not a reading.

float() parses both, and a NaN fails every comparison, so it passed each check on its way in. On
v0.9.0 the adapter handed the engine degree minutes and a tank temperature of NaN; one NaN power
sample made its billing period NaN, recorded as a peak, and as the month's first peak or in place of
the day's it made every later period read as safe ("Safe margin: nan kW below peak", a 30 kW one
too); and an offset entity reading "inf" raised OverflowError out of the write path, which must
never raise. Home Assistant refuses a non-finite value for a sensor entity that declares a unit, a
device class or a state class, but not every state reaches it that way (a state set over the REST
API or by another integration directly). Each reader refuses it as it refuses text.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.core import State
from homeassistant.helpers.update_coordinator import UpdateFailed
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter
from custom_components.effektguard.utils.power import power_kw_from_state

NOT_NUMBERS = ["nan", "inf", "-inf"]
DM = "sensor.degree_minutes"
BT7 = "sensor.bt7"


def _adapter(states: dict[str, str]) -> NibeAdapter:
    hass = MagicMock()

    def get(entity_id):
        if entity_id not in states:
            return None
        state = MagicMock()
        state.state = states[entity_id]
        state.attributes = {"unit_of_measurement": "°C"}
        state.last_reported = dt_util.utcnow()
        state.last_updated = state.last_reported
        return state

    hass.states.get.side_effect = get
    adapter = NibeAdapter(hass, {"nibe_entity": "number.offset", "degree_minutes_entity": DM})
    adapter._entity_cache.update(
        outdoor_temp="sensor.bt1", supply_temp="sensor.bt2", dhw_top_temp=BT7
    )
    adapter._discover_nibe_entities = AsyncMock()
    return adapter


def _pump(**overrides: str) -> dict[str, str]:
    return {
        "sensor.bt1": "0",
        "sensor.bt2": "35",
        DM: "-200",
        BT7: "48",
        "number.offset": "0",
        **overrides,
    }


@pytest.mark.parametrize("value", NOT_NUMBERS)
async def test_the_adapter_reads_no_number_where_the_sensor_says_none(value):
    """Degree minutes are required, so the cycle is refused; the tank is optional, so it is unread."""
    with pytest.raises(UpdateFailed, match="degree minutes"):
        await _adapter(_pump(**{DM: value})).get_current_state()

    state = await _adapter(_pump(**{BT7: value})).get_current_state()
    assert state.dhw_top_temp is None, state.dhw_top_temp


@pytest.mark.parametrize("value", NOT_NUMBERS)
def test_a_power_meter_that_says_nan_has_no_reading(value):
    assert power_kw_from_state(State("sensor.grid", value, {"unit_of_measurement": "kW"})) is None


@pytest.mark.parametrize("value", NOT_NUMBERS)
async def test_an_offset_entity_that_says_inf_does_not_break_the_write(value):
    """The register's value is unread, as for text, and the write goes ahead from 0."""
    hass = MagicMock()
    offset = MagicMock()
    offset.state = value
    offset.attributes = {}
    hass.states.get.return_value = offset
    hass.services.async_call = AsyncMock()
    adapter = NibeAdapter(hass, {"nibe_entity": "number.offset"})

    assert await adapter.set_curve_offset(3.0) == 3

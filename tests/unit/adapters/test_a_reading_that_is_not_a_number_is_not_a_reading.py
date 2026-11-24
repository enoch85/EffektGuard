"""A sensor state of "nan" or "inf" is not a reading.

float() parses both, and a NaN fails every comparison, so it passed each check on its way in. On
v0.9.0 the adapter handed the engine degree minutes and a tank temperature of NaN; one NaN power
sample made its billing period NaN, recorded as a peak, and as the month's first peak or in place of
the day's it made every later period read as safe ("Safe margin: nan kW below peak", a 30 kW one
too); an offset entity reading "inf" raised OverflowError out of the write path, which must never
raise, and out of setup on a restart with saved learning, so the integration did not load; and the
tank history read back at start-up kept a NaN. Home Assistant refuses a non-finite value for a
sensor entity that declares a unit, a device class or a state class, but not every state reaches it
that way (a state set over the REST API or by another integration directly). Each reader refuses it
as it refuses text.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import State
from homeassistant.helpers.update_coordinator import UpdateFailed
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import MockConfigEntry

from custom_components.effektguard import _create_coordinator
from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter
from custom_components.effektguard.const import (
    CONF_NIBE_ENTITY,
    DOMAIN,
    LEARNING_STORAGE_VERSION,
    STORAGE_KEY_LEARNING,
)
from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler
from custom_components.effektguard.utils.power import power_kw_from_state
from tests.conftest import create_mock_price_analyzer

NOT_NUMBERS = ["nan", "inf", "-inf"]
OFFSET = "number.offset"
STORED_OFFSET = -2
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


@pytest.mark.parametrize("value", NOT_NUMBERS)
async def test_setup_survives_an_offset_entity_that_says_inf(hass, hass_storage, value):
    """A restart with saved learning syncs the register's value from the offset entity at setup:
    int() of "inf" raised OverflowError out of setup, and the integration did not load. Unread as
    text is, the stored value stands in."""
    hass_storage[STORAGE_KEY_LEARNING] = {
        "version": LEARNING_STORAGE_VERSION,
        "minor_version": 1,
        "key": STORAGE_KEY_LEARNING,
        "data": {"version": LEARNING_STORAGE_VERSION, "last_offset": {"value": STORED_OFFSET}},
    }
    hass.states.async_set(OFFSET, value)
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_NIBE_ENTITY: OFFSET}, options={})
    entry.add_to_hass(hass)

    coordinator = await _create_coordinator(hass, entry)

    assert coordinator.last_applied_offset == STORED_OFFSET


@pytest.mark.parametrize("value", NOT_NUMBERS)
async def test_the_tank_history_read_at_start_up_skips_it(value):
    """The recorder's BT7 history, read back at start-up for the hygiene-cycle detection, is a
    tank temperature too."""
    scheduler = IntelligentDHWScheduler(price_analyzer=create_mock_price_analyzer())
    now = dt_util.utcnow()
    history = [
        State(BT7, reading, last_changed=now - timedelta(hours=hours))
        for hours, reading in ((3, "48"), (2, value), (1, "47"))
    ]
    with (
        patch("homeassistant.components.recorder.is_entity_recorded", return_value=True),
        patch("homeassistant.components.recorder.get_instance") as recorder,
    ):
        recorder.return_value.async_add_executor_job = AsyncMock(return_value={BT7: history})
        await scheduler.initialize_from_history(MagicMock(), BT7)

    assert [temp for _, temp in scheduler.bt7_history] == [48.0, 47.0]

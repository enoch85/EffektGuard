"""A room thermometer that has not reported for a while is a steady room, not a dead sensor.

Most room thermometers report when their value changes. The 30-minute staleness guard (F-015) is
for the pump's own readings, which drive it directly. Applied to the owner's room thermometers it
dropped a quiet one from the indoor median and let it back in when it reported, so the median
hopped 0.3-0.6 °C with no room changing - 484 warnings in 16.5 hours. Only Home Assistant's own
unavailable/unknown takes a room thermometer out.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter

QUIET = timedelta(minutes=79)  # the owner's motion sensor, the evening of 2026-10-17


def _adapter(states: dict[str, tuple[str, timedelta]]) -> NibeAdapter:
    """An adapter whose entities hold (state, time since last reported)."""
    hass = MagicMock()

    def get(entity_id):
        if entity_id not in states:
            return None
        value, age = states[entity_id]
        state = MagicMock()
        state.state = value
        state.attributes = {"unit_of_measurement": "°C"}
        state.last_reported = state.last_updated = dt_util.utcnow() - age
        return state

    hass.states.get.side_effect = get
    adapter = NibeAdapter(hass, {"nibe_entity": "number.offset"})
    adapter._entity_cache = {
        "outdoor_temp": "sensor.bt1",
        "supply_temp": "sensor.bt2",
        "indoor_temp": "sensor.bt50",
        "degree_minutes": "sensor.dm",
    }
    adapter._additional_indoor_sensors = ["sensor.hall", "sensor.living"]
    return adapter


FRESH = timedelta(minutes=1)
PUMP = {"sensor.bt1": ("-3.0", FRESH), "sensor.bt2": ("35.0", FRESH), "sensor.dm": ("-150", FRESH)}


@pytest.mark.asyncio
async def test_a_quiet_room_thermometer_stays_in_the_median():
    adapter = _adapter(
        {
            **PUMP,
            "sensor.bt50": ("21.0", FRESH),
            "sensor.hall": ("20.0", FRESH),
            "sensor.living": ("23.0", QUIET),
        }
    )

    state = await adapter.get_current_state()

    assert state.indoor_temp == pytest.approx(21.0), "median of 21.0, 20.0 and 23.0"


@pytest.mark.asyncio
async def test_an_unavailable_room_thermometer_is_left_out():
    adapter = _adapter(
        {
            **PUMP,
            "sensor.bt50": ("21.0", FRESH),
            "sensor.hall": ("20.0", FRESH),
            "sensor.living": ("unavailable", FRESH),
        }
    )

    state = await adapter.get_current_state()

    assert state.indoor_temp == pytest.approx(20.5), "median of 21.0 and 20.0"


@pytest.mark.asyncio
async def test_the_pumps_own_room_sensor_still_ages_out():
    """F-015 stands for the pump's readings: a BT50 nobody has confirmed is not a reading."""
    adapter = _adapter(
        {
            **PUMP,
            "sensor.bt50": ("21.0", QUIET),
            "sensor.hall": ("20.0", FRESH),
            "sensor.living": ("23.0", FRESH),
        }
    )

    state = await adapter.get_current_state()

    assert state.indoor_temp == pytest.approx(21.5), "median of 20.0 and 23.0, without BT50"

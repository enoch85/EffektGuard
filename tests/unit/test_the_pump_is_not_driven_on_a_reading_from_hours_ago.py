"""An MQTT sensor that stops being published keeps its last value, and stays available forever.

The adapter refuses `unavailable` and `unknown`, so an upstream integration that DIES is caught:
MyUplink and nibe_heatpump use a DataUpdateCoordinator, so when their polling fails the entities go
unavailable and EffektGuard raises UpdateFailed rather than control the pump on incomplete data.

`manifest.json` also lists **mqtt** and **modbus** as NIBE sources, and they do not behave that way.
An MQTT sensor holds its last retained value indefinitely. Nothing marks it unavailable. If the
bridge publishing the pump's degree minutes stops - broker down, bridge crashed, topic renamed - the
sensor goes on cheerfully reporting the number it was given hours ago, and every check this adapter
makes passes.

So the pump keeps being driven on it. Degree minutes could have fallen to -1400 while the sensor
still reads -150, and the integration would go on trimming the curve offset for price, because as
far as it can tell the house is comfortable and the pump is coping.

Age is the only thing that distinguishes a reading from a memory. Home Assistant records
`last_reported` on every state write - even when the value is unchanged - precisely so that "the
pump has been steady at -150 for twenty minutes" can be told apart from "nothing has said anything
about the pump for twenty minutes".

A stale required reading is not a special case. It is the case the adapter already handles: it is a
reading it does not have. It takes the same path - `None`, then UpdateFailed, then entities
unavailable and the pump left on its last offset - which is the safe thing to do with a heat pump
you have stopped being able to see.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter
from custom_components.effektguard.const import NIBE_READING_MAX_AGE_MINUTES


def _adapter_with(entity_id: str, value: str, age: timedelta) -> NibeAdapter:
    """An adapter whose sensor last said anything `age` ago."""
    state = MagicMock()
    state.state = value
    state.last_reported = dt_util.utcnow() - age
    state.last_updated = dt_util.utcnow() - age

    hass = MagicMock()
    hass.states.get.return_value = state

    return NibeAdapter(hass, {"nibe_entity": "number.offset", "degree_minutes_entity": entity_id})


def test_the_max_age_is_generous_enough_not_to_break_a_working_setup():
    """A guard that rejects healthy data is worse than the bug it was meant to fix.

    The coordinator runs every five minutes. Any NIBE source that reports less often than this
    threshold cannot support five-minute heat-pump control anyway, so nothing that works today can
    be broken by it.
    """
    assert NIBE_READING_MAX_AGE_MINUTES >= 15, (
        f"A max age of {NIBE_READING_MAX_AGE_MINUTES} minutes is tight enough to reject a healthy "
        f"but slow NIBE integration, and refusing to control a working heat pump is a worse failure "
        f"than the one this guards against."
    )


@pytest.mark.asyncio
async def test_a_fresh_reading_is_used():
    """The precondition. If this fails, the guard is rejecting everything."""
    adapter = _adapter_with("sensor.dm", "-150", age=timedelta(minutes=1))

    value = await adapter._read_entity_float("sensor.dm", default=None)

    assert value == -150.0


@pytest.mark.asyncio
async def test_a_reading_nobody_has_confirmed_for_hours_is_not_a_reading():
    """The MQTT case: available, unchanged, and hours old."""
    stale = timedelta(minutes=NIBE_READING_MAX_AGE_MINUTES + 60)
    adapter = _adapter_with("sensor.dm", "-150", age=stale)

    value = await adapter._read_entity_float("sensor.dm", default=None)

    assert value is None, (
        f"A degree-minute sensor that last reported {stale} ago was read as -150.0 and used to "
        f"drive the heat pump. Nothing has confirmed that number since. The real degree minutes "
        f"could be anywhere - including past the auxiliary-heat limit - and the integration would "
        f"go on trimming the curve for price, because as far as it can tell the pump is coping."
    )


@pytest.mark.asyncio
async def test_a_stale_required_reading_stops_the_integration_controlling():
    """It must take the same path as a missing one: refuse to drive on data we do not have."""
    from homeassistant.helpers.update_coordinator import UpdateFailed

    adapter = _adapter_with("sensor.dm", "-150", age=timedelta(hours=6))

    with pytest.raises(UpdateFailed):
        await adapter.get_current_state()

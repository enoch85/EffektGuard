"""A required NIBE reading nobody has confirmed for hours is not a reading; it must not drive the pump.

An MQTT/modbus sensor (both listed as NIBE sources in manifest.json) holds its last retained value
indefinitely and is never marked unavailable, so if its publisher stops the adapter's other checks
all pass while the number goes stale. Age is the only thing that separates a reading from a memory:
`_read_entity_float` rejects a value older than NIBE_READING_MAX_AGE_MINUTES, and a required sensor
that comes back None raises UpdateFailed - the pump is left on its last offset, the safe thing to do
with a heat pump you can no longer see. The threshold stays generous enough not to break a slow but
working NIBE integration.
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

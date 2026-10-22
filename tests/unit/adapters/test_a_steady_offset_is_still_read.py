"""A steady offset register is still the offset register, however long since it last reported.

The register changes only when it is written. The README's local Modbus recipe exposes it as a
template number, and a template entity reports only when its value changes: on the bench the
offset last reported at startup while its Modbus source reported every 5 s (2026-10-22). The
30-minute staleness guard (F-015) then read it as 0.0 while the register held its value - the
anti-windup could not see a held raise, `sensor.effektguard_current_offset` showed 0, and every
cycle logged that the pump "will not be driven on it", which it still was.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from homeassistant.helpers.update_coordinator import UpdateFailed
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter
from custom_components.effektguard.const import NIBE_READING_MAX_AGE_MINUTES

FRESH = timedelta(minutes=1)
STEADY = timedelta(minutes=NIBE_READING_MAX_AGE_MINUTES + 60)


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
        "offset": "number.offset",
        "outdoor_temp": "sensor.bt1",
        "supply_temp": "sensor.bt2",
        "degree_minutes": "sensor.dm",
    }
    return adapter


PUMP = {"sensor.bt1": ("-3.0", FRESH), "sensor.bt2": ("35.0", FRESH), "sensor.dm": ("-150", FRESH)}


@pytest.mark.asyncio
async def test_a_steady_offset_is_read_as_the_register_holds(caplog):
    adapter = _adapter({**PUMP, "number.offset": ("3.0", STEADY)})

    with caplog.at_level(logging.WARNING):
        state = await adapter.get_current_state()

    assert state.current_offset == 3.0, (
        f"The register holds +3 and last reported {STEADY} ago, as a template number does when "
        f"nothing changes it, and it was read as {state.current_offset}. The anti-windup judges a "
        f"held raise on this number."
    )
    assert "number.offset last reported" not in caplog.text


@pytest.mark.asyncio
async def test_an_unavailable_offset_still_reads_as_zero():
    adapter = _adapter({**PUMP, "number.offset": ("unavailable", FRESH)})

    state = await adapter.get_current_state()

    assert state.current_offset == 0.0


@pytest.mark.asyncio
async def test_a_stale_pump_reading_still_stops_control():
    """F-015 stands for the readings that prove the pump is alive; only the offset is exempt."""
    adapter = _adapter({**PUMP, "sensor.dm": ("-150", STEADY), "number.offset": ("3.0", FRESH)})

    with pytest.raises(UpdateFailed):
        await adapter.get_current_state()

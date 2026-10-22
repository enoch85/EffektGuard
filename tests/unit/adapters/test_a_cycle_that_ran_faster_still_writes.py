"""Each control cycle may write once; a cycle that ran faster than the one before keeps its write.

The write limit was five minutes since the last write - the control interval itself. The last write
is stamped when a cycle has finished its read and decide work, which varies from cycle to cycle
(the weather service call, the executor, recorder queries). Whenever a cycle ran faster than the one
before it, by any amount, its write fell inside the limit and was skipped, and the register followed
a cycle late: about every other write on consecutive cycles (runtime review F4).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter
from custom_components.effektguard.const import CONF_NIBE_ENTITY


def _adapter() -> tuple[NibeAdapter, dict]:
    """An adapter writing a number entity that holds whatever was last written."""
    register = {"value": "0"}
    hass = MagicMock(spec=HomeAssistant)

    def get(_entity_id):
        state = MagicMock()
        state.state = register["value"]
        state.attributes = {}
        return state

    async def call(_domain, _service, data, **_):
        register["value"] = str(data["value"])

    hass.states = MagicMock()
    hass.states.get = get
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock(side_effect=call)
    return NibeAdapter(hass, {CONF_NIBE_ENTITY: "number.offset"}), register


async def _write_at(adapter: NibeAdapter, moment: datetime, offset: float, **kwargs):
    with patch(
        "custom_components.effektguard.adapters.nibe_adapter.dt_util.utcnow", return_value=moment
    ):
        return await adapter.set_curve_offset(offset, **kwargs)


@pytest.mark.asyncio
async def test_a_cycle_fifty_milliseconds_faster_than_the_last_one_still_writes():
    adapter, register = _adapter()
    assert await _write_at(adapter, datetime(2026, 1, 15, 10, 0, 10, 900_000, timezone.utc), 1.0)

    written = await _write_at(adapter, datetime(2026, 1, 15, 10, 5, 10, 850_000, timezone.utc), 2.0)

    assert written == 2, (
        f"The 10:05:10 cycle finished its work 50 ms sooner than the 10:00:10 one and its write "
        f"of 2 returned {written}: the register stays at {register['value']} for another five "
        f"minutes."
    )


@pytest.mark.asyncio
async def test_a_second_write_in_the_same_control_interval_is_still_refused():
    adapter, register = _adapter()
    assert await _write_at(adapter, datetime(2026, 1, 15, 10, 5, 10, tzinfo=timezone.utc), 1.0)

    assert (
        await _write_at(adapter, datetime(2026, 1, 15, 10, 9, 0, tzinfo=timezone.utc), 3.0) is None
    )
    assert register["value"] == "1"


@pytest.mark.asyncio
async def test_a_forced_write_goes_through_in_the_same_interval():
    adapter, _ = _adapter()
    assert await _write_at(adapter, datetime(2026, 1, 15, 10, 5, 10, tzinfo=timezone.utc), 1.0)

    forced = await _write_at(
        adapter, datetime(2026, 1, 15, 10, 6, 0, tzinfo=timezone.utc), 3.0, force_write=True
    )

    assert forced == 3

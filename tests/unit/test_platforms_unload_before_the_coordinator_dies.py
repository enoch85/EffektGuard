"""Platforms unload FIRST; the coordinator dies only after they actually did.

async_unload_entry used to shut the coordinator down and THEN ask Home Assistant to unload
the platforms. If a platform refused - which HA reports by returning False and keeping the
entry loaded - the user was left with a loaded entry full of live entities served by a dead
coordinator: every sensor frozen on its last value, the control loop gone, nothing saying so.
That is the "watching a heat pump that is not there" failure, manufactured during teardown.

HA's own pattern is the other order: unload platforms, and only on success tear down what
they were reading from.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.effektguard import async_unload_entry
from custom_components.effektguard.const import DOMAIN


def _env(unload_ok: bool):
    hass = MagicMock()
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=unload_ok)
    hass.services.has_service = MagicMock(return_value=False)

    entry = MagicMock()
    entry.entry_id = "test-entry"

    coordinator = MagicMock()
    coordinator.async_shutdown = AsyncMock()
    hass.data = {DOMAIN: {"test-entry": coordinator}}
    return hass, entry, coordinator


@pytest.mark.asyncio
async def test_a_refused_platform_unload_leaves_the_coordinator_alive():
    hass, entry, coordinator = _env(unload_ok=False)

    result = await async_unload_entry(hass, entry)

    assert result is False
    coordinator.async_shutdown.assert_not_awaited()
    assert hass.data[DOMAIN]["test-entry"] is coordinator, (
        "The entry is still loaded - HA keeps serving its entities - so the coordinator "
        "must still be the live object behind them."
    )


@pytest.mark.asyncio
async def test_a_successful_unload_shuts_the_coordinator_down_after():
    hass, entry, coordinator = _env(unload_ok=True)

    result = await async_unload_entry(hass, entry)

    assert result is True
    coordinator.async_shutdown.assert_awaited_once()
    assert "test-entry" not in hass.data[DOMAIN]

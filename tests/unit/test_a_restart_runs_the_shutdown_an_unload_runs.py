"""A Home Assistant restart runs the coordinator's shutdown, as an unload does.

Home Assistant does not unload config entries when it stops (ConfigEntries._async_shutdown only
cancels retried setups), so the shutdown that `async_unload_entry` runs never ran on a restart.
That shutdown is what stops a hot-water boost EffektGuard started: after a restart mid-boost the
temporary-lux switch stayed on to NIBE's own lux timeout, and the new instance had lost the record
that the boost was its own (review DHW F7, run on Home Assistant 2026.10.0: after hass.async_stop()
the entry was still loaded, the cleanup had not run and the switch was on).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import async_setup_entry
from custom_components.effektguard.const import DOMAIN


async def test_stopping_home_assistant_shuts_the_coordinator_down(hass):
    coordinator = MagicMock()
    coordinator.async_config_entry_first_refresh = AsyncMock()
    coordinator.async_shutdown = AsyncMock()
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)

    with (
        patch(
            "custom_components.effektguard._create_coordinator",
            AsyncMock(return_value=coordinator),
        ),
        patch("custom_components.effektguard._async_register_services", AsyncMock()),
        patch.object(hass.config_entries, "async_forward_entry_setups", AsyncMock()),
    ):
        assert await async_setup_entry(hass, entry)

    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()

    assert coordinator.async_shutdown.await_count >= 1, (
        "Home Assistant stopped and the coordinator's shutdown did not run: a hot-water boost "
        "EffektGuard started stays on to NIBE's own lux timeout"
    )

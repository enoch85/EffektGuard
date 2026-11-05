"""A Home Assistant restart runs the coordinator's shutdown, as an unload does - while the pump can
still be reached.

Home Assistant does not unload config entries when it stops (ConfigEntries._async_shutdown only
cancels retried setups), so the shutdown that `async_unload_entry` runs never ran on a restart.
That shutdown is what stops a hot-water boost EffektGuard started: after a restart mid-boost the
temporary-lux switch stayed on to NIBE's own lux timeout (review DHW F7). Run on
EVENT_HOMEASSISTANT_STOP it raced the local transports, which close on that event: on the Modbus
bench the stop was sent after the connection closed, and the boost stayed on.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import Event, callback
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import async_setup_entry
from custom_components.effektguard.const import DOMAIN


async def test_stopping_home_assistant_shuts_the_coordinator_down_before_transports_close(hass):
    order: list[str] = []

    async def shutdown() -> None:
        await asyncio.sleep(0)  # the stop is a service call that reaches the pump
        order.append("coordinator shut down")

    coordinator = MagicMock()
    coordinator.async_config_entry_first_refresh = AsyncMock()
    coordinator.async_shutdown = AsyncMock(side_effect=shutdown)
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

    @callback
    def close_transport(_event: Event) -> None:
        # modbus and nibe_heatpump close their connection to the pump here
        order.append("transport closed")

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, close_transport)
    await hass.async_stop()

    assert order == ["coordinator shut down", "transport closed"], (
        f"stopping Home Assistant ran {order}: a hot-water boost EffektGuard started is stopped "
        f"only if the stop reaches the pump before its connection closes"
    )

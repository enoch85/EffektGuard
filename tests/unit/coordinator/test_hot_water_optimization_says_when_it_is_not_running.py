"""On an S-series pump, hot-water optimisation must raise a repair issue, not fail in a debug line.

EffektGuard drives DHW via NIBE's temporary-lux switch (register 50004), which Home Assistant's
NIBE integration maps for the F-SERIES ONLY. On an S-series pump no such entity exists, so the
whole DHW feature silently does nothing while the UI still shows a hot-water status, a
recommendation and a scheduled start time that can never fire. A _LOGGER.debug is not telling
anyone - so this now raises the same kind of repair issue the missing price source does (F-123).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.effektguard.const import DHW_CONTROL_ISSUE_ID, DOMAIN
from custom_components.effektguard.coordinator import EffektGuardCoordinator

NOW = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)


def _coordinator(lux_entity: str | None) -> EffektGuardCoordinator:
    coordinator = EffektGuardCoordinator.__new__(EffektGuardCoordinator)
    coordinator.hass = MagicMock()
    coordinator.hass.services.async_call = AsyncMock()
    coordinator.temp_lux_entity = lux_entity
    coordinator._dhw_issue_active = False
    coordinator._lux_boost_is_ours = False
    coordinator._service_boost_until = None
    coordinator._last_dhw_control_time = None
    # `__new__` skips `__init__`, so anything the real object always carries has to be set here or
    # the fake is not the object. Home Assistant's DataUpdateCoordinator.__init__ sets this, and the
    # hot-water switch door reads it: a coordinator whose entry has unloaded does not start a boost.
    coordinator._shutdown_requested = False
    coordinator.last_update_success = True
    coordinator.data = {}
    coordinator.entry = MagicMock()
    coordinator.entry.data = {"enable_hot_water_optimization": True}
    coordinator.entry.options = {}

    state = MagicMock()
    state.state = "off"
    coordinator.hass.states.get.return_value = state
    return coordinator


def _decision():
    decision = MagicMock()
    decision.should_heat = True
    decision.priority_reason = "cheap window"
    return decision


@pytest.mark.asyncio
async def test_an_s_series_pump_raises_a_repair_issue():
    coordinator = _coordinator(lux_entity=None)

    with patch("custom_components.effektguard.coordinator.async_create_issue") as create_issue:
        await coordinator._apply_dhw_control(_decision(), current_dhw_temp=45.0, now_time=NOW)

    create_issue.assert_called_once()
    args, kwargs = create_issue.call_args
    assert args[1] == DOMAIN
    assert args[2] == DHW_CONTROL_ISSUE_ID
    assert kwargs["translation_key"] == DHW_CONTROL_ISSUE_ID


@pytest.mark.asyncio
async def test_the_issue_is_raised_once_not_on_every_cycle():
    """The coordinator ticks every five minutes. Do not re-raise it 288 times a day."""
    coordinator = _coordinator(lux_entity=None)

    with patch("custom_components.effektguard.coordinator.async_create_issue") as create_issue:
        for _ in range(5):
            await coordinator._apply_dhw_control(_decision(), current_dhw_temp=45.0, now_time=NOW)

    assert create_issue.call_count == 1


@pytest.mark.asyncio
async def test_a_pump_that_has_the_switch_clears_the_issue():
    """An F-series pump must not be nagged - and a stale issue from a restart must be cleared."""
    coordinator = _coordinator(lux_entity="switch.temporary_lux_50004")

    with (
        patch("custom_components.effektguard.coordinator.async_delete_issue") as delete_issue,
        patch("custom_components.effektguard.coordinator.async_create_issue") as create_issue,
    ):
        await coordinator._apply_dhw_control(_decision(), current_dhw_temp=45.0, now_time=NOW)

    create_issue.assert_not_called()
    delete_issue.assert_called_once()


@pytest.mark.asyncio
async def test_the_f_series_pump_still_actually_controls_hot_water():
    """The regression guard: raising an issue must not break the pumps that work."""
    coordinator = _coordinator(lux_entity="switch.temporary_lux_50004")

    with patch("custom_components.effektguard.coordinator.async_delete_issue"):
        await coordinator._apply_dhw_control(_decision(), current_dhw_temp=45.0, now_time=NOW)

    turn_ons = [
        call
        for call in coordinator.hass.services.async_call.await_args_list
        if call.args[:2] == ("homeassistant", "turn_on")
    ]
    assert turn_ons, "an F-series pump with a cheap window must still get its hot-water boost"

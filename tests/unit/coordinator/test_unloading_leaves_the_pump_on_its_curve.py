"""Disabling or removing EffektGuard leaves the heat pump on its normal curve.

Unloading used to stop the coordinator and leave the curve offset wherever the last cycle put it,
-10 in a peak quarter, with nothing left to bring it back. A user who disables the integration on a
cold night kept a house that was being steered down (hacs/default#12414). OFF already returned the
pump to its curve; unloading now does the same, through the same routine.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.exceptions import HomeAssistantError

from custom_components.effektguard import async_unload_entry
from custom_components.effektguard.const import CONF_ENABLE_OPTIMIZATION, DOMAIN
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.effect_layer import EffectManager


def _setup(optimization_enabled: bool = True):
    hass = MagicMock()
    hass.config.latitude = 59.33
    hass.config.longitude = 18.07
    hass.config_entries.async_update_entry = MagicMock()
    hass.config_entries.async_unload_platforms = AsyncMock(return_value=True)
    hass.services.has_service = MagicMock(return_value=False)

    nibe = MagicMock()
    nibe.set_curve_offset = AsyncMock(return_value=0)
    nibe.set_enhanced_ventilation = AsyncMock(return_value=True)
    nibe.is_enhanced_ventilation_active = AsyncMock(return_value=False)
    nibe.has_ventilation_control = False

    entry = MagicMock()
    entry.entry_id = "test-entry"
    entry.data = {CONF_ENABLE_OPTIMIZATION: optimization_enabled}
    entry.options = {}

    coordinator = EffektGuardCoordinator(
        hass, nibe, MagicMock(), MagicMock(), MagicMock(), EffectManager(hass), entry
    )
    coordinator.learning_store = MagicMock()
    coordinator.learning_store.async_save = AsyncMock()
    coordinator.effect.async_save = AsyncMock()
    hass.data = {DOMAIN: {"test-entry": coordinator}}
    return hass, entry, coordinator, nibe


@pytest.mark.asyncio
async def test_unloading_puts_the_offset_back_to_zero_before_the_coordinator_stops():
    hass, entry, coordinator, nibe = _setup()
    shut_down_at_write = []
    nibe.set_curve_offset.side_effect = lambda *a, **k: (
        shut_down_at_write.append(coordinator._shutdown_requested) or 0
    )

    assert await async_unload_entry(hass, entry) is True

    nibe.set_curve_offset.assert_awaited_once_with(0.0, force_write=True)
    assert shut_down_at_write == [False], (
        "the neutral offset has to be written while the coordinator is still a writer; once "
        "async_shutdown has run, its own guard refuses every write and the pump keeps the last "
        "cycle's offset"
    )
    assert coordinator._shutdown_requested is True


@pytest.mark.asyncio
async def test_unloading_returns_an_enhanced_fan_to_normal():
    hass, entry, coordinator, nibe = _setup()
    nibe.has_ventilation_control = True
    nibe.is_enhanced_ventilation_active = AsyncMock(return_value=True)

    await async_unload_entry(hass, entry)

    nibe.set_enhanced_ventilation.assert_awaited_once_with(False, force_write=True)


@pytest.mark.asyncio
async def test_unloading_while_off_writes_nothing():
    """OFF already put the pump on its curve; whatever is on it now is the owner's."""
    hass, entry, coordinator, nibe = _setup(optimization_enabled=False)

    assert await async_unload_entry(hass, entry) is True

    nibe.set_curve_offset.assert_not_awaited()
    nibe.set_enhanced_ventilation.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_pump_that_does_not_answer_does_not_block_the_unload(caplog):
    hass, entry, coordinator, nibe = _setup()
    nibe.set_curve_offset = AsyncMock(side_effect=HomeAssistantError("entity unavailable"))

    with caplog.at_level(logging.WARNING):
        assert await async_unload_entry(hass, entry) is True

    assert coordinator._shutdown_requested is True
    assert "test-entry" not in hass.data[DOMAIN]
    assert "Check the heating offset on the pump" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("cycle_first", [True, False])
async def test_a_cycle_in_flight_does_not_get_the_last_word(cycle_first):
    """A control cycle racing the unload either writes first and is overwritten, or is refused."""
    hass, entry, coordinator, nibe = _setup()
    holding = asyncio.Event()
    release = asyncio.Event()

    async def read_and_decide(
        apply: bool = False, explicit_command: bool = False, scheduled: bool = False
    ):
        holding.set()
        await release.wait()
        if apply:
            await coordinator._write_curve_offset(-10.0)
        return {}

    with patch.object(coordinator, "_read_and_decide", read_and_decide):
        if cycle_first:
            cycle = asyncio.create_task(coordinator._drive_the_pump())
            await holding.wait()
            unload = asyncio.create_task(async_unload_entry(hass, entry))
            await asyncio.sleep(0)
            release.set()
        else:
            # The unload queues for the lock first; the cycle queues behind it.
            async with coordinator._control_lock:
                unload = asyncio.create_task(async_unload_entry(hass, entry))
                for _ in range(5):
                    await asyncio.sleep(0)
                cycle = asyncio.create_task(coordinator._drive_the_pump())
                for _ in range(5):
                    await asyncio.sleep(0)
            release.set()
        await unload
        await cycle

    assert nibe.set_curve_offset.await_args_list[-1].args[0] == 0.0, (
        f"the last write to the pump was {nibe.set_curve_offset.await_args_list}: a control cycle "
        f"in flight during the unload got the last word and left the curve steered"
    )

"""EffektGuard started a hot-water boost, then unloaded and left it running.

EffektGuard drives DHW by turning NIBE's temporary-lux switch ON, and turns it OFF again on the
tick that decides the cycle is done. Nothing turned it off on UNLOAD.

So a reload, an options change, or a Home Assistant restart landing in the middle of an
EffektGuard-initiated boost left the heat pump running that boost until NIBE's own timeout expired,
with nothing left alive to stop it. A full high-temperature hot-water cycle - at the top of the
tank, which is where the immersion heater does the work - that nobody asked for.

Only OUR boost is cancelled. The owner may start one from the heat pump's own panel or from their
own automation, and that one is none of our business. The DHW control path already says exactly
this, about its own turn-off branch:

    Stopping the lux boost cannot harm the pump - it only stops an EffektGuard-initiated boost.

which is true of the turn-off it was written for, and was NOT true of unload, because unload did
not turn anything off at all.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from custom_components.effektguard.coordinator import EffektGuardCoordinator

LUX = "switch.temporary_lux_50004"


def _coordinator(lux_state: str | None, boost_is_ours: bool) -> EffektGuardCoordinator:
    """A real coordinator with __init__ bypassed - only what the shutdown path touches."""
    coordinator = EffektGuardCoordinator.__new__(EffektGuardCoordinator)
    coordinator._shutdown_requested = False
    coordinator._unsub_aligned_refresh = None
    coordinator._power_sensor_listener = None
    coordinator.adaptive_learning = None
    coordinator.thermal_predictor = None
    coordinator.weather_learner = None
    coordinator.effect = MagicMock()
    coordinator.effect.async_save = AsyncMock()
    coordinator._save_learned_data = AsyncMock()
    coordinator._clock_aligned = True

    coordinator.temp_lux_entity = LUX
    coordinator._lux_boost_is_ours = boost_is_ours

    coordinator.hass = MagicMock()
    coordinator.hass.services.async_call = AsyncMock()
    if lux_state is None:
        coordinator.hass.states.get.return_value = None
    else:
        state = MagicMock()
        state.state = lux_state
        coordinator.hass.states.get.return_value = state

    return coordinator


async def _unload(coordinator, monkeypatch) -> None:
    async def fake_base_shutdown(self) -> None:
        self._shutdown_requested = True

    monkeypatch.setattr(DataUpdateCoordinator, "async_shutdown", fake_base_shutdown)
    await coordinator.async_shutdown()


def _turn_off_calls(coordinator) -> list:
    return [
        call
        for call in coordinator.hass.services.async_call.await_args_list
        if call.args[:2] == ("switch", "turn_off")
    ]


@pytest.mark.asyncio
async def test_our_own_boost_is_cancelled_on_unload(monkeypatch):
    coordinator = _coordinator(lux_state=STATE_ON, boost_is_ours=True)

    await _unload(coordinator, monkeypatch)

    calls = _turn_off_calls(coordinator)
    assert calls, (
        "EffektGuard unloaded while a hot-water boost IT had started was still running, and did "
        "not turn it off. The pump runs that boost to NIBE's own timeout with nothing left alive "
        "to stop it - a full high-temperature DHW cycle nobody asked for, heated at the top of the "
        "tank where the immersion heater does the work."
    )
    assert calls[0].args[2] == {"entity_id": LUX}


@pytest.mark.asyncio
async def test_a_boost_the_owner_started_is_left_alone(monkeypatch):
    """The switch is ON, but it was not us. Turning it off would be overriding the owner."""
    coordinator = _coordinator(lux_state=STATE_ON, boost_is_ours=False)

    await _unload(coordinator, monkeypatch)

    assert not _turn_off_calls(coordinator), (
        "EffektGuard turned off a temporary-lux boost it did not start. The owner may run one from "
        "the heat pump's own panel or from their own automation, and unloading EffektGuard must "
        "not cancel their hot water."
    )


@pytest.mark.asyncio
async def test_nothing_is_written_when_the_boost_has_already_finished(monkeypatch):
    """Ours, but NIBE already timed it out. Do not write for the sake of writing."""
    coordinator = _coordinator(lux_state=STATE_OFF, boost_is_ours=True)

    await _unload(coordinator, monkeypatch)

    assert not _turn_off_calls(coordinator)
    assert coordinator._lux_boost_is_ours is False


@pytest.mark.asyncio
async def test_a_pump_with_no_lux_switch_unloads_cleanly(monkeypatch):
    """An S1155 exposes no temporary-lux entity at all. Unload must not raise."""
    coordinator = _coordinator(lux_state=None, boost_is_ours=False)
    coordinator.temp_lux_entity = None

    await _unload(coordinator, monkeypatch)

    assert not _turn_off_calls(coordinator)


@pytest.mark.asyncio
async def test_the_rest_of_shutdown_still_runs(monkeypatch):
    """The regression guard: cancelling the boost must not skip saving state."""
    coordinator = _coordinator(lux_state=STATE_ON, boost_is_ours=True)

    await _unload(coordinator, monkeypatch)

    coordinator.effect.async_save.assert_awaited_once()
    assert coordinator._shutdown_requested is True

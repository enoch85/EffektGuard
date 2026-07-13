"""Unload must actually stop the coordinator. Two writers on one heat pump is unacceptable.

EffektGuard disables the base scheduler (`update_interval=None`) and drives itself from a
clock-aligned timer, re-armed in `_do_aligned_refresh`'s `finally` block so that a single
bad cycle cannot kill the update loop.

That `finally` created a second, subtler hazard, and this file pins it shut.

THE ORPHAN-TIMER RACE
---------------------
`_do_aligned_refresh` runs on a task created with `hass.async_create_task` - NOT
`entry.async_create_task` - so Home Assistant cannot cancel it when the entry unloads:

    T+0.00  timer fires -> hass.async_create_task(_do_aligned_refresh())
    T+0.02  user hits Reload -> async_unload_entry -> coordinator.async_shutdown()
    T+0.03  coordinator popped from hass.data; platforms unloaded
    T+0.05  _do_aligned_refresh finishes -> finally -> _schedule_aligned_refresh()
            ^^^ RE-ARMS A TIMER ON THE DEAD COORDINATOR
    T+0.06  async_setup_entry runs again -> a SECOND coordinator, with its own timer
    T+5min  BOTH fire -> both call nibe.set_curve_offset()

Each coordinator has its own rate limiter and its own `last_applied_offset`, so they fight.
Every reload would add another writer, permanently.

The guard is `_shutdown_requested`, which only exists if `async_shutdown()` calls
`super().async_shutdown()`. It previously did not - so the flag was never set, the base
refresh handle was never cancelled, and the request debouncer was never shut down (a
trailing 10 s debounced refresh queued by a service call could fire *after* unload and
write an offset to the pump).

The same missing `super()` call also meant shutdown ran TWICE per unload - the base
registers `config_entry.async_on_unload(self.async_shutdown)` in its own __init__, and
`async_unload_entry` calls it explicitly - double-saving learning data and effect peaks.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from custom_components.effektguard.coordinator import EffektGuardCoordinator


def make_coordinator() -> EffektGuardCoordinator:
    """A REAL EffektGuardCoordinator with __init__ bypassed.

    It must be a real instance: `super()` inside async_shutdown requires
    `isinstance(self, EffektGuardCoordinator)`, so a MagicMock cannot stand in here.
    Only the attributes the shutdown/scheduling paths touch are populated.
    """
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
    coordinator.hass = MagicMock()
    coordinator._clock_aligned = True
    # Shutdown now also cancels an EffektGuard-initiated hot-water boost, so it touches these.
    coordinator.temp_lux_entity = None
    coordinator._lux_boost_is_ours = False
    return coordinator


async def shutdown(coordinator, base_shutdown_calls: list, monkeypatch) -> None:
    """Run the real async_shutdown, with super().async_shutdown() faithfully emulated."""

    async def fake_base_shutdown(self) -> None:
        base_shutdown_calls.append(True)
        # Exactly what the real base class does, and it is the whole point of the fix:
        self._shutdown_requested = True

    monkeypatch.setattr(DataUpdateCoordinator, "async_shutdown", fake_base_shutdown)
    await coordinator.async_shutdown()


class TestShutdownActuallyStopsIt:
    @pytest.mark.asyncio
    async def test_shutdown_calls_super(self, monkeypatch):
        """Without super(), `_shutdown_requested` is never set and nothing below works."""
        coordinator = make_coordinator()
        base_calls: list = []

        await shutdown(coordinator, base_calls, monkeypatch)

        assert base_calls, (
            "async_shutdown() did not call super().async_shutdown(). The base sets "
            "_shutdown_requested, cancels the refresh handle and shuts down the debouncer. "
            "Without it, unload does not actually stop the coordinator."
        )
        assert coordinator._shutdown_requested is True

    @pytest.mark.asyncio
    async def test_an_inflight_refresh_cannot_rearm_a_dead_coordinator(self, monkeypatch):
        """THE ORPHAN-TIMER RACE. This is the one that puts two writers on one pump."""
        coordinator = make_coordinator()
        await shutdown(coordinator, [], monkeypatch)

        # A refresh task was already in flight when the entry unloaded. Its `finally`
        # block now runs and tries to re-arm the timer.
        coordinator._schedule_aligned_refresh()

        assert coordinator._unsub_aligned_refresh is None, (
            "A shut-down coordinator re-armed its aligned-refresh timer. The reloaded entry "
            "creates a second coordinator, and BOTH will write curve offsets to the same "
            "heat pump - fighting each other, and adding another writer on every reload."
        )

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, monkeypatch):
        """It runs twice per unload: once via async_on_unload, once from async_unload_entry."""
        coordinator = make_coordinator()
        base_calls: list = []

        await shutdown(coordinator, base_calls, monkeypatch)
        await shutdown(coordinator, base_calls, monkeypatch)

        assert coordinator._save_learned_data.await_count == 0  # no learning modules here
        assert coordinator.effect.async_save.await_count == 1, (
            "Effect peaks were saved twice on a single unload. async_shutdown runs twice "
            "(the base auto-registers it AND async_unload_entry calls it) and must be "
            "idempotent."
        )
        assert len(base_calls) == 1, "super().async_shutdown() must not run twice either."


class TestTheUpdateLoopStillRearmsWhenAlive:
    """Do not over-correct: a LIVE coordinator must still re-arm, or the loop dies."""

    def test_a_live_coordinator_rearms(self):
        coordinator = make_coordinator()
        coordinator._calculate_next_aligned_time = MagicMock()

        coordinator._schedule_aligned_refresh()

        # It reached the scheduling call rather than returning early.
        coordinator._calculate_next_aligned_time.assert_called_once()

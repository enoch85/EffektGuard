"""Regression tests for clock-aligned coordinator scheduling.

Ensures EffektGuard's aligned refresh timer is not stored in Home Assistant's
DataUpdateCoordinator internal `_unsub_refresh`, which Home Assistant cancels
during `async_set_updated_data()`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.effektguard.coordinator import EffektGuardCoordinator


def _make_minimal_hass() -> MagicMock:
    hass = MagicMock()
    hass.data = {}
    hass.config = MagicMock()
    hass.config.latitude = 59.3
    hass.config.config_dir = "/tmp/test"
    hass.loop = MagicMock()
    hass.loop.call_soon_threadsafe = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda func, *args: func(*args))
    hass.async_create_task = MagicMock()
    return hass


def _make_minimal_entry() -> MagicMock:
    entry = MagicMock()
    entry.data = MagicMock()
    entry.data.get.side_effect = lambda key, default=None: default
    entry.options = MagicMock()
    entry.options.get.side_effect = lambda key, default=None: default
    return entry


@pytest.mark.asyncio
async def test_aligned_timer_not_canceled_by_async_set_updated_data(monkeypatch):
    canceled = False

    def fake_async_track_point_in_time(_hass, _action, _point_in_time):
        def _cancel():
            nonlocal canceled
            canceled = True

        return _cancel

    monkeypatch.setattr(
        "custom_components.effektguard.coordinator.async_track_point_in_time",
        fake_async_track_point_in_time,
    )

    fixed_next = datetime(2025, 12, 19, 1, 0, 10, tzinfo=timezone.utc)
    monkeypatch.setattr(
        EffektGuardCoordinator,
        "_calculate_next_aligned_time",
        lambda self: fixed_next,
    )

    coordinator = EffektGuardCoordinator(
        hass=_make_minimal_hass(),
        nibe_adapter=MagicMock(),
        gespot_adapter=MagicMock(),
        weather_adapter=MagicMock(),
        decision_engine=MagicMock(),
        effect_manager=MagicMock(),
        entry=_make_minimal_entry(),
    )

    coordinator._schedule_aligned_refresh()

    # Must not use the base class internal scheduler handle.
    assert coordinator._unsub_refresh is None
    assert coordinator._unsub_aligned_refresh is not None

    # Home Assistant cancels DataUpdateCoordinator._unsub_refresh here; our aligned timer must remain.
    coordinator.async_set_updated_data({"ok": True})

    assert canceled is False
    assert coordinator._unsub_aligned_refresh is not None


@pytest.mark.asyncio
async def test_schedule_aligned_refresh_cancels_previous_timer(monkeypatch):
    cancel_calls = 0

    def fake_async_track_point_in_time(_hass, _action, _point_in_time):
        def _cancel():
            nonlocal cancel_calls
            cancel_calls += 1

        return _cancel

    monkeypatch.setattr(
        "custom_components.effektguard.coordinator.async_track_point_in_time",
        fake_async_track_point_in_time,
    )

    fixed_next = datetime(2025, 12, 19, 1, 0, 10, tzinfo=timezone.utc)
    monkeypatch.setattr(
        EffektGuardCoordinator,
        "_calculate_next_aligned_time",
        lambda self: fixed_next,
    )

    coordinator = EffektGuardCoordinator(
        hass=_make_minimal_hass(),
        nibe_adapter=MagicMock(),
        gespot_adapter=MagicMock(),
        weather_adapter=MagicMock(),
        decision_engine=MagicMock(),
        effect_manager=MagicMock(),
        entry=_make_minimal_entry(),
    )

    coordinator._schedule_aligned_refresh()
    coordinator._schedule_aligned_refresh()

    assert cancel_calls == 1

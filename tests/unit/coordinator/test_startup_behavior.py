"""Tests for coordinator startup behavior.

Covers:
- Startup pending path still schedules next aligned retry (update_interval=None).
- Startup grace period blocks any control actions (e.g., airflow control).
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from homeassistant.helpers.update_coordinator import UpdateFailed

from custom_components.effektguard.const import CONF_ENABLE_AIRFLOW_OPTIMIZATION
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.decision_engine import OptimizationDecision


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
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.bus = MagicMock()
    hass.bus.async_listen = MagicMock(return_value=lambda: None)
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.components = MagicMock()
    hass.components.persistent_notification = MagicMock()
    hass.components.persistent_notification.async_create = MagicMock()
    return hass


@pytest.mark.asyncio
async def test_startup_pending_arms_aligned_timer(monkeypatch):
    scheduled = False

    def fake_async_track_point_in_time(_hass, _action, _point_in_time):
        nonlocal scheduled
        scheduled = True

        def _cancel():
            return None

        return _cancel

    monkeypatch.setattr(
        "custom_components.effektguard.coordinator.async_track_point_in_time",
        fake_async_track_point_in_time,
    )

    fixed_next = datetime(2025, 12, 19, 1, 0, 10, tzinfo=timezone.utc)
    monkeypatch.setattr(EffektGuardCoordinator, "_calculate_next_aligned_time", lambda self: fixed_next)

    hass = _make_minimal_hass()

    nibe = MagicMock()
    nibe.get_current_state = AsyncMock(side_effect=UpdateFailed("not ready"))

    coordinator = EffektGuardCoordinator(
        hass=hass,
        nibe_adapter=nibe,
        gespot_adapter=MagicMock(),
        weather_adapter=MagicMock(),
        decision_engine=MagicMock(),
        effect_manager=MagicMock(),
        entry=MagicMock(data={}, options={}),
    )

    data = await coordinator._async_update_data()

    assert data.get("startup_pending") is True
    assert scheduled is True
    assert coordinator._unsub_aligned_refresh is not None


@pytest.mark.asyncio
async def test_airflow_control_not_applied_during_startup_grace(monkeypatch):
    hass = _make_minimal_hass()

    nibe_data = SimpleNamespace(
        indoor_temp=20.0,
        outdoor_temp=0.0,
        flow_temp=30.0,
        degree_minutes=-60.0,
        compressor_hz=None,
        timestamp=datetime(2025, 12, 19, 1, 0, 0, tzinfo=timezone.utc),
        dhw_top_temp=None,
        phase1_current=None,
        phase2_current=None,
        phase3_current=None,
        power_kw=None,
    )

    nibe = MagicMock()
    nibe.get_current_state = AsyncMock(return_value=nibe_data)

    engine = MagicMock()
    engine.calculate_decision = MagicMock(
        return_value=OptimizationDecision(offset=0.0, reasoning="test", layers=[])
    )
    engine.price = MagicMock()
    engine.price.get_current_classification = MagicMock(return_value="normal")

    effect = MagicMock()
    effect.async_save = AsyncMock()

    entry = MagicMock(
        data={
            "enable_optimization": True,
            CONF_ENABLE_AIRFLOW_OPTIMIZATION: True,
        },
        options={},
    )

    coordinator = EffektGuardCoordinator(
        hass=hass,
        nibe_adapter=nibe,
        gespot_adapter=MagicMock(get_prices=AsyncMock(return_value=None)),
        weather_adapter=MagicMock(get_forecast=AsyncMock(return_value=None)),
        decision_engine=engine,
        effect_manager=effect,
        entry=entry,
    )

    # Avoid unrelated side-effects; we only care about airflow control gating.
    coordinator._update_peak_tracking = AsyncMock()
    coordinator._record_learning_observations = AsyncMock()
    coordinator._schedule_aligned_refresh = MagicMock()

    coordinator.airflow_optimizer = MagicMock()
    coordinator.airflow_optimizer.evaluate_from_nibe = MagicMock(
        return_value=SimpleNamespace(
            reason="x",
            should_enhance=True,
            expected_gain_kw=0.5,
        )
    )

    coordinator._apply_airflow_decision = AsyncMock()

    result = await coordinator._async_update_data()

    # First successful update is within grace period by default.
    coordinator._apply_airflow_decision.assert_not_awaited()
    assert result.get("airflow_decision") is not None

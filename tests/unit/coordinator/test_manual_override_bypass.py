"""Manual force_offset commands must bypass the volatile-reversal blocker.

Live-reproduced regression risk: after a +4°C offset, a user-commanded
effektguard.force_offset with offset 0 was accepted by the service and the
decision engine, but the coordinator's volatility blocker deferred it as a
volatile reversal and kept the previous +4°C for 45 minutes. User commands
are authoritative and must apply immediately; the blocker still applies to
automatic decisions.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from homeassistant.util import dt as dt_util

from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.decision_engine import OptimizationDecision


def _make_minimal_hass() -> MagicMock:
    hass = MagicMock()
    hass.data = {}
    hass.config = MagicMock()
    hass.config.latitude = 59.3
    hass.config.config_dir = "/tmp/test"
    # A real Store computes its path via hass.config.path(); an unstubbed MagicMock there
    # makes os.makedirs create a literal ./MagicMock/... directory in the repo root.
    hass.config.path = MagicMock(side_effect=lambda *parts: "/".join(("/tmp/test", *parts)))
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
    return hass


def _make_coordinator(decision: OptimizationDecision) -> EffektGuardCoordinator:
    """Coordinator past the startup grace period, one update from `decision`."""
    hass = _make_minimal_hass()

    nibe_data = SimpleNamespace(
        indoor_temp=21.0,
        outdoor_temp=0.0,
        flow_temp=32.0,
        degree_minutes=-150.0,
        current_offset=4.0,
        compressor_hz=40,
        timestamp=datetime.now(timezone.utc),
        power_kw=2.0,
        is_heating=True,
        is_hot_water=False,
        dhw_top_temp=None,
        dhw_amount_minutes=None,
        phase1_current=None,
        phase2_current=None,
        phase3_current=None,
    )
    nibe = MagicMock()
    nibe.get_current_state = AsyncMock(return_value=nibe_data)
    nibe.set_curve_offset = AsyncMock(side_effect=lambda offset, **_: round(offset))
    nibe.power_sensor_entity = None
    nibe._power_sensor_entity = None

    engine = MagicMock()
    engine.calculate_decision = MagicMock(return_value=decision)
    engine.price = MagicMock()
    engine.price.get_current_classification = MagicMock(return_value="normal")

    effect = MagicMock()
    effect.async_save = AsyncMock()

    coordinator = EffektGuardCoordinator(
        hass=hass,
        nibe_adapter=nibe,
        gespot_adapter=MagicMock(get_prices=AsyncMock(return_value=None)),
        weather_adapter=MagicMock(get_forecast=AsyncMock(return_value=None)),
        decision_engine=engine,
        effect_manager=effect,
        entry=MagicMock(data={"enable_optimization": True}, options={}),
    )

    # Defeat the startup grace period: control actions are live
    coordinator._startup_grace_timeout = dt_util.now() - timedelta(seconds=5)
    coordinator._startup_update_count = coordinator._startup_grace_updates

    # Avoid unrelated side-effects
    coordinator._update_peak_tracking = AsyncMock()
    coordinator._record_learning_observations = AsyncMock()
    coordinator._schedule_aligned_refresh = MagicMock()

    # History exactly as in the live repro: offset went 0 -> +4 just now,
    # so a drop back to 0 is a large opposite-direction change within the
    # volatility window
    coordinator._offset_volatility_tracker.record_change(0.0, "baseline")
    coordinator._offset_volatility_tracker.record_change(4.0, "raise")

    return coordinator


@pytest.mark.asyncio
async def test_manual_reduction_applies_immediately_after_raise():
    """force_offset(0) right after +4°C must be applied, not deferred."""
    manual = OptimizationDecision(
        offset=0.0,
        reasoning="Manual override active: 0.0°C",
        is_manual_override=True,
    )
    coordinator = _make_coordinator(manual)

    await coordinator._drive_the_pump()

    assert coordinator.current_offset == 0.0
    coordinator.nibe.set_curve_offset.assert_awaited_with(0.0, force_write=False)
    # The tracker adopted the manual value as the new baseline
    assert coordinator._offset_volatility_tracker.last_offset == 0.0


@pytest.mark.asyncio
async def test_automatic_reversal_still_blocked():
    """The volatility blocker must keep protecting against AUTOMATIC
    rapid reversals - only user commands bypass it."""
    automatic = OptimizationDecision(
        offset=0.0,
        reasoning="price reduction",
        is_manual_override=False,
    )
    coordinator = _make_coordinator(automatic)

    await coordinator._drive_the_pump()

    # Blocked: previous +4°C retained
    assert coordinator.current_offset == 4.0

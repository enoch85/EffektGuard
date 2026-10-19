"""A hot-water cycle NIBE ran while Home Assistant was down is read back from the recorder.

The restore used to run at start-up, after discovering NIBE's entities only if the entity cache was
empty. The adapter seeds that cache with the configured offset and degree-minute entities, so it was
never empty, BT7 was never bound there, and the restore never ran. It now runs once, on the first
cycle that reads BT7.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.decision_engine import OptimizationDecision

from tests.conftest import create_mock_hass

START = datetime(2026, 10, 17, 15, 40, tzinfo=timezone.utc)
BT7 = "sensor.f750_hot_water_top_bt7"


def _coordinator() -> EffektGuardCoordinator:
    cycle = iter(range(3))

    async def read():
        return SimpleNamespace(
            indoor_temp=21.0,
            outdoor_temp=8.0,
            flow_temp=30.0,
            degree_minutes=-100.0,
            compressor_hz=40,
            compressor_hz_max=None,
            is_hot_water=False,
            timestamp=START + timedelta(minutes=5 * next(cycle)),
            dhw_top_temp=50.0,
            dhw_amount_minutes=None,
            phase1_current=None,
            phase2_current=None,
            phase3_current=None,
            power_kw=None,
            current_offset=0.0,
        )

    nibe = MagicMock()
    nibe.get_current_state = AsyncMock(side_effect=read)
    nibe.is_enhanced_ventilation_active = AsyncMock(return_value=False)
    # What the adapter holds once its first read bound BT7, next to the configured overrides
    nibe.entity_cache = {
        "offset": "number.f750_heating_offset",
        "degree_minutes": "sensor.f750_degree_minutes",
        "dhw_top_temp": BT7,
    }
    engine = MagicMock()
    engine.calculate_decision = MagicMock(
        return_value=OptimizationDecision(offset=0.0, reasoning="test", layers=[])
    )
    effect = MagicMock()
    effect.async_save = AsyncMock()
    coordinator = EffektGuardCoordinator(
        hass=create_mock_hass(),
        nibe_adapter=nibe,
        gespot_adapter=MagicMock(get_prices=AsyncMock(return_value=None)),
        weather_adapter=MagicMock(get_forecast=AsyncMock(return_value=None)),
        decision_engine=engine,
        effect_manager=effect,
        entry=MagicMock(data={"enable_optimization": True}, options={}),
    )
    coordinator._update_peak_tracking = AsyncMock()
    coordinator._record_learning_observations = AsyncMock()
    coordinator._schedule_aligned_refresh = MagicMock()
    coordinator.dhw_optimizer.initialize_from_history = AsyncMock()
    return coordinator


@pytest.mark.asyncio
async def test_the_recorder_is_read_once_bt7_is_bound():
    coordinator = _coordinator()
    for _ in range(3):
        await coordinator._drive_the_pump()

    coordinator.dhw_optimizer.initialize_from_history.assert_awaited_once_with(
        coordinator.hass, BT7
    )

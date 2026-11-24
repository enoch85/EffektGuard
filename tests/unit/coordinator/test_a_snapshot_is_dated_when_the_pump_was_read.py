"""The predictor's snapshot carries the time the pump was read, not the time the cycle ended.

The predictor counts its history in control cycles and allows one missed cycle. A cycle reads the
pump, decides in an executor, writes, and only then records. Stamped when it recorded, a cycle that
read at 10:00:10 and ended at 10:05:20 was filed under the 10:05:10 cycle, which never ran (the
next aligned cycle is 10:10:10): the missed cycle disappeared from the history, and a day with two
missed cycles passed as one with one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.decision_engine import OptimizationDecision

from tests.conftest import create_mock_hass

READ_AT = datetime(2026, 1, 15, 10, 0, 10, tzinfo=timezone.utc)  # a cycle start
ENDED_AT = READ_AT + timedelta(minutes=5, seconds=10)  # past the next cycle's start


@pytest.mark.asyncio
async def test_a_cycle_that_ends_late_dates_its_snapshot_at_the_read(freezer):
    freezer.move_to(READ_AT)

    async def read():
        return NibeState(
            outdoor_temp=-5.0,
            indoor_temp=21.0,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-150.0,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=dt_util.utcnow(),
        )

    def slow_decision(*_args, **_kwargs):
        freezer.move_to(ENDED_AT)
        return OptimizationDecision(offset=0.0, reasoning="slow")

    nibe = MagicMock()
    nibe.get_current_state = AsyncMock(side_effect=read)
    nibe.is_enhanced_ventilation_active = AsyncMock(return_value=False)
    nibe.set_curve_offset = AsyncMock(return_value=0)
    coordinator = EffektGuardCoordinator(
        hass=create_mock_hass(),
        nibe_adapter=nibe,
        gespot_adapter=MagicMock(get_prices=AsyncMock(return_value=None)),
        weather_adapter=MagicMock(get_forecast=AsyncMock(return_value=None)),
        decision_engine=MagicMock(calculate_decision=MagicMock(side_effect=slow_decision)),
        effect_manager=MagicMock(
            async_save=AsyncMock(),
            get_monthly_peak_summary=MagicMock(return_value={"count": 0, "highest": 0.0}),
        ),
        entry=MagicMock(data={"enable_optimization": True}, options={}),
    )
    coordinator._startup_grace_timeout = READ_AT - timedelta(seconds=5)
    coordinator._startup_update_count = coordinator._startup_grace_updates
    coordinator._update_peak_tracking = AsyncMock()
    coordinator._save_learned_data = AsyncMock()
    coordinator._save_thermal_predictor_immediate = AsyncMock()
    coordinator._schedule_aligned_refresh = MagicMock()

    await coordinator._drive_the_pump(scheduled=True)

    assert dt_util.utcnow() == ENDED_AT, "the cycle must really end after the next one was due"
    [snapshot] = coordinator.thermal_predictor.state_history
    assert snapshot.timestamp == READ_AT, (
        f"The pump was read at {READ_AT:%H:%M:%S} and the snapshot is dated "
        f"{snapshot.timestamp:%H:%M:%S}: it is filed under a cycle that never read the pump."
    )

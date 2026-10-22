"""A read-only refresh decides and publishes; it must not drive the pump, now or later.

Reads come from anything: `homeassistant.update_entity` on an EffektGuard entity, the power-sensor
listener, `boost_dhw`, `reset_peak_tracking`. Each one decided and recorded its decision in the
volatility tracker as if it had been applied. The next control cycle's reversal check compared
against that never-written offset, and when it refused, it wrote the tracker's offset: the pump went
to an offset only a read had decided (runtime review F2, reproduced end to end on a template
register). A read in start-up observation also counted as an observation cycle.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.decision_engine import OptimizationDecision
from custom_components.effektguard.utils.volatile_helpers import OffsetVolatilityTracker

from tests.conftest import create_mock_hass

START = datetime(2026, 1, 15, 10, 0, 10, tzinfo=timezone.utc)


def _coordinator(proposals: list[float], *, past_startup: bool = True):
    """A coordinator whose engine proposes `proposals` in turn, and the pump it drives."""
    clock = {"now": START}
    pump = {"offset": 0}

    async def read():
        return NibeState(
            outdoor_temp=-5.0,
            indoor_temp=21.0,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-150.0,
            current_offset=float(pump["offset"]),
            is_heating=True,
            is_hot_water=False,
            timestamp=clock["now"],
        )

    async def write(offset, **_):
        pump["offset"] = int(offset)
        return pump["offset"]

    nibe = MagicMock()
    nibe.get_current_state = AsyncMock(side_effect=read)
    nibe.is_enhanced_ventilation_active = AsyncMock(return_value=False)
    nibe.set_curve_offset = AsyncMock(side_effect=write)
    engine = MagicMock()
    engine.calculate_decision = MagicMock(
        side_effect=lambda *a, **k: OptimizationDecision(
            offset=proposals.pop(0), reasoning="proposed"
        )
    )
    coordinator = EffektGuardCoordinator(
        hass=create_mock_hass(),
        nibe_adapter=nibe,
        gespot_adapter=MagicMock(get_prices=AsyncMock(return_value=None)),
        weather_adapter=MagicMock(get_forecast=AsyncMock(return_value=None)),
        decision_engine=engine,
        effect_manager=MagicMock(
            async_save=AsyncMock(),
            get_monthly_peak_summary=MagicMock(return_value={"count": 0, "highest": 0.0}),
        ),
        entry=MagicMock(data={"enable_optimization": True}, options={}),
    )
    coordinator._startup_grace_timeout = dt_util.now() - timedelta(seconds=5)
    if past_startup:
        coordinator._startup_update_count = coordinator._startup_grace_updates
    coordinator._offset_volatility_tracker = OffsetVolatilityTracker(
        time_source=lambda: clock["now"].timestamp()
    )
    coordinator._update_peak_tracking = AsyncMock()
    coordinator._record_learning_observations = AsyncMock()
    coordinator._save_learned_data = AsyncMock()
    coordinator._schedule_aligned_refresh = MagicMock()
    return coordinator, pump, clock


@pytest.mark.asyncio
async def test_an_offset_only_a_read_decided_never_reaches_the_pump():
    coordinator, pump, clock = _coordinator([1.0, 3.0, 1.0])

    await coordinator._drive_the_pump()  # control cycle A: +1
    assert pump["offset"] == 1

    clock["now"] += timedelta(minutes=2)
    await coordinator._async_update_data()  # a read decides +3 and writes nothing
    assert pump["offset"] == 1

    clock["now"] += timedelta(minutes=3)
    data = await coordinator._drive_the_pump()  # control cycle B: +1 again

    assert pump["offset"] == 1, (
        f"Cycle B proposed +1 and the register holds {pump['offset']}: the read's +3 became the "
        f"tracker's baseline, B's +1 was refused as a reversal from it, and the refusal wrote +3 "
        f"({data['decision'].reasoning!r})."
    )


@pytest.mark.asyncio
async def test_a_read_publishes_what_a_control_cycle_would_and_records_nothing():
    """Held as a control cycle would hold it, so the display matches the register."""
    coordinator, pump, clock = _coordinator([3.0, 0.0])
    await coordinator._drive_the_pump()  # +3 applied and recorded
    tracker = coordinator._offset_volatility_tracker
    recorded = tracker._last_change

    clock["now"] += timedelta(minutes=2)
    data = await coordinator._async_update_data()  # a read proposes 0: a reversal of 3

    assert data["decision"].offset == 3.0
    assert "reversal blocked" in data["decision"].reasoning
    assert tracker._last_change is recorded, "the read re-stamped the gate's 45-minute window"
    assert pump["offset"] == 3


@pytest.mark.asyncio
async def test_a_read_is_not_a_start_up_observation_cycle():
    coordinator, pump, clock = _coordinator([0.5] * 4, past_startup=False)

    for _ in range(3):
        await coordinator._async_update_data()
    assert coordinator._startup_update_count == 0, "three reads ended the observation early"

    await coordinator._drive_the_pump()
    assert coordinator._startup_update_count == 1

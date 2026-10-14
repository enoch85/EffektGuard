"""The wear guard's trigger is fed by the cycles that drive the pump, and the engine gets its verdict.

CompressorSaturation reads the compressor's run one pump-driving cycle at a time. A read-only refresh
- anything may trigger one, unserialised with the control loop - is not a sample of that run, so it
must not feed the tracker. The coordinator tells it about hot water and the ventilation switch, and
hands the engine the verdict, or the guard never fires.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.effektguard.const import COMPRESSOR_HZ_MAX
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.decision_engine import OptimizationDecision

from tests.conftest import create_mock_hass

START = datetime(2026, 1, 15, 2, 0, tzinfo=timezone.utc)
CYCLES = 5


def _coordinator(hot_water: bool, enhanced) -> EffektGuardCoordinator:
    """A real coordinator and tracker; the pump at its inverter maximum, DM holding."""
    cycle = iter(range(CYCLES))
    now = {"cycle": 0}

    async def read():
        now["cycle"] = next(cycle)
        return SimpleNamespace(
            indoor_temp=21.0,
            outdoor_temp=-10.0,
            flow_temp=40.0,
            degree_minutes=-300.0,
            compressor_hz=COMPRESSOR_HZ_MAX,
            is_hot_water=hot_water,
            timestamp=START + timedelta(minutes=5 * now["cycle"]),
            dhw_top_temp=None,
            phase1_current=None,
            phase2_current=None,
            phase3_current=None,
            power_kw=None,
            current_offset=0.0,
        )

    nibe = MagicMock()
    nibe.get_current_state = AsyncMock(side_effect=read)
    nibe.is_enhanced_ventilation_active = AsyncMock(side_effect=lambda: enhanced(now["cycle"]))
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
    return coordinator


async def _verdicts(read_only=False, hot_water=False, enhanced=lambda cycle: False) -> list[bool]:
    """The saturation verdict the engine was handed on each of five cycles, five minutes apart."""
    coordinator = _coordinator(hot_water, enhanced)
    verdicts = []
    for _ in range(CYCLES):
        if read_only:
            await coordinator._async_update_data()
        else:
            await coordinator._drive_the_pump()
        verdicts.append(coordinator.engine.calculate_decision.call_args.args[-1])
    return verdicts


@pytest.mark.asyncio
async def test_the_cycles_that_drive_the_pump_feed_it_and_the_engine_gets_the_verdict():
    """Flat out with DM holding: saturated once half a window has been read."""
    assert await _verdicts() == [False, False, False, True, True]


@pytest.mark.asyncio
async def test_a_read_only_refresh_does_not_feed_it():
    assert await _verdicts(read_only=True) == [False] * CYCLES


@pytest.mark.asyncio
async def test_hot_water_pauses_it():
    assert await _verdicts(hot_water=True) == [False] * CYCLES


@pytest.mark.asyncio
async def test_the_ventilation_switch_turning_ends_the_run():
    """Enhanced ventilation lifts an exhaust-air pump's ceiling: the run before it proves nothing."""
    verdicts = await _verdicts(enhanced=lambda cycle: cycle == CYCLES - 1)

    assert verdicts[-2:] == [True, False]

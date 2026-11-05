"""The volatility gate does not hold back the wear guard's cap, or the heater guard's cut.

The gate defers an offset reversal of 2 °C or more toward zero for 45 minutes, so that price-driven
votes cannot flip the curve faster than the compressor can follow. A raise into a saturated
compressor is not that: kept on the register it buys no compressor heat, only a deeper degree-minute
deficit and the immersion heater sooner (F-124). On the live HA bench - an F750 at 120 Hz, DM -1400 -
the engine capped +3 to 0 and the register stayed at +3.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    COMPRESSOR_HZ_MAX,
    TOLERANCE_RANGE_MULTIPLIER,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.decision_engine import OptimizationDecision
from custom_components.effektguard.utils.volatile_helpers import OffsetVolatilityTracker

from tests.conftest import create_mock_hass

START = datetime(2026, 1, 15, 2, 0, tzinfo=timezone.utc)
CYCLE = timedelta(minutes=UPDATE_INTERVAL_MINUTES)
CYCLES = 5
TARGET = 21.0
TOLERANCE = 0.5


async def _registers(engine, register: int = 0, cycles: int = CYCLES) -> list[int]:
    """The register after each pump-driving cycle, five minutes apart, past start-up: the F750 at
    its inverter maximum, DM holding at -1400, the room at the top of its band."""
    clock = {"now": START}
    pump = {"offset": register}

    async def read():
        return NibeState(
            outdoor_temp=-12.0,
            indoor_temp=TARGET + TOLERANCE * TOLERANCE_RANGE_MULTIPLIER,
            supply_temp=40.0,
            return_temp=35.0,
            degree_minutes=-1400.0,
            current_offset=float(pump["offset"]),
            is_heating=True,
            is_hot_water=False,
            timestamp=clock["now"],
            compressor_hz=COMPRESSOR_HZ_MAX,
            power_kw=3.0,
        )

    async def write(offset, **_):
        pump["offset"] = int(offset)
        return pump["offset"]

    nibe = MagicMock()
    nibe.get_current_state = AsyncMock(side_effect=read)
    nibe.is_enhanced_ventilation_active = AsyncMock(return_value=False)
    nibe.set_curve_offset = AsyncMock(side_effect=write)
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
    coordinator._startup_update_count = coordinator._startup_grace_updates
    coordinator._offset_volatility_tracker = OffsetVolatilityTracker(
        time_source=lambda: clock["now"].timestamp()
    )
    coordinator._update_peak_tracking = AsyncMock()
    coordinator._record_learning_observations = AsyncMock()
    coordinator._save_learned_data = AsyncMock()
    coordinator._schedule_aligned_refresh = MagicMock()

    registers = []
    for _ in range(cycles):
        await coordinator._drive_the_pump()
        registers.append(pump["offset"])
        clock["now"] += CYCLE
    return registers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("guard", "taken_back_to"),
    [
        pytest.param("saturation_capped", 0.0, id="the-wear-guard-withdraws-the-raise"),
        pytest.param("heater_capped", -2.0, id="the-heater-guard-cuts-past-zero"),
    ],
)
async def test_a_guard_taking_back_a_raise_reaches_the_register_at_once(guard, taken_back_to):
    """The engine raises the pump for three cycles, then a guard takes the raise back - fifteen
    minutes after it, well inside the window in which the gate holds a reversal of 2 C or more."""
    engine = MagicMock()
    engine.calculate_decision = MagicMock(
        side_effect=[OptimizationDecision(offset=3.0, reasoning="a raise")] * 3
        + [OptimizationDecision(offset=taken_back_to, reasoning="capped", **{guard: True})] * 2
    )

    registers = await _registers(engine)

    assert registers == [
        3,
        3,
        3,
        taken_back_to,
        taken_back_to,
    ], f"the gate kept a raise a guard took back: {registers}"


@pytest.mark.asyncio
async def test_only_the_withdrawal_passes():
    """A cut on the register is not a raise to withdraw. Lifting it onto a saturated compressor at
    once would raise S1 where the supply cannot follow - the deficit the cap exists to stop - so the
    gate still holds it."""
    engine = MagicMock()
    engine.calculate_decision = MagicMock(
        return_value=OptimizationDecision(offset=0.0, reasoning="capped", saturation_capped=True)
    )

    assert await _registers(engine, register=-4, cycles=1) == [-4]

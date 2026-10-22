"""One hot-water charge of the tank is one observation of its heating rate, through the control cycle.

The rate was learned from the BT7 curve wherever it was asked for - three times a cycle, so one
charge was 36 observations an hour (review DHW F11) - and then once per rise, which a reading that
pauses mid-charge split in two. A charge is now the pump's own: its priority is hot water.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from freezegun import freeze_time

from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.decision_engine import OptimizationDecision
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer

from tests.conftest import create_mock_hass

START = datetime(2026, 10, 17, 15, 40, tzinfo=timezone.utc)
CYCLE = timedelta(minutes=5)
# BT7 in the owner's F750's 0.5-1 C steps: 44 -> 51 C in 45 minutes, one reading repeated mid-way.
CHARGE = [44.0, 45.0, 46.0, 47.0, 48.0, 48.0, 49.0, 50.0, 51.0]


def _coordinator(readings: list[tuple[float, bool]]) -> EffektGuardCoordinator:
    cycle = iter(readings)

    async def read():
        tank, hot_water = next(cycle)
        return SimpleNamespace(
            indoor_temp=21.0,
            outdoor_temp=8.0,
            flow_temp=30.0,
            degree_minutes=-100.0,
            compressor_hz=40,
            compressor_hz_max=None,
            is_hot_water=hot_water,
            timestamp=START,
            dhw_top_temp=tank,
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
    nibe.entity_cache = {}
    detector = ClimateZoneDetector(latitude=59.33)
    engine = MagicMock()
    engine.climate_detector = detector
    engine.emergency_layer = EmergencyLayer(detector, heating_type="radiator")
    engine.price = PriceAnalyzer()
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


@pytest.mark.asyncio
async def test_a_charge_is_one_observation_however_its_readings_fall():
    # Idle, the charge, then an hour back on space heating.
    readings = [(46.0, False)] * 3 + [(t, True) for t in CHARGE] + [(51.0, False)] * 12
    coordinator = _coordinator(readings)

    with freeze_time(START) as clock:
        for _ in readings:
            await coordinator._drive_the_pump()
            clock.tick(CYCLE)

    scheduler = coordinator.dhw_optimizer
    assert (
        scheduler.heating_rate_observations == 1
    ), f"one charge became {scheduler.heating_rate_observations} observations of the heating rate"
    # 44 -> 51 C from the first hot-water reading to the first one after it: 45 minutes.
    assert scheduler.learned_heating_rate == pytest.approx(7.0 / 0.75)

"""A hot-water boost EffektGuard started is stopped when the optimiser can no longer decide on it.

The optimiser's boost runs while the optimiser decides to heat, and the coordinator applied DHW
control only inside the block that has a tank reading. With BT7 unavailable, stale or unreadable
the optimiser decided nothing and a boost it had started was neither abort-checked nor stopped:
it ran unsupervised until BT7 came back or NIBE's own lux timeout (review DHW F12). Without a
decision the boost is now stopped, and nothing new is started.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from freezegun import freeze_time

from homeassistant.const import STATE_OFF, STATE_ON

from custom_components.effektguard.const import CONF_NIBE_TEMP_LUX_ENTITY
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.decision_engine import OptimizationDecision
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer

from tests.conftest import create_mock_hass

NOW = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
LUX = "switch.temporary_lux_50004"


def _coordinator(pump: dict, switch_calls: list) -> EffektGuardCoordinator:
    async def read():
        return SimpleNamespace(
            indoor_temp=21.0,
            outdoor_temp=0.0,
            flow_temp=35.0,
            degree_minutes=-200.0,
            compressor_hz=None,
            is_hot_water=False,
            timestamp=NOW,
            dhw_top_temp=pump["bt7"],
            dhw_amount_minutes=None,
            power_kw=None,
            current_offset=0.0,
        )

    async def switch(domain, service, data, blocking=False):
        switch_calls.append(service)
        pump["lux"] = STATE_ON if service == "turn_on" else STATE_OFF

    hass = create_mock_hass()
    hass.services.async_call = AsyncMock(side_effect=switch)
    hass.states.get = MagicMock(
        side_effect=lambda entity_id: (
            SimpleNamespace(state=pump["lux"], last_changed=NOW) if entity_id == LUX else None
        )
    )
    nibe = MagicMock(entity_cache={})
    nibe.get_current_state = AsyncMock(side_effect=read)
    nibe.set_curve_offset = AsyncMock(return_value=0)
    nibe.is_enhanced_ventilation_active = AsyncMock(return_value=False)
    detector = ClimateZoneDetector(latitude=59.33)
    engine = MagicMock(
        climate_detector=detector,
        emergency_layer=EmergencyLayer(detector, heating_type="radiator"),
        price=PriceAnalyzer(),
    )
    engine.calculate_decision = MagicMock(
        return_value=OptimizationDecision(offset=0.0, reasoning="test", layers=[])
    )
    coordinator = EffektGuardCoordinator(
        hass=hass,
        nibe_adapter=nibe,
        gespot_adapter=MagicMock(get_prices=AsyncMock(return_value=None)),
        weather_adapter=MagicMock(get_forecast=AsyncMock(return_value=None)),
        decision_engine=engine,
        effect_manager=MagicMock(async_save=AsyncMock()),
        entry=MagicMock(
            data={
                "enable_optimization": True,
                "enable_hot_water_optimization": True,
                "target_indoor_temp": 21.0,
                CONF_NIBE_TEMP_LUX_ENTITY: LUX,
            },
            options={},
        ),
    )
    coordinator._update_peak_tracking = AsyncMock()
    coordinator._record_learning_observations = AsyncMock()
    coordinator._save_learned_data = AsyncMock()
    coordinator._schedule_aligned_refresh = MagicMock()
    coordinator._startup_grace_timeout = NOW - timedelta(hours=1)
    coordinator._startup_update_count = coordinator._startup_grace_updates
    coordinator.dhw_optimizer.last_legionella_boost = NOW - timedelta(days=2)
    return coordinator


@pytest.mark.asyncio
async def test_our_boost_is_stopped_when_the_tank_can_no_longer_be_read():
    pump = {"bt7": 40.0, "lux": STATE_OFF}
    switch_calls: list[str] = []
    with freeze_time(NOW) as clock:
        coordinator = _coordinator(pump, switch_calls)
        coordinator.data = await coordinator._drive_the_pump()
        assert switch_calls == ["turn_on"], "precondition: the optimiser started its boost"

        pump["bt7"] = None  # BT7 unavailable, stale or unreadable from here on
        for _ in range(3):
            clock.tick(timedelta(minutes=5))
            coordinator.data = await coordinator._drive_the_pump()

    assert switch_calls == ["turn_on", "turn_off"], (
        f"the tank stopped reporting fifteen minutes ago and the boost EffektGuard started is still "
        f"on (switch calls {switch_calls}): nothing decides on it any more"
    )

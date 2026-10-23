"""A room cooling fast stops a running hot-water boost, as it refuses a new one.

`calculate_recommendation` blocks hot water when the room is more than DHW_TREND_DEFICIT_THRESHOLD
below target and falling faster than DHW_TREND_RATE_THRESHOLD. It said so with `decision=None`,
and the coordinator applies DHW control only on a decision - so for as long as the room cooled, a
boost already running was neither abort-checked nor stopped (review DHW F6). A hot-water run in
the cold is one of the things that cools the room.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from homeassistant.const import STATE_ON

from custom_components.effektguard.const import (
    DHW_TREND_DEFICIT_THRESHOLD,
    DHW_TREND_RATE_THRESHOLD,
)
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer

NOW = datetime(2026, 1, 15, 10, 30, tzinfo=ZoneInfo("Europe/Stockholm"))
TARGET = 21.0
LUX = "switch.temporary_lux_50004"


def _recommendation():
    detector = ClimateZoneDetector(latitude=59.33)
    scheduler = IntelligentDHWScheduler(
        climate_detector=detector,
        emergency_layer=EmergencyLayer(detector, heating_type="radiator"),
    )
    return scheduler.calculate_recommendation(
        current_dhw_temp=45.0,
        thermal_debt=-200.0,
        space_heating_demand=1.0,
        outdoor_temp=-5.0,
        indoor_temp=TARGET - DHW_TREND_DEFICIT_THRESHOLD - 0.2,
        target_indoor=TARGET,
        price_classification="cheap",
        current_time=NOW,
        price_periods=None,
        thermal_trend_rate=DHW_TREND_RATE_THRESHOLD - 0.2,
    )


def _coordinator_with_our_boost_running():
    """Duck-typed, as in test_dhw_safety_stop_not_rate_limited: the real control and switch door."""
    coordinator = MagicMock()
    coordinator.temp_lux_entity = LUX
    coordinator._last_dhw_control_time = NOW - timedelta(minutes=5)
    coordinator.last_update_success = True
    coordinator.data = {"dhw_planning": {"thermal_debt": -200.0, "indoor_temperature": 20.5}}
    coordinator.entry.options = {}
    coordinator.entry.data = {"target_indoor_temp": TARGET}
    lux = MagicMock()
    lux.state = STATE_ON
    coordinator.hass.states.get.return_value = lux
    coordinator.hass.services.async_call = AsyncMock()
    coordinator._shutdown_requested = False
    coordinator._service_boost_until = None
    coordinator._lux_boost_is_ours = True
    # Started at the last control action, five minutes ago: inside its minimum run, which only
    # safety cuts short - and a room cooling fast is safety.
    coordinator._lux_boost_started_at = coordinator._last_dhw_control_time
    coordinator._set_temporary_lux = lambda on: EffektGuardCoordinator._set_temporary_lux(
        coordinator, on
    )
    return coordinator


@pytest.mark.asyncio
async def test_a_room_cooling_fast_stops_a_running_boost():
    recommendation = _recommendation()
    decision = recommendation.decision

    assert decision is not None, (
        f"{recommendation.details['priority_reason']} gave the coordinator no decision, and it "
        f"applies DHW control only on one: a running boost is neither abort-checked nor stopped"
    )
    assert not decision.should_heat
    assert decision.recommended_start_time is not None

    coordinator = _coordinator_with_our_boost_running()
    await EffektGuardCoordinator._apply_dhw_control(coordinator, decision, 45.0, NOW)

    calls = [call.args[1] for call in coordinator.hass.services.async_call.call_args_list]
    assert calls == ["turn_off"], f"the running boost was not stopped: switch calls {calls}"

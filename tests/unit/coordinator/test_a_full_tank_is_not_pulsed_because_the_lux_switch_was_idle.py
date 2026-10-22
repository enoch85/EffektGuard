"""A tank that needs nothing is not given a five-minute boost because the lux switch sat idle.

The 36-hour rule (RULE 3.5) heated at cheap or normal price when the temporary-lux switch had not
been ON for DHW_MAX_WAIT_HOURS, whatever the tank's temperature. Its trigger was the switch's own
history, so the boost it started reset it: on the next cycle the switch had been ON five minutes
ago, the rule no longer applied, and the rules below it decided as if it had never fired. Where
they would not heat, the boost was stopped - a five-minute lux pulse and a compressor start and
stop every 36 hours on a tank NIBE keeps hot by itself (review DHW F10). Where they would, they
would have started it on the previous cycle too. That is everything the rule could do, so it is
gone, and with it the recorder query that fed it on every cycle.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from freezegun import freeze_time

from homeassistant.const import STATE_OFF, STATE_ON

from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler
from custom_components.effektguard.optimization.prediction_layer import ThermalStatePredictor
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer

NOW = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)
LUX = "switch.temporary_lux_50004"
IDLE = timedelta(hours=40)
TANK = 60.0


def _coordinator(lux: dict) -> EffektGuardCoordinator:
    coordinator = EffektGuardCoordinator.__new__(EffektGuardCoordinator)
    coordinator.hass = MagicMock()
    coordinator.hass.states.get = MagicMock(
        side_effect=lambda entity_id: SimpleNamespace(
            state=lux["state"], last_changed=lux["changed"]
        )
    )

    async def switch(domain, service, data, blocking=False):
        lux["calls"].append(service)
        lux["state"] = STATE_ON if service == "turn_on" else STATE_OFF
        lux["changed"] = datetime.now(timezone.utc)

    coordinator.hass.services.async_call = AsyncMock(side_effect=switch)
    coordinator.entry = MagicMock(data={"target_indoor_temp": 21.0}, options={})
    coordinator.data = {}
    coordinator.last_update_success = True
    coordinator.temp_lux_entity = LUX
    coordinator._shutdown_requested = False
    coordinator._lux_boost_is_ours = False
    coordinator._service_boost_until = None
    coordinator._last_dhw_control_time = None
    coordinator._raise_dhw_control_issue = MagicMock()
    coordinator._clear_dhw_control_issue = MagicMock()
    coordinator.thermal_predictor = ThermalStatePredictor()
    detector = ClimateZoneDetector(latitude=59.33)
    coordinator.engine = MagicMock(climate_detector=detector)
    coordinator.dhw_optimizer = IntelligentDHWScheduler(
        climate_detector=detector,
        emergency_layer=EmergencyLayer(detector, heating_type="radiator"),
        user_target_temp=50.0,
    )
    coordinator.dhw_optimizer.last_legionella_boost = NOW - timedelta(days=2)
    return coordinator


@pytest.mark.asyncio
async def test_a_full_tank_gets_no_boost_after_a_long_idle_spell():
    lux = {"state": STATE_OFF, "changed": NOW - IDLE + timedelta(hours=1), "calls": []}
    coordinator = _coordinator(lux)
    nibe = SimpleNamespace(
        degree_minutes=-200.0,
        power_kw=1.0,
        outdoor_temp=0.0,
        indoor_temp=21.0,
        dhw_amount_minutes=None,
    )
    # The switch's recorded history, for any code that asks: last ON 40 hours ago.
    history = {
        LUX: [
            SimpleNamespace(state=STATE_ON, last_changed=NOW - IDLE),
            SimpleNamespace(state=STATE_OFF, last_changed=NOW - IDLE + timedelta(hours=1)),
        ]
    }
    recorder = MagicMock(async_add_executor_job=AsyncMock(return_value=history))

    with (
        freeze_time(NOW) as clock,
        patch("homeassistant.components.recorder.get_instance", return_value=recorder),
    ):
        for _ in range(2):
            when = datetime.now(timezone.utc)
            result = await coordinator._calculate_dhw_recommendation(nibe, None, None, TANK, when)
            await coordinator._apply_dhw_control(result.decision, TANK, when)
            clock.tick(timedelta(minutes=5))

    assert lux["calls"] == [], (
        f"a {TANK} C tank at a normal price was switched {lux['calls']} because the lux switch "
        f"had been idle for {IDLE}"
    )

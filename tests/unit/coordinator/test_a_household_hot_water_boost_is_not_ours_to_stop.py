"""A hot-water boost the household started is not the optimiser's to stop.

EffektGuard records whether it started the running temporary-lux boost (`_lux_boost_is_ours`),
and the unload cleanup stopped only its own. The control loop did not ask: any decision not to
heat, or an abort condition, turned off a lux the household had switched on at the pump's panel
or in MyUplink at the next cycle (review DHW F8). It now stops only what it started - and a lux
seen off ends its claim, so a household boost started later is not mistaken for its own.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from homeassistant.const import STATE_OFF, STATE_ON

from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler

NOW = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)
LUX = "switch.temporary_lux_50004"

WAIT = SimpleNamespace(
    should_heat=False, abort_conditions=[], priority_reason="DHW_ADEQUATE_WAITING_CHEAP_EXPENSIVE"
)
HEAT_BUT_ABORT = SimpleNamespace(
    should_heat=True, abort_conditions=["thermal_debt < -300"], priority_reason="DHW_COMFORT_LOW"
)


def _coordinator(lux: dict, *, ours: bool) -> EffektGuardCoordinator:
    coordinator = EffektGuardCoordinator.__new__(EffektGuardCoordinator)
    coordinator.hass = MagicMock()
    coordinator.hass.states.get = MagicMock(side_effect=lambda _: SimpleNamespace(state=lux["s"]))

    async def switch(domain, service, data, blocking=False):
        lux["calls"].append(service)
        lux["s"] = STATE_ON if service == "turn_on" else STATE_OFF

    coordinator.hass.services.async_call = AsyncMock(side_effect=switch)
    coordinator.entry = MagicMock(data={"target_indoor_temp": 21.0}, options={})
    coordinator.data = {"dhw_planning": {"thermal_debt": -400.0, "indoor_temperature": 21.0}}
    coordinator.last_update_success = True
    coordinator.temp_lux_entity = LUX
    coordinator._shutdown_requested = False
    coordinator._lux_boost_is_ours = ours
    coordinator._service_boost_until = None
    coordinator._last_dhw_control_time = NOW - timedelta(hours=2)
    coordinator._raise_dhw_control_issue = MagicMock()
    coordinator._clear_dhw_control_issue = MagicMock()
    coordinator.dhw_optimizer = IntelligentDHWScheduler()
    return coordinator


@pytest.mark.parametrize("decision", [WAIT, HEAT_BUT_ABORT], ids=["a_price_hold", "an_abort"])
@pytest.mark.asyncio
async def test_a_lux_the_household_switched_on_is_left_on(decision):
    lux = {"s": STATE_ON, "calls": []}
    coordinator = _coordinator(lux, ours=False)

    await coordinator._apply_dhw_control(decision, 48.0, NOW)

    assert lux["calls"] == [], (
        f"the household switched temporary lux on and the optimiser turned it off ({lux['calls']}) "
        f"on {decision.priority_reason}"
    )


@pytest.mark.asyncio
async def test_a_household_boost_after_ours_ended_is_still_the_households():
    """Ours was switched off by someone else; the household then switched lux on again."""
    lux = {"s": STATE_OFF, "calls": []}
    coordinator = _coordinator(lux, ours=True)

    await coordinator._apply_dhw_control(WAIT, 48.0, NOW)
    lux["s"] = STATE_ON
    await coordinator._apply_dhw_control(WAIT, 48.0, NOW + timedelta(minutes=5))

    assert lux["calls"] == [], "the household's boost was taken for the one EffektGuard had started"

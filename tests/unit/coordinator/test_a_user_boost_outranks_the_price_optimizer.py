"""A hot-water boost the USER commanded is not the price optimizer's to cancel.

`boost_dhw` records HOW LONG the user asked for, and while that window is open:
- the ordinary price-based stop path defers to it,
- the thermal-debt SAFETY abort still stops it (and closes the window),
- expiry stops it through the same owned door the unload cleanup uses,
- and `duration` therefore does something real, instead of being validated and discarded.

Only safety outranks the user; cost optimization does not.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from homeassistant.exceptions import HomeAssistantError

from custom_components.effektguard.coordinator import EffektGuardCoordinator

NOW = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)


def _coordinator(lux_state: str = "off") -> EffektGuardCoordinator:
    coordinator = EffektGuardCoordinator.__new__(EffektGuardCoordinator)
    coordinator.hass = MagicMock()
    coordinator.hass.services.async_call = AsyncMock()
    lux = MagicMock()
    lux.state = lux_state
    coordinator.hass.states.get = MagicMock(return_value=lux)
    coordinator.entry = MagicMock()
    coordinator.entry.data = {"target_indoor_temp": 21.0}
    coordinator.entry.options = {}
    coordinator.data = {}
    coordinator.last_update_success = True
    coordinator.temp_lux_entity = "switch.temporary_lux_50004"
    coordinator._shutdown_requested = False
    coordinator._lux_boost_is_ours = False
    coordinator._service_boost_until = None
    coordinator._last_dhw_control_time = NOW - timedelta(hours=2)
    coordinator.dhw_optimizer = MagicMock()
    coordinator._raise_dhw_control_issue = MagicMock()
    coordinator._clear_dhw_control_issue = MagicMock()
    return coordinator


def _stop_decision():
    """What the optimizer says when prices are high: stop heating water."""
    return SimpleNamespace(should_heat=False, abort_conditions=[], priority_reason="EXPENSIVE")


@pytest.mark.asyncio
async def test_the_price_stop_does_not_cancel_a_user_boost():
    coordinator = _coordinator(lux_state="off")
    await coordinator.async_start_dhw_boost(duration_minutes=60, now_time=NOW)
    assert coordinator._lux_boost_is_ours is True

    # Next cycle: lux is on, prices are high, the optimizer wants it off.
    coordinator.hass.states.get.return_value.state = "on"
    coordinator.hass.services.async_call.reset_mock()

    await coordinator._apply_dhw_control(_stop_decision(), 45.0, NOW + timedelta(minutes=5))

    coordinator.hass.services.async_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_boost_ends_when_its_duration_expires():
    coordinator = _coordinator(lux_state="off")
    await coordinator.async_start_dhw_boost(duration_minutes=60, now_time=NOW)

    coordinator.hass.states.get.return_value.state = "on"
    coordinator.hass.services.async_call.reset_mock()

    await coordinator._apply_dhw_control(_stop_decision(), 45.0, NOW + timedelta(minutes=61))

    coordinator.hass.services.async_call.assert_awaited_once()
    assert coordinator.hass.services.async_call.await_args.args[1] == "turn_off"
    assert coordinator._service_boost_until is None


@pytest.mark.asyncio
async def test_the_safety_abort_still_stops_a_user_boost():
    """Only safety outranks the user: deep thermal debt ends the boost, window and all."""
    coordinator = _coordinator(lux_state="on")
    coordinator._service_boost_until = NOW + timedelta(minutes=60)
    coordinator._lux_boost_is_ours = True
    coordinator.dhw_optimizer.check_abort_conditions = MagicMock(
        return_value=(True, "thermal debt DM -800")
    )

    decision = SimpleNamespace(
        should_heat=True, abort_conditions=["dm"], priority_reason="USER_BOOST"
    )
    await coordinator._apply_dhw_control(decision, 45.0, NOW + timedelta(minutes=5))

    coordinator.hass.services.async_call.assert_awaited_once()
    assert coordinator.hass.services.async_call.await_args.args[1] == "turn_off"
    assert coordinator._service_boost_until is None


@pytest.mark.asyncio
async def test_a_boost_is_refused_while_optimization_is_off():
    """OFF means safety monitoring only - it does not fire the immersion heater on request."""
    coordinator = _coordinator()
    coordinator.entry.data = {"enable_optimization": False}

    with pytest.raises(HomeAssistantError):
        await coordinator.async_start_dhw_boost(duration_minutes=60, now_time=NOW)

    coordinator.hass.services.async_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_unload_cleanup_closes_the_window_too():
    coordinator = _coordinator(lux_state="on")
    await coordinator.async_start_dhw_boost(duration_minutes=60, now_time=NOW)

    await coordinator._cancel_our_dhw_boost()

    assert coordinator._service_boost_until is None
    assert coordinator._lux_boost_is_ours is False


def test_the_service_no_longer_advertises_a_temperature_it_cannot_set():
    """Temporary lux is a switch: the pump owns the temperature. services.yaml must not lie."""
    from pathlib import Path

    import yaml

    services = yaml.safe_load(
        Path("custom_components/effektguard/services.yaml").read_text(encoding="utf-8")
    )
    fields = services["boost_dhw"].get("fields", {})
    assert "target_temp" not in fields
    assert "duration" in fields

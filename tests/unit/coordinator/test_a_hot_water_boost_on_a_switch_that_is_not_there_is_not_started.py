"""A hot-water boost on a lux switch that is missing or unavailable is not reported as started.

Home Assistant answers `homeassistant.turn_on` on such an entity with a log line ("Referenced
entities ... are missing or not currently available") and returns normally. The boost was then
claimed as ours, the boost_dhw window opened and the service reported it activated; in the automatic
path the 60-minute start limiter was stamped, so a boost due during a cloud outage counted as done.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.exceptions import HomeAssistantError
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import (
    _async_register_services,
    _check_service_cooldown,
    _create_coordinator,
    _service_last_called,
)
from custom_components.effektguard.const import (
    CONF_NIBE_TEMP_LUX_ENTITY,
    DHW_BOOST_COOLDOWN_MINUTES,
    DOMAIN,
    SERVICE_BOOST_DHW,
)

LUX = "switch.myuplink_temporary_lux"
NOW = datetime(2026, 1, 15, 10, 0, 10, tzinfo=timezone.utc)
DHW_TOP_TEMP = 45.0


@pytest.fixture
async def coordinator(hass):
    """The coordinator as setup builds it, with the myUplink lux switch configured."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "switch", {})
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_NIBE_TEMP_LUX_ENTITY: LUX}, options={})
    entry.add_to_hass(hass)
    coordinator = await _create_coordinator(hass, entry)
    hass.data[DOMAIN] = {entry.entry_id: coordinator}
    await _async_register_services(hass)
    _service_last_called.clear()
    yield coordinator
    await coordinator.async_shutdown()
    _service_last_called.clear()


@pytest.mark.parametrize("lux_state", [None, STATE_UNAVAILABLE], ids=["missing", "unavailable"])
async def test_boost_dhw_on_a_switch_it_cannot_reach_fails_and_holds_nothing(
    hass, coordinator, lux_state
):
    if lux_state is not None:
        hass.states.async_set(LUX, lux_state)

    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(DOMAIN, SERVICE_BOOST_DHW, {}, blocking=True)

    held = (coordinator._lux_boost_is_ours, coordinator._service_boost_until)
    assert held == (False, None), f"nothing was switched on, yet (ours, window) = {held}"
    retry_allowed, _ = _check_service_cooldown(SERVICE_BOOST_DHW, DHW_BOOST_COOLDOWN_MINUTES)
    assert retry_allowed, "the boost never started, and its cooldown refuses the retry"


async def test_a_due_boost_on_an_unavailable_switch_does_not_count_as_started(hass, coordinator):
    """The next cycle may try again: the start limiter runs from a boost that happened."""
    hass.states.async_set(LUX, STATE_UNAVAILABLE)
    coordinator.data = {}
    due = SimpleNamespace(should_heat=True, abort_conditions=[], priority_reason="CHEAP")

    await coordinator._apply_dhw_control(due, DHW_TOP_TEMP, NOW)

    held = (coordinator._lux_boost_is_ours, coordinator._last_dhw_control_time)
    assert held == (False, None), f"nothing was switched on, yet (ours, limiter) = {held}"

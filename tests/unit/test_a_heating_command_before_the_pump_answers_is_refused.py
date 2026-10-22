"""force_offset and boost_heating before the heat pump has answered fail, and leave nothing armed.

While the pump has not answered since start-up (myUplink still loading after a Home Assistant
restart), a cycle publishes placeholder data and waits. An explicit command took the same path: the
service reported success, consumed its cooldown and wrote nothing, and the override stayed armed for
the first ordinary cycle to apply once the pump appeared - without the command's forced write.
"""

from __future__ import annotations

import pytest
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import (
    _async_register_services,
    _check_service_cooldown,
    _create_coordinator,
    _service_last_called,
)
from custom_components.effektguard.const import (
    ATTR_DURATION,
    ATTR_OFFSET,
    CONF_NIBE_ENTITY,
    DOMAIN,
    HEATING_BOOST_COOLDOWN_MINUTES,
    MAX_OFFSET,
    SERVICE_BOOST_HEATING,
    SERVICE_FORCE_OFFSET,
    SERVICE_RATE_LIMIT_MINUTES,
)

OFFSET = "number.nibe_offset"  # configured, but its integration has not loaded yet
DURATION_MINUTES = 60


@pytest.fixture
async def coordinator(hass):
    """The coordinator as setup builds it, before any NIBE entity exists."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_NIBE_ENTITY: OFFSET}, options={})
    entry.add_to_hass(hass)
    coordinator = await _create_coordinator(hass, entry)
    hass.data[DOMAIN] = {entry.entry_id: coordinator}
    await _async_register_services(hass)
    _service_last_called.clear()
    yield coordinator
    await coordinator.async_shutdown()
    _service_last_called.clear()


@pytest.mark.parametrize(
    ("service", "data", "cooldown_minutes"),
    [
        pytest.param(
            SERVICE_FORCE_OFFSET,
            {ATTR_OFFSET: MAX_OFFSET, ATTR_DURATION: DURATION_MINUTES},
            SERVICE_RATE_LIMIT_MINUTES,
            id=SERVICE_FORCE_OFFSET,
        ),
        pytest.param(
            SERVICE_BOOST_HEATING,
            {ATTR_DURATION: DURATION_MINUTES},
            HEATING_BOOST_COOLDOWN_MINUTES,
            id=SERVICE_BOOST_HEATING,
        ),
    ],
)
async def test_a_heating_command_before_the_pump_answers_fails_and_arms_nothing(
    hass, coordinator, service, data, cooldown_minutes
):
    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(DOMAIN, service, data, blocking=True)

    assert coordinator.engine._check_manual_override() is None, (
        f"{service} failed, and its override is still armed for the first cycle after the pump "
        f"answers"
    )
    retry_allowed, _ = _check_service_cooldown(service, cooldown_minutes)
    assert retry_allowed, f"{service} wrote nothing, and its cooldown refuses the retry"

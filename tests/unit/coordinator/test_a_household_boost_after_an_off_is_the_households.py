"""A lux the household switches off and on again between two control cycles is the household's.

Ownership was read only at each control cycle, so an off and on in between left EffektGuard's claim
standing, and the household's new boost was switched off as EffektGuard's own: at the end of the
old boost_dhw window, or on the next reload.
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_mock_service

from custom_components.effektguard import _create_coordinator, async_setup_entry
from custom_components.effektguard.const import CONF_NIBE_TEMP_LUX_ENTITY, DOMAIN

LUX = "switch.temporary_lux_48132"


async def test_a_household_boost_after_an_off_is_left_on(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_NIBE_TEMP_LUX_ENTITY: LUX}, options={})
    entry.add_to_hass(hass)
    coordinator = await _create_coordinator(hass, entry)
    coordinator.async_config_entry_first_refresh = AsyncMock()
    with (
        patch(
            "custom_components.effektguard._create_coordinator",
            AsyncMock(return_value=coordinator),
        ),
        patch("custom_components.effektguard._async_register_services", AsyncMock()),
        patch.object(hass.config_entries, "async_forward_entry_setups", AsyncMock()),
    ):
        assert await async_setup_entry(hass, entry)

    # EffektGuard's boost_dhw, an hour to run
    hass.states.async_set(LUX, STATE_ON)
    coordinator._lux_boost_is_ours = True
    coordinator._service_boost_until = dt_util.utcnow() + timedelta(hours=1)

    # The household switches it off and on again before the next control cycle
    hass.states.async_set(LUX, STATE_OFF)
    await hass.async_block_till_done()
    hass.states.async_set(LUX, STATE_ON)
    await hass.async_block_till_done()

    turned_off = async_mock_service(hass, "homeassistant", "turn_off")
    await coordinator._end_user_dhw_boost_when_due(
        SimpleNamespace(degree_minutes=-100.0, indoor_temp=21.0),
        dt_util.utcnow() + timedelta(hours=2),  # the old window has ended
    )
    await coordinator._cancel_our_dhw_boost()  # a reload

    assert (
        turned_off == []
    ), f"the household's boost was switched off as EffektGuard's own {len(turned_off)} time(s)"

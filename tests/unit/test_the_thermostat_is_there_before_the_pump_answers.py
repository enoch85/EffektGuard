"""The thermostat is added while the pump is still starting, and follows it once it answers.

Until the pump's entities answer, the coordinator publishes its startup data - no decision, no
reading - and Home Assistant adds the platforms on it. The climate entity read the decision's
reasoning from that data, so adding it raised AttributeError and the thermostat was never added
(production, 2026-10-19 21:32, an HA start before myUplink had loaded; EG-01 of the 2026-10-20 log
investigation). Home Assistant's own entity platform adds it here, as at a restart.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from homeassistant.helpers.update_coordinator import UpdateFailed
from pytest_homeassistant_custom_component.common import MockConfigEntry, MockEntityPlatform

from custom_components.effektguard.climate import EffektGuardClimate
from custom_components.effektguard.const import DOMAIN
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.decision_engine import OptimizationDecision


async def test_the_thermostat_is_added_on_the_startup_data_and_follows_the_pump(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    nibe = MagicMock()
    nibe.get_current_state = AsyncMock(side_effect=UpdateFailed("required sensors unreadable"))
    coordinator = EffektGuardCoordinator(
        hass, nibe, MagicMock(), MagicMock(), MagicMock(), MagicMock(), entry
    )
    coordinator.data = await coordinator._async_update_data()
    coordinator._unsub_aligned_refresh()  # the retry it arms is not under test
    assert coordinator.data["startup_pending"] is True

    platform = MockEntityPlatform(hass, domain="climate", platform_name=DOMAIN)
    platform.config_entry = entry
    await platform.async_add_entities([EffektGuardClimate(coordinator, entry)])

    added = hass.states.get("climate.effektguard")
    assert added is not None, "Home Assistant could not add the thermostat on the startup data"
    assert added.state == "heat"
    assert added.attributes["current_temperature"] is None

    # The pump answers. The entity added at startup follows the coordinator, no reload.
    pump = SimpleNamespace(
        indoor_temp=21.3, outdoor_temp=-3.2, supply_temp=35.8, degree_minutes=-150
    )
    coordinator.async_set_updated_data(
        {"nibe": pump, "decision": OptimizationDecision(offset=1.0, reasoning="answered")}
    )

    following = hass.states.get("climate.effektguard")
    assert following.attributes["current_temperature"] == 21.3
    assert following.attributes["optimization_reasoning"] == "answered"

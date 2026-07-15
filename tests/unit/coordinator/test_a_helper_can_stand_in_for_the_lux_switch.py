"""A Modbus user's input_boolean helper is a valid temporary-lux actuator (issue #18).

MyUplink exposes temporary lux as a `switch`; nibe_heatpump and generic Modbus do not, so
those users bridge it with a helper + automation. The lux door hardcoded the `switch`
service domain, and the config flow only accepted `switch` entities - locking every
non-MyUplink install out of hot-water optimization for no reason: `homeassistant.turn_on`
/`turn_off` drive both domains through the same one door.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.effektguard.coordinator import EffektGuardCoordinator


def _coordinator(lux_entity: str) -> EffektGuardCoordinator:
    coordinator = EffektGuardCoordinator.__new__(EffektGuardCoordinator)
    coordinator.hass = MagicMock()
    coordinator.hass.services.async_call = AsyncMock()
    coordinator.temp_lux_entity = lux_entity
    coordinator._shutdown_requested = False
    coordinator._lux_boost_is_ours = False
    return coordinator


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lux_entity",
    ["switch.temporary_lux_50004", "input_boolean.nibe_temp_lux_bridge"],
)
async def test_the_lux_door_drives_any_toggleable_entity(lux_entity):
    coordinator = _coordinator(lux_entity)

    assert await coordinator._set_temporary_lux(True) is True

    call = coordinator.hass.services.async_call.await_args
    assert call.args[0] == "homeassistant", (
        f"The lux door called the {call.args[0]!r} service domain for {lux_entity}. An "
        f"input_boolean helper - the only bridge a Modbus/nibe_heatpump user has - does not "
        f"answer switch.turn_on; homeassistant.turn_on drives both."
    )
    assert call.args[1] == "turn_on"
    assert call.args[2] == {"entity_id": lux_entity}


def test_the_config_flow_accepts_a_helper_for_temporary_lux():
    import re
    from pathlib import Path

    source = Path("custom_components/effektguard/config_flow.py").read_text(encoding="utf-8")
    lux_selectors = re.findall(
        r"CONF_NIBE_TEMP_LUX_ENTITY[^)]*?EntitySelectorConfig\(domain=(\[[^\]]*\]|\"[a-z_]+\")",
        source,
        flags=re.DOTALL,
    )
    assert lux_selectors, "could not find the temp-lux entity selector in the config flow"
    for domains in lux_selectors:
        assert "input_boolean" in domains and "switch" in domains, (
            f"The temporary-lux selector accepts only {domains}. A nibe_heatpump/Modbus user "
            f"has no lux switch - their bridge is an input_boolean helper, and the selector "
            f"must let them pick it (issue #18)."
        )

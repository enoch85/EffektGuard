"""A hot-water boost EffektGuard's own service started must be recognised as ours on unload.

`_cancel_our_dhw_boost` turns off, on unload, only a temporary-lux boost EffektGuard started - told
apart from the owner's by `_lux_boost_is_ours`, which is set in exactly one place:
`_set_temporary_lux`. The `boost_dhw` service must reach the switch through that method (via the
coordinator), or the flag is never set and the boost it started is left running to NIBE's lux
timeout on the immersion heater after the entry unloads (reconfigure, reload, removal, restart).

The structural test pins the one-door invariant: every `switch.turn_on/off` on the lux entity goes
through `_set_temporary_lux`, the only place that records who started the boost.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.const import STATE_ON

from custom_components.effektguard import _async_register_services
from custom_components.effektguard.const import CONF_NIBE_TEMP_LUX_ENTITY, DOMAIN
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.effect_layer import EffectManager

LUX = "switch.temporary_lux_50004"


def _hass_and_coordinator() -> tuple[MagicMock, EffektGuardCoordinator, dict]:
    hass = MagicMock()
    hass.config.latitude = 59.33
    hass.config.longitude = 18.07
    hass.services.async_call = AsyncMock()
    hass.services.has_service = MagicMock(return_value=False)

    registered: dict = {}
    hass.services.async_register = MagicMock(
        side_effect=lambda domain, service, handler, **kw: registered.__setitem__(service, handler)
    )

    entry = MagicMock()
    entry.data = {CONF_NIBE_TEMP_LUX_ENTITY: LUX}
    entry.options = {}

    coordinator = EffektGuardCoordinator(
        hass, MagicMock(), MagicMock(), MagicMock(), MagicMock(), EffectManager(hass), entry
    )
    coordinator.data = {}
    coordinator.temp_lux_entity = LUX
    coordinator.learning_store = MagicMock()
    coordinator.learning_store.async_save = AsyncMock()
    coordinator.effect.async_save = AsyncMock()

    hass.data = {DOMAIN: {"entry_1": coordinator}}

    # The lux switch reads ON once a boost is running.
    lux_state = MagicMock()
    lux_state.state = STATE_ON
    hass.states.get = MagicMock(return_value=lux_state)

    return hass, coordinator, registered


def _turn_offs(hass) -> list:
    return [
        call
        for call in hass.services.async_call.await_args_list
        if call.args[0] == "homeassistant" and call.args[1] == "turn_off"
    ]


@pytest.mark.asyncio
async def test_a_boost_our_own_service_started_is_cancelled_on_unload():
    """The service starts the boost; the cleanup must recognise it as ours on unload."""
    hass, coordinator, registered = _hass_and_coordinator()
    await _async_register_services(hass)

    call = MagicMock()
    call.data = {}
    with patch.object(coordinator, "async_request_refresh", AsyncMock()):
        await registered["boost_dhw"](call)

    assert coordinator._lux_boost_is_ours is True, (
        "the effektguard.boost_dhw service turned the temporary-lux switch on and did not record "
        "that EffektGuard is the one who did it. `_cancel_our_dhw_boost` reads exactly that flag."
    )

    hass.services.async_call.reset_mock()
    await coordinator.async_shutdown()  # reconfigure / manual reload / removal / restart

    assert len(_turn_offs(hass)) == 1, (
        f"the integration unloaded and left a hot-water boost running that IT had started "
        f"({len(_turn_offs(hass))} turn_off calls). Nothing is left to stop it, so it runs to NIBE's "
        f"own temporary-lux timeout on the immersion heater at COP 1.0. The cleanup for this exists "
        f"and was simply never told the boost was ours."
    )


@pytest.mark.asyncio
async def test_a_boost_the_owner_started_is_left_alone():
    """The other half, and it is why the flag exists at all.

    A boost the HOUSEHOLD started - somebody pressed temporary lux on the pump, or in MyUplink -
    is not EffektGuard's to cancel. Unloading the integration must not switch off somebody's shower.
    """
    hass, coordinator, _ = _hass_and_coordinator()
    # Nobody called our service and the optimizer never ran: the switch is on, but not by us.
    assert coordinator._lux_boost_is_ours is False

    await coordinator.async_shutdown()

    assert _turn_offs(hass) == [], (
        "unloading EffektGuard cancelled a hot-water boost it did not start. That is the owner's "
        "boost, and taking it away is worse than leaving ours running."
    )


@pytest.mark.asyncio
async def test_a_shut_down_coordinator_cannot_start_a_boost():
    """The same race as the curve offset and the fan: an unloaded entry does not command the pump.

    Turning a boost OFF during shutdown must still work - that is the cleanup itself - so the guard
    can only refuse to START one.
    """
    hass, coordinator, _ = _hass_and_coordinator()
    await coordinator.async_shutdown()
    hass.services.async_call.reset_mock()

    started = await coordinator._set_temporary_lux(True)

    assert started is False
    assert hass.services.async_call.await_count == 0, (
        "a shut-down coordinator started a hot-water boost. The entry is unloaded, and nothing is "
        "left that would ever switch it off again."
    )


def test_there_is_exactly_one_door_to_the_hot_water_switch():
    """Every `switch.turn_on/off` on the temporary-lux entity must go through `_set_temporary_lux` -
    the only place that records who started the boost, which is what lets the unload cleanup tell
    ours from the owner's.
    """
    import ast
    import pathlib

    def commands_the_lux_switch(call: ast.Call) -> bool:
        """A `switch.turn_on/off` aimed at the TEMPORARY-LUX entity specifically.

        Scoped to the lux entity because the NIBE adapter also drives a different `switch` (the
        enhanced-ventilation one), which has its own guard.
        """
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "async_call"
            and len(call.args) >= 3
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == "homeassistant"
        ):
            return False
        return "temp_lux_entity" in ast.dump(call.args[2])

    doors: list[tuple[str, str]] = []
    for path in sorted(pathlib.Path("custom_components/effektguard").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and commands_the_lux_switch(inner):
                    doors.append((path.name, node.name))

    assert doors == [("coordinator.py", "_set_temporary_lux")], (
        f"the hot-water switch is commanded from {doors}. Every call must go through "
        f"`_set_temporary_lux`, which is the only place that records whether the boost is ours - "
        f"the fact the unload cleanup reads."
    )

"""Which user actions actually tear the entry down - the two facts the shutdown guards depend on.

An options change calls the update listener, which HOT-RELOADS (`async_update_config`): the entry
stays loaded, so the shutdown guards must NOT fire on it and the entities must be re-rendered
(`async_update_listeners`) since they are views of the entry.

What DOES unload the entry: the reconfigure flow (changing entity selections - power meter, weather,
pump model), which ends in `async_update_reload_and_abort` and schedules a full reload; plus manual
reload, removal, and restart. The reconfigure case is exactly when a stray write from the old
coordinator would land, which is the defect the shutdown guards close.
"""

from __future__ import annotations

import ast
import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.effektguard import async_reload_entry
from custom_components.effektguard.const import DOMAIN


@pytest.mark.asyncio
async def test_changing_an_option_hot_reloads_and_does_not_unload():
    """The listener that fires on an options change must not tear the entry down."""
    hass = MagicMock()
    hass.config_entries.async_reload = AsyncMock()
    hass.config_entries.async_unload = AsyncMock()

    coordinator = MagicMock()
    coordinator.async_update_config = AsyncMock()

    entry = MagicMock()
    entry.entry_id = "entry_1"
    entry.data = {"target_indoor_temp": 21.0}
    entry.options = {"thermal_mass": 1.8}
    hass.data = {DOMAIN: {"entry_1": coordinator}}

    await async_reload_entry(hass, entry)

    coordinator.async_update_config.assert_awaited_once()
    applied = coordinator.async_update_config.await_args.args[0]
    assert applied["thermal_mass"] == 1.8, "the changed option must actually reach the coordinator"

    assert hass.config_entries.async_reload.await_count == 0, (
        "changing an option tore the entry down. This integration hot-reloads on purpose - it is "
        "what preserves the startup grace period, the entities, and the accumulated learning state. "
        "If this ever becomes a real reload, every comment that says an options change does NOT "
        "unload becomes wrong, and the shutdown guards start firing on an ordinary settings change."
    )
    assert hass.config_entries.async_unload.await_count == 0


@pytest.mark.asyncio
async def test_the_entities_are_told_when_the_entry_changes():
    """Hot-reloading the config must re-render the entities that are VIEWS of it.

    Switches read `entry.data` in `is_on` and the thermostat's `hvac_mode` reads the same
    `enable_optimization` key, but a view only updates when told to. Without
    `async_update_listeners`, the switch kept displaying its old value until the next aligned refresh.
    """
    hass = MagicMock()
    coordinator = MagicMock()
    coordinator.async_update_config = AsyncMock()

    entry = MagicMock()
    entry.entry_id = "entry_1"
    entry.data = {"enable_optimization": False}
    entry.options = {}
    hass.data = {DOMAIN: {"entry_1": coordinator}}

    await async_reload_entry(hass, entry)

    coordinator.async_update_listeners.assert_called_once_with()


def test_the_reconfigure_flow_is_the_one_that_reloads():
    """And it is a real user action: swapping the power meter or the weather entity.

    Checked structurally: the reconfigure step ends in `async_update_reload_and_abort`, Home
    Assistant's "apply these entity selections and reload the entry" - the FULL teardown the
    shutdown guards exist for.
    """
    source = pathlib.Path("custom_components/effektguard/config_flow.py").read_text()
    tree = ast.parse(source)

    reloaders = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "async_update_reload_and_abort"
            for inner in ast.walk(node)
        )
    ]

    assert reloaders, (
        "no step in the config flow calls `async_update_reload_and_abort`. Something must force a "
        "full reload when the entity selections change - the adapters are built from entry.data at "
        "setup and would otherwise keep pointing at the old entities."
    )
    assert all("reconfigure" in name for name in reloaders), (
        f"{reloaders} force a full entry reload. Only the reconfigure step should: it is the one "
        f"that changes which entities the adapters are built from."
    )

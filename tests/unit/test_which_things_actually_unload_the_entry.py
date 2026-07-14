"""Which user actions tear the entry down - because I asserted the wrong one, four times.

The two commits before this one fixed real defects: a coordinator whose entry had unloaded could
still write a curve offset, command the fan, and start a hot-water boost that nothing would ever
switch off. Those are real, and they are fixed.

But I described the TRIGGER as "a reload, which is what Home Assistant does every time an option is
changed", and I wrote that in two commit messages, two pull-request comments and four code comments
without ever executing it. IT IS FALSE.

This integration installs an update listener that HOT-RELOADS:

    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    async def async_reload_entry(...):
        \"\"\"Handle a config-entry update by hot-reloading the runtime settings.
        Hot-reloading (rather than tearing the entry down) is what preserves ...\"\"\"
        await coordinator.async_update_config(merged_config)

The name says reload. The body does not reload. Changing an option calls that listener, and the entry
stays loaded. Measured against a running Home Assistant, submitting the real options flow:

    Unloading EffektGuard:                          0
    Options updated, applying changes (no restart): 1
    thermal mass: 1.80                                 (the change took effect)

WHAT DOES UNLOAD THE ENTRY, and every one of these is a real thing a real user does:

    the RECONFIGURE flow   - changing the entity selections: the power meter, the weather entity,
                             the pump model. It ends in `async_update_reload_and_abort`, which
                             schedules a FULL reload. Measured: 1 unload, 1 setup.
    a manual reload        - the Reload button, or `homeassistant.reload_config_entry`. Measured.
    removing the integration
    restarting Home Assistant

So the defects stand and the fixes stand - the reconfigure flow is exactly when somebody swaps the
power meter, and a stray write from the old coordinator landing after that is precisely the bug - but
the sentence I used to justify them was wrong, and a fix justified by a mechanism nobody ran is the
thing this whole audit exists to catch.

These tests pin the two facts, so the next person to reason about it reads something that was
executed:
  * the update listener hot-reloads and does NOT tear the entry down;
  * the reconfigure flow DOES.
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


def test_the_reconfigure_flow_is_the_one_that_reloads():
    """And it is a real user action: swapping the power meter or the weather entity.

    Structural, because the whole point is that I reasoned about this instead of executing it. The
    reconfigure step ends in `async_update_reload_and_abort`, which is Home Assistant's "apply these
    entity selections and reload the entry" - the FULL teardown the shutdown guards exist for.
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

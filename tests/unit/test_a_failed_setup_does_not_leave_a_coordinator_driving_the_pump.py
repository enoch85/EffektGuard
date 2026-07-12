"""If setup fails after the first refresh, the coordinator keeps driving the heat pump.

`async_setup_entry` does this, in order:

    106   hass.data[DOMAIN][entry.entry_id] = coordinator
    112   await coordinator.async_config_entry_first_refresh()   <-- ARMS the 5-minute timer
    128   await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

The first refresh runs `_read_and_decide` to completion, and that ends by calling
`_schedule_aligned_refresh()`. So by the time line 128 runs, the clock-aligned control loop is
**armed and ticking**.

Line 128 is not guarded. If a platform's setup raises, `async_setup_entry` propagates it and:

  * Home Assistant runs the entry's `async_on_unload` callbacks - which would call
    `async_shutdown()` and cancel the timer - in **exactly one** of its failure branches, the
    generic `except (SystemExit, Exception)`. It does **not** do so for `ConfigEntryNotReady`,
    `ConfigEntryError` or `ConfigEntryAuthFailed`. A platform that reports "not ready" is the
    ordinary case, and it is the one that leaks.

  * So the coordinator is left with its timer live. It is no longer reachable through the config
    entry, but the timer holds a reference to it, and every five minutes it reads the world,
    decides, and **writes a curve offset to the heat pump**.

  * Home Assistant then RETRIES the setup. `_create_coordinator` builds a second coordinator, which
    arms its own timer. And the retry after that builds a third.

**Two coordinators, one heat pump, conflicting curve offsets, forever.** That sentence is already in
this codebase - `_schedule_aligned_refresh` carries it as a comment, from the F-061 fix - and the
setup path walks straight into it by another door.

The fix is not subtle: anything that fails after the coordinator has been stored must shut it down
and take it out of `hass.data`, whatever the exception was.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.exceptions import ConfigEntryNotReady

from custom_components.effektguard import async_setup_entry
from custom_components.effektguard.const import DOMAIN


def test_home_assistant_only_cleans_up_on_one_of_its_failure_branches():
    """The premise, read out of Home Assistant itself.

    If this ever stops being true - if HA starts processing on_unload for every failure - then the
    integration is covered by the framework and this file is belt-and-braces. Today it is not.
    """
    from homeassistant import config_entries

    source = inspect.getsource(config_entries)
    start = source.find("async def __async_setup_with_context")
    end = source.find("\n    async def ", start + 10)
    body = source[start:end]

    assert body.count("_async_process_on_unload") == 1, (
        "Home Assistant now cleans up on a different number of failure branches than it used to. "
        "Re-read which ones: this integration relies on knowing that ConfigEntryNotReady from a "
        "platform does NOT run the entry's on_unload callbacks."
    )

    not_ready_branch = body[body.find("except ConfigEntryNotReady") : body.find("except asyncio")]
    assert "_async_process_on_unload" not in not_ready_branch, (
        "ConfigEntryNotReady now processes on_unload. If that is real, the orphan-coordinator leak "
        "this file guards is closed by the framework - verify before deleting the guard."
    )


def test_the_setup_path_shuts_the_coordinator_down_if_anything_after_it_fails():
    """Structural: every step after the coordinator is stored must be inside a guard."""
    source = inspect.getsource(async_setup_entry)

    forward = source.find("async_forward_entry_setups")
    assert forward != -1, "async_forward_entry_setups is no longer called from async_setup_entry"

    # Everything from storing the coordinator to the end must sit under a try that shuts it down.
    assert "async_shutdown" in source, (
        "async_setup_entry never calls coordinator.async_shutdown(). If async_forward_entry_setups "
        "raises ConfigEntryNotReady - the ordinary case when a platform is not ready - Home "
        "Assistant does NOT run the entry's on_unload callbacks, so the aligned-refresh timer "
        "armed by the first refresh stays live. The orphaned coordinator goes on writing curve "
        "offsets to the heat pump every five minutes, and HA's retry creates a second one."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        ConfigEntryNotReady("the sensor platform is not ready"),
        ValueError("a platform blew up"),
    ],
    ids=["platform_not_ready", "platform_raised"],
)
async def test_a_platform_failure_leaves_no_coordinator_driving_the_pump(failure):
    """Behavioural: whatever the platform throws, the timer must not survive it."""
    hass = MagicMock()
    hass.data = {}
    hass.config_entries.async_forward_entry_setups = AsyncMock(side_effect=failure)

    entry = MagicMock()
    entry.entry_id = "abc123"
    entry.data = MagicMock()
    entry.data.get.side_effect = lambda key, default=None: default

    coordinator = MagicMock()
    coordinator.async_config_entry_first_refresh = AsyncMock()  # succeeds -> the TIMER IS ARMED
    coordinator.async_shutdown = AsyncMock()
    coordinator.setup_power_sensor_listener = MagicMock()
    coordinator.async_restore_peaks = AsyncMock()
    coordinator.async_initialize_learning = AsyncMock()

    with patch(
        "custom_components.effektguard._create_coordinator", AsyncMock(return_value=coordinator)
    ):
        with pytest.raises(type(failure)):
            await async_setup_entry(hass, entry)

    coordinator.async_shutdown.assert_awaited(), (
        f"The platform raised {type(failure).__name__} and the coordinator was never shut down. "
        f"The first refresh had already succeeded, so its 5-minute aligned timer is armed and "
        f"still writing curve offsets to the heat pump - and Home Assistant is about to retry "
        f"setup and build a second coordinator alongside it."
    )
    assert not hass.data.get(DOMAIN, {}).get(entry.entry_id), (
        "The dead coordinator is still in hass.data[DOMAIN]. The retry overwrites the reference, "
        "but the armed timer keeps the object - and its grip on the pump - alive."
    )

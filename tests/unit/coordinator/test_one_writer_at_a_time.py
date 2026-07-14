"""Two things may drive the heat pump. They must never drive it at once.

The write path has exactly two entry points: the aligned control loop, every five minutes, and a
service that explicitly commands the pump (force_offset, boost_heating, the optimization switch).
Nothing serialises them, and both are long coroutines that await at every step - reading entities
through Home Assistant, saving state, calling the NIBE adapter. asyncio interleaves them freely.

So this sequence is not hypothetical, it is ordinary:

    12:05:10  the aligned refresh starts. It reads the world and begins deciding.
    12:05:11  the user calls force_offset(+3). The override is set on the engine, and the service
              reads, decides (+3, honouring the override) and writes +3 to the pump.
    12:05:12  the aligned refresh - which snapshotted the engine BEFORE the override existed -
              finishes its decision and writes +0.5.

The forced offset is gone, overwritten by a decision that predates it. The user sees the service
succeed and the pump ignore it. The same interleaving corrupts _apply_offset's rate limiting, which
reads self.last_offset_timestamp and then writes it.

One writer at a time. The read path is unaffected: reads are free to overlap, and do.
"""

from __future__ import annotations

import asyncio
import ast
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.effektguard.coordinator import EffektGuardCoordinator


def _make_minimal_hass() -> MagicMock:
    hass = MagicMock()
    hass.data = {}
    hass.config = MagicMock()
    hass.config.latitude = 59.3
    hass.config.config_dir = "/tmp/test"
    hass.loop = MagicMock()
    hass.loop.call_soon_threadsafe = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda func, *args: func(*args))
    hass.async_create_task = MagicMock()
    return hass


def _make_minimal_entry() -> MagicMock:
    entry = MagicMock()
    entry.data = MagicMock()
    entry.data.get.side_effect = lambda key, default=None: default
    entry.options = MagicMock()
    entry.options.get.side_effect = lambda key, default=None: default
    return entry


def _make_coordinator() -> EffektGuardCoordinator:
    return EffektGuardCoordinator(
        hass=_make_minimal_hass(),
        nibe_adapter=MagicMock(),
        gespot_adapter=MagicMock(),
        weather_adapter=MagicMock(),
        decision_engine=MagicMock(),
        effect_manager=MagicMock(),
        entry=_make_minimal_entry(),
    )


@pytest.mark.asyncio
async def test_the_control_loop_and_a_service_never_write_together(monkeypatch):
    """The two writers, launched together. They must take the pump in turns."""
    coordinator = _make_coordinator()

    in_flight = 0
    overlapped = False

    async def slow_cycle(
        apply: bool,
        explicit_command: bool = False,
    ) -> dict[str, object]:
        """Stand-in for the real read-decide-write cycle, which awaits at every step."""
        nonlocal in_flight, overlapped
        in_flight += 1
        if in_flight > 1:
            overlapped = True
        await asyncio.sleep(0)  # asyncio's chance to interleave, exactly as the real body gives it
        in_flight -= 1
        return {"applied": apply}

    monkeypatch.setattr(coordinator, "_read_and_decide", slow_cycle)
    monkeypatch.setattr(coordinator, "async_set_updated_data", MagicMock())
    monkeypatch.setattr(coordinator, "_schedule_aligned_refresh", MagicMock())

    await asyncio.gather(
        coordinator._do_aligned_refresh(),  # the control loop
        coordinator.async_refresh_and_apply(),  # a service commanding the pump
    )

    assert not overlapped, (
        "The aligned control loop and a service were both driving the heat pump at the same "
        "moment. Whichever decision finishes last wins - and that may be the OLDER one, computed "
        "before the user's force_offset override even existed. The forced offset is silently "
        "overwritten, and _apply_offset's rate limiting reads state another writer is changing."
    )


@pytest.mark.asyncio
async def test_reads_are_still_free_to_overlap(monkeypatch):
    """The lock guards the pump, not the sensors. Serialising reads would be a needless stall."""
    coordinator = _make_coordinator()

    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_cycle(
        apply: bool,
        explicit_command: bool = False,
    ) -> dict[str, object]:
        started.set()
        await release.wait()
        return {}

    monkeypatch.setattr(coordinator, "_read_and_decide", blocking_cycle)
    monkeypatch.setattr(coordinator, "async_set_updated_data", MagicMock())
    monkeypatch.setattr(coordinator, "_schedule_aligned_refresh", MagicMock())

    writer = asyncio.create_task(coordinator.async_refresh_and_apply())
    await started.wait()  # the writer now holds whatever it holds

    # A read must not be stuck behind it. HA calls this hook on its own schedule and on reload;
    # blocking it on a write in progress would stall the entities for no reason.
    monkeypatch.setattr(coordinator, "_read_and_decide", AsyncMock(return_value={}))
    await asyncio.wait_for(coordinator._async_update_data(), timeout=1.0)

    release.set()
    await writer


def test_nothing_can_write_without_taking_the_lock():
    """Structural: `apply=True` exists in exactly one place, and that place holds the lock.

    The behavioural test above proves the two callers we have today serialise. This one keeps the
    next caller honest - a third `_read_and_decide(apply=True)` added elsewhere would reintroduce
    the race in a way no existing test would notice.
    """
    source = inspect.getsource(EffektGuardCoordinator)
    tree = ast.parse(source)
    writers = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_read_and_decide"
        and any(
            keyword.arg == "apply"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
    )

    assert writers == 1, (
        f"`_read_and_decide(apply=True)` is called from {writers} places. The write path must have "
        f"exactly one owner, and that owner must hold the control lock. Route new writers through "
        f"it rather than calling the cycle directly."
    )

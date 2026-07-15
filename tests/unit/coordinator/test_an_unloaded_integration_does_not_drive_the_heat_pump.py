"""A coordinator whose entry has unloaded must not write to the heat pump.

`_do_aligned_refresh` runs on `hass.async_create_task`, so HA cannot cancel it on unload, and it
is mid-flight for seconds while `_read_and_decide` awaits the weather forecast, the price adapter
and the learning modules. `_shutdown_requested` guarded the timer RE-ARM but not the WRITE, so an
in-flight refresh drove the pump after unload. The entry unloads on the reconfigure flow, a manual
reload, a removal or a restart (NOT on an options change, which hot-reloads) - and this stray write
can land after the reload's new coordinator, or be a deleted integration getting the last word.

The write path now has one guarded door per actuator, and each refuses once the entry is gone.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.effect_layer import EffectManager


def _coordinator() -> EffektGuardCoordinator:
    hass = MagicMock()
    hass.config.latitude = 59.33
    hass.config.longitude = 18.07
    hass.config_entries.async_update_entry = MagicMock()

    nibe = MagicMock()
    nibe.set_curve_offset = AsyncMock(return_value=2)
    nibe.set_enhanced_ventilation = AsyncMock(return_value=True)
    nibe.is_enhanced_ventilation_active = AsyncMock(return_value=False)
    nibe.has_ventilation_control = False

    entry = MagicMock()
    entry.data = {}
    entry.options = {}

    coordinator = EffektGuardCoordinator(
        hass, nibe, MagicMock(), MagicMock(), MagicMock(), EffectManager(hass), entry
    )
    # Storage is not what is under test here, and HA's Store wants a real event loop executor.
    coordinator.learning_store = MagicMock()
    coordinator.learning_store.async_save = AsyncMock()
    coordinator.effect.async_save = AsyncMock()
    return coordinator


@pytest.mark.asyncio
async def test_a_refresh_in_flight_when_the_entry_unloads_does_not_write():
    """THE RACE, run for real: unload lands while the refresh is awaiting the weather forecast."""
    coordinator = _coordinator()

    reached_the_awaits = asyncio.Event()
    let_it_finish = asyncio.Event()

    async def slow_read_and_decide(apply: bool = False, explicit_command: bool = False):
        # Stands in for the real one, which awaits the weather service call, the price adapter and
        # the learning modules. Seconds of awaits - and the unload lands in the middle of them.
        #
        # It reaches the pump the way the real one does, through `_write_curve_offset`. Calling
        # `nibe.set_curve_offset` directly here would be testing a code path production no longer
        # has - and the assertion is on the ADAPTER, so nothing about the guard is assumed: the
        # question is only whether the heat pump was touched.
        reached_the_awaits.set()
        await let_it_finish.wait()
        if apply:
            await coordinator._write_curve_offset(2.0)
        return {}

    with patch.object(coordinator, "_read_and_decide", slow_read_and_decide):
        refresh = asyncio.create_task(coordinator._do_aligned_refresh())
        await reached_the_awaits.wait()

        # The user swaps the power meter in the reconfigure flow. The entry unloads.
        await coordinator.async_shutdown()
        assert coordinator._shutdown_requested is True

        # Home Assistant cannot cancel this task - the coordinator's own comment says so.
        let_it_finish.set()
        await refresh

    assert coordinator.nibe.set_curve_offset.await_count == 0, (
        f"the coordinator wrote to the heat pump {coordinator.nibe.set_curve_offset.await_count} "
        f"time(s) AFTER the entry was unloaded: "
        f"{coordinator.nibe.set_curve_offset.await_args_list}. The entry unloads on the reconfigure "
        f"flow, a manual reload, a removal or a restart - and this write can land after the NEW "
        f"coordinator's, leaving the pump on a decision computed by a coordinator built from the "
        f"entities the user has just replaced. On a removal, it is the deleted integration getting "
        f"the last word on the heat pump."
    )


@pytest.mark.asyncio
async def test_a_live_coordinator_still_writes():
    """The control. The guard must refuse dead coordinators, not working ones."""
    coordinator = _coordinator()

    async def read_and_decide(apply: bool = False, explicit_command: bool = False):
        if apply:
            await coordinator._write_curve_offset(2.0)
        return {}

    with patch.object(coordinator, "_read_and_decide", read_and_decide):
        await coordinator._do_aligned_refresh()

    assert coordinator.nibe.set_curve_offset.await_count == 1, (
        "a running coordinator must drive the pump - that is the whole job. The shutdown guard "
        "must not be reachable while the entry is loaded."
    )


@pytest.mark.asyncio
async def test_switching_optimization_off_after_unload_does_not_write():
    """The other write path. `set_optimization_enabled(False)` resets the offset to neutral.

    It is a user command and perfectly legitimate while the entry is loaded - but if it is in flight
    when the entry unloads, it reaches the pump from a dead coordinator exactly as the control loop
    does. One guarded way to the pump, not two.
    """
    coordinator = _coordinator()
    await coordinator.async_shutdown()

    await coordinator.set_optimization_enabled(False)

    assert coordinator.nibe.set_curve_offset.await_count == 0, (
        f"a shut-down coordinator reset the pump's offset to neutral "
        f"({coordinator.nibe.set_curve_offset.await_args_list}). The entry is gone; it has no "
        f"business writing anything."
    )


@pytest.mark.asyncio
async def test_switching_optimization_off_forces_neutral_through_cooldown():
    """OFF is a safety transition, not an ordinary rate-limited adjustment."""
    coordinator = _coordinator()
    coordinator.nibe.set_curve_offset = AsyncMock(return_value=0)

    await coordinator.set_optimization_enabled(False)

    coordinator.nibe.set_curve_offset.assert_awaited_once_with(0.0, force_write=True)
    assert coordinator.last_applied_offset == 0.0
    disabled_data = coordinator.hass.config_entries.async_update_entry.call_args.kwargs["data"]
    assert disabled_data["enable_optimization"] is False


@pytest.mark.asyncio
async def test_an_unloaded_coordinator_does_not_command_the_fan_either():
    """The heating curve is not the only thing this integration writes to the pump.

    `set_enhanced_ventilation` raises the exhaust fan on an F750/F730 from the control loop, so it
    rides the same in-flight refresh and the same race. On a reload the old coordinator can switch
    the fan ON while the new one starts up believing it is off, leaving it running with nothing left
    to turn it off.
    """
    coordinator = _coordinator()
    coordinator.nibe.set_enhanced_ventilation = AsyncMock(return_value=True)
    await coordinator.async_shutdown()

    wrote = await coordinator._write_enhanced_ventilation(True)

    assert wrote is False
    assert coordinator.nibe.set_enhanced_ventilation.await_count == 0, (
        "a shut-down coordinator switched enhanced ventilation on. The entry is unloaded; the fan "
        "is not its to command, and nothing is left to switch it off again."
    )


def test_there_is_exactly_one_door_to_each_thing_the_pump_can_be_told():
    """A structural guard, and it is the one that keeps the others honest.

    The tests above prove the guarded doors refuse a dead coordinator; they cannot prove nobody has
    cut a NEW door beside them. So: every `self.nibe.set_*` call in the coordinator must live inside
    a `_write_*` method - the only places that ask whether the entry is still loaded. A new way to
    command the pump either routes through one of them or changes this test in a reviewed diff.
    """
    source = pathlib.Path("custom_components/effektguard/coordinator.py").read_text()
    tree = ast.parse(source)

    # A LIST of (command, the method that issues it), not a dict keyed by the command.
    #
    # The first version of this collected `doors[command] = enclosing_method`, and a mutation test
    # walked straight through it: a second `self.nibe.set_enhanced_ventilation(...)` in the airflow
    # loop simply OVERWROTE the dict entry, so two doors looked exactly like one. A container that
    # silently collapses duplicates cannot count duplicates - which is the whole job here.
    doors: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and isinstance(inner.func.value, ast.Attribute)
                and inner.func.value.attr == "nibe"
                and inner.func.attr.startswith("set_")
            ):
                doors.append((inner.func.attr, node.name))

    assert sorted(doors) == [
        ("set_curve_offset", "_write_curve_offset"),
        ("set_enhanced_ventilation", "_write_enhanced_ventilation"),
    ], (
        f"the heat pump is commanded from {sorted(doors)}. Every write must go through a `_write_*` "
        f"method - exactly once - because those are the only ones that ask whether the entry is "
        f"still loaded. A door that bypasses them is how an unloaded integration gets the last word "
        f"on somebody's heating."
    )


@pytest.mark.asyncio
async def test_the_shutdown_flag_is_actually_consulted_on_the_write_path():
    """A structural guard, because the flag existed and was simply never read here.

    `_shutdown_requested` was checked in two places - the code that re-arms the timer, and the code
    that sets it - and in neither of the two places that drive the heat pump. The bug was not a wrong
    value; it was a value nobody asked for.
    """
    coordinator = _coordinator()
    await coordinator.async_shutdown()

    written = await coordinator._write_curve_offset(3.0)

    assert written is None
    assert coordinator.nibe.set_curve_offset.await_count == 0

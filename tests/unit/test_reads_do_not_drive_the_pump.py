"""Reading the state of the world must not command the heat pump.

`_async_update_data` is HA's READ hook, and `async_request_refresh()` is public, debounced, and
called from reloads, options changes and bookkeeping services (reset_peak_tracking clears a
counter). So the read path must contain no write. Writes belong to `_do_aligned_refresh`, the one
scheduled owner of the control loop; services that genuinely command the pump (force_offset,
boost_heating) go through the explicit `async_refresh_and_apply` path and take effect at once.
"""

import inspect
from pathlib import Path

import pytest

from custom_components.effektguard.coordinator import EffektGuardCoordinator

# Everything that reaches the heat pump.
WRITES = (
    "set_curve_offset",
    "set_enhanced_ventilation",
    "_apply_dhw_control",
    "_apply_airflow_decision",
)


def test_the_read_hook_contains_no_write():
    """Checked structurally, not by execution.

    `_async_update_data` gathers state from half a dozen adapters; a test that stubbed all of them
    would prove only that the stubs were right. What matters is that the source of the READ path
    contains no call that reaches the pump.
    """
    source = inspect.getsource(EffektGuardCoordinator._async_update_data)

    found = [call for call in WRITES if call in source]

    assert not found, (
        f"_async_update_data is Home Assistant's READ hook and it writes to the heat pump: "
        f"{', '.join(found)}. Everything that calls async_request_refresh() therefore drives the "
        f"pump - including reset_peak_tracking, which only clears a counter."
    )


def test_the_control_loop_is_the_one_that_writes():
    """If the scheduled loop does not drive the pump, nothing ever will.

    With `update_interval=None`, `_do_aligned_refresh` is the only thing on a clock. Taking the
    writes out of the read hook without putting them here would leave the pump on whatever offset
    it last held, forever, and every entity would still look healthy.
    """
    source = inspect.getsource(EffektGuardCoordinator._do_aligned_refresh)

    assert "_drive_the_pump" in source, (
        "_do_aligned_refresh is the scheduled owner of the write path, and with update_interval "
        "None it is the only thing on a clock. If it does not drive the pump, nothing does."
    )


def test_a_service_can_still_command_the_pump_at_once():
    """Splitting read from write must not make force_offset wait for the next aligned tick."""
    assert hasattr(EffektGuardCoordinator, "async_refresh_and_apply"), (
        "Services that genuinely command the pump need an explicit way to read, decide and apply "
        "immediately - otherwise force_offset would take up to a full update interval to land."
    )

    source = inspect.getsource(EffektGuardCoordinator.async_refresh_and_apply)
    assert "_drive_the_pump" in source, (
        "async_refresh_and_apply exists to reach the pump, and must do so through the one owner of "
        "the write path - which is what holds the control lock."
    )


def _service_handler(marker: str) -> str:
    """The source of the service handler containing `marker`, to its closing boundary.

    Sliced at the next `async def`, not at a byte count: a fixed window silently stops covering
    the handler the moment anyone adds a line to it, and the test then passes for the wrong reason.
    """
    source = (
        Path(__file__).resolve().parents[2] / "custom_components" / "effektguard" / "__init__.py"
    ).read_text(encoding="utf-8")

    start = source.index(marker)
    end = source.find("\n    async def ", start)
    return source[start:end] if end != -1 else source[start:]


def test_bookkeeping_services_do_not_touch_the_pump():
    """reset_peak_tracking clears a counter. That is all it may do."""
    handler = _service_handler("Reset peak tracking service called")

    assert "async_refresh_and_apply" not in handler, (
        "reset_peak_tracking clears a stored counter and must not drive the heat pump. It may ask "
        "for a refresh so the entities catch up; it may not ask for an apply."
    )


@pytest.mark.parametrize(
    "marker",
    ["Force offset service called", "Boost heating service called"],
)
def test_the_services_that_command_the_pump_do_apply(marker):
    """force_offset and boost_heating mean what they say, and must land at once."""
    handler = _service_handler(marker)

    assert "async_apply_manual_override" in handler, (
        f"{marker!r} exists to drive the heat pump. With the read path no longer writing, it must "
        f"use the shared explicit-command path, or it does nothing until the next aligned tick."
    )

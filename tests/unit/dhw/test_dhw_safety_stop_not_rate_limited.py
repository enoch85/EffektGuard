"""A DHW stop must never be deferred by the rate limiter.

The DHW rate limiter (DHW_CONTROL_MIN_INTERVAL_MINUTES = 60) used to guard BOTH
directions, and its `return` sat above the turn-off branch in _apply_dhw_control.

That produced a real safety hole:

  1. Every `should_heat=False` return in should_start_dhw() carries an EMPTY
     abort_conditions list, so the early-abort branch above the limiter is skipped.
  2. `_last_dhw_control_time` is stamped by the turn-ON, so the 60-minute clock starts at
     the beginning of the very cycle we later want to stop.

Result: 03:00 lux ON in a cheap window. 03:05 a cold front arrives, DM crashes past the
T2 block threshold, the decision flips to CRITICAL_THERMAL_DEBT - and DHW keeps the
compressor away from space heating until 04:00 while thermal debt deepens. That is the
exact "DHW during heating demand = thermal debt accumulation" failure the rulebook names.

Stopping the lux boost cannot harm the pump: it only cancels an EffektGuard-initiated
boost (NIBE's own DHW schedule is untouched). Throttling it has no safety benefit and a
real safety cost. Starts remain rate limited, which is what bounds oscillation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.effektguard.const import DHW_CONTROL_MIN_INTERVAL_MINUTES
from custom_components.effektguard.coordinator import EffektGuardCoordinator

LUX_ENTITY = "switch.temporary_lux_50004"
NOW = datetime(2026, 1, 15, 3, 5)

# The lux boost was started 5 minutes ago - deep inside the 60-minute rate-limit window.
STARTED_5_MIN_AGO = NOW - timedelta(minutes=5)


@dataclass
class FakeDHWDecision:
    """Mirrors the shape of DHWScheduleDecision on the paths under test."""

    should_heat: bool
    priority_reason: str
    # Every should_heat=False return in should_start_dhw() sets this to []. That is
    # precisely why the abort branch cannot rescue us and the limiter had to be fixed.
    abort_conditions: list[str] = field(default_factory=list)


def make_coordinator(lux_is_on: bool, last_control_time: datetime | None):
    """Duck-typed stand-in exposing only what _apply_dhw_control touches.

    Calling the unbound method with this avoids standing up a full HA config entry, and
    keeps the test deterministic.
    """
    coordinator = MagicMock()
    coordinator.temp_lux_entity = LUX_ENTITY
    coordinator._last_dhw_control_time = last_control_time
    coordinator.last_update_success = True
    coordinator.data = {"dhw_planning": {"thermal_debt": -1100.0, "indoor_temperature": 20.4}}
    coordinator.entry.options = {}
    coordinator.entry.data = {"target_indoor_temp": 21.0}

    lux_state = MagicMock()
    lux_state.state = "on" if lux_is_on else "off"
    coordinator.hass.states.get.return_value = lux_state
    coordinator.hass.services.async_call = AsyncMock()

    # Bind the real rate-limit helper so the test exercises production logic.
    coordinator._is_dhw_start_rate_limited = (
        lambda now: EffektGuardCoordinator._is_dhw_start_rate_limited(coordinator, now)
    )
    # And the real switch door, for the same reason: it is the only place that records whether a
    # running hot-water boost is EffektGuard's to cancel, and `_apply_dhw_control` now goes through
    # it. A MagicMock would answer the call cheerfully and record nothing.
    #
    # `_shutdown_requested` must be a real False, not an auto-mock: the door refuses to START a boost
    # when it is set, and every MagicMock attribute is truthy. The fake has to be the object.
    coordinator._shutdown_requested = False
    # Same for the user-boost window: None means "no service boost", an auto-mock means chaos.
    coordinator._service_boost_until = None
    coordinator._set_temporary_lux = lambda on: EffektGuardCoordinator._set_temporary_lux(
        coordinator, on
    )
    return coordinator


async def apply(coordinator, decision) -> None:
    await EffektGuardCoordinator._apply_dhw_control(coordinator, decision, 45.0, NOW)


def switch_calls(coordinator) -> list[str]:
    """The switch services actually invoked, e.g. ['turn_off']."""
    return [
        call.args[1]
        for call in coordinator.hass.services.async_call.call_args_list
        if call.args and call.args[0] == "switch"
    ]


class TestSafetyStopIsNotRateLimited:
    @pytest.mark.asyncio
    async def test_critical_thermal_debt_stops_dhw_inside_the_rate_limit_window(self):
        """The F-029 scenario: stop must happen at 03:05, not wait until 04:00."""
        coordinator = make_coordinator(lux_is_on=True, last_control_time=STARTED_5_MIN_AGO)

        await apply(
            coordinator,
            FakeDHWDecision(should_heat=False, priority_reason="CRITICAL_THERMAL_DEBT"),
        )

        assert "turn_off" in switch_calls(coordinator), (
            "DHW was NOT stopped despite CRITICAL_THERMAL_DEBT, because the rate limiter "
            f"deferred it ({DHW_CONTROL_MIN_INTERVAL_MINUTES} min window, boost started "
            "5 min ago). DHW keeps stealing the compressor from space heating while "
            "thermal debt deepens."
        )

    @pytest.mark.asyncio
    async def test_stop_works_with_empty_abort_conditions(self):
        """The abort branch cannot rescue us: should_heat=False always sets [].

        This pins the reason the limiter had to change rather than the abort path.
        """
        coordinator = make_coordinator(lux_is_on=True, last_control_time=STARTED_5_MIN_AGO)

        decision = FakeDHWDecision(
            should_heat=False,
            priority_reason="SPACE_HEATING_EMERGENCY",
            abort_conditions=[],
        )
        await apply(coordinator, decision)

        assert "turn_off" in switch_calls(coordinator)


class TestStartsRemainRateLimited:
    """Bounding oscillation is what the limiter is for - that must still hold."""

    @pytest.mark.asyncio
    async def test_start_is_still_rate_limited(self):
        coordinator = make_coordinator(lux_is_on=False, last_control_time=STARTED_5_MIN_AGO)

        await apply(
            coordinator,
            FakeDHWDecision(should_heat=True, priority_reason="DHW_SCHEDULED"),
        )

        assert switch_calls(coordinator) == [], (
            "A DHW start inside the rate-limit window must still be deferred - otherwise "
            "the pump can be cycled every coordinator tick."
        )

    @pytest.mark.asyncio
    async def test_start_proceeds_once_the_window_has_passed(self):
        coordinator = make_coordinator(
            lux_is_on=False,
            last_control_time=NOW - timedelta(minutes=DHW_CONTROL_MIN_INTERVAL_MINUTES + 1),
        )

        await apply(
            coordinator,
            FakeDHWDecision(should_heat=True, priority_reason="DHW_SCHEDULED"),
        )

        assert "turn_on" in switch_calls(coordinator)

    @pytest.mark.asyncio
    async def test_first_ever_start_is_not_rate_limited(self):
        coordinator = make_coordinator(lux_is_on=False, last_control_time=None)

        await apply(
            coordinator,
            FakeDHWDecision(should_heat=True, priority_reason="DHW_SCHEDULED"),
        )

        assert "turn_on" in switch_calls(coordinator)

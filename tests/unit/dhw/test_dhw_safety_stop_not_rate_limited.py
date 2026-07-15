"""A DHW stop must never be deferred by the rate limiter.

The rate limiter (DHW_CONTROL_MIN_INTERVAL_MINUTES = 60) once guarded BOTH directions, and its
clock is stamped by the turn-ON - so a boost started at 03:00 could not be stopped until 04:00. If
a cold front then crashes DM into the CRITICAL_THERMAL_DEBT block at 03:05, DHW keeps the compressor
off space heating while thermal debt deepens ("DHW during heating demand"). The abort branch cannot
rescue it either: every should_heat=False return carries an empty abort_conditions list.

Stopping an EffektGuard lux boost cannot harm the pump (NIBE's own schedule is untouched), so STARTS
stay rate-limited to bound oscillation while STOPS are always allowed.
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
        """Stop must happen at 03:05, not be deferred to 04:00."""
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

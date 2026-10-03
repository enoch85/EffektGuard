"""The ventilation fan must not cycle every tick when a decision oscillates around its threshold.

The old anti-cycle guard was 5 minutes - exactly one coordinator tick - so a turn-off was permitted
on the very next cycle and it prevented nothing. It also only guarded the turn-OFF, with no rest
period before re-enhancing, so an oscillating decision (what a marginal COP gain produces) flipped
the fan twelve times an hour, each flip perturbing the source air an exhaust-air F750 draws from.

The optimizer's own `duration_minutes` (15-60 min by deficit), previously logged and discarded, is
now the minimum run time, and NIBE_VENTILATION_MIN_REST_DURATION bounds the other direction.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.const import (
    AIRFLOW_DURATION_SMALL_DEFICIT,
    NIBE_VENTILATION_MIN_ENHANCED_DURATION,
    NIBE_VENTILATION_MIN_REST_DURATION,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.airflow_optimizer import FlowDecision, FlowMode

START = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)


def _decision(should_enhance: bool, duration: int = AIRFLOW_DURATION_SMALL_DEFICIT) -> FlowDecision:
    return FlowDecision(
        mode=FlowMode.ENHANCED if should_enhance else FlowMode.STANDARD,
        duration_minutes=duration if should_enhance else 0,
        expected_gain_kw=0.4 if should_enhance else 0.0,
        reason="marginal COP gain, oscillating around the threshold",
        timestamp=START,
    )


class _Fan:
    """A NIBE whose ventilation switch actually remembers what it was told."""

    def __init__(self) -> None:
        self.enhanced = False
        self.changes = 0

    async def is_enhanced_ventilation_active(self) -> bool:
        return self.enhanced

    async def set_enhanced_ventilation(self, on: bool, *, force_write: bool = False) -> bool:
        if on != self.enhanced:
            self.changes += 1
        self.enhanced = on
        return True


def _coordinator(fan: _Fan) -> EffektGuardCoordinator:
    coordinator = EffektGuardCoordinator.__new__(EffektGuardCoordinator)
    coordinator.nibe = fan
    coordinator._airflow_enhance_start = None
    coordinator._airflow_enhance_minutes = NIBE_VENTILATION_MIN_ENHANCED_DURATION
    coordinator._airflow_normal_since = None
    # `__new__` skips `__init__`, so every attribute the real object always has must be set here or
    # the fake is not the object. Home Assistant's DataUpdateCoordinator.__init__ sets this one, and
    # the fan write now consults it: a coordinator whose entry has unloaded does not command the fan.
    coordinator._shutdown_requested = False
    return coordinator


async def _run_an_oscillating_hour(coordinator, monkeypatch) -> None:
    """Twelve ticks, the decision flipping on every one of them."""
    for step in range(12):
        now = START + timedelta(minutes=UPDATE_INTERVAL_MINUTES * step)
        monkeypatch.setattr(dt_util, "utcnow", lambda _n=now: _n)
        await coordinator._apply_airflow_decision(_decision(should_enhance=step % 2 == 0))


def test_the_old_guard_was_exactly_one_tick_long():
    """The premise. A minimum that equals the sampling interval constrains nothing."""
    assert NIBE_VENTILATION_MIN_ENHANCED_DURATION > UPDATE_INTERVAL_MINUTES, (
        f"The minimum enhanced duration ({NIBE_VENTILATION_MIN_ENHANCED_DURATION} min) is not "
        f"longer than one coordinator tick ({UPDATE_INTERVAL_MINUTES} min), so a turn-off is "
        f"permitted on the very next cycle and the guard prevents nothing."
    )


def test_the_minimum_is_at_least_the_shortest_enhancement_ever_recommended():
    assert NIBE_VENTILATION_MIN_ENHANCED_DURATION >= AIRFLOW_DURATION_SMALL_DEFICIT, (
        f"The minimum run time ({NIBE_VENTILATION_MIN_ENHANCED_DURATION} min) is shorter than the "
        f"shortest duration the optimizer ever asks for ({AIRFLOW_DURATION_SMALL_DEFICIT} min), so "
        f"it could never enforce even the mildest of its own recommendations."
    )


@pytest.mark.asyncio
async def test_an_oscillating_decision_does_not_cycle_the_fan(monkeypatch):
    """Twelve state changes an hour, before. The bound is now set by the constants, not the tick."""
    fan = _Fan()

    await _run_an_oscillating_hour(_coordinator(fan), monkeypatch)

    # A full cycle cannot be shorter than one minimum run plus one minimum rest, so an hour
    # permits at most that many cycles, and each cycle is two state changes.
    period = NIBE_VENTILATION_MIN_ENHANCED_DURATION + NIBE_VENTILATION_MIN_REST_DURATION
    allowed = 2 * (60 // period)

    assert fan.changes <= allowed, (
        f"The ventilation fan changed state {fan.changes} times in one hour while the decision "
        f"oscillated around its threshold; the constants bound it to {allowed}. The old guard was "
        f"five minutes and a tick is five minutes, so a turn-off was allowed on the very next "
        f"cycle - and nothing guarded the turn-on at all, which produced twelve. On an exhaust-air "
        f"F750 every change perturbs the source air the compressor is drawing from."
    )
    assert fan.changes < 12, "the unbounded behaviour was twelve changes an hour"


@pytest.mark.asyncio
async def test_the_enhancement_runs_for_the_duration_the_optimizer_asked_for(monkeypatch):
    """`duration_minutes` was computed on every decision, logged, and thrown away."""
    fan = _Fan()
    coordinator = _coordinator(fan)

    monkeypatch.setattr(dt_util, "utcnow", lambda: START)
    await coordinator._apply_airflow_decision(_decision(True, duration=45))
    assert fan.enhanced is True

    # The decision flips immediately. It must not be obeyed until the 45 minutes are up.
    for minutes in (5, 20, 44):
        moment = START + timedelta(minutes=minutes)
        monkeypatch.setattr(dt_util, "utcnow", lambda _m=moment: _m)
        await coordinator._apply_airflow_decision(_decision(False))
        assert fan.enhanced is True, (
            f"The optimizer asked for 45 minutes of enhanced ventilation and the fan was switched "
            f"off after {minutes}. That number was being logged and discarded."
        )

    moment = START + timedelta(minutes=46)
    monkeypatch.setattr(dt_util, "utcnow", lambda _m=moment: _m)
    await coordinator._apply_airflow_decision(_decision(False))
    assert fan.enhanced is False, "after the recommended duration it must be free to stop"


@pytest.mark.asyncio
async def test_the_fan_rests_before_it_can_be_enhanced_again(monkeypatch):
    """The guard that never existed. Without it, the run time only sets the oscillation period."""
    fan = _Fan()
    coordinator = _coordinator(fan)
    coordinator._airflow_normal_since = START

    monkeypatch.setattr(dt_util, "utcnow", lambda: START + timedelta(minutes=1))
    await coordinator._apply_airflow_decision(_decision(True))

    assert fan.enhanced is False, (
        f"The fan was re-enhanced one minute after returning to normal. It must rest for "
        f"{NIBE_VENTILATION_MIN_REST_DURATION} min first."
    )

    rested = START + timedelta(minutes=NIBE_VENTILATION_MIN_REST_DURATION + 1)
    monkeypatch.setattr(dt_util, "utcnow", lambda _m=rested: _m)
    await coordinator._apply_airflow_decision(_decision(True))

    assert fan.enhanced is True, "once rested, a real gain must still be taken"


@pytest.mark.asyncio
async def test_a_steady_beneficial_decision_still_enhances(monkeypatch):
    """The regression guard: do not switch the feature off while bounding it."""
    fan = _Fan()
    coordinator = _coordinator(fan)

    monkeypatch.setattr(dt_util, "utcnow", lambda: START)
    await coordinator._apply_airflow_decision(_decision(True))

    assert fan.enhanced is True
    assert fan.changes == 1


@pytest.mark.asyncio
async def test_a_pump_with_no_ventilation_switch_is_left_alone(monkeypatch):
    """A ground-source pump has no exhaust-air fan to enhance."""
    nibe = MagicMock()
    nibe.is_enhanced_ventilation_active = AsyncMock(return_value=None)
    nibe.set_enhanced_ventilation = AsyncMock()
    coordinator = _coordinator(_Fan())
    coordinator.nibe = nibe

    monkeypatch.setattr(dt_util, "utcnow", lambda: START)
    await coordinator._apply_airflow_decision(_decision(True))

    nibe.set_enhanced_ventilation.assert_not_awaited()

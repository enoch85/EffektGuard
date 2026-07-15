"""An hour the meter mostly did not see must not be billed at all.

When the meter goes `unavailable`, nothing is billed FROM the estimate - but the billing HOUR
used to carry on and, at close, bill whatever the meter last said before it went quiet,
stretched across the silence. A 9 kW reading at 10:00 followed by a blackout until 10:55
became a fabricated 8.33 kW hour ((9*55 + 1*5)/60), which stands for the rest of the month
because the effect tariff bills the three highest hours - throttling the pump to defend a
number that happened in no hour.

The guard: an hour containing a silence longer than MAX_BILLING_OBSERVATION_GAP_MINUTES is
refused. Missing a real peak is recoverable; inventing one is not.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    MAX_BILLING_OBSERVATION_GAP_MINUTES,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.effect_layer import EffectManager

STOCKHOLM = ZoneInfo("Europe/Stockholm")


def _coordinator() -> EffektGuardCoordinator:
    hass = MagicMock()
    hass.config.latitude = 59.33
    hass.config.longitude = 18.07

    nibe = MagicMock()
    nibe._power_sensor_entity = "sensor.house_power"
    nibe.power_sensor_entity = "sensor.house_power"

    entry = MagicMock()
    entry.data = {}
    entry.options = {}

    coordinator = EffektGuardCoordinator(
        hass, nibe, MagicMock(), MagicMock(), MagicMock(), EffectManager(hass), entry
    )
    coordinator.peak_today = 0.0
    coordinator.peak_this_month = 0.0
    coordinator._power_sensor_available = True  # it HAS answered before
    coordinator.effect.record_period_measurement = AsyncMock(return_value=None)
    return coordinator


def _pump() -> NibeState:
    return NibeState(
        outdoor_temp=-5.0,
        indoor_temp=21.0,
        supply_temp=42.0,
        return_temp=37.0,
        degree_minutes=-150.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=STOCKHOLM),
        phase1_current=None,
    )


def _meter(hass, kw: float | None) -> None:
    """`None` is a meter that has gone `unavailable` - which real meters do, routinely."""
    state = MagicMock()
    if kw is None:
        state.state = "unavailable"
        state.attributes = {}
    else:
        state.state = str(kw)
        state.attributes = {"unit_of_measurement": "kW"}
    hass.states.get.return_value = state


async def _run_the_hour(coordinator, monkeypatch, reading_at) -> None:
    """10:00 through 11:00, on the coordinator's real update cadence."""
    for minute in range(0, 60, UPDATE_INTERVAL_MINUTES):
        monkeypatch.setattr(
            dt_util,
            "now",
            lambda tz=None, _m=minute: datetime(2026, 1, 15, 10, _m, tzinfo=STOCKHOLM),
        )
        _meter(coordinator.hass, reading_at(minute))
        await coordinator._update_peak_tracking(_pump())

    # The first sample of the next hour is what closes this one.
    monkeypatch.setattr(
        dt_util, "now", lambda tz=None: datetime(2026, 1, 15, 11, 0, tzinfo=STOCKHOLM)
    )
    _meter(coordinator.hass, 2.0)
    await coordinator._update_peak_tracking(_pump())


def _billed(coordinator) -> list[float]:
    return [
        round(call.kwargs["power_kw"], 2)
        for call in coordinator.effect.record_period_measurement.await_args_list
    ]


@pytest.mark.asyncio
async def test_an_hour_the_meter_slept_through_is_not_billed(monkeypatch):
    """The bug: 8.33 kW billed from two readings, fifty minutes of it unobserved."""
    coordinator = _coordinator()

    # 9 kW at the top of the hour. Then the meter dies until 10:55, and returns reading 1 kW.
    def reading_at(minute: int) -> float | None:
        if minute == 0:
            return 9.0
        if minute == 55:
            return 1.0
        return None

    await _run_the_hour(coordinator, monkeypatch, reading_at)

    assert _billed(coordinator) == [], (
        f"the coordinator billed {_billed(coordinator)} kW for an hour in which the meter answered "
        f"twice and was `unavailable` for fifty of the sixty minutes. That figure is the 9 kW "
        f"reading taken at 10:00, stretched across a blackout nobody watched. It becomes one of the "
        f"month's three billed peaks, and the pump is throttled for the rest of the month to defend "
        f"it. The code logs 'Peak billing is suspended until it does' ten times while doing this."
    )


@pytest.mark.asyncio
async def test_a_fully_observed_hour_is_still_billed(monkeypatch):
    """The control. The guard must refuse blackouts, not customers."""
    coordinator = _coordinator()

    await _run_the_hour(coordinator, monkeypatch, lambda minute: 6.0)

    assert _billed(coordinator) == [6.0], (
        f"a meter that answered on every one of the twelve cycles of the hour billed "
        f"{_billed(coordinator)}. A fully observed 6 kW hour is a 6 kW bill."
    )


@pytest.mark.asyncio
async def test_a_brief_dropout_is_tolerated(monkeypatch):
    """Sensors miss a beat. That is jitter, not a blackout, and the hour was still measured.

    One missed cycle leaves a gap of 2 x UPDATE_INTERVAL_MINUTES between readings, which is inside
    MAX_BILLING_OBSERVATION_GAP_MINUTES. Refusing this would throw away most real hours and buy
    nothing: the reading either side of a five-minute blink is the same reading.
    """
    coordinator = _coordinator()

    await _run_the_hour(coordinator, monkeypatch, lambda minute: None if minute == 25 else 6.0)

    assert _billed(coordinator) == [6.0], (
        f"a single missed update cycle threw the whole hour away ({_billed(coordinator)}). Home "
        f"Assistant misses cycles routinely; a guard that discards an hour for one blink discards "
        f"most of them, and the tariff record goes empty."
    )


@pytest.mark.asyncio
async def test_the_gap_that_is_tolerated_is_bounded_by_the_update_interval(monkeypatch):
    """The threshold is a judgement, so it is pinned where it can be argued with."""
    assert (
        MAX_BILLING_OBSERVATION_GAP_MINUTES > UPDATE_INTERVAL_MINUTES
    ), "the tolerated gap must exceed one update interval, or every ordinary hour is discarded"
    assert (
        MAX_BILLING_OBSERVATION_GAP_MINUTES < 60
    ), "a tolerated gap of an hour or more means no hour can ever be refused, which is the bug"


@pytest.mark.asyncio
async def test_a_meter_that_dies_and_never_returns_does_not_bill_the_rest_of_the_hour(monkeypatch):
    """The silence that runs from the last reading to the hour boundary is a gap too.

    The meter answers at 10:00 and 10:05, then stays `unavailable`. Every gap BETWEEN readings is a
    healthy five minutes, so a guard that only inspects those gaps would see a well-observed hour -
    but the last reading is carried across fifty-five minutes of silence to the boundary, and that
    trailing span must be measured as a gap.
    """
    coordinator = _coordinator()

    await _run_the_hour(coordinator, monkeypatch, lambda minute: 9.0 if minute <= 5 else None)

    assert _billed(coordinator) == [], (
        f"billed {_billed(coordinator)} for an hour whose meter answered twice - at 10:00 and 10:05 "
        f"- and was `unavailable` for the remaining fifty-five minutes. The 9 kW reading was carried "
        f"to the boundary and billed as though it had been watched the whole way."
    )


@pytest.mark.asyncio
async def test_a_long_blackout_is_refused_even_when_the_power_was_low(monkeypatch):
    """It is not about the magnitude. An unobserved hour is unobserved, whatever it reads.

    A LOW reading stretched across a blackout is just as false as a high one - it simply fails
    quietly, by under-recording a peak that did happen, and leaving the month unprotected.
    """
    coordinator = _coordinator()

    def reading_at(minute: int) -> float | None:
        return 1.0 if minute in (0, 55) else None

    await _run_the_hour(coordinator, monkeypatch, reading_at)

    assert _billed(coordinator) == [], (
        f"billed {_billed(coordinator)} for an hour the meter slept through. The house may have "
        f"drawn 9 kW for fifty unwatched minutes; a 1 kW bill would leave the month undefended."
    )

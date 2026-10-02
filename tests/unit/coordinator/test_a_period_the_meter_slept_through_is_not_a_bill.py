"""A billing period the meter mostly did not see must not be billed at all.

When the meter goes `unavailable`, nothing is billed FROM the estimate - but the billing PERIOD
used to carry on and, at close, bill whatever the meter last said before it went quiet, stretched
across the silence. A fabricated peak stands for the rest of the month because the effect tariff
bills the three highest periods - throttling the pump to defend a number that happened in no
observed period.

The guard: a period containing a silence longer than MAX_BILLING_OBSERVATION_GAP_MINUTES (10 min,
i.e. more than one dropped 5-minute cycle inside a 15-minute period) is refused. Missing a real
peak is recoverable; inventing one is not.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    BILLING_PERIOD_MINUTES,
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


async def _run_the_period(coordinator, monkeypatch, reading_at) -> None:
    """One 15-minute billing period (10:00-10:15), on the coordinator's real update cadence."""
    for minute in range(0, 15, UPDATE_INTERVAL_MINUTES):
        monkeypatch.setattr(
            dt_util,
            "now",
            lambda tz=None, _m=minute: datetime(2026, 1, 15, 10, _m, tzinfo=STOCKHOLM),
        )
        _meter(coordinator.hass, reading_at(minute))
        await coordinator._update_peak_tracking(_pump())

    # The first sample of the next period is what closes this one.
    monkeypatch.setattr(
        dt_util, "now", lambda tz=None: datetime(2026, 1, 15, 10, 15, tzinfo=STOCKHOLM)
    )
    _meter(coordinator.hass, 2.0)
    await coordinator._update_peak_tracking(_pump())


def _billed(coordinator) -> list[float]:
    return [
        round(call.kwargs["power_kw"], 2)
        for call in coordinator.effect.record_period_measurement.await_args_list
    ]


@pytest.mark.asyncio
async def test_a_period_the_meter_slept_through_is_not_billed(monkeypatch):
    """The bug shape: one reading at the top, then silence across two-thirds of the period."""
    coordinator = _coordinator()

    # 9 kW at 10:00. Then the meter is `unavailable` for the rest of the quarter.
    await _run_the_period(coordinator, monkeypatch, lambda minute: 9.0 if minute == 0 else None)

    assert _billed(coordinator) == [], (
        f"the coordinator billed {_billed(coordinator)} kW for a period in which the meter answered "
        f"ONCE and was `unavailable` for the remaining ten minutes. That figure is the 9 kW reading "
        f"taken at 10:00, stretched across a blackout nobody watched. It becomes one of the month's "
        f"three billed peaks, and the pump is throttled for the rest of the month to defend it."
    )


@pytest.mark.asyncio
async def test_a_fully_observed_hour_is_still_billed(monkeypatch):
    """The control. The guard must refuse blackouts, not customers."""
    coordinator = _coordinator()

    await _run_the_period(coordinator, monkeypatch, lambda minute: 6.0)

    assert _billed(coordinator) == [6.0], (
        f"a meter that answered on every cycle of the period billed {_billed(coordinator)}. A "
        f"fully observed 6 kW period is a 6 kW bill."
    )


@pytest.mark.asyncio
async def test_a_brief_dropout_is_tolerated(monkeypatch):
    """Sensors miss a beat. That is jitter, not a blackout, and the hour was still measured.

    One missed cycle leaves a gap of 2 x UPDATE_INTERVAL_MINUTES between readings, which is exactly
    MAX_BILLING_OBSERVATION_GAP_MINUTES. Refusing this would throw away most real periods and buy
    nothing: the reading either side of a five-minute blink is the same reading.
    """
    coordinator = _coordinator()

    await _run_the_period(coordinator, monkeypatch, lambda minute: None if minute == 5 else 6.0)

    assert _billed(coordinator) == [6.0], (
        f"a single missed update cycle threw the whole period away ({_billed(coordinator)}). Home "
        f"Assistant misses cycles routinely; a guard that discards a period for one blink discards "
        f"most of them, and the tariff record goes empty."
    )


@pytest.mark.asyncio
async def test_the_gap_that_is_tolerated_is_bounded_by_the_update_interval(monkeypatch):
    """The threshold is a judgement, so it is pinned where it can be argued with."""
    assert (
        MAX_BILLING_OBSERVATION_GAP_MINUTES > UPDATE_INTERVAL_MINUTES
    ), "the tolerated gap must exceed one update interval, or every ordinary period is discarded"
    assert MAX_BILLING_OBSERVATION_GAP_MINUTES < BILLING_PERIOD_MINUTES, (
        "the tolerated gap must be strictly shorter than the period, or a single-sample period - "
        "one reading resting to the boundary - could never be refused and the rule would not bite"
    )


@pytest.mark.asyncio
async def test_a_meter_that_dies_mid_period_bills_only_what_it_observed(monkeypatch):
    """The trailing silence counts as a gap, and the tolerance is deliberate.

    The meter answers at 10:00 and 10:05, then stays `unavailable`. The trailing span to the 10:15
    boundary is ten minutes - exactly the tolerated gap, i.e. one dropped cycle - so THIS period is
    billed from what was observed. The unobserved periods after it accumulate no samples at all and
    are never billed: a dead meter must not keep generating bills.
    """
    coordinator = _coordinator()

    for minute in (0, 5):
        monkeypatch.setattr(
            dt_util,
            "now",
            lambda tz=None, _m=minute: datetime(2026, 1, 15, 10, _m, tzinfo=STOCKHOLM),
        )
        _meter(coordinator.hass, 9.0)
        await coordinator._update_peak_tracking(_pump())
    # The meter is dead for 40 minutes; the next billable reading arrives at 10:45.
    for minute in (10, 15, 20, 25, 30, 35, 40):
        monkeypatch.setattr(
            dt_util,
            "now",
            lambda tz=None, _m=minute: datetime(2026, 1, 15, 10, _m, tzinfo=STOCKHOLM),
        )
        _meter(coordinator.hass, None)
        await coordinator._update_peak_tracking(_pump())
    monkeypatch.setattr(
        dt_util, "now", lambda tz=None: datetime(2026, 1, 15, 10, 45, tzinfo=STOCKHOLM)
    )
    _meter(coordinator.hass, 2.0)
    await coordinator._update_peak_tracking(_pump())

    assert _billed(coordinator) == [9.0], (
        f"billed {_billed(coordinator)}. The 10:00 period was observed for two of its three cycles "
        f"(one dropped cycle is tolerated by design) and bills 9.0; the quarters the meter slept "
        f"through entirely must bill NOTHING - a dead meter must not keep generating bills."
    )


@pytest.mark.asyncio
async def test_a_long_blackout_is_refused_even_when_the_power_was_low(monkeypatch):
    """It is not about the magnitude. An unobserved hour is unobserved, whatever it reads.

    A LOW reading stretched across a blackout is just as false as a high one - it simply fails
    quietly, by under-recording a peak that did happen, and leaving the month unprotected.
    """
    coordinator = _coordinator()

    def reading_at(minute: int) -> float | None:
        return 1.0 if minute == 0 else None

    await _run_the_period(coordinator, monkeypatch, reading_at)

    assert _billed(coordinator) == [], (
        f"billed {_billed(coordinator)} for a period the meter slept through. The house may have "
        f"drawn 9 kW for fifty unwatched minutes; a 1 kW bill would leave the month undefended."
    )

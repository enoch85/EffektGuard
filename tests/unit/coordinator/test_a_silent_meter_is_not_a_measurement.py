"""A power-meter state nobody has written for longer than a cycle is not this cycle's measurement.

A meter whose publisher has gone silent (an MQTT or P1 bridge that stopped, a template whose source
froze) keeps its last numeric state and stays `available`, so the availability, unit and parsing
checks all pass a number from hours ago. The coordinator re-read that one state every five minutes,
stamped it with each cycle's time, billed it as a period the meter never saw, and handed it to the
effect layer: a 9 kW state frozen above a 6 kW monthly peak voted CRITICAL (offset -3, weight 1.0)
on every cycle it stood - a heating cut held for as long as the meter stayed silent.

`State.last_reported` is the receipt time, moved by Home Assistant on every write even when the value
is unchanged. A state older than POWER_METER_MAX_AGE_MINUTES is refused exactly as an unavailable one
is: nothing billed, the frozen value never reaches the decision, and the stale sample is never
relabelled with the control loop's time. A steady draw from a live meter stays a measurement.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.core import State
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter, NibeState
from custom_components.effektguard.const import (
    BILLING_PERIOD_MINUTES,
    POWER_SOURCE_EXTERNAL_METER,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.effect_layer import EffectManager

METER = "sensor.house_power"
PERIOD_START = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
# The probe's case: the publisher stopped two hours ago, the entity still shows its last number.
TWO_HOURS_AGO = PERIOD_START - timedelta(hours=2)


def _coordinator() -> EffektGuardCoordinator:
    hass = MagicMock()
    hass.config.latitude = 59.33
    hass.config.longitude = 18.07

    nibe = MagicMock()
    nibe._power_sensor_entity = METER
    nibe.power_sensor_entity = METER

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
    """No phase currents, so a refused meter leaves only an estimate - which is never recorded."""
    return NibeState(
        outdoor_temp=-5.0,
        indoor_temp=21.0,
        supply_temp=42.0,
        return_temp=37.0,
        degree_minutes=-150.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=PERIOD_START,
        phase1_current=None,
    )


def _meter_state(watts: str, written_at: datetime) -> State:
    """A real Home Assistant State: `last_updated` is when the VALUE last changed, `last_reported`
    when the entity was last WRITTEN. A silent meter's `last_reported` stops moving; a live meter
    with a steady draw keeps writing the same value, so only `last_reported` moves."""
    return State(
        METER,
        watts,
        {"unit_of_measurement": "W"},
        last_updated=TWO_HOURS_AGO,
        last_reported=written_at,
    )


async def _cycle(coordinator, monkeypatch, now: datetime, watts: str, written_at: datetime) -> None:
    monkeypatch.setattr(dt_util, "now", lambda tz=None: now)
    monkeypatch.setattr(dt_util, "utcnow", lambda: now)
    coordinator.hass.states.get.return_value = _meter_state(watts, written_at)
    await coordinator._update_peak_tracking(_pump())


def _cycle_times(start: datetime) -> list[datetime]:
    """The cycles of the period starting at `start`, plus the first of the next - which closes it."""
    return [
        start + timedelta(minutes=minute)
        for minute in range(0, BILLING_PERIOD_MINUTES + 1, UPDATE_INTERVAL_MINUTES)
    ]


def _billed(coordinator) -> list[float]:
    return [
        round(call.kwargs["power_kw"], 2)
        for call in coordinator.effect.record_period_measurement.await_args_list
    ]


@pytest.mark.asyncio
async def test_a_state_nobody_has_written_for_hours_is_neither_billed_nor_decided_on(monkeypatch):
    """The defect: 9 kW, available, numeric, and two hours stale on every cycle of the period."""
    coordinator = _coordinator()

    for now in _cycle_times(PERIOD_START):
        await _cycle(coordinator, monkeypatch, now, "9000", written_at=TWO_HOURS_AGO)
        assert coordinator.peak_today_source != POWER_SOURCE_EXTERNAL_METER, (
            f"at {now:%H:%M} the coordinator took a meter state last written at "
            f"{TWO_HOURS_AGO:%H:%M} as this cycle's measurement ({coordinator.current_power_kw} kW, "
            f"source {coordinator.peak_today_source}). That number goes to the effect layer, which "
            f"votes CRITICAL at weight 1.0 against the month's peak - a heating cut that lasts as "
            f"long as the meter stays silent."
        )

    assert _billed(coordinator) == [], (
        f"billed {_billed(coordinator)} kW for a period in which the meter wrote nothing. The state "
        f"was last written at {TWO_HOURS_AGO:%H:%M}; stamping it with each cycle's time bills a "
        f"period nobody measured, and the tariff defends that peak for the rest of the month."
    )


@pytest.mark.asyncio
async def test_a_steady_draw_from_a_live_meter_is_still_a_measurement(monkeypatch):
    """The control. The value has not changed for two hours; the meter keeps writing it.

    `last_updated` stays at the last change, so a guard that read it would refuse every steady
    draw. `last_reported` moves on every write, and that is the one that says the meter is alive.
    """
    coordinator = _coordinator()

    for now in _cycle_times(PERIOD_START):
        await _cycle(coordinator, monkeypatch, now, "6000", written_at=now)

    assert _billed(coordinator) == [6.0], (
        f"a meter that wrote 6000 W on every cycle billed {_billed(coordinator)}. Its value had not "
        f"changed since {TWO_HOURS_AGO:%H:%M}, but a steady house is not a silent meter: the receipt "
        f"time is this cycle's, and the period was measured."
    )
    assert coordinator.current_power_kw == 6.0
    assert coordinator.peak_today_source == POWER_SOURCE_EXTERNAL_METER


@pytest.mark.asyncio
async def test_a_period_the_meter_wrote_nothing_in_is_not_billed(monkeypatch):
    """The meter's last write was a minute before the period began - fresh for the first cycle only.

    This is what pins the age limit to ONE update interval. At two intervals the :05 cycle would still
    accept a state written at 09:59, the period's longest silence would be exactly the tolerated gap,
    and 9 kW would be billed for a quarter in which the meter wrote nothing at all. The meter is back
    for the :15 cycle, which is what closes the period - as in production, where a period is closed
    by the next billable sample.
    """
    coordinator = _coordinator()
    last_write = PERIOD_START - timedelta(minutes=1)

    *silent, meter_back = _cycle_times(PERIOD_START)
    for now in silent:
        await _cycle(coordinator, monkeypatch, now, "9000", written_at=last_write)
    await _cycle(coordinator, monkeypatch, meter_back, "9000", written_at=meter_back)

    assert _billed(coordinator) == [], (
        f"billed {_billed(coordinator)} kW for a period in which the meter wrote nothing - its last "
        f"write was at {last_write:%H:%M}, before the period began. Each later cycle re-read that "
        f"state and relabelled it with its own time."
    )


@pytest.mark.asyncio
async def test_billing_resumes_when_the_meter_reports_again(monkeypatch):
    """Silent through one period, back for the next: only the measured period is billed."""
    coordinator = _coordinator()
    next_period = PERIOD_START + timedelta(minutes=BILLING_PERIOD_MINUTES)

    for now in _cycle_times(PERIOD_START)[:-1]:
        await _cycle(coordinator, monkeypatch, now, "9000", written_at=TWO_HOURS_AGO)
    for now in _cycle_times(next_period):
        await _cycle(coordinator, monkeypatch, now, "6000", written_at=now)

    assert _billed(coordinator) == [6.0], (
        f"billed {_billed(coordinator)}. The 10:00 period passed in silence and is not a bill; the "
        f"10:15 period was measured at 6 kW on every cycle and is. A meter that comes back must "
        f"bill again from its first fresh write, and nothing from before it."
    )
    assert coordinator.current_power_kw == 6.0
    assert coordinator.peak_today_source == POWER_SOURCE_EXTERNAL_METER


@pytest.mark.asyncio
async def test_the_savings_figure_does_not_read_a_silent_meter_either(monkeypatch):
    """The adapter reads the same entity for `NibeState.power_kw`, which prices the spot savings.

    A stale state must not come back as a measurement there: it either yields nothing or the
    temperature estimate, flagged as one, which the savings accumulator already refuses.
    """
    now = PERIOD_START
    monkeypatch.setattr(dt_util, "utcnow", lambda: now)
    hass = MagicMock()
    hass.states.get.return_value = _meter_state("9000", written_at=TWO_HOURS_AGO)
    adapter = NibeAdapter(hass, {"nibe_entity": "number.offset", "power_sensor_entity": METER})

    power, estimated = await adapter.get_power_consumption()

    assert power is None or estimated, (
        f"get_power_consumption() returned {power} kW as a MEASUREMENT from a meter state last "
        f"written at {TWO_HOURS_AGO:%H:%M}. That figure is multiplied by the spot price and reported "
        f"to the owner as kronor saved, every five minutes, for as long as the meter stays silent."
    )

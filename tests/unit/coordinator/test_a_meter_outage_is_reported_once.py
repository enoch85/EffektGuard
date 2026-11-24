"""A power-meter outage is reported when it begins and when it ends, not on every cycle it lasts.

A change-driven template can hold the same available state for hours. Availability-based billing
continues, but an old report must not authorize heat-reducing peak control. A fresh report with an
unchanged value remains suitable for control.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import State
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    BILLING_PERIOD_MINUTES,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.effect_layer import EffectManager

METER = "sensor.house_power"
START = datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc)
CYCLE = timedelta(minutes=UPDATE_INTERVAL_MINUTES)


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
    coordinator._power_sensor_available = True  # it HAS answered before
    coordinator.effect.record_period_measurement = AsyncMock(return_value=None)
    return coordinator


def _pump(now: datetime) -> NibeState:
    return NibeState(
        outdoor_temp=15.0,
        indoor_temp=21.0,
        supply_temp=30.0,
        return_temp=27.0,
        degree_minutes=-100.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=now,
        phase1_current=None,
    )


async def _cycle(coordinator, monkeypatch, now: datetime, state: State) -> None:
    monkeypatch.setattr(dt_util, "now", lambda tz=None: now)
    monkeypatch.setattr(dt_util, "utcnow", lambda: now)
    coordinator.hass.states.get.return_value = state
    await coordinator._update_peak_tracking(_pump(now))


def _meter(value: str, written: datetime) -> State:
    return State(
        METER, value, {"unit_of_measurement": "W"}, last_updated=written, last_reported=written
    )


@pytest.mark.asyncio
async def test_an_outage_is_reported_when_it_begins_and_when_it_ends(monkeypatch, caplog):
    coordinator = _coordinator()
    caplog.set_level(logging.DEBUG, logger="custom_components.effektguard")
    states = [_meter("6000", START)] + [_meter(STATE_UNAVAILABLE, START + CYCLE)] * 3
    states.append(_meter("6000", START + 4 * CYCLE))

    for step, state in enumerate(states):
        await _cycle(coordinator, monkeypatch, START + step * CYCLE, state)

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    back = [r.getMessage() for r in caplog.records if "reporting again" in r.getMessage()]
    assert (
        len(warnings) == 1 and "not reporting" in warnings[0]
    ), f"three cycles of one outage logged {len(warnings)} warnings: {warnings}"
    assert len(back) == 1, f"the meter's return was reported {len(back)} times: {back}"


@pytest.mark.asyncio
@pytest.mark.parametrize("power_w", [0, 20000])
async def test_a_held_value_bills_without_authorizing_peak_control(monkeypatch, caplog, power_w):
    """A held value is not an outage, and a held high draw must not keep cutting heat."""
    coordinator = _coordinator()
    steady = _meter(str(power_w), START - timedelta(hours=2))
    period = BILLING_PERIOD_MINUTES // UPDATE_INTERVAL_MINUTES

    for step in range(period + 1):
        await _cycle(coordinator, monkeypatch, START + step * CYCLE, steady)

    billed = [
        c.kwargs["power_kw"] for c in coordinator.effect.record_period_measurement.await_args_list
    ]
    assert billed == [power_w / 1000.0], billed
    assert coordinator.current_power_kw is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warnings, warnings

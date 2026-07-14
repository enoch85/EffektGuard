"""A billing hour's provenance is decided by every sample in it, not by the closing one.

The accumulator stored (timestamp, power) and nothing else; the coordinator stamped the
completed hour with whatever source the BOUNDARY cycle happened to have. So an hour whose
middle was measured at the pump's phase currents - because the grid meter dropped out -
became a billable whole-house-meter hour the moment the meter answered again at the top of
the next hour. The tariff bills whole-house grid import; fifty minutes of pump-only samples
are not that.

The rule: every sample from the grid meter -> billable meter hour. Meter and pump-current
samples mixed (or pump-only) -> control-grade, never shown as a bill. Anything weaker in the
mix -> not a measurement at all.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    POWER_SOURCE_EXTERNAL_METER,
    POWER_SOURCE_NIBE_CURRENTS,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.billing_period import BillingPeriodAccumulator
from custom_components.effektguard.optimization.effect_layer import EffectManager

STOCKHOLM = ZoneInfo("Europe/Stockholm")


def _hour(minute: int, hour: int = 10) -> datetime:
    return datetime(2026, 1, 15, hour, minute, tzinfo=STOCKHOLM)


class TestTheAccumulatorTracksSources:
    def test_a_pure_meter_hour_stays_a_meter_hour(self):
        acc = BillingPeriodAccumulator()
        for minute in range(0, 60, 5):
            acc.add(_hour(minute), 4.0, POWER_SOURCE_EXTERNAL_METER)
        completed = acc.add(_hour(0, hour=11), 2.0, POWER_SOURCE_EXTERNAL_METER)

        assert completed is not None
        assert completed.source == POWER_SOURCE_EXTERNAL_METER

    def test_one_pump_only_sample_degrades_the_hour_to_control_grade(self):
        acc = BillingPeriodAccumulator()
        for minute in range(0, 60, 5):
            source = POWER_SOURCE_NIBE_CURRENTS if minute == 30 else POWER_SOURCE_EXTERNAL_METER
            acc.add(_hour(minute), 4.0, source)
        completed = acc.add(_hour(0, hour=11), 2.0, POWER_SOURCE_EXTERNAL_METER)

        assert completed is not None
        assert completed.source == POWER_SOURCE_NIBE_CURRENTS


def _coordinator() -> EffektGuardCoordinator:
    hass = MagicMock()
    hass.config.latitude = 59.33
    hass.config.longitude = 18.07

    nibe = MagicMock()
    nibe._power_sensor_entity = "sensor.house_power"
    nibe.power_sensor_entity = "sensor.house_power"
    nibe.calculate_power_from_currents = MagicMock(return_value=9.0)

    entry = MagicMock()
    entry.data = {}
    entry.options = {}

    coordinator = EffektGuardCoordinator(
        hass, nibe, MagicMock(), MagicMock(), MagicMock(), EffectManager(hass), entry
    )
    coordinator.peak_today = 0.0
    coordinator.peak_this_month = 0.0
    coordinator._power_sensor_available = True
    coordinator.effect.record_period_measurement = AsyncMock(return_value=None)
    return coordinator


def _pump(with_currents: bool) -> NibeState:
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
        phase1_current=8.0 if with_currents else None,
        phase2_current=8.0 if with_currents else None,
        phase3_current=8.0 if with_currents else None,
    )


def _meter(hass, kw: float | None) -> None:
    state = MagicMock()
    if kw is None:
        state.state = "unavailable"
        state.attributes = {}
    else:
        state.state = str(kw)
        state.attributes = {"unit_of_measurement": "kW"}
    hass.states.get.return_value = state


@pytest.mark.asyncio
async def test_a_meter_dropout_hour_is_not_billed_as_a_meter_hour(monkeypatch):
    """Meter for the first half, pump currents for the second, meter again at the boundary.

    The boundary cycle's source is the METER - and the old stamping would have recorded the
    whole hour as a billable meter measurement. Half of it never saw the house.
    """
    coordinator = _coordinator()

    for minute in range(0, 60, UPDATE_INTERVAL_MINUTES):
        monkeypatch.setattr(dt_util, "now", lambda tz=None, _m=minute: _hour(_m))
        meter_alive = minute < 30
        _meter(coordinator.hass, 4.0 if meter_alive else None)
        await coordinator._update_peak_tracking(_pump(with_currents=not meter_alive))

    monkeypatch.setattr(dt_util, "now", lambda tz=None: _hour(0, hour=11))
    _meter(coordinator.hass, 2.0)
    await coordinator._update_peak_tracking(_pump(with_currents=False))

    calls = coordinator.effect.record_period_measurement.await_args_list
    assert len(calls) == 1, "the 10:00 hour was continuously sampled and must be recorded"
    assert calls[0].kwargs["source"] == POWER_SOURCE_NIBE_CURRENTS, (
        f"The hour was recorded with source {calls[0].kwargs['source']!r}. Fifty-five minutes "
        f"of it are fine, but 25 minutes were measured at the PUMP, not the grid connection - "
        f"the tariff bills whole-house import, so this hour is control-grade, not billable."
    )

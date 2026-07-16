"""A billing hour's provenance is decided by every sample in it, not by the closing cycle.

The accumulator stamps a completed hour with the WEAKEST source among its samples. So an hour
whose middle fell back to pump phase currents (the grid meter dropped out) is control-grade,
even if the meter answered again at the hour boundary - the tariff bills whole-house grid
import, and fifty minutes of pump-only samples are not that. A pure grid-meter hour stays
billable; anything weaker in the mix degrades it.
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
    def test_a_pure_meter_period_stays_a_meter_period(self):
        acc = BillingPeriodAccumulator()
        for minute in range(0, 15, 5):
            acc.add(_hour(minute), 4.0, POWER_SOURCE_EXTERNAL_METER)
        completed = acc.add(_hour(15), 2.0, POWER_SOURCE_EXTERNAL_METER)

        assert completed is not None
        assert completed.source == POWER_SOURCE_EXTERNAL_METER

    def test_one_pump_only_sample_degrades_the_period_to_control_grade(self):
        acc = BillingPeriodAccumulator()
        for minute in range(0, 15, 5):
            source = POWER_SOURCE_NIBE_CURRENTS if minute == 5 else POWER_SOURCE_EXTERNAL_METER
            acc.add(_hour(minute), 4.0, source)
        completed = acc.add(_hour(15), 2.0, POWER_SOURCE_EXTERNAL_METER)

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
async def test_a_meter_dropout_period_is_not_billed_as_a_meter_period(monkeypatch):
    """Meter for the first cycle, pump currents for the rest, meter again at the boundary.

    The boundary cycle's source is the METER - and the old stamping would have recorded the
    whole period as a billable meter measurement. Two-thirds of it never saw the house.
    """
    coordinator = _coordinator()

    for minute in range(0, 15, UPDATE_INTERVAL_MINUTES):
        monkeypatch.setattr(dt_util, "now", lambda tz=None, _m=minute: _hour(_m))
        meter_alive = minute < 5
        _meter(coordinator.hass, 4.0 if meter_alive else None)
        await coordinator._update_peak_tracking(_pump(with_currents=not meter_alive))

    monkeypatch.setattr(dt_util, "now", lambda tz=None: _hour(15))
    _meter(coordinator.hass, 2.0)
    await coordinator._update_peak_tracking(_pump(with_currents=False))

    calls = coordinator.effect.record_period_measurement.await_args_list
    assert len(calls) == 1, "the 10:00 period was continuously sampled and must be recorded"
    assert calls[0].kwargs["source"] == POWER_SOURCE_NIBE_CURRENTS, (
        f"The period was recorded with source {calls[0].kwargs['source']!r}. Ten minutes of it "
        f"were measured at the PUMP, not the grid connection - the tariff bills whole-house "
        f"import, so this period is control-grade, not billable."
    )

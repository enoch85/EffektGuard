"""A savings figure is money, and must not be computed from an estimated power reading.

`NibeState.power_kw` is filled by `get_power_consumption()`, which falls back to a temperature
curve fit when no power sensor is configured - a guess in the same field as a measurement, clamped
to never read below 1.0 kW even with the compressor off. The coordinator fed that into
`_daily_spot_savings`, which the owner reads as kronor: a savings report every day derived from a
formula that never saw a watt.

The estimate stays available to layers that want a rough magnitude, but it must now carry a
`power_is_estimated` flag, and anything that reports or bills money must ask it first.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter, NibeState


def _nibe(power_kw: float, estimated: bool) -> NibeState:
    return NibeState(
        outdoor_temp=-5.0,
        indoor_temp=21.0,
        supply_temp=42.0,
        return_temp=37.0,
        degree_minutes=-150.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=dt_util.utcnow(),
        power_kw=power_kw,
        power_is_estimated=estimated,
    )


@pytest.fixture
def coordinator_for_savings():
    """A coordinator with a real savings calculator and a SEK price unit."""
    from custom_components.effektguard.coordinator import EffektGuardCoordinator
    from custom_components.effektguard.optimization.effect_layer import EffectManager

    hass = MagicMock()
    hass.config.latitude = 59.33
    hass.config.longitude = 18.07

    gespot = MagicMock()
    gespot.price_unit = "SEK/kWh"

    entry = MagicMock()
    entry.data = {}
    entry.options = {}

    coordinator = EffektGuardCoordinator(
        hass, MagicMock(), gespot, MagicMock(), MagicMock(), EffectManager(hass), entry
    )
    coordinator._daily_spot_savings = 0.0
    return coordinator


def _a_day_of_prices() -> PriceData:
    """A day with a genuinely cheap current quarter, so savings would be non-zero if computed."""
    midnight = dt_util.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today = [
        QuarterPeriod(start_time=midnight + timedelta(minutes=15 * q), price=100.0)
        for q in range(96)
    ]
    now_index = PriceData(today=today, tomorrow=[], has_tomorrow=False).get_period_index(
        dt_util.now()
    )
    assert now_index is not None, "precondition: some quarter must contain 'now'"
    today[now_index] = QuarterPeriod(start_time=today[now_index].start_time, price=1.0)
    return PriceData(today=today, tomorrow=[], has_tomorrow=False)


@pytest.mark.asyncio
async def test_an_estimate_is_marked_as_one():
    """The adapter must say which it gave you.

    No power sensor is configured, so the only thing left is the temperature curve fit.
    """
    state = MagicMock()
    state.state = "40.0"
    state.last_reported = dt_util.utcnow()
    state.last_updated = dt_util.utcnow()

    hass = MagicMock()
    hass.states.get.return_value = state

    adapter = NibeAdapter(hass, {"nibe_entity": "number.offset"})
    adapter._entity_cache = {"supply_temp": "sensor.supply", "outdoor_temp": "sensor.outdoor"}

    power, estimated = await adapter.get_power_consumption()

    assert power is not None, "precondition: the temperature fallback should produce a number"
    assert estimated is True, (
        f"get_power_consumption() returned {power:.2f} kW derived from supply and outdoor "
        f"temperature and reported it as a measurement. Nothing downstream can now tell it from a "
        f"reading off a real meter."
    )


@pytest.mark.asyncio
async def test_a_measurement_is_not_marked_as_an_estimate():
    """The precondition in the other direction: a real meter must not be dismissed as a guess."""
    state = MagicMock()
    state.state = "4200"
    state.attributes = {"unit_of_measurement": "W"}
    state.last_reported = dt_util.utcnow()
    state.last_updated = dt_util.utcnow()

    hass = MagicMock()
    hass.states.get.return_value = state

    adapter = NibeAdapter(
        hass, {"nibe_entity": "number.offset", "power_sensor_entity": "sensor.house_power"}
    )

    power, estimated = await adapter.get_power_consumption()

    assert power == pytest.approx(4.2)
    assert estimated is False


@pytest.mark.asyncio
async def test_no_savings_are_reported_from_estimated_power(coordinator_for_savings):
    """The whole point. No power sensor means no savings figure - not a plausible one."""
    coordinator = coordinator_for_savings

    coordinator._accumulate_spot_savings(_nibe(power_kw=4.4, estimated=True), _a_day_of_prices())

    assert coordinator._daily_spot_savings == 0.0, (
        f"{coordinator._daily_spot_savings:.2f} kr of savings were accumulated from a power figure "
        f"that was estimated from supply and outdoor temperature. The owner reads that number as "
        f"money saved."
    )


@pytest.mark.asyncio
async def test_savings_are_still_reported_from_measured_power(coordinator_for_savings):
    """And the regression guard: an owner WITH a power meter must not lose their savings report."""
    coordinator = coordinator_for_savings

    coordinator._accumulate_spot_savings(_nibe(power_kw=4.4, estimated=False), _a_day_of_prices())

    assert coordinator._daily_spot_savings != 0.0, (
        "A measured power reading produced no savings figure at all. The guard against estimated "
        "power has been drawn too wide and now refuses real measurements."
    )

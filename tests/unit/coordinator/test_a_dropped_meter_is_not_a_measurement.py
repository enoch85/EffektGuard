"""A configured power meter that drops out must not have its estimate billed as a meter reading.

A meter goes `unavailable` routinely (a Zigbee plug loses its router, an MQTT bridge restarts).
The old billing guard asked whether a power sensor was CONFIGURED, not whether one had just
MEASURED anything - so once the meter dropped out, the compressor-Hz estimate that replaced it
was recorded as a tariff peak and stamped with source "external_meter". Provenance was falsified,
and effect tariffs bill the top-3 hours of the month, so a phantom peak stands for weeks.

The fix: a measurement carries where it came from, and the billing guard asks that (via
PEAK_CONTROL_POWER_SOURCES), not the config entry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.effect_layer import EffectManager


@pytest.fixture
def coordinator_with_external_meter():
    """A coordinator whose owner has configured a whole-house power meter."""
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
    coordinator.effect.record_period_measurement = AsyncMock(return_value=None)
    return coordinator


def _pump_running_but_unmetered() -> NibeState:
    """The compressor is working. No phase-current sensors, so Hz is all that is left."""
    return NibeState(
        outdoor_temp=-5.0,
        indoor_temp=21.0,
        supply_temp=42.0,
        return_temp=37.0,
        degree_minutes=-150.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        phase1_current=None,
        phase2_current=None,
        phase3_current=None,
        compressor_hz=60,
    )


async def _run_a_complete_billing_period(coordinator, nibe_data, monkeypatch) -> None:
    """Samples one 15-minute period whole, so it completes and is recorded.

    The owner's effect tariff bills the 15-minute period mean (operator models vary - F-107).
    """
    for hour, minute in [(10, m) for m in range(0, 15, 5)] + [(10, 15)]:
        monkeypatch.setattr(
            dt_util,
            "now",
            lambda tz=None, hour=hour, minute=minute: datetime(
                2026, 1, 15, hour, minute, tzinfo=timezone.utc
            ),
        )
        await coordinator._update_peak_tracking(nibe_data)


@pytest.mark.asyncio
async def test_a_meter_that_drops_out_does_not_keep_billing(
    coordinator_with_external_meter, monkeypatch
):
    """The meter answered once, hours ago. It is not answering now."""
    coordinator = coordinator_with_external_meter

    # It worked at startup. That is what latches the flag - and unsubscribes the listener.
    coordinator._power_sensor_available = True

    dropped_out = MagicMock()
    dropped_out.state = "unavailable"
    dropped_out.attributes = {}
    coordinator.hass.states.get.return_value = dropped_out

    await _run_a_complete_billing_period(coordinator, _pump_running_but_unmetered(), monkeypatch)

    coordinator.effect.record_period_measurement.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_meter_reporting_garbage_does_not_keep_billing(
    coordinator_with_external_meter, monkeypatch
):
    """The other way in: the state is present and unparseable.

    `except (ValueError, TypeError)` warns and leaves `current_power` as None - and then the very same
    fall-through to the estimate happens, with the very same "the entity is configured, so this must be
    a real measurement" conclusion at the end.
    """
    coordinator = coordinator_with_external_meter
    coordinator._power_sensor_available = True

    garbage = MagicMock()
    garbage.state = "n/a"
    garbage.attributes = {"unit_of_measurement": "W"}
    coordinator.hass.states.get.return_value = garbage

    await _run_a_complete_billing_period(coordinator, _pump_running_but_unmetered(), monkeypatch)

    coordinator.effect.record_period_measurement.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_estimate_is_never_stamped_as_a_meter_reading(
    coordinator_with_external_meter, monkeypatch
):
    """Whatever else happens, the record must not LIE about where the number came from.

    The daily peak is allowed to hold an estimate - it is a display value. What it must never do is
    claim the estimate came from the external meter, because that is the one field anyone would consult
    to find out whether a peak can be trusted.
    """
    coordinator = coordinator_with_external_meter
    coordinator._power_sensor_available = True

    dropped_out = MagicMock()
    dropped_out.state = "unavailable"
    dropped_out.attributes = {}
    coordinator.hass.states.get.return_value = dropped_out

    await _run_a_complete_billing_period(coordinator, _pump_running_but_unmetered(), monkeypatch)

    assert coordinator.peak_today_source != "external_meter", (
        f"A peak of {coordinator.peak_today:.2f} kW, estimated from compressor Hz because the meter "
        f"was unavailable, was recorded with source 'external_meter'. Nothing downstream - and nobody "
        f"reading the logs - can now tell it apart from a real reading."
    )


@pytest.mark.asyncio
async def test_a_working_meter_still_bills(coordinator_with_external_meter, monkeypatch):
    """The precondition, and the thing that must not regress.

    A guard that refuses real measurements is worse than the bug it fixes: it would silently stop peak
    tracking for every owner whose meter works. This is the test that says the fix costs them nothing.
    """
    coordinator = coordinator_with_external_meter
    coordinator._power_sensor_available = True

    working = MagicMock()
    working.state = "4200"
    working.attributes = {"unit_of_measurement": "W"}
    coordinator.hass.states.get.return_value = working

    await _run_a_complete_billing_period(coordinator, _pump_running_but_unmetered(), monkeypatch)

    coordinator.effect.record_period_measurement.assert_awaited_once()
    recorded = coordinator.effect.record_period_measurement.await_args.kwargs
    assert recorded["power_kw"] == pytest.approx(4.2)
    assert coordinator.peak_today_source == "external_meter"

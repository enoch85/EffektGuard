"""The power meter goes away, and its estimate is billed as if the meter were still there.

The coordinator is scrupulous about this. It says so three times:

    # PRIORITY 3: Estimate from compressor Hz (NOT FOR PEAK TRACKING!)
    # WARNING: Never record estimated peaks - billing must use real measurements only
    ...
    # CRITICAL: Only record monthly peaks with REAL measurements
    # Estimates are NEVER used for monthly peak tracking - billing must be accurate
    has_real_measurement = has_external_power_sensor or nibe_data.phase1_current is not None
    if not has_real_measurement:
        return

Read the guard again. `has_external_power_sensor` is:

    has_external_power_sensor = hasattr(self.nibe, "_power_sensor_entity") and bool(
        self.nibe.power_sensor_entity
    )

That is **"is a power sensor configured"**, not **"did a power sensor just measure something"**. The
guard asks about the config entry. It cannot ask about the measurement, because by the time it runs,
`current_power` is a bare float with no memory of where it came from.

And the sensor's availability flag is a one-way latch. It is set True the first time the meter is seen
alive, and the listener that set it then **unsubscribes itself** - "we don't need this listener anymore".
Nothing ever sets it back to False. So the coordinator has no mechanism to notice a meter going away.

Put those together and a configured meter that drops out - a Zigbee plug losing its router, an MQTT
bridge restarting, a Shelly rebooting; the ordinary weather of a Home Assistant install - walks straight
through:

  1. state is `unavailable`, so the reader is skipped and `current_power` stays None;
  2. `elif not self._power_sensor_available:` is False, because the flag latched True hours ago, so the
     early return never fires;
  3. PRIORITY 3 estimates the power from compressor Hz, logging "[ESTIMATE ONLY - not used for peak
     billing]";
  4. `has_real_measurement` is True, because the entity is still *configured*;
  5. the estimate is accumulated into the quarter mean and **recorded as a tariff peak**.

In the same cycle, the log says the number must never be used for billing, and then it is used for
billing.

It is also **stamped as a real measurement**: `measurement_source` is derived from
`has_external_power_sensor and current_power >= 0.5`, which the estimate satisfies, so the peak is
recorded with source "external_meter". The provenance is not merely lost. It is falsified, and there is
nothing left in the record to tell the owner - or the next maintainer - that the number was invented.

Swedish effect tariffs bill the top-3 quarter means of the month. A phantom peak survives the whole
month: it corrupts what EffektGuard believes the bill will be, what it reports to the owner, and every
decision the effect layer makes against it.

The fix is not a better guess. It is to stop passing power around as a bare float. A measurement has to
carry where it came from, and the billing guard has to ask *that* - not the config entry.
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


async def _run_a_complete_billing_hour(coordinator, nibe_data, monkeypatch) -> None:
    """Samples from 10:00 through 11:00, so the HOUR is observed whole and recorded.

    It used to run 10:00-10:15 and call that a billing period. The Swedish effect tariff bills the
    HOURLY mean - Ellevio: "the measurement uses hourly averages" - so a quarter-hour never
    completes a billing period at all.
    """
    for hour, minute in [(10, m) for m in range(0, 60, 5)] + [(11, 0)]:
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

    await _run_a_complete_billing_hour(coordinator, _pump_running_but_unmetered(), monkeypatch)

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

    await _run_a_complete_billing_hour(coordinator, _pump_running_but_unmetered(), monkeypatch)

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

    await _run_a_complete_billing_hour(coordinator, _pump_running_but_unmetered(), monkeypatch)

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

    await _run_a_complete_billing_hour(coordinator, _pump_running_but_unmetered(), monkeypatch)

    coordinator.effect.record_period_measurement.assert_awaited_once()
    recorded = coordinator.effect.record_period_measurement.await_args.kwargs
    assert recorded["power_kw"] == pytest.approx(4.2)
    assert coordinator.peak_today_source == "external_meter"

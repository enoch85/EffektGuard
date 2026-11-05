"""A peak the coordinator defends or a sensor shows belongs to the day or month it is shown for.

The month's peaks are the EffectManager's stored record: saved every cycle, dated, and pruned to the
month they are loaded in. The coordinator kept a copy of the highest one, and the copy drifted:
`reset_peak_tracking` cleared the record and left the old peak shown and handed to the effect layer,
and after a restart in a new month the `peak_this_month` sensor restored last month's peak into the
copy. The day's peak lives only in memory, so a restart restores it from the sensor - which handed
back yesterday's peak as today's when Home Assistant was down across midnight.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from homeassistant.core import State
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import _async_register_services, _create_coordinator
from custom_components.effektguard.const import (
    DOMAIN,
    EFFECT_STORAGE_VERSION,
    POWER_SOURCE_EXTERNAL_METER,
    SERVICE_RESET_PEAK_TRACKING,
    STORAGE_KEY,
)
from custom_components.effektguard.optimization.effect_layer import PeakEvent
from custom_components.effektguard.sensor import SENSORS, EffektGuardSensor

STOCKHOLM = ZoneInfo("Europe/Stockholm")
OCTOBER_PEAK_KW = 6.2
OCTOBER_PEAK_AT = datetime(2026, 10, 20, 8, 0, tzinfo=STOCKHOLM)
MID_OCTOBER = datetime(2026, 10, 22, 10, 0, 10, tzinfo=STOCKHOLM)
FIRST_RESTART_OF_NOVEMBER = datetime(2026, 11, 1, 0, 30, 10, tzinfo=STOCKHOLM)
DAILY_PEAK_KW = 5.1
DAILY_PEAK_PERIOD = 0
YESTERDAY_EVENING = datetime(2026, 10, 31, 23, 40, tzinfo=STOCKHOLM)
EARLIER_TODAY = datetime(2026, 11, 1, 0, 10, tzinfo=STOCKHOLM)
DAYTIME_PERIOD = 32


async def _coordinator_after_a_restart(hass, hass_storage, freezer, now):
    """The coordinator as setup builds it, at `now`, over a store holding October's peak."""
    await hass.config.async_set_time_zone("Europe/Stockholm")
    freezer.move_to(now)
    hass_storage[STORAGE_KEY] = {
        "version": EFFECT_STORAGE_VERSION,
        "minor_version": 1,
        "key": STORAGE_KEY,
        "data": {
            "peaks": [
                PeakEvent(
                    timestamp=OCTOBER_PEAK_AT,
                    period_of_day=DAYTIME_PERIOD,
                    actual_power=OCTOBER_PEAK_KW,
                    effective_power=OCTOBER_PEAK_KW,
                    is_daytime=True,
                    source=POWER_SOURCE_EXTERNAL_METER,
                ).to_dict()
            ]
        },
    }
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    return await _create_coordinator(hass, entry), entry


def _sensor(coordinator, entry, key: str, last_state: State) -> EffektGuardSensor:
    description = next(d for d in SENSORS if d.key == key)
    sensor = EffektGuardSensor(coordinator, entry, description)
    sensor.hass = coordinator.hass
    sensor.entity_id = f"sensor.effektguard_{key}"
    sensor.async_get_last_state = AsyncMock(return_value=last_state)
    return sensor


async def test_a_reset_clears_the_peak_the_effect_layer_is_handed(hass, hass_storage, freezer):
    """After reset_peak_tracking the month has no peak - in the record and in what is defended."""
    coordinator, entry = await _coordinator_after_a_restart(
        hass, hass_storage, freezer, MID_OCTOBER
    )
    assert coordinator.peak_this_month == OCTOBER_PEAK_KW  # anchor: the stored peak is defended

    hass.data[DOMAIN] = {entry.entry_id: coordinator}
    await _async_register_services(hass)
    coordinator.async_request_refresh = AsyncMock()
    await hass.services.async_call(DOMAIN, SERVICE_RESET_PEAK_TRACKING, {}, blocking=True)

    assert coordinator.effect.get_monthly_peak_summary()["count"] == 0
    assert coordinator.peak_this_month == 0.0, (
        f"reset_peak_tracking cleared the stored peaks, and the coordinator still hands the effect "
        f"layer {coordinator.peak_this_month} kW to defend and the sensor shows it - until the next "
        f"new peak, and without a meter never."
    )


async def test_last_month_s_peak_is_not_restored_into_a_new_month(hass, hass_storage, freezer):
    """A restart on the 1st: the store drops October, and the sensor's last state must not undo it."""
    coordinator, entry = await _coordinator_after_a_restart(
        hass, hass_storage, freezer, FIRST_RESTART_OF_NOVEMBER
    )
    sensor = _sensor(
        coordinator,
        entry,
        "peak_this_month",
        State("sensor.effektguard_peak_this_month", str(OCTOBER_PEAK_KW)),
    )

    await sensor.async_added_to_hass()

    assert (coordinator.peak_this_month, sensor.native_value) == (0.0, 0.0), (
        f"November has no peak yet, but the coordinator defends {coordinator.peak_this_month} kW "
        f"and the sensor shows {sensor.native_value} kW: October's peak, restored from the "
        f"sensor's last state."
    )


@pytest.mark.parametrize(
    ("peak_time", "restored"),
    [
        pytest.param(YESTERDAY_EVENING, (0.0, None), id="yesterday's peak"),
        pytest.param(EARLIER_TODAY, (DAILY_PEAK_KW, EARLIER_TODAY), id="today's peak"),
    ],
)
async def test_the_daily_peak_restored_after_a_restart_is_today_s(
    hass, hass_storage, freezer, peak_time, restored
):
    """Restored when its peak_time is today - with that time, so the next restart can date it."""
    coordinator, entry = await _coordinator_after_a_restart(
        hass, hass_storage, freezer, FIRST_RESTART_OF_NOVEMBER
    )
    sensor = _sensor(
        coordinator,
        entry,
        "peak_today",
        State(
            "sensor.effektguard_peak_today",
            str(DAILY_PEAK_KW),
            {
                "peak_time": peak_time.isoformat(),
                "peak_billing_period": DAILY_PEAK_PERIOD,
                "measurement_source": POWER_SOURCE_EXTERNAL_METER,
            },
        ),
    )

    await sensor.async_added_to_hass()

    assert (coordinator.peak_today, coordinator.peak_today_time) == restored, (
        f"restarted at {FIRST_RESTART_OF_NOVEMBER:%Y-%m-%d %H:%M} with a {DAILY_PEAK_KW} kW peak "
        f"from {peak_time:%Y-%m-%d %H:%M}: the coordinator holds {coordinator.peak_today} kW "
        f"at {coordinator.peak_today_time}"
    )

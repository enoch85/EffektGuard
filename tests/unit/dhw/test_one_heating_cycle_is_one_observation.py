"""One heating cycle of the tank is one observation of its heating rate.

`calculate_heating_rate` found the most recent heating cycle in the 4-hour BT7 history and fed it
to the learned rate on every call. It is called three times a coordinator cycle (Lane 1's estimate
and `_check_upcoming_demand_period`, which runs twice), so one cycle was learned 36 times an hour
for as long as it stayed in the history, and the weighted average was the last cycle's rate
(review DHW F11). Learning now happens where the readings arrive, once per finished cycle.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.dhw_optimizer import (
    DHWDemandPeriod,
    IntelligentDHWScheduler,
)
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer

STOCKHOLM = ZoneInfo("Europe/Stockholm")
START = datetime(2026, 1, 15, 0, 0, tzinfo=STOCKHOLM)
STEP = timedelta(minutes=5)
RATE = 14.0  # °C/hour: 44 -> 58 C over the hour below


def test_a_heating_cycle_is_learned_once_however_often_it_is_asked_about():
    detector = ClimateZoneDetector(latitude=59.33)
    scheduler = IntelligentDHWScheduler(
        demand_periods=[DHWDemandPeriod(availability_hour=7, target_temp=50.0, duration_hours=2)],
        climate_detector=detector,
        emergency_layer=EmergencyLayer(detector, heating_type="radiator"),
        user_target_temp=50.0,
    )
    cooling = [46.0, 45.4, 44.8, 44.0]
    heating = [44.0 + RATE * k / 12 for k in range(1, 13)]
    readings = cooling + heating

    now = START
    for temp in readings:
        scheduler.update_bt7_temperature(temp, now)
        now += STEP

    # An hour of coordinator cycles after the cycle: a reading and a recommendation each.
    for _ in range(12):
        scheduler.update_bt7_temperature(readings[-1], now)
        scheduler.calculate_recommendation(
            current_dhw_temp=readings[-1],
            thermal_debt=-200.0,
            space_heating_demand=1.0,
            outdoor_temp=0.0,
            indoor_temp=21.0,
            target_indoor=21.0,
            price_classification="normal",
            current_time=now,
            price_periods=None,
        )
        now += STEP

    assert scheduler.heating_rate_observations == 1, (
        f"one heating cycle became {scheduler.heating_rate_observations} observations of the "
        f"heating rate in an hour"
    )
    assert scheduler.learned_heating_rate == pytest.approx(RATE)

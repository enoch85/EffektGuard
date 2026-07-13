"""Two readers of one power sensor, disagreeing by a factor of a thousand.

The same entity - the owner's whole-house meter - is read in two places, and they default differently
when it has no unit:

    nibe_adapter.get_power_consumption()
        unit = state.attributes.get("unit_of_measurement", "").lower()
        if unit == "w":
            power = power / 1000.0                      # absent unit -> kept as kW

    coordinator._update_peak_tracking()
        power_unit = str(power_state.attributes.get("unit_of_measurement", "W")).lower()
        if power_unit == "w":
            current_power = current_power / WATTS_PER_KILOWATT    # absent unit -> divided by 1000

A unit-less sensor reporting `6000` is therefore **6000 kW** to the adapter and **6.0 kW** to the
coordinator, in the same process, in the same five-minute cycle. One of them feeds savings and model
validation; the other feeds peak protection and the tariff record.

The reverse case is the dangerous one, and the coordinator's own comment describes it exactly:

    # Convert to kW only when the meter reports watts -
    # a kW meter must not be divided a second time
    # (a 6.0 kW whole-house meter would become 0.006 kW,
    # invalidating peak protection and peak records)

That comment was written for a real regression - there is a test class named
`TestKilowattMeterNotDividedTwice`. But the fix only taught the code about an explicit "kW" unit, while
leaving the *absent* unit defaulting to "W". So a unit-less meter already reporting kilowatts still
becomes 0.006 kW, still invalidates peak protection, and still does it silently. **The comment
describes the bug the code still has.**

Neither reader knows about MW, and neither notices that a `kWh` *energy* sensor - an easy thing to pick
from an entity dropdown, and cumulative, so it climbs forever - is not a power sensor at all.

There is no defensible default here. Watts and kilowatts are a factor of a thousand apart, and this
number decides whether the house is about to set a monthly billing peak. A power sensor with no unit is
a misconfiguration, and the honest response is the one this codebase already uses for a missing price
source: refuse to guess, withdraw the feature that depends on it, and raise a repair issue that tells
the owner exactly what to fix.

What must never happen again is that two places invent two different answers to the same question.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter, NibeState
from custom_components.effektguard.const import POWER_SOURCE_EXTERNAL_METER
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.effect_layer import EffectManager

POWER_ENTITY = "sensor.house_power"


def _hass_with_power_sensor(value: str, unit: str | None) -> MagicMock:
    state = MagicMock()
    state.state = value
    state.attributes = {} if unit is None else {"unit_of_measurement": unit}
    state.last_reported = dt_util.utcnow()
    state.last_updated = dt_util.utcnow()

    hass = MagicMock()
    hass.config.latitude = 59.33
    hass.config.longitude = 18.07
    hass.states.get.return_value = state
    return hass


async def _adapter_says(value: str, unit: str | None) -> float | None:
    """What the adapter reports as a MEASUREMENT. None when it declines to accept the sensor.

    A refused sensor falls through to `_estimate_power_from_temps`, which is a legitimate thing for
    the adapter to do - the estimate is flagged, and layers that only need a magnitude may use it.
    It is not a reading of this sensor, so it is not what this file is about.
    """
    hass = _hass_with_power_sensor(value, unit)
    adapter = NibeAdapter(
        hass, {"nibe_entity": "number.offset", "power_sensor_entity": POWER_ENTITY}
    )
    power, estimated = await adapter.get_power_consumption()
    return None if estimated else power


async def _coordinator_says(value: str, unit: str | None) -> float | None:
    """What the coordinator took FROM THE METER. None when it declined to accept the sensor.

    Same distinction as `_adapter_says`: a refused sensor still leaves the coordinator estimating a
    power figure for the decision layers, but that estimate is not billable and is not a reading of
    this sensor. `peak_today_source` is how the coordinator records which it was.
    """
    hass = _hass_with_power_sensor(value, unit)

    nibe = MagicMock()
    nibe._power_sensor_entity = POWER_ENTITY
    nibe.power_sensor_entity = POWER_ENTITY

    entry = MagicMock()
    entry.data = {}
    entry.options = {}

    coordinator = EffektGuardCoordinator(
        hass, nibe, MagicMock(), MagicMock(), MagicMock(), EffectManager(hass), entry
    )
    coordinator.peak_today = 0.0
    coordinator.peak_this_month = 0.0
    coordinator._power_sensor_available = True
    coordinator.effect.record_quarter_measurement = AsyncMock(return_value=None)

    await coordinator._update_peak_tracking(
        NibeState(
            outdoor_temp=-5.0,
            indoor_temp=21.0,
            supply_temp=42.0,
            return_temp=37.0,
            degree_minutes=-150.0,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        )
    )
    if coordinator.peak_today_source != POWER_SOURCE_EXTERNAL_METER:
        return None
    return coordinator.current_power_kw


@pytest.mark.asyncio
@pytest.mark.parametrize(("value", "unit"), [("6000", "W"), ("6.0", "kW"), ("6000", "MW")])
async def test_both_readers_of_the_same_sensor_give_the_same_answer(value, unit):
    """Whatever the right answer is, there cannot be two of them."""
    adapter = await _adapter_says(value, unit)
    coordinator = await _coordinator_says(value, unit)

    assert adapter == coordinator, (
        f"A power sensor reporting {value!r} with unit {unit!r} is read as {adapter} kW by the NIBE "
        f"adapter and {coordinator} kW by the coordinator - the same entity, the same instant. One "
        f"drives savings and model validation; the other drives peak protection and the tariff "
        f"record."
    )


@pytest.mark.asyncio
async def test_a_sensor_with_no_unit_is_refused_by_both_readers():
    """The 1000x split. Neither reader may take the number, and neither may take a different one.

    The adapter used to keep `6000` as 6000 kW; the coordinator divided the same 6000 down to 6.0 kW.
    Six megawatts and six kilowatts, from one sensor, in one cycle. There is no answer that makes both
    right, so neither is allowed to invent one.
    """
    assert await _adapter_says("6000", None) is None, (
        "The NIBE adapter accepted a power sensor with no declared unit, keeping 6000 verbatim as "
        "6000 kW - six megawatts, fed to savings and model validation."
    )
    assert await _coordinator_says("6000", None) is None, (
        "The coordinator accepted a power sensor with no declared unit, assuming watts and dividing "
        "by 1000. The adapter, reading the SAME entity in the SAME cycle, assumed kilowatts."
    )


@pytest.mark.asyncio
async def test_a_kilowatt_meter_with_no_unit_does_not_become_six_watts():
    """The failure the coordinator's own comment warns about, which its default still creates.

    A 6.0 kW whole-house meter that carries no unit is divided by 1000 into 0.006 kW. Peak protection
    then sees a house drawing six watts and never fires - all month, silently.
    """
    coordinator = await _coordinator_says("6.0", None)

    assert coordinator is None or coordinator > 0.5, (
        f"A meter reading 6.0 with no declared unit was taken as {coordinator} kW. If it is a "
        f"kilowatt meter - and 6.0 is a kilowatt-shaped number; a watt meter would say 6000 - then "
        f"peak protection has just been told the house is drawing six watts, and it will not fire "
        f"again this month."
    )


@pytest.mark.asyncio
async def test_an_energy_sensor_is_not_a_power_sensor():
    """kWh is cumulative. It only ever climbs, and it is one dropdown entry away from the right one.

    Picked by mistake, it is read as if it were instantaneous power: a house that has consumed 4300 kWh
    this year reports a 4300 kW peak, and every subsequent decision is made against it.
    """
    assert await _adapter_says("4300", "kWh") is None, (
        "A cumulative ENERGY sensor (kWh) was accepted as instantaneous power. It never falls, so the "
        "recorded peak becomes the meter's lifetime total and stays there."
    )
    assert (
        await _coordinator_says("4300", "kWh") is None
    ), "A cumulative ENERGY sensor (kWh) was accepted as instantaneous power for peak billing."

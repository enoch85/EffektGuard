"""The effect tariff counts at most ONE peak per day - the three must come from THREE days.

Ellevio, "Så fungerar effektavgiften": the monthly charge is the mean of the three highest hourly
peaks, and "only one power peak per day is counted, so the three peaks must come from three
different days." https://www.ellevio.se/abonnemang/elnatspriser/ny-prismodell-baserad-pa-effekt/

Date-blind top-3 let one cold day fill all three slots. That overstates the bill and understates
the margin the pump is then throttled against (9/8/7 from one Saturday vs a real third day of 4 kW).
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.effektguard.const import POWER_SOURCE_EXTERNAL_METER
from custom_components.effektguard.optimization.effect_layer import EffectManager


@pytest.fixture
def manager():
    mgr = EffectManager(MagicMock())
    mgr.async_save = AsyncMock()
    return mgr


async def _record(mgr, power_kw, day, hour):
    return await mgr.record_period_measurement(
        power_kw=power_kw,
        period=hour,
        timestamp=datetime(2026, 1, day, hour, 0),
        source=POWER_SOURCE_EXTERNAL_METER,
    )


@pytest.mark.asyncio
async def test_three_hours_on_one_day_count_as_one_peak(manager):
    """9, 8 and 7 kW on the same date must yield ONE tracked peak, not three."""
    await _record(manager, 9.0, day=10, hour=7)
    await _record(manager, 8.0, day=10, hour=18)
    await _record(manager, 7.0, day=10, hour=20)

    assert len(manager._monthly_peaks) == 1
    assert manager._monthly_peaks[0].actual_power == 9.0


@pytest.mark.asyncio
async def test_a_higher_hour_replaces_its_own_day(manager):
    """The day's counted peak is its highest hour - a later, higher hour takes the slot over."""
    await _record(manager, 6.0, day=10, hour=8)
    event = await _record(manager, 9.0, day=10, hour=17)

    assert event is not None
    assert len(manager._monthly_peaks) == 1
    assert manager._monthly_peaks[0].actual_power == 9.0


@pytest.mark.asyncio
async def test_a_lower_hour_on_an_already_counted_day_cannot_evict_another_day(manager):
    """The trap the date-blind top-3 walks into.

    Day 10 peaked at 9 kW, day 11 at 5, day 12 at 4. A 6 kW hour on day 10 beats day 12's
    4 kW - but day 10 is already counted at 9, so the 6 must not evict day 12. Without the
    one-per-day rule the bill gains a second day-10 entry and loses a real billing day.
    """
    await _record(manager, 9.0, day=10, hour=7)
    await _record(manager, 5.0, day=11, hour=7)
    await _record(manager, 4.0, day=12, hour=7)

    event = await _record(manager, 6.0, day=10, hour=19)

    assert event is None
    days = sorted(p.timestamp.day for p in manager._monthly_peaks)
    assert days == [10, 11, 12]
    assert sorted(p.actual_power for p in manager._monthly_peaks) == [4.0, 5.0, 9.0]


@pytest.mark.asyncio
async def test_three_days_fill_three_slots_and_a_fourth_evicts_the_lowest_day(manager):
    await _record(manager, 9.0, day=10, hour=7)
    await _record(manager, 8.0, day=11, hour=7)
    await _record(manager, 7.0, day=12, hour=7)

    event = await _record(manager, 8.5, day=13, hour=7)

    assert event is not None
    assert len(manager._monthly_peaks) == 3
    days = sorted(p.timestamp.day for p in manager._monthly_peaks)
    assert days == [10, 11, 13]


@pytest.mark.asyncio
async def test_replacement_within_a_day_compares_effective_power(manager):
    """A 9 kW night hour bills as 4.5 - a later 5 kW day hour outbills it and takes the day."""
    await _record(manager, 9.0, day=10, hour=2)  # night: effective 4.5
    event = await _record(manager, 5.0, day=10, hour=12)  # day: effective 5.0

    assert event is not None
    assert len(manager._monthly_peaks) == 1
    assert manager._monthly_peaks[0].effective_power == 5.0

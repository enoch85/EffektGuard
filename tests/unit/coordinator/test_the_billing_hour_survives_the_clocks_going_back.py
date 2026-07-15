"""The billing hour must survive DST: the autumn fold must not delete a month's peak.

When the clocks go back, wall-clock hour 02 happens twice (02:00 CEST, then 02:00 CET). PEP 495
ignores `fold` when comparing two aware datetimes with the same tzinfo, so an hour-boundary check
that compares local wall-clock times sees 02:00 CEST == 02:00 CET: the rollover never fires, the
two hours merge, and sample deltas across the fold run backwards - subtracting the earlier hour's
energy instead of recording it. 02:00 is exactly where the optimiser puts its load (cheap night
power), so this deletes the hour most likely to be the month's peak.

The fix keeps the accumulator arithmetic on the absolute (UTC) time line, where the two 02:00 hours
are an hour apart, while the LABEL stays local (the night discount and the month a peak belongs to
are local-clock facts). The spring gap is tested too: wall-clock 02:00 never happens and must not
be invented.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import POWER_SOURCE_EXTERNAL_METER
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.effect_layer import EffectManager

STOCKHOLM = ZoneInfo("Europe/Stockholm")
UTC = ZoneInfo("UTC")

# The real transitions, from the tz database.
AUTUMN_FALL_BACK = datetime(2026, 10, 25, 0, 0, tzinfo=UTC)  # 02:00 CEST; 02:xx runs twice
SPRING_FORWARD = datetime(2026, 3, 29, 0, 0, tzinfo=UTC)  # 01:00 CET; 02:xx never happens


@contextmanager
def a_swedish_installation():
    """HA's `dt_util.as_local` resolves against the timezone HA is CONFIGURED with.

    The test harness leaves that at UTC, and the coordinator asks `as_local` which month a completed
    billing hour belongs to. A test that does not set it is not testing a Swedish install - it is
    testing a UTC one, where the month boundary cannot go wrong and the assertion would pass for the
    wrong reason. (It is set here rather than in a fixture because
    pytest-homeassistant-custom-component asserts at teardown that nobody has left the default zone
    moved, and a fixture's undo loses that race.)
    """
    previous = dt_util.DEFAULT_TIME_ZONE
    dt_util.DEFAULT_TIME_ZONE = STOCKHOLM
    try:
        yield
    finally:
        dt_util.DEFAULT_TIME_ZONE = previous


def _coordinator() -> EffektGuardCoordinator:
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
    coordinator._power_sensor_available = True
    coordinator.effect.record_period_measurement = AsyncMock(return_value=None)
    return coordinator


def _pump() -> NibeState:
    return NibeState(
        outdoor_temp=-5.0,
        indoor_temp=21.0,
        supply_temp=42.0,
        return_temp=37.0,
        degree_minutes=-150.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 10, 25, 2, 0, tzinfo=UTC),
    )


def _meter(hass, kw: float) -> None:
    state = MagicMock()
    state.state = str(kw)
    state.attributes = {"unit_of_measurement": "kW"}
    hass.states.get.return_value = state


async def _drive(coordinator, monkeypatch, start_utc, minutes, power_at) -> None:
    """Step real (absolute) time in 5-minute coordinator cycles, as HA actually would.

    Time is advanced on the UTC line and handed to the coordinator as LOCAL time - which is exactly
    what dt_util.now() gives it, fold and all. Nothing here fakes the transition; the tz database
    does it.
    """
    for step in range(0, minutes, 5):
        instant = start_utc + timedelta(minutes=step)
        local = instant.astimezone(STOCKHOLM)
        monkeypatch.setattr(dt_util, "now", lambda tz=None, _local=local: _local)
        _meter(coordinator.hass, power_at(instant))
        await coordinator._update_peak_tracking(_pump())


def _recorded(coordinator) -> list[tuple[int, float]]:
    """(billing hour, mean kW) for every hour the coordinator actually recorded."""
    return [
        (call.kwargs["period"], round(call.kwargs["power_kw"], 2))
        for call in coordinator.effect.record_period_measurement.await_args_list
    ]


@pytest.mark.asyncio
async def test_the_repeated_hour_does_not_delete_the_months_peak(monkeypatch):
    """9 kW through the first 02:00, 1 kW through the second. Both are real, billable hours."""
    coordinator = _coordinator()

    # 9 kW for the first 02:00-03:00 (CEST, i.e. 00:00-01:00 UTC), 1 kW for the second.
    def power_at(instant: datetime) -> float:
        return 9.0 if instant < AUTUMN_FALL_BACK + timedelta(hours=1) else 1.0

    # Three real hours: 02:00 CEST, 02:00 CET, 03:00 CET.
    await _drive(coordinator, monkeypatch, AUTUMN_FALL_BACK, 180, power_at)

    recorded = _recorded(coordinator)
    means = [mean for _, mean in recorded]

    assert 9.0 in means, (
        f"the coordinator recorded {recorded}. A full hour at 9 kW - the highest of the month, and "
        f"the hour the optimiser itself chose to load, because night power is cheap - was never "
        f"recorded. On the night the clocks go back, wall-clock 02:00 occurs twice, and PEP 495 "
        f"makes 02:00 CEST == 02:00 CET for an aware-datetime comparison with the same tzinfo. So "
        f"the hour never rolls over, the two hours merge, and the sample deltas across the fold run "
        f"backwards - which subtracts the 9 kW hour instead of recording it. The effect tariff bills "
        f"the mean of the month's three highest hours: a peak that is never recorded is never "
        f"defended, for the rest of the month."
    )


@pytest.mark.asyncio
async def test_both_halves_of_the_repeated_hour_are_recorded(monkeypatch):
    """Two real hours went by. Two hours must be billed - not one, and not three."""
    coordinator = _coordinator()
    await _drive(coordinator, monkeypatch, AUTUMN_FALL_BACK, 180, lambda i: 5.0)

    recorded = _recorded(coordinator)

    assert len(recorded) == 2, (
        f"three real hours elapsed (02:00 CEST, 02:00 CET, 03:00 CET) and the coordinator completed "
        f"{len(recorded)} of the first two: {recorded}. Each repeated hour is separately metered and "
        f"separately billable."
    )
    assert [period for period, _ in recorded] == [2, 2], (
        f"both completed hours are the local hour 2 - that is the point, they print the same digits. "
        f"Got {recorded}."
    )
    for _, mean in recorded:
        assert mean == pytest.approx(5.0, abs=0.01), (
            f"a flat 5 kW through a whole hour has an hourly mean of 5 kW. Got {recorded}. A mean "
            f"that is not 5 means the window it was divided by was not one hour."
        )


@pytest.mark.asyncio
async def test_the_spring_gap_does_not_invent_an_hour(monkeypatch):
    """The other transition. Wall-clock 02:00 never happens - it must not be billed."""
    coordinator = _coordinator()

    # 01:00 CET -> 03:00 CEST. Two real hours: 01:00 and 03:00. There is no 02:00.
    await _drive(coordinator, monkeypatch, SPRING_FORWARD, 120, lambda i: 4.0)

    recorded = _recorded(coordinator)
    hours = [period for period, _ in recorded]

    assert 2 not in hours, (
        f"the coordinator billed an hour 2 on the spring-forward day: {recorded}. Wall-clock 02:00 "
        f"does not exist that night - no meter recorded it, and no bill will contain it."
    )
    for _, mean in recorded:
        assert mean == pytest.approx(
            4.0, abs=0.01
        ), f"a flat 4 kW hour has a mean of 4 kW. Got {recorded} - the divisor was not an hour."


@pytest.mark.asyncio
async def test_the_first_hour_of_a_month_is_billed_to_that_month(monkeypatch):
    """The completed hour is stamped local, so it is bucketed into the right calendar month.

    The accumulator runs on the UTC time line, but the effect layer buckets peaks by calendar month
    (`peak.timestamp.year, peak.timestamp.month`), a local-clock fact. In Stockholm the billing hour
    00:00-01:00 on 1 November is 23:00-00:00 on 31 October in UTC - hand the layer the raw UTC stamp
    and a November peak is filed against an already-billed October, while November loses its first
    hour.
    """
    coordinator = _coordinator()
    # 23:00 UTC on 31 Oct == 00:00 local on 1 Nov (CET, +01:00). Two whole local hours.
    november_first = datetime(2026, 10, 31, 23, 0, tzinfo=UTC)

    with a_swedish_installation():
        await _drive(coordinator, monkeypatch, november_first, 120, lambda i: 7.0)

    stamps = [
        call.kwargs["timestamp"]
        for call in coordinator.effect.record_period_measurement.await_args_list
    ]
    assert stamps, "no hour was recorded at all"
    for stamp in stamps:
        assert (stamp.year, stamp.month) == (2026, 11), (
            f"an hour of 1 November was handed to the effect layer stamped {stamp.isoformat()}, "
            f"which is month {stamp.month}. The layer files peaks by calendar month, so this peak "
            f"lands in October - a month already billed - and November loses its first hour."
        )


@pytest.mark.asyncio
async def test_an_ordinary_hour_is_unchanged(monkeypatch):
    """The control. Whatever the fix does to DST, a January hour must still bill exactly as before."""
    coordinator = _coordinator()
    january = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)

    await _drive(coordinator, monkeypatch, january, 120, lambda i: 6.0)

    recorded = _recorded(coordinator)

    assert len(recorded) == 1 and recorded[0][1] == pytest.approx(
        6.0, abs=0.01
    ), f"a flat 6 kW hour on an ordinary day must record exactly one hour at 6.0 kW. Got {recorded}."

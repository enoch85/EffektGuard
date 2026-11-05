"""The control loop ticks every five REAL minutes, at :XX:10 local, through both clock changes.

`_calculate_next_aligned_time` did `dt_util.now() + timedelta(...)` on the local time. That is
wall-clock arithmetic: at 02:55:10 CEST on the night the clocks go back the next tick was 03:00:10
CET, 65 minutes away - an hour with no control cycle - and inside the repeated hour every tick
landed in the first pass (`+ timedelta` resets `fold`), up to 55 minutes in the past, which Home
Assistant fires at once: back-to-back cycles until 02:55 CET. The tick is now computed in UTC and
handed to `async_track_point_in_time` as such; every UTC offset is a multiple of 15 minutes, so it
still lands at :XX:10 on the local clock.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard import coordinator as coordinator_module
from custom_components.effektguard.const import UPDATE_INTERVAL_MINUTES
from custom_components.effektguard.coordinator import EffektGuardCoordinator

STOCKHOLM = ZoneInfo("Europe/Stockholm")
UTC = timezone.utc
TICK = timedelta(minutes=UPDATE_INTERVAL_MINUTES)


def _clock_at(monkeypatch, instant_utc: datetime) -> None:
    """Both of dt_util's clocks from one instant: `now()` local with `fold`, as HA hands it out."""
    monkeypatch.setattr(dt_util, "now", lambda tz=None: instant_utc.astimezone(tz or STOCKHOLM))
    monkeypatch.setattr(dt_util, "utcnow", lambda: instant_utc)


def _coordinator() -> EffektGuardCoordinator:
    coordinator = EffektGuardCoordinator.__new__(EffektGuardCoordinator)
    coordinator.hass = MagicMock()
    coordinator._shutdown_requested = False
    coordinator._unsub_aligned_refresh = None
    coordinator._clock_aligned = False
    return coordinator


def _next_tick_utc(monkeypatch, coordinator: EffektGuardCoordinator, instant_utc: datetime):
    """The instant the coordinator arms for its next cycle, as Home Assistant will fire it."""
    armed: list[datetime] = []

    def fake_async_track_point_in_time(_hass, _action, point_in_time):
        armed.append(point_in_time)
        return lambda: None

    monkeypatch.setattr(
        coordinator_module, "async_track_point_in_time", fake_async_track_point_in_time
    )
    _clock_at(monkeypatch, instant_utc)
    coordinator._schedule_aligned_refresh()
    # async_track_point_in_utc_time converts exactly so before scheduling
    return dt_util.as_utc(armed[-1])


@pytest.mark.parametrize(
    "night, first_cycle_utc, first_tick_utc, ticks",
    [
        (
            "the clocks go back (2026-10-25, 03:00 CEST -> 02:00 CET)",
            datetime(2026, 10, 24, 23, 52, 34, tzinfo=UTC),  # 01:52:34 CEST
            datetime(2026, 10, 24, 23, 55, 10, tzinfo=UTC),
            28,  # through 02:00 CEST twice, to 03:10:10 CET
        ),
        (
            "the clocks go forward (2027-03-28, 02:00 CET -> 03:00 CEST)",
            datetime(2027, 3, 28, 0, 52, 34, tzinfo=UTC),  # 01:52:34 CET
            datetime(2027, 3, 28, 0, 55, 10, tzinfo=UTC),
            16,  # past the hour that never happens, to 04:10:10 CEST
        ),
    ],
)
def test_every_tick_is_five_real_minutes_after_the_last(
    monkeypatch, night, first_cycle_utc, first_tick_utc, ticks
):
    """From an unaligned start, each cycle arms the next :XX:10, five real minutes on."""
    coordinator = _coordinator()
    wrong = []

    cycle_at, expected = first_cycle_utc, first_tick_utc
    for _ in range(ticks):
        armed = _next_tick_utc(monkeypatch, coordinator, cycle_at)
        if armed != expected:
            wrong.append(
                f"at {cycle_at.astimezone(STOCKHOLM):%H:%M:%S %Z} armed "
                f"{armed.astimezone(STOCKHOLM):%H:%M:%S %Z}, {(armed - cycle_at)} later"
            )
        cycle_at, expected = expected, expected + TICK

    assert not wrong, (
        f"on the night {night} the aligned timer went wrong {len(wrong)} times: "
        + "; ".join(wrong)
        + ". A tick an hour out is an hour the pump is not steered; a tick in the past fires at "
        "once, and the loop re-arms into the past again."
    )

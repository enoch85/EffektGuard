"""forecast_hours[N] must mean "N hours from now"; the adapter must make that true.

Every consumer slices WeatherData.forecast_hours positionally (thermal_layer
forecast_hours[:3] is the cold-snap trigger; weather_layer [:24]; prediction_layer
[:horizon]). The adapter used to append every entry the weather entity published, in
its published order, including hours already past - so a stalled-but-"available"
integration (unavailable never trips) could hold stale weather at index 0 and push a
real cold snap outside every horizon.

get_forecast() now drops hours that have already ended and sorts the rest. A forecast
entirely in the past becomes empty, and the layers abstain when there is no forecast.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.weather_adapter import WeatherAdapter
from custom_components.effektguard.const import CONF_WEATHER_ENTITY

# The clock is read INSIDE each test (fixture below), never at module import: a module-level NOW
# captured at collection time would diverge from the adapter's run-time clock under a frozen clock.


@pytest.fixture
def now():
    return dt_util.utcnow()


# A cold snap arriving within the hour, behind six hours of stale mild weather.
STALE_LEADING_HOURS = [(-6, 5.0), (-5, 4.0), (-4, 3.0), (-3, 2.0), (-2, 1.0), (-1, 0.0)]
THE_COLD_SNAP = [(0, -1.0), (1, -8.0), (2, -14.0), (3, -18.0)]


def _adapter(now, hours: list[tuple[int, float]]) -> WeatherAdapter:
    state = MagicMock()
    state.state = "cloudy"
    state.attributes = {
        "temperature": -1.0,
        "temperature_unit": "°C",
        "forecast": [
            {
                "datetime": (now + timedelta(hours=offset)).isoformat(),
                "temperature": temp,
                "condition": "cloudy",
            }
            for offset, temp in hours
        ],
    }
    hass = MagicMock()
    hass.states.get.return_value = state
    return WeatherAdapter(hass, {CONF_WEATHER_ENTITY: "weather.home"})


@pytest.mark.asyncio
async def test_the_first_forecast_hour_is_actually_in_the_future(now):
    data = await _adapter(now, STALE_LEADING_HOURS + THE_COLD_SNAP).get_forecast()

    first = data.forecast_hours[0]
    hours_away = (first.datetime - now).total_seconds() / 3600

    assert hours_away > -1.0, (
        f"forecast_hours[0] is {hours_away:+.0f} hours from now, and reads {first.temperature:+.1f} "
        f"C. Every layer slices this list positionally and treats index 0 as the next hour - so the "
        f"cold-snap trigger was reading the weather from this morning."
    )


@pytest.mark.asyncio
async def test_the_cold_snap_is_inside_the_three_hour_trigger_window(now):
    """The whole point. thermal_layer reads forecast_hours[:3] to decide whether cold is coming."""
    data = await _adapter(now, STALE_LEADING_HOURS + THE_COLD_SNAP).get_forecast()

    next_three = [hour.temperature for hour in data.forecast_hours[:3]]

    assert min(next_three) < -5.0, (
        f"The next three forecast hours read {next_three} C, and a cold snap reaching -18 C arrives "
        f"within the hour. Six hours of already-past weather were sitting at the front of the list, "
        f"pushing the snap out of every horizon the layers look at."
    )


@pytest.mark.asyncio
async def test_the_past_hours_are_dropped_entirely(now):
    data = await _adapter(now, STALE_LEADING_HOURS + THE_COLD_SNAP).get_forecast()

    assert len(data.forecast_hours) == len(THE_COLD_SNAP)
    assert all((hour.datetime - now).total_seconds() / 3600 > -1.0 for hour in data.forecast_hours)


@pytest.mark.asyncio
async def test_the_hours_come_back_in_order(now):
    """A positional read is meaningless on an unsorted list, and nothing guaranteed the order."""
    shuffled = [THE_COLD_SNAP[2], THE_COLD_SNAP[0], THE_COLD_SNAP[3], THE_COLD_SNAP[1]]

    data = await _adapter(now, shuffled).get_forecast()
    times = [hour.datetime for hour in data.forecast_hours]

    assert times == sorted(times)
    assert data.forecast_hours[0].temperature == pytest.approx(-1.0)


@pytest.mark.asyncio
async def test_a_forecast_entirely_in_the_past_is_no_forecast_at_all(now):
    """A stalled weather integration stays 'available' forever. It must not drive the pre-heat."""
    data = await _adapter(now, STALE_LEADING_HOURS).get_forecast()

    assert data is None, (
        "Every hour this weather entity published has already passed - it has stalled, and its "
        "entity is still 'available', so the existing unavailable-check never trips. Driving the "
        "pre-heat on it means pre-heating for weather that has already happened. The layers already "
        "abstain when there is no forecast, which is the correct behaviour here."
    )


@pytest.mark.asyncio
async def test_a_healthy_forecast_is_untouched(now):
    """The regression guard."""
    data = await _adapter(now, THE_COLD_SNAP).get_forecast()

    assert [hour.temperature for hour in data.forecast_hours] == pytest.approx(
        [temp for _, temp in THE_COLD_SNAP]
    )


@pytest.mark.asyncio
async def test_the_current_hour_is_kept(now):
    """A period that began forty minutes ago is still the weather now, not a memory."""
    data = await _adapter(now, [(0, -1.0), (1, -8.0)]).get_forecast()

    assert len(data.forecast_hours) == 2

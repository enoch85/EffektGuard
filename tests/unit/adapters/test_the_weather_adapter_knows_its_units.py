"""The weather adapter must convert forecast temperatures to °C, like nibe_adapter.

A HA weather entity reports temperatures in the user's unit and declares it in
`temperature_unit`; get_forecast() must convert via TemperatureConverter. Without it, on an
imperial install a -5 C cold snap arrives as "23" (F) and is read as +23 C - a 28-degree
error that withdraws the pre-heat exactly when it is needed and disagrees with nibe_adapter,
which does convert. Both current_temp and every forecast hour must be converted; a missing
unit is assumed Celsius (HA's default).
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from homeassistant.const import UnitOfTemperature
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.weather_adapter import WeatherAdapter
from custom_components.effektguard.const import CONF_WEATHER_ENTITY

# The clock is read INSIDE each test (fixture below), never at module import: a module-level NOW
# captured at collection time would diverge from the adapter's run-time clock under a frozen clock.


@pytest.fixture
def now():
    return dt_util.utcnow()


# -5 C, -10 C, -15 C: a Nordic cold snap, spelled in each unit system.
COLD_SNAP_C = [-5.0, -10.0, -15.0]
COLD_SNAP_F = [23.0, 14.0, 5.0]


def _weather_entity(now, current: float, forecast: list[float], unit: str) -> MagicMock:
    state = MagicMock()
    state.state = "cloudy"
    state.attributes = {
        "temperature": current,
        "temperature_unit": unit,
        "forecast": [
            {
                "datetime": (now + timedelta(hours=i)).isoformat(),
                "temperature": t,
                "condition": "cloudy",
            }
            for i, t in enumerate(forecast)
        ],
    }
    return state


def _adapter(state: MagicMock) -> WeatherAdapter:
    hass = MagicMock()
    hass.states.get.return_value = state
    return WeatherAdapter(hass, {CONF_WEATHER_ENTITY: "weather.home"})


@pytest.mark.asyncio
async def test_a_fahrenheit_cold_snap_is_not_read_as_a_warm_spell(now):
    """23 F is -5 C. Read as Celsius it is a mild spring day, and the pre-heat stands down."""
    adapter = _adapter(_weather_entity(now, 23.0, COLD_SNAP_F, UnitOfTemperature.FAHRENHEIT))

    data = await adapter.get_forecast()

    assert data is not None
    assert data.current_temp == pytest.approx(-5.0, abs=0.1), (
        f"A weather entity reporting 23 degrees FAHRENHEIT (-5 C) was read as "
        f"{data.current_temp:.1f} C. That is a 28-degree error, in the direction of 'the house does "
        f"not need heat' - so the pre-heat is withdrawn at exactly the moment it is needed, while "
        f"nibe_adapter reports the outdoor sensor correctly as -5 C."
    )


@pytest.mark.asyncio
async def test_the_whole_fahrenheit_forecast_is_converted_not_just_the_current_reading(now):
    """The forecast drives the cold-snap trigger. It is the half that matters most."""
    adapter = _adapter(_weather_entity(now, 23.0, COLD_SNAP_F, UnitOfTemperature.FAHRENHEIT))

    data = await adapter.get_forecast()

    got = [round(h.temperature, 1) for h in data.forecast_hours[: len(COLD_SNAP_C)]]
    assert got == pytest.approx(COLD_SNAP_C, abs=0.1), (
        f"The forecast came back as {got} C from a Fahrenheit entity; it should be {COLD_SNAP_C}. "
        f"The cold-snap trigger reads the FORECAST - a slab must start charging days ahead - so an "
        f"unconverted forecast means the pre-heat never fires for an imperial user."
    )


@pytest.mark.asyncio
async def test_celsius_is_untouched(now):
    """The regression guard: every existing (metric) install must be bit-for-bit unchanged."""
    adapter = _adapter(_weather_entity(now, -5.0, COLD_SNAP_C, UnitOfTemperature.CELSIUS))

    data = await adapter.get_forecast()

    assert data.current_temp == pytest.approx(-5.0)
    assert [round(h.temperature, 1) for h in data.forecast_hours[:3]] == pytest.approx(COLD_SNAP_C)


@pytest.mark.asyncio
async def test_an_entity_that_declares_no_unit_is_assumed_celsius(now):
    """Home Assistant's own default. Do not refuse to work with a sparse weather integration."""
    state = _weather_entity(now, -5.0, COLD_SNAP_C, UnitOfTemperature.CELSIUS)
    del state.attributes["temperature_unit"]

    data = await _adapter(state).get_forecast()

    assert data.current_temp == pytest.approx(-5.0)

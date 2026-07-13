"""The weather adapter read Fahrenheit as Celsius. The NIBE adapter, reading the same weather, did not.

A Home Assistant weather entity reports its temperatures in the user's configured unit system and
declares which one in `temperature_unit`. The weather adapter never looked. `nibe_adapter` does -
it has used `TemperatureConverter` since F-016 was fixed - so on an imperial install the two
primary temperature sources silently disagreed by about 28 degrees:

    a -5 C cold snap arrives from the weather entity as "23"
    nibe_adapter reports the outdoor sensor correctly as -5

23 is what the weather, prediction and pre-heating layers were handed. So the pre-heat is withdrawn
at precisely the moment it is needed, and the cold-snap detection - the feature the owner cares
most about, because a concrete slab must start charging DAYS ahead - never fires.

Sweden is metric, so the owner's own install was never affected. The integration nonetheless claims
to adapt "from Arctic (-30C) to Mild (5C) climates without configuration", and a US or UK user on
an imperial HA install gets this.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from homeassistant.const import UnitOfTemperature

from custom_components.effektguard.adapters.weather_adapter import WeatherAdapter
from custom_components.effektguard.const import CONF_WEATHER_ENTITY

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)

# -5 C, -10 C, -15 C: a Nordic cold snap, spelled in each unit system.
COLD_SNAP_C = [-5.0, -10.0, -15.0]
COLD_SNAP_F = [23.0, 14.0, 5.0]


def _weather_entity(current: float, forecast: list[float], unit: str) -> MagicMock:
    state = MagicMock()
    state.state = "cloudy"
    state.attributes = {
        "temperature": current,
        "temperature_unit": unit,
        "forecast": [
            {
                "datetime": (NOW + timedelta(hours=i)).isoformat(),
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
async def test_a_fahrenheit_cold_snap_is_not_read_as_a_warm_spell():
    """23 F is -5 C. Read as Celsius it is a mild spring day, and the pre-heat stands down."""
    adapter = _adapter(_weather_entity(23.0, COLD_SNAP_F, UnitOfTemperature.FAHRENHEIT))

    data = await adapter.get_forecast()

    assert data is not None
    assert data.current_temp == pytest.approx(-5.0, abs=0.1), (
        f"A weather entity reporting 23 degrees FAHRENHEIT (-5 C) was read as "
        f"{data.current_temp:.1f} C. That is a 28-degree error, in the direction of 'the house does "
        f"not need heat' - so the pre-heat is withdrawn at exactly the moment it is needed, while "
        f"nibe_adapter reports the outdoor sensor correctly as -5 C."
    )


@pytest.mark.asyncio
async def test_the_whole_fahrenheit_forecast_is_converted_not_just_the_current_reading():
    """The forecast drives the cold-snap trigger. It is the half that matters most."""
    adapter = _adapter(_weather_entity(23.0, COLD_SNAP_F, UnitOfTemperature.FAHRENHEIT))

    data = await adapter.get_forecast()

    got = [round(h.temperature, 1) for h in data.forecast_hours[: len(COLD_SNAP_C)]]
    assert got == pytest.approx(COLD_SNAP_C, abs=0.1), (
        f"The forecast came back as {got} C from a Fahrenheit entity; it should be {COLD_SNAP_C}. "
        f"The cold-snap trigger reads the FORECAST - a slab must start charging days ahead - so an "
        f"unconverted forecast means the pre-heat never fires for an imperial user."
    )


@pytest.mark.asyncio
async def test_celsius_is_untouched():
    """The regression guard: every existing (metric) install must be bit-for-bit unchanged."""
    adapter = _adapter(_weather_entity(-5.0, COLD_SNAP_C, UnitOfTemperature.CELSIUS))

    data = await adapter.get_forecast()

    assert data.current_temp == pytest.approx(-5.0)
    assert [round(h.temperature, 1) for h in data.forecast_hours[:3]] == pytest.approx(COLD_SNAP_C)


@pytest.mark.asyncio
async def test_an_entity_that_declares_no_unit_is_assumed_celsius():
    """Home Assistant's own default. Do not refuse to work with a sparse weather integration."""
    state = _weather_entity(-5.0, COLD_SNAP_C, UnitOfTemperature.CELSIUS)
    del state.attributes["temperature_unit"]

    data = await _adapter(state).get_forecast()

    assert data.current_temp == pytest.approx(-5.0)

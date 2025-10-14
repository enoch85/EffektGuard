"""Weather adapter for reading forecast data.

This adapter reads weather forecast data from Home Assistant's weather
entities (typically Met.no) for predictive optimization.

Used for:
- Pre-heating before cold periods
- Thermal model predictions
- Rate-of-change analysis
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from homeassistant.core import HomeAssistant

from ..const import CONF_WEATHER_ENTITY

_LOGGER = logging.getLogger(__name__)


@dataclass
class WeatherForecastHour:
    """Single hour of weather forecast."""

    datetime: datetime
    temperature: float  # °C
    condition: str | None = None


@dataclass
class WeatherData:
    """Weather forecast data for optimization."""

    current_temp: float  # Current outdoor temperature
    forecast_hours: list[WeatherForecastHour]  # Next 24-48 hours


class WeatherAdapter:
    """Adapter for reading weather forecast entities."""

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]):
        """Initialize weather adapter.

        Args:
            hass: Home Assistant instance
            config: Configuration dictionary with entity IDs
        """
        self.hass = hass
        self._weather_entity = config.get(CONF_WEATHER_ENTITY)

    async def get_forecast(self) -> WeatherData | None:
        """Read weather forecast from entity.

        Returns:
            WeatherData with forecast, or None if unavailable
        """
        if not self._weather_entity:
            _LOGGER.debug("No weather entity configured")
            return None

        # Read weather entity state
        state = self.hass.states.get(self._weather_entity)
        if not state or state.state in ["unknown", "unavailable"]:
            _LOGGER.warning("Weather entity %s unavailable", self._weather_entity)
            return None

        # Get current temperature
        current_temp = state.attributes.get("temperature")
        if current_temp is None:
            _LOGGER.warning("No current temperature in weather entity")
            return None

        # Get forecast from attributes
        forecast_raw = state.attributes.get("forecast", [])
        if not forecast_raw:
            _LOGGER.debug("No forecast data available")
            return None

        # Parse forecast hours
        forecast_hours = []
        for item in forecast_raw[:48]:  # Take up to 48 hours
            try:
                dt = item.get("datetime")
                if isinstance(dt, str):
                    from homeassistant.util import dt as dt_util

                    dt = dt_util.parse_datetime(dt)

                temp = item.get("temperature")
                if dt is None or temp is None:
                    continue

                forecast_hours.append(
                    WeatherForecastHour(
                        datetime=dt,
                        temperature=float(temp),
                        condition=item.get("condition"),
                    )
                )
            except (ValueError, TypeError, KeyError) as err:
                _LOGGER.debug("Skipping forecast item: %s", err)
                continue

        _LOGGER.debug("Loaded weather forecast: %d hours", len(forecast_hours))

        return WeatherData(
            current_temp=float(current_temp),
            forecast_hours=forecast_hours,
        )

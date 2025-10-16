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

        Supports multiple weather integration patterns:
        - Met.no: forecast attribute in weather entity
        - OpenWeatherMap: requires weather.get_forecasts service call
        - AccuWeather: forecast attribute

        Returns:
            WeatherData with forecast, or None if unavailable
        """
        if not self._weather_entity:
            _LOGGER.info("Weather forecast disabled - no entity configured in setup")
            return None

        _LOGGER.debug("Reading weather forecast from entity: %s", self._weather_entity)

        # Read weather entity state
        state = self.hass.states.get(self._weather_entity)
        if not state or state.state in ["unknown", "unavailable"]:
            _LOGGER.warning(
                "Weather entity %s unavailable (state: %s)",
                self._weather_entity,
                state.state if state else "missing",
            )
            return None

        # Get current temperature
        current_temp = state.attributes.get("temperature")
        if current_temp is None:
            _LOGGER.warning("No current temperature in weather entity")
            return None

        # Try attribute access first (Met.no, AccuWeather, etc.)
        forecast_raw = state.attributes.get("forecast", [])

        # If no forecast attribute, try service call (OpenWeatherMap, etc.)
        if not forecast_raw:
            _LOGGER.debug("No forecast attribute found, trying weather.get_forecasts service call")
            try:
                forecast_data = await self.hass.services.async_call(
                    "weather",
                    "get_forecasts",
                    {"type": "hourly", "entity_id": self._weather_entity},
                    blocking=True,
                    return_response=True,
                )
                # Extract forecast from service response
                forecast_raw = forecast_data.get(self._weather_entity, {}).get("forecast", [])

                if forecast_raw:
                    _LOGGER.debug(
                        "Weather forecast retrieved via service call: %d hours",
                        len(forecast_raw),
                    )
            except Exception as err:
                _LOGGER.debug(
                    "Failed to get forecast via service call: %s. "
                    "Weather-based optimization disabled for this integration.",
                    err,
                )
                return None

        if not forecast_raw:
            _LOGGER.debug(
                "No forecast data available from %s. "
                "Optimization will work without weather forecast, but less predictive.",
                self._weather_entity,
            )
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
                _LOGGER.debug("Skipping invalid forecast entry: %s", err)
                continue

        if not forecast_hours:
            _LOGGER.warning("No valid forecast hours parsed")
            return None

        # Validate forecast length
        forecast_count = len(forecast_hours)
        if forecast_count < 12:
            _LOGGER.warning(
                "Weather forecast from %s: only %d hours (minimum 12h recommended for optimal pre-heating)",
                self._weather_entity,
                forecast_count,
            )
        else:
            _LOGGER.info(
                "Weather forecast from %s: %d hours available (sufficient for optimization)",
                self._weather_entity,
                forecast_count,
            )

        return WeatherData(
            current_temp=float(current_temp),
            forecast_hours=forecast_hours,
        )

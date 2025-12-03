"""Weather adapter for reading forecast data.

This adapter reads weather forecast data from Home Assistant's weather
entities (typically Met.no) for predictive optimization.

Used for:
- Pre-heating before cold periods
- Thermal model predictions
- Rate-of-change analysis
"""

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

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
    source_entity: str = ""  # Weather entity ID
    source_method: str = ""  # "attribute" or "service_call"


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
        self._last_forecast_time: datetime | None = None  # When we last got a valid forecast
        self._last_service_call_time: datetime | None = None  # When we last tried service call
        self._next_random_attempt: datetime | None = None  # Next scheduled random attempt

    def _schedule_next_random_attempt(self) -> None:
        """Schedule next random forecast attempt within the next hour.

        This helps avoid API rate limits while still retrying periodically
        when forecast is unavailable.
        """
        now = dt_util.utcnow()
        # Random time between 5-55 minutes from now
        minutes_delay = random.randint(5, 55)
        self._next_random_attempt = now + timedelta(minutes=minutes_delay)
        _LOGGER.debug(
            "Next random forecast attempt scheduled in %d minutes (at %s)",
            minutes_delay,
            self._next_random_attempt.strftime("%H:%M"),
        )

    async def get_forecast(self) -> WeatherData | None:
        """Read weather forecast from entity.

        Supports multiple weather integration patterns:
        - Met.no: forecast attribute in weather entity
        - OpenWeatherMap: requires weather.get_forecasts service call
        - AccuWeather: forecast attribute

        Implements smart retry logic:
        - If forecast unavailable, retry at random times each hour
        - Checks before calling to avoid unnecessary API calls
        - Rate limiting to prevent API abuse

        Returns:
            WeatherData with forecast, or None if unavailable
        """
        if not self._weather_entity:
            _LOGGER.info("Weather forecast disabled - no entity configured in setup")
            return None

        now = dt_util.utcnow()

        _LOGGER.debug("Reading weather forecast from entity: %s", self._weather_entity)

        # Read weather entity state
        state = self.hass.states.get(self._weather_entity)
        if not state or state.state in ["unknown", "unavailable"]:
            _LOGGER.warning(
                "Weather entity %s unavailable (state: %s)",
                self._weather_entity,
                state.state if state else "missing",
            )
            # Schedule next random attempt if not already scheduled
            if self._next_random_attempt is None:
                self._schedule_next_random_attempt()
            return None

        # Get current temperature
        current_temp = state.attributes.get("temperature")
        if current_temp is None:
            _LOGGER.warning("No current temperature in weather entity")
            return None

        # Try attribute access first (Met.no, AccuWeather, etc.)
        forecast_raw = state.attributes.get("forecast", [])
        source_method = "attribute" if forecast_raw else ""

        # If no forecast attribute, try service call (OpenWeatherMap, etc.)
        if not forecast_raw:
            # Check if we should attempt service call now
            # Conditions:
            # 1. No next attempt scheduled (first try), OR
            # 2. Current time is past the scheduled attempt time
            should_attempt = self._next_random_attempt is None or now >= self._next_random_attempt

            if not should_attempt:
                minutes_until_next = (
                    (self._next_random_attempt - now).total_seconds() / 60
                    if self._next_random_attempt
                    else 0
                )
                _LOGGER.debug(
                    "Skipping forecast service call, next attempt in %.1f minutes",
                    minutes_until_next,
                )
                return None

            _LOGGER.debug(
                "No forecast attribute in %s, trying weather.get_forecasts service call",
                self._weather_entity,
            )
            try:
                # Call weather.get_forecasts service for hourly forecast
                # Required for OpenWeatherMap and some other integrations
                forecast_data = await self.hass.services.async_call(
                    "weather",
                    "get_forecasts",
                    {"type": "hourly", "entity_id": self._weather_entity},
                    blocking=True,
                    return_response=True,
                )

                self._last_service_call_time = now  # Track service call time

                _LOGGER.debug(
                    "Service call response keys: %s",
                    list(forecast_data.keys()) if forecast_data else "None",
                )

                # Extract forecast from service response
                # Response format: {entity_id: {"forecast": [...]}}
                if forecast_data and self._weather_entity in forecast_data:
                    forecast_raw = forecast_data[self._weather_entity].get("forecast", [])
                    if forecast_raw:
                        _LOGGER.debug(
                            "Weather forecast retrieved via service call: %d hours",
                            len(forecast_raw),
                        )
                        # Success! Reset random attempt timer
                        self._next_random_attempt = None
                        source_method = "service_call"
                    else:
                        _LOGGER.warning(
                            "Service call succeeded but returned no forecast data. " "Response: %s",
                            forecast_data,
                        )
                        # Schedule next random attempt
                        self._schedule_next_random_attempt()
                else:
                    _LOGGER.warning(
                        "Service call succeeded but returned no forecast data. " "Response: %s",
                        forecast_data,
                    )
                    # Schedule next random attempt
                    self._schedule_next_random_attempt()
            except (AttributeError, KeyError, ValueError, TypeError, OSError) as err:
                _LOGGER.warning(
                    "Failed to get forecast via service call from %s: %s. "
                    "Weather-based optimization disabled. "
                    "This is normal for OpenWeatherMap free tier (no OneCall 3.0 access). "
                    "Consider switching to Met.no for free forecast access.",
                    self._weather_entity,
                    err,
                    exc_info=True,
                )
                # Schedule next random attempt after error
                self._schedule_next_random_attempt()
                return None

        if not forecast_raw:
            _LOGGER.warning(
                "No forecast data available from %s. "
                "Possible reasons: "
                "1) OpenWeatherMap free tier (no OneCall 3.0 access), "
                "2) API key not configured, "
                "3) Integration not set up correctly. "
                "Weather-based optimization will be disabled. "
                "Optimization will work without weather forecast, but less predictive. "
                "Consider switching to Met.no for free forecast access.",
                self._weather_entity,
            )
            return None

        # Parse forecast hours
        forecast_hours = []
        for item in forecast_raw[:48]:  # Take up to 48 hours
            try:
                dt = item.get("datetime")
                if isinstance(dt, str):
                    # dt_util already imported at module level
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
        # For UFH concrete slab (10h prediction horizon), 24h forecast is ideal
        # Met.no provides 9 days, OpenWeatherMap v3.0 provides 48h
        forecast_count = len(forecast_hours)
        if forecast_count < 24:
            _LOGGER.warning(
                "Weather forecast from %s: only %d hours available "
                "(24h recommended for optimal pre-heating with concrete slab UFH). "
                "If using OpenWeatherMap Free tier, upgrade to v3.0 (One Call API 3.0) "
                "for 48h hourly forecast, or switch to Met.no for free 9-day forecast.",
                self._weather_entity,
                forecast_count,
            )
        else:
            _LOGGER.info(
                "Weather forecast from %s: %d hours available (sufficient for optimization)",
                self._weather_entity,
                forecast_count,
            )

        # Successfully retrieved forecast - update timestamp and reset random attempts
        self._last_forecast_time = now
        self._next_random_attempt = None

        return WeatherData(
            current_temp=float(current_temp),
            forecast_hours=forecast_hours,
            source_entity=self._weather_entity,
            source_method=source_method,
        )

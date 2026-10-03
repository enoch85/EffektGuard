"""The coordinator's update loop must survive any single bad cycle and always re-arm.

The base scheduler is disabled (`update_interval=None`), so `_do_aligned_refresh` is the sole
owner of the timer: if it returns without calling `_schedule_aligned_refresh()`, nothing re-arms
it and the coordinator is permanently dead - silently, since `last_update_success` stays True and
entities keep serving stale values. So the refresh catches a broad `except Exception` and re-arms
in a `finally`. The update path can raise HomeAssistantError (weather with no hourly forecast),
IndexError (DST price lookup), ZeroDivisionError (savings), RuntimeError/numpy (learning) - and a
daily-only weather entity raises on every cycle, so the first such raise must not kill the loop.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound
from homeassistant.helpers.update_coordinator import UpdateFailed

from custom_components.effektguard.adapters.weather_adapter import WeatherAdapter
from custom_components.effektguard.coordinator import EffektGuardCoordinator

WEATHER_ENTITY = "weather.daily_only_provider"


def make_coordinator(update_error: Exception | None):
    """Duck-typed stand-in exposing only what _do_aligned_refresh touches.

    It drives the pump through `_drive_the_pump` - the sole owner of the write path - not through
    Home Assistant's read hook. Stubbing the wrong one is not a harmless mismatch: the real
    `_drive_the_pump` would be reached on a MagicMock, fail to await, and be swallowed by the very
    `except Exception` under test. The error cases would then pass on a TypeError instead of on the
    error they name.
    """
    coordinator = MagicMock()
    coordinator.last_update_success = True

    if update_error is None:
        coordinator._drive_the_pump = AsyncMock(return_value={"ok": True})
    else:
        coordinator._drive_the_pump = AsyncMock(side_effect=update_error)

    coordinator._schedule_aligned_refresh = MagicMock()
    coordinator.async_set_updated_data = MagicMock()
    return coordinator


async def run_refresh(coordinator) -> None:
    await EffektGuardCoordinator._do_aligned_refresh(coordinator)


class TestUpdateLoopAlwaysRearms:
    """Whatever happens, the next update must be scheduled."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            HomeAssistantError("Weather entity does not support 'hourly' forecast"),
            ServiceNotFound("weather", "get_forecasts"),
            IndexError("list index out of range"),  # DST 92/100-quarter day
            ZeroDivisionError("float division by zero"),  # savings maths
            RuntimeError("something unexpected"),
            UpdateFailed("required NIBE sensors unreadable"),
        ],
        ids=[
            "HomeAssistantError",
            "ServiceNotFound",
            "IndexError_dst",
            "ZeroDivisionError_savings",
            "RuntimeError",
            "UpdateFailed_expected",
        ],
    )
    async def test_timer_is_rearmed_after_any_error(self, error):
        coordinator = make_coordinator(update_error=error)

        # The loop must not propagate - a dead task means a dead coordinator.
        await run_refresh(coordinator)

        coordinator._schedule_aligned_refresh.assert_called_once(), (
            f"{type(error).__name__} left the aligned-refresh timer un-armed. With "
            "update_interval=None, the coordinator is now permanently dead."
        )

    @pytest.mark.asyncio
    async def test_failure_marks_the_update_unsuccessful(self):
        """Entities must go unavailable rather than serving stale data as if healthy."""
        coordinator = make_coordinator(update_error=HomeAssistantError("boom"))

        await run_refresh(coordinator)

        assert coordinator.last_update_success is False, (
            "The coordinator reported success after a failed update. Entities would keep "
            "serving their last value and look healthy while control had stopped."
        )

    @pytest.mark.asyncio
    async def test_timer_is_rearmed_on_success_too(self):
        coordinator = make_coordinator(update_error=None)

        await run_refresh(coordinator)

        coordinator._schedule_aligned_refresh.assert_called_once()
        assert coordinator.last_update_success is True
        coordinator.async_set_updated_data.assert_called_once()


class TestWeatherAdapterSurvivesUnsupportedForecast:
    """A daily-only weather entity must degrade, not take the integration down."""

    @pytest.mark.asyncio
    async def test_unsupported_forecast_returns_none_instead_of_raising(self):
        hass = MagicMock()

        state = MagicMock()
        state.state = "cloudy"
        # Current HA weather entities publish no `forecast` state attribute, so the
        # service-call path is always taken.
        state.attributes = {"temperature": 4.2}
        hass.states.get.return_value = state

        hass.services.async_call = AsyncMock(
            side_effect=HomeAssistantError(
                f"Weather entity '{WEATHER_ENTITY}' does not support 'hourly' forecast"
            )
        )

        adapter = WeatherAdapter(hass, {"weather_entity": WEATHER_ENTITY})

        result = await adapter.get_forecast()

        assert result is None, "Weather is optional - it must degrade to None, not raise."
        # And it must back off rather than hammering the service every cycle.
        assert adapter._next_random_attempt is not None

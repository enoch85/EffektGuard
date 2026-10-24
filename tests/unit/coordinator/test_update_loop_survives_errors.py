"""The coordinator's update loop must survive any single bad cycle and always re-arm.

The base scheduler is disabled (`update_interval=None`), so `_do_aligned_refresh` is the sole
owner of the timer: if it returns without calling `_schedule_aligned_refresh()`, nothing re-arms
it and the coordinator is permanently dead - silently, since `last_update_success` stays True and
entities keep serving stale values. So the refresh catches a broad `except Exception` and re-arms
in a `finally`. The update path can raise HomeAssistantError (weather with no hourly forecast),
IndexError (DST price lookup), ZeroDivisionError (savings), RuntimeError/numpy (learning) - and a
daily-only weather entity raises on every cycle, so the first such raise must not kill the loop.
"""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound
from homeassistant.helpers.update_coordinator import UpdateFailed
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    MockEntityPlatform,
    async_fire_time_changed,
)

from custom_components.effektguard.adapters.weather_adapter import WeatherAdapter
from custom_components.effektguard.climate import EffektGuardClimate
from custom_components.effektguard.const import DOMAIN, HEAT_PUMP_READ_RETRY_SECONDS
from custom_components.effektguard.coordinator import EffektGuardCoordinator, HeatPumpUnreadable
from custom_components.effektguard.optimization.decision_engine import OptimizationDecision

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
            UpdateFailed("the heat pump never answered after start-up"),
            HeatPumpUnreadable("required NIBE sensors unreadable"),
        ],
        ids=[
            "HomeAssistantError",
            "ServiceNotFound",
            "IndexError_dst",
            "ZeroDivisionError_savings",
            "RuntimeError",
            "UpdateFailed_expected",
            "HeatPumpUnreadable_tried_again",
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
    async def test_timer_is_rearmed_on_success_too(self):
        coordinator = make_coordinator(update_error=None)

        await run_refresh(coordinator)

        coordinator._schedule_aligned_refresh.assert_called_once()
        coordinator.async_set_updated_data.assert_called_once()


def _good_cycle(indoor_temp: float) -> dict:
    pump = SimpleNamespace(
        indoor_temp=indoor_temp, outdoor_temp=-3.2, supply_temp=35.8, degree_minutes=-150
    )
    return {"nibe": pump, "decision": OptimizationDecision(offset=0.0, reasoning="good")}


async def _coordinator_with_a_thermostat(hass, nibe) -> EffektGuardCoordinator:
    """A real coordinator past start-up, its last cycle good, and the thermostat on screen."""
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    coordinator = EffektGuardCoordinator(
        hass, nibe, MagicMock(), MagicMock(), MagicMock(), MagicMock(), entry
    )
    coordinator._schedule_aligned_refresh = MagicMock()  # the aligned timer is not under test
    coordinator._first_successful_update = True  # past startup, a failed read is an error
    coordinator.async_set_updated_data(_good_cycle(21.3))
    platform = MockEntityPlatform(hass, domain="climate", platform_name=DOMAIN)
    platform.config_entry = entry
    await platform.async_add_entities([EffektGuardClimate(coordinator, entry)])
    assert hass.states.get("climate.effektguard").state == "heat"
    return coordinator


async def _second_try_is_due(hass) -> None:
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=HEAT_PUMP_READ_RETRY_SECONDS)
    )
    await hass.async_block_till_done()


async def test_the_entities_show_a_failed_cycle_and_the_recovery(hass):
    """Unavailable while the pump cannot be read, back when it can - and nothing written meanwhile.

    The loop only set `last_update_success` and told no listener, so until the next good cycle every
    entity went on showing the last one as if healthy (production 2026-10-20, five cycles while
    myUplink failed; EG-02 of the log investigation). A cycle is shown failed once its second try
    has failed too.
    """
    nibe = MagicMock()
    nibe.get_current_state = AsyncMock(side_effect=UpdateFailed("required sensors unreadable"))
    coordinator = await _coordinator_with_a_thermostat(hass, nibe)

    await coordinator._do_aligned_refresh()

    assert hass.states.get("climate.effektguard").state == "heat"

    await _second_try_is_due(hass)

    assert nibe.get_current_state.await_count == 2
    assert hass.states.get("climate.effektguard").state == "unavailable"
    nibe.set_curve_offset.assert_not_called()

    coordinator._drive_the_pump = AsyncMock(return_value=_good_cycle(21.3))
    await coordinator._do_aligned_refresh()

    assert hass.states.get("climate.effektguard").state == "heat"


async def test_a_second_try_that_reads_the_pump_does_the_cycle(hass):
    """One missed poll of the pump's source no longer costs the cycle or blanks the entities.

    Owner's production 2026-10-14 to 10-24: 11 cycles could not read the pump, every entity went
    unavailable for 4.4 to 40 minutes each time, and in 6 of them the readings were back 70 s later.
    """
    coordinator = await _coordinator_with_a_thermostat(hass, MagicMock())
    coordinator._drive_the_pump = AsyncMock(
        side_effect=[HeatPumpUnreadable("required sensors unreadable"), _good_cycle(21.6)]
    )

    await coordinator._do_aligned_refresh()
    await _second_try_is_due(hass)

    assert coordinator._drive_the_pump.await_count == 2
    state = hass.states.get("climate.effektguard")
    assert state.state == "heat"
    assert state.attributes["current_temperature"] == 21.6


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

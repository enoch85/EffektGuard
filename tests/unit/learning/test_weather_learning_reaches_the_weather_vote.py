"""A learned unusual forecast must reach the weather vote built by Home Assistant."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import _create_coordinator
from custom_components.effektguard.adapters.weather_adapter import WeatherData, WeatherForecastHour
from custom_components.effektguard.const import DOMAIN

NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)
FORECAST_HOURS = 24
NORMAL_OUTDOOR = 0.0
COLD_OUTDOOR = -30.0


async def test_setup_passes_learned_unusual_weather_to_the_live_vote(hass, freezer):
    freezer.move_to(NOW)
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    coordinator = await _create_coordinator(hass, entry)
    for year in (2023, 2024, 2025):
        coordinator.weather_learner.record_weather_pattern(
            NOW.replace(year=year), [NORMAL_OUTDOOR] * FORECAST_HOURS
        )
    required = coordinator.engine.weather_comp.calculate_optimal_flow_temp(
        coordinator.engine.target_temp, NORMAL_OUTDOOR
    ).flow_temp
    state = SimpleNamespace(
        outdoor_temp=NORMAL_OUTDOOR, flow_temp=required + 2.0, degree_minutes=-50.0
    )

    def forecast(temperature):
        return WeatherData(
            NORMAL_OUTDOOR,
            [WeatherForecastHour(NOW + timedelta(hours=h), temperature) for h in range(1, 25)],
        )

    normal = coordinator.engine.weather_comp_layer.evaluate_layer(
        state, forecast(NORMAL_OUTDOOR), coordinator.engine.target_temp
    )
    cold = coordinator.engine.weather_comp_layer.evaluate_layer(
        state, forecast(COLD_OUTDOOR), coordinator.engine.target_temp
    )

    assert cold.unusual_weather
    assert cold.safety_margin > normal.safety_margin
    assert cold.weight > normal.weight
    assert cold.offset > normal.offset

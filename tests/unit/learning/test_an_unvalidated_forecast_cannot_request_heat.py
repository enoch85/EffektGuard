"""History coverage is not evidence that the room trajectory predicts the native heating curve."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import _create_coordinator
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.adapters.weather_adapter import WeatherData, WeatherForecastHour
from custom_components.effektguard.const import DOMAIN, SAMPLES_PER_HOUR, UPDATE_INTERVAL_MINUTES

NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)


@pytest.mark.parametrize("mass,learned", [(0.7, False), (1.3, False), (1.8, False), (1.8, True)])
async def test_a_fresh_stationary_day_cannot_request_heat_from_the_unvalidated_trajectory(
    hass, freezer, mass, learned
):
    freezer.move_to(NOW)
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"enable_price_optimization": False},
        options={"target_indoor_temp": 22.0, "thermal_mass": mass},
    )
    entry.add_to_hass(hass)
    coordinator = await _create_coordinator(hass, entry)
    predictor = coordinator.thermal_predictor
    if learned:
        for i in range(672):
            coordinator.adaptive_learning.record_observation(
                NOW - timedelta(hours=671 - i),
                21.0 + (i % 13) * 0.12,
                0.0,
                0.0 if i % 13 == 0 else 1.0,
            )
        assert coordinator.adaptive_learning.get_parameters().confidence > 0.9
    for i in range(24 * SAMPLES_PER_HOUR + 1):
        predictor.record_state(
            NOW - timedelta(minutes=UPDATE_INTERVAL_MINUTES * (24 * SAMPLES_PER_HOUR - i)),
            22.0,
            0.0,
            0.0,
            35.0,
            -100.0,
        )
    state = NibeState(
        outdoor_temp=0.0,
        indoor_temp=22.0,
        supply_temp=35.0,
        return_temp=30.0,
        degree_minutes=-100.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=NOW,
    )
    weather = WeatherData(
        0.0, [WeatherForecastHour(NOW + timedelta(hours=h), 0.0) for h in range(1, 49)]
    )
    # The flawed diagnostic trajectory sees cooling even though native heat held the room steady.
    proposal = predictor.evaluate_layer(state, weather, 22.0, coordinator.adaptive_learning)
    assert proposal.offset > 0.0 and proposal.weight > 0.0
    decision = coordinator.engine.calculate_decision(state, None, weather, 0.0, 0.0)
    vote = next(layer for layer in decision.layers if layer.name == "Learned Pre-heat")
    assert vote.weight == 0.0 and vote.offset == 0.0
    assert predictor.get_current_trend()["rate_per_hour"] == pytest.approx(0.0)
    # Containment leaves the measured-room safety response in charge when the room is cold.
    cold = coordinator.engine.calculate_decision(
        replace(state, indoor_temp=17.0), None, weather, 0.0, 0.0
    )
    assert cold.is_emergency
    assert cold.offset > 0.0

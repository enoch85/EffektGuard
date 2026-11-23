"""A NIBE with no room sensor must not be driven on the placeholder indoor temperature.

A pump with no BT50 is a supported configuration: it runs on degree minutes and the heating curve.
The adapter substitutes DEFAULT_INDOOR_TEMP (21.0) for display and sets indoor_temp_valid=False so
comfort-reasoning layers abstain. The comfort layer must honour that flag: any target below the
placeholder would otherwise read as a permanent, uncorrectable overshoot and coast the pump to
minimum output all winter, on a house nobody is measuring.

Invariant: with indoor_temp_valid=False the comfort layer abstains (weight 0, offset 0); with a
real reading it still corrects a genuine overshoot or a genuinely cold house.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import _create_coordinator
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.adapters.weather_adapter import (
    WeatherData,
    WeatherForecastHour,
)
from custom_components.effektguard.const import (
    CONF_TARGET_INDOOR_TEMP,
    CONF_THERMAL_MASS,
    DEFAULT_INDOOR_TEMP,
    DOMAIN,
    MIN_TEMP_LIMIT,
    PREDICTION_LEARNED_PREHEAT_MIN_HOURS,
    SAMPLES_PER_HOUR,
    UFH_CONCRETE_PREDICTION_HORIZON,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.optimization.comfort_layer import ComfortLayer
from custom_components.effektguard.optimization.prediction_layer import ThermalStatePredictor
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

# Targets a real owner can set. All of them sit BELOW the placeholder, which is the whole problem.
COOL_TARGETS = [20.5, 20.0, 19.0, 18.5]
# One above it: a room nothing measures then reads as a degree too cold
WARM_TARGET = 22.0
CONCRETE_SLAB = 1.8
START = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)


def _sensorless_pump() -> NibeState:
    """Exactly what the adapter builds when there is no BT50: the placeholder, flagged invalid."""
    return NibeState(
        outdoor_temp=-5.0,
        indoor_temp=DEFAULT_INDOOR_TEMP,
        supply_temp=42.0,
        return_temp=37.0,
        degree_minutes=-150.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        indoor_temp_valid=False,
    )


def _pump_with_a_real_sensor(indoor: float) -> NibeState:
    state = _sensorless_pump()
    state.indoor_temp = indoor
    state.indoor_temp_valid = True
    return state


def test_the_placeholder_is_above_every_cool_target_which_is_why_this_bites():
    """The precondition. If DEFAULT_INDOOR_TEMP ever drops, re-derive these numbers."""
    assert DEFAULT_INDOOR_TEMP == 21.0
    assert all(t < DEFAULT_INDOOR_TEMP for t in COOL_TARGETS)
    assert min(COOL_TARGETS) >= MIN_TEMP_LIMIT, "an allowed target must be above the safety floor"


@pytest.mark.parametrize("target", COOL_TARGETS)
def test_the_comfort_layer_abstains_with_no_room_sensor(target):
    """No measurement, no comfort opinion. Not a small one - none."""
    decision = ComfortLayer(target_temp=target).evaluate_layer(_sensorless_pump())

    assert decision.weight == 0.0, (
        f"With no room sensor and a target of {target} C, the comfort layer commanded "
        f"{decision.offset:+.2f} C at weight {decision.weight:.2f} - '{decision.reason}'. That "
        f"deviation is measured against DEFAULT_INDOOR_TEMP ({DEFAULT_INDOOR_TEMP} C), a "
        f"placeholder. Nothing is measuring this house, so the 'overshoot' can never be corrected "
        f"and the pump stays coasted for the whole winter."
    )
    assert decision.offset == 0.0


async def test_a_day_without_a_room_sensor_teaches_the_learning_no_room(hass, freezer):
    """A day of placeholder readings under a target above it made the learned pre-heat vote +3.0 at
    weight 0.65 for a room nothing measured, and gave the house a room trend of its own invention.
    The house as Home Assistant builds it, its observations recorded as each cycle records them."""
    freezer.move_to(START)
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={},
        options={CONF_TARGET_INDOOR_TEMP: WARM_TARGET, CONF_THERMAL_MASS: CONCRETE_SLAB},
    )
    entry.add_to_hass(hass)
    coordinator = await _create_coordinator(hass, entry)
    state = _sensorless_pump()
    for _ in range(PREDICTION_LEARNED_PREHEAT_MIN_HOURS * SAMPLES_PER_HOUR):
        state.timestamp = dt_util.utcnow()
        await coordinator._record_learning_observations(state, None, 0.0)
        freezer.tick(timedelta(minutes=UPDATE_INTERVAL_MINUTES))
    state.timestamp = dt_util.utcnow()
    forecast = WeatherData(
        current_temp=state.outdoor_temp,
        forecast_hours=[
            WeatherForecastHour(state.timestamp + timedelta(hours=hour), state.outdoor_temp)
            for hour in range(1, int(UFH_CONCRETE_PREDICTION_HORIZON) + 1)
        ],
    )

    decision = coordinator.engine.calculate_decision(state, None, forecast, 0.0, 0.0)

    learned = next(layer for layer in decision.layers if layer.name == "Learned Pre-heat")
    assert learned.weight == 0.0, f"{learned.offset:+.1f}@{learned.weight} ({learned.reason})"
    assert coordinator.thermal_predictor.get_current_trend()["trend"] == "unknown"
    assert not coordinator.adaptive_learning.observations


@pytest.mark.parametrize("measured", [True, False], ids=["room sensor", "the sensor dropped out"])
def test_the_learned_pre_heat_speaks_only_for_a_measured_room(measured):
    """A day of a measured room 1 C under its target gives the learned layer a vote; the moment the
    reading becomes the placeholder, the vote is gone - its overshoot would be the placeholder's."""
    predictor = ThermalStatePredictor()
    for cycle in range(PREDICTION_LEARNED_PREHEAT_MIN_HOURS * SAMPLES_PER_HOUR):
        predictor.record_state(
            START + cycle * timedelta(minutes=UPDATE_INTERVAL_MINUTES),
            WARM_TARGET - 1.0,
            0.0,
            0.0,
            35.0,
            -100.0,
        )
    state = _pump_with_a_real_sensor(WARM_TARGET - 1.0)
    state.indoor_temp_valid = measured
    forecast = WeatherData(
        current_temp=0.0,
        forecast_hours=[WeatherForecastHour(START + timedelta(hours=h), 0.0) for h in range(1, 25)],
    )

    vote = predictor.evaluate_layer(state, forecast, WARM_TARGET, ThermalModel(CONCRETE_SLAB))

    assert (vote.weight > 0.0) == measured, f"{vote.offset:+.1f}@{vote.weight} ({vote.reason})"


class TestTheLayerStillWorksWhenItCanSee:
    """The regression guard on the guard: abstaining must not break a normal house."""

    def test_a_real_overshoot_is_still_corrected(self):
        decision = ComfortLayer(target_temp=21.0).evaluate_layer(_pump_with_a_real_sensor(23.0))

        assert decision.weight > 0.0
        assert decision.offset < 0.0, "a house that is genuinely 2 C too warm must still coast"

    def test_a_real_cold_house_is_still_heated(self):
        decision = ComfortLayer(target_temp=21.0).evaluate_layer(_pump_with_a_real_sensor(19.5))

        assert decision.weight > 0.0
        assert decision.offset > 0.0, "a house that is genuinely 1.5 C too cold must still heat"

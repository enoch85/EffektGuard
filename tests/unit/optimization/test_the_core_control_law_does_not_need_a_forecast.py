"""The weather entity is optional; the weather-compensation CONTROL LAW is not.

Math WC is the EN 442 emitter law: given the outdoor temperature and indoor setpoint, what flow
temperature do the emitters need? Its inputs are the pump's OWN sensors (nibe_state.outdoor_temp and
flow_temp), which are always present; it does not read the forecast. So `evaluate_layer` must NOT
early-return when weather_data is None - a blank optional weather dropdown would otherwise silently
switch off the layer that votes on every cycle.

Invariant: with weather_data=None the Math WC layer still votes (weight > 0) and computes the SAME
offset it would with a forecast, keeping its sign across the whole winter; the pre-heat layer, which
genuinely needs a forecast, still abstains without one.
"""

from __future__ import annotations

import pytest

from custom_components.effektguard.optimization.weather_layer import (
    AdaptiveClimateSystem,
    WeatherCompensationCalculator,
    WeatherCompensationLayer,
    WeatherPredictionLayer,
)


class _NibeState:
    """Only the fields the emitter law reads. All of them come from the pump itself."""

    def __init__(self, outdoor_temp: float, flow_temp: float, degree_minutes: float = -30.0):
        self.outdoor_temp = outdoor_temp
        self.flow_temp = flow_temp
        self.degree_minutes = degree_minutes


def _layer() -> WeatherCompensationLayer:
    return WeatherCompensationLayer(
        weather_comp=WeatherCompensationCalculator(),
        climate_system=AdaptiveClimateSystem(latitude=59.3),  # Stockholm
        weather_learner=None,
    )


def test_math_wc_still_votes_when_no_weather_entity_is_configured():
    """The bug. A blank optional dropdown silently switched off the primary control law."""
    decision = _layer().evaluate_layer(
        nibe_state=_NibeState(outdoor_temp=-5.0, flow_temp=30.0),
        weather_data=None,  # exactly what WeatherAdapter returns with no entity configured
        target_temp=21.0,
    )

    assert decision.weight > 0.0, (
        f"Math WC returned weight={decision.weight} reason={decision.reason!r} because no weather "
        f"entity is configured. Math WC is the EN 442 emitter law over the pump's OWN outdoor and "
        f"flow sensors - it does not read the forecast. Switching it off leaves the air-source "
        f"F2040 pinned at maximum offset against a saturated compressor: 13x more immersion heat "
        f"than its capacity deficit forced, and 1265 minutes above the comfort ceiling."
    )


def test_the_offset_is_the_same_with_and_without_a_forecast():
    """It must not merely vote - it must compute the SAME answer. The forecast is not an input."""
    nibe_state = _NibeState(outdoor_temp=-5.0, flow_temp=30.0)

    without = _layer().evaluate_layer(nibe_state=nibe_state, weather_data=None, target_temp=21.0)
    with_forecast = _layer().evaluate_layer(
        nibe_state=nibe_state,
        weather_data=_FORECAST_THAT_CHANGES_NOTHING,
        target_temp=21.0,
    )

    assert without.offset == pytest.approx(with_forecast.offset), (
        f"the emitter law returned {without.offset} without a forecast and {with_forecast.offset} "
        f"with one. Its inputs are the outdoor temperature, the flow temperature and the setpoint. "
        f"A forecast that changes the answer means the forecast leaked into a calculation that is "
        f"defined not to use it."
    )
    assert without.weight == pytest.approx(with_forecast.weight)


@pytest.mark.parametrize("outdoor_temp", [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0])
def test_a_cold_house_is_still_told_to_add_heat_with_no_forecast(outdoor_temp):
    """The law must keep its sign across the whole winter, not just at one temperature.

    Flow is held far below what the emitters need, so the correct answer is always "add heat".
    Before the fix this returned a flat 0.0 at every outdoor temperature - the DM ran away, the
    other layers pinned the offset at maximum, and the immersion heater picked up the difference.
    """
    decision = _layer().evaluate_layer(
        nibe_state=_NibeState(outdoor_temp=outdoor_temp, flow_temp=22.0),
        weather_data=None,
        target_temp=21.0,
    )

    assert decision.offset > 0.0 and decision.weight > 0.0, (
        f"at {outdoor_temp}C with the flow 22C - well under what the radiators need - Math WC "
        f"proposed offset={decision.offset} weight={decision.weight}. With no forecast the law "
        f"went quiet and the house was left to the layers that cannot see a heating curve."
    )


def test_the_forecast_layer_itself_still_stands_down_without_a_forecast():
    """The other half. Pre-heat genuinely needs a forecast, and must NOT invent one."""
    preheat = WeatherPredictionLayer(thermal_mass=1.0, forecast_horizon=12)

    decision = preheat.evaluate_layer(
        nibe_state=_NibeState(outdoor_temp=-5.0, flow_temp=30.0),
        weather_data=None,
        thermal_trend={},
    )

    assert decision.weight == 0.0, (
        "weather PRE-HEAT is forecast-driven by definition - with no forecast it must abstain, not "
        "guess. Fixing Math WC must not drag it along."
    )


class _Hour:
    def __init__(self, temperature: float):
        self.temperature = temperature


class _Forecast:
    def __init__(self):
        self.current_temp = -5.0
        self.forecast_hours = [_Hour(-5.0) for _ in range(48)]
        self.source_entity = "test"


_FORECAST_THAT_CHANGES_NOTHING = _Forecast()

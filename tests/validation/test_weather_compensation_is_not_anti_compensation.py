"""Weather compensation must ask for a flow temperature that can actually heat the house.

The "Math WC" layer is enabled on every installation (decision_engine reads
`config.get("enable_weather_compensation", True)`; no config-flow option switches it off). It can
still take the early exit `if not weather_data or not weather_data.forecast_hours: weight=0.0`, so
an installation with a blank weather entity runs with it silently disabled - see
tests/unit/optimization/test_the_core_control_law_does_not_need_a_forecast.py.

The test house is the standard Swedish low-temperature radiator design: 22 C indoor, 150 W/K, 50 C
supply at the -15 C design outdoor. At -15 C the emitters MUST run at 50 C or the house cannot hold
22 C - a matter of the emitter law, not opinion. The removed Kuehne model targeted far less (its
curve rose only ~0.22 C of supply per -1 C outdoor where this house needs 0.76), cutting hardest
exactly when the house needs heat most; and nothing downstream catches it, because lowering the
offset lowers S1 and DM = integral(BT25 - S1), so degree minutes IMPROVE as the house cools (F-120).

These tests assert properties ANY correct model has, so they outlive the model that satisfies them:
  ADEQUACY - at the design outdoor temperature the flow target must be able to heat the house.
  NO CUTS  - with the house on target and the curve already correct, the layer must not take heat
             away. It is a trim, not a replacement curve.
  BOUNDED  - the correction stays inside WEATHER_COMP_MAX_OFFSET in both directions.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import WEATHER_COMP_MAX_OFFSET
from custom_components.effektguard.adapters.weather_adapter import (
    WeatherData,
    WeatherForecastHour,
)
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

# The test house: standard Swedish low-temperature radiator design.
TARGET_INDOOR = 22.0
DESIGN_OUTDOOR = -15.0
DESIGN_FLOW = 50.0  # supply needed at DESIGN_OUTDOOR to hold TARGET_INDOOR
HEAT_LOSS_COEFFICIENT = 150.0

# A correctly tuned curve for that house: flow(-15) == 50, and flow == room when no heat is
# needed (outdoor == room). Slope = (50 - 22) / (22 - -15) = 0.757 C of supply per C outdoor.
CURVE_SLOPE = (DESIGN_FLOW - TARGET_INDOOR) / (TARGET_INDOOR - DESIGN_OUTDOOR)

# How far below the design flow the model may fall at the design point before the house can no
# longer be heated. Generous: the true shortfall under Kuehne is 18.3 C.
DESIGN_FLOW_SHORTFALL_ALLOWED = 2.0

# The deepest heat CUT that can be justified while the house sits exactly on target and the curve
# already delivers what the house needs - which is to say, almost none. A small POSITIVE trim is
# not bounded here: adding heat is the safe direction, and WEATHER_COMP_MAX_OFFSET already caps
# the magnitude in both directions. What must never happen is the layer taking heat AWAY from a
# house that is exactly where it should be.
MAX_DEFENSIBLE_CUT = -1.0

NOW = datetime(2026, 1, 15, 12, 0)


def _correct_curve_flow(outdoor: float) -> float:
    """Supply temperature the correctly tuned curve delivers at this outdoor temperature."""
    return TARGET_INDOOR + CURVE_SLOPE * (TARGET_INDOOR - outdoor)


@pytest.fixture
def engine() -> DecisionEngine:
    config = {
        "target_indoor_temp": TARGET_INDOOR,
        "tolerance": 0.5,
        "optimization_mode": "balanced",
        "enable_weather_compensation": True,
        "enable_peak_protection": True,
        "enable_price_optimization": True,
        "latitude": 59.33,
        "heating_type": "radiator",
        "heat_loss_coefficient": HEAT_LOSS_COEFFICIENT,
        "thermal_mass": 0.7,
        "insulation_quality": 1.0,
    }
    return DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(0.7, 1.0),
        config=config,
        heat_pump_model=NibeF750Profile(),
    )


def _evaluate(engine: DecisionEngine, outdoor: float):
    """Math WC's decision with the house on target and the curve already correct.

    The layer is evaluated directly rather than fished out of `decision.layers`, because the
    aggregate flattens each layer into a `LayerDecision` that carries only name/offset/weight and
    drops `optimal_flow_temp` - the flow target is exactly what these tests need to see.

    The outdoor temperature is steady (flat forecast), so nothing the layer does here can be a
    legitimate response to weather that is about to change.
    """
    forecast = [
        WeatherForecastHour(datetime=NOW + timedelta(hours=h), temperature=outdoor)
        for h in range(1, 49)
    ]
    flow = _correct_curve_flow(outdoor)
    state = NibeState(
        outdoor_temp=outdoor,
        indoor_temp=TARGET_INDOOR,  # exactly on target
        supply_temp=round(flow, 1),
        return_temp=round(flow - 5.0, 1),
        degree_minutes=-30.0,  # healthy
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=NOW,
        compressor_hz=50,
        power_kw=2.0,
    )
    return engine.weather_comp_layer.evaluate_layer(
        nibe_state=state,
        weather_data=WeatherData(
            current_temp=outdoor, forecast_hours=forecast, source_entity="test"
        ),
        target_temp=TARGET_INDOOR,
    )


def test_flow_target_at_design_temperature_can_actually_heat_the_house(engine):
    """At the design outdoor temperature the flow target must be able to heat the house.

    This is the decisive invariant and it needs no arbitrary threshold: at -15 C this house
    requires 50 C of supply to hold 22 C. A weather-compensation model that targets less than
    that is asking the emitters to deliver the design heat load at below the design temperature,
    which the emitter law forbids. Whatever model is used, it must clear its own design point.
    """
    layer = _evaluate(engine, DESIGN_OUTDOOR)
    target_flow = layer.optimal_flow_temp

    assert target_flow >= DESIGN_FLOW - DESIGN_FLOW_SHORTFALL_ALLOWED, (
        f"At the {DESIGN_OUTDOOR:.0f} C design temperature this house needs {DESIGN_FLOW:.1f} C "
        f"of supply to hold {TARGET_INDOOR:.0f} C indoor. Weather compensation targets "
        f"{target_flow:.1f} C - a {DESIGN_FLOW - target_flow:.1f} C shortfall - and so commands "
        f"{layer.offset:+.2f} C of curve offset at the coldest hour of the winter."
    )


def test_compensation_never_cuts_heat_from_a_house_that_is_already_correct(engine):
    """The layer must not take heat AWAY from a house on target with a correct curve.

    This is the defect itself, stated as an invariant. Across the whole operating range the pump
    is already delivering exactly what the house needs, so there is nothing to cut - and the
    colder it gets, the less defensible a cut becomes. Kuehne cut deeper and deeper: -2.71 at
    +10 C, -6.09 at 0 C, -11.06 at -15 C.

    A small POSITIVE trim is fine and is not failed here; adding heat is the safe direction, and
    WEATHER_COMP_MAX_OFFSET bounds the magnitude both ways.
    """
    cuts = []
    for outdoor in (10.0, 5.0, 0.0, -5.0, -10.0, -15.0, -20.0):
        layer = _evaluate(engine, outdoor)
        if layer.offset < MAX_DEFENSIBLE_CUT:
            cuts.append(
                f"{outdoor:+.0f} C: curve delivers {_correct_curve_flow(outdoor):.1f} C, "
                f"layer wants only {layer.optimal_flow_temp:.1f} C, commands {layer.offset:+.2f}"
            )

    assert not cuts, (
        "Weather compensation cuts heat from a house that is exactly on target with a correctly "
        f"tuned curve (deepest defensible cut {MAX_DEFENSIBLE_CUT:+.1f} C):\n  " + "\n  ".join(cuts)
    )


def test_compensation_offset_is_bounded(engine):
    """The correction is a trim and stays inside its declared bound, in both directions.

    An unbounded offset is how a mis-configured design point turns into a large swing at the
    pump. The old implementation had no clamp at all and could return -11.4.
    """
    for outdoor in (15.0, 10.0, 0.0, -10.0, -20.0, -30.0):
        offset = _evaluate(engine, outdoor).offset
        assert abs(offset) <= WEATHER_COMP_MAX_OFFSET + 1e-9, (
            f"At {outdoor:+.0f} C weather compensation commands {offset:+.2f} C, outside its "
            f"declared bound of +/-{WEATHER_COMP_MAX_OFFSET:.1f} C."
        )

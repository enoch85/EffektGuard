"""Weather compensation must command ~zero on a curve that is already correct.

A layer that adds a constant to every decision is not a controller, it is a bias - the removed
Kuehne model carried a persistent negative one that under-heated the house while presenting the
shortfall as savings. The direction is not what made it a bug; being a bias is.

So this asserts the property the failure shared with its replacement: with the house exactly on
target, degree minutes healthy, a steady forecast, and the pump's own curve already delivering what
the emitter law asks for, there is nothing to correct and the offset must be ~0. The climate-zone
safety margin is what breaks this - it exists to pull up a curve running COLD in a hard winter, but
adding it unconditionally tells a perfectly-tuned curve to add heat too. A margin is permission to
run warm, not an instruction to.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.const import DEFAULT_HEAT_LOSS_COEFFICIENT, INTERNAL_GAINS_W
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.adapters.weather_adapter import (
    WeatherData,
    WeatherForecastHour,
)
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

TARGET_INDOOR = 22.0
DESIGN_OUTDOOR = -15.0
DESIGN_FLOW = 50.0
DESIGN_SPREAD = 5.0
EMITTER_EXPONENT = 1.3

# The correction that remains when there is genuinely nothing to correct. Not zero, because the
# pump's curve is quantised and the emitter law is continuous - but a fraction of one offset step.
NO_CORRECTION_NEEDED = 0.35

NOW = datetime(2026, 1, 15, 12, 0)


def _emitter_law_flow(outdoor: float) -> float:
    """The flow a PERFECTLY tuned curve delivers - taken from OpenEnergyMonitor, not from us.

    A reference has to come from outside, or it reproduces the code's own bug. This is
    OpenEnergyMonitor's weather-compensation tool (weathercomp.js):

        DT    = (heat_demand / rated_emitter_output_dt50) ** (1/1.3) * 50
        flowT = room_temperature + DT + systemDT * 0.5        <- systemDT, NOT systemDT * phi

    The flow-return spread is CONSTANT (a heat pump modulates its circulator). Anchored on our
    design point rather than theirs, which is the same equation rewritten.
    """
    balance = TARGET_INDOOR - INTERNAL_GAINS_W / DEFAULT_HEAT_LOSS_COEFFICIENT
    load = balance - outdoor
    design_load = balance - DESIGN_OUTDOOR
    design_excess = DESIGN_FLOW - DESIGN_SPREAD / 2 - TARGET_INDOOR
    phi = load / design_load
    return TARGET_INDOOR + design_excess * phi ** (1 / EMITTER_EXPONENT) + DESIGN_SPREAD / 2


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
        "heat_loss_coefficient": 150.0,
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


def _offset_on_a_perfect_curve(engine: DecisionEngine, outdoor: float) -> float:
    flow = _emitter_law_flow(outdoor)
    forecast = [
        WeatherForecastHour(datetime=NOW + timedelta(hours=h), temperature=outdoor)
        for h in range(1, 49)
    ]
    state = NibeState(
        outdoor_temp=outdoor,
        indoor_temp=TARGET_INDOOR,
        supply_temp=round(flow, 1),
        return_temp=round(flow - DESIGN_SPREAD, 1),
        degree_minutes=-30.0,
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
    ).offset


def test_no_dc_bias_when_the_curve_is_already_perfect(engine):
    """The pump is delivering exactly the emitter law's answer. Ask for nothing."""
    biased = []
    for outdoor in (10.0, 5.0, 0.0, -5.0, -10.0, -15.0, -20.0):
        offset = _offset_on_a_perfect_curve(engine, outdoor)
        if abs(offset) > NO_CORRECTION_NEEDED:
            biased.append(
                f"{outdoor:+.0f} C: curve delivers {_emitter_law_flow(outdoor):.2f} C, exactly "
                f"what the emitter law asks - yet the layer commands {offset:+.2f}"
            )

    assert not biased, (
        "Weather compensation carries a DC bias: it corrects a curve that needs no correction "
        f"(tolerance +/-{NO_CORRECTION_NEEDED}):\n  " + "\n  ".join(biased)
    )


def test_the_bias_does_not_merely_average_out(engine):
    """A bias that cancels across the range would be noise; one that does not is a setback.

    The sign is irrelevant - a persistent +1.5 C over-heats the house and raises the bill just as
    reliably as a negative bias under-heats it and lowers it.
    """
    walk = [10.0, 5.0, 0.0, -5.0, -10.0, -15.0, -20.0]
    offsets = [_offset_on_a_perfect_curve(engine, t) for t in walk]
    mean = sum(offsets) / len(offsets)

    assert abs(mean) <= NO_CORRECTION_NEEDED, (
        f"Mean offset {mean:+.2f} C across the operating range on a perfectly tuned curve. "
        f"This is a permanent setback, not a correction. Offsets: "
        + ", ".join(f"{t:+.0f}C:{o:+.2f}" for t, o in zip(walk, offsets))
    )


def test_a_cold_curve_is_still_pulled_up(engine):
    """The margin's safety purpose must survive: an under-supplying curve gets corrected."""
    outdoor = -15.0
    short_by = 4.0
    flow = _emitter_law_flow(outdoor) - short_by
    forecast = [
        WeatherForecastHour(datetime=NOW + timedelta(hours=h), temperature=outdoor)
        for h in range(1, 49)
    ]
    state = NibeState(
        outdoor_temp=outdoor,
        indoor_temp=TARGET_INDOOR,
        supply_temp=round(flow, 1),
        return_temp=round(flow - DESIGN_SPREAD, 1),
        degree_minutes=-30.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=NOW,
        compressor_hz=50,
        power_kw=2.0,
    )
    offset = engine.weather_comp_layer.evaluate_layer(
        nibe_state=state,
        weather_data=WeatherData(
            current_temp=outdoor, forecast_hours=forecast, source_entity="test"
        ),
        target_temp=TARGET_INDOOR,
    ).offset

    assert offset > 1.0, (
        f"A curve running {short_by:.0f} C COLD at the design temperature must be pulled up. "
        f"The layer commands {offset:+.2f}."
    )

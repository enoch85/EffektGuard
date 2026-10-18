"""A winter reading with the elpatron running is normal, not an every-cycle warning.

typical_electrical_range_kw is the compressor draw alone (0.27-2.06 kW). The validator compared
the whole-machine reading against it and flagged "exceeds max" on every cold cycle where the
immersion heater was doing its job. The machine's plausible ceiling is compressor + immersion
heater: below it, aux-range draw is normal; above it, the reading is implausible for the
hardware. That is a debug line, not a warning: the configured sensor is often the whole house.
"""

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.models.nibe.f750 import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel


def _engine() -> DecisionEngine:
    engine = DecisionEngine.__new__(DecisionEngine)
    engine.heat_pump_model = NibeF750Profile()
    return engine


def test_compressor_plus_elpatron_draw_is_valid_and_quiet():
    # 2.0 kW compressor + 3.5 kW delivery-setting immersion: a cold January morning.
    result = _engine()._validate_power_consumption(5.5, outdoor_temp=-10.0)

    assert result["valid"] is True
    assert result["warning"] is None, (
        f"A draw the machine's own immersion heater fully explains was flagged: "
        f"{result['warning']!r}. This fired every cycle, all winter."
    )


def test_a_draw_no_f750_can_produce_is_flagged():
    # Ceiling is (compressor max 2.06 + immersion 3.5) x margin = 6.67 kW; 12 kW is not this machine.
    result = _engine()._validate_power_consumption(12.0, outdoor_temp=-10.0)

    assert result["valid"] is False
    assert result["warning"] is not None


def test_a_whole_house_reading_is_not_a_warning(caplog):
    """The owner's power sensor is the house's grid import: 8.8 kW on an evening with the car
    charging was a warning every cycle - 37 in three hours on 2026-10-17."""
    engine = DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(2.0, 1.5),
        config={"target_indoor_temp": 21.0, "tolerance": 1.2, "latitude": 55.6},
        heat_pump_model=NibeF750Profile(),
    )
    state = NibeState(
        outdoor_temp=5.0,
        indoor_temp=21.5,
        supply_temp=30.0,
        return_temp=27.0,
        degree_minutes=-200.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 10, 17, 19, 0, tzinfo=timezone.utc),
        compressor_hz=40,
        power_kw=8.8,
    )

    with caplog.at_level(logging.DEBUG, logger="custom_components.effektguard.optimization"):
        engine.calculate_decision(state, None, None, 0.0, 8.8)

    lines = [r for r in caplog.records if "exceeds what a" in r.getMessage()]
    assert [r.levelno for r in lines] == [logging.DEBUG]

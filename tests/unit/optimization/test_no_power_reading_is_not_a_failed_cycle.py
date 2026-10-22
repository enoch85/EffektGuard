"""A cycle with no power reading decides like any other; it must not fail into a written 0 °C.

`NibeState.power_kw` is None when there is no power sensor and the adapter's own estimate fails. The
engine's power check compared None with the machine's ceiling, raised TypeError, and the coordinator
caught it and applied the safe default - offset 0.0, written over whatever the register held
(control review F-6: every fuzz case with power None, 4,991 of 20,000).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel


def test_a_cycle_without_a_power_reading_still_decides():
    engine = DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(1.0, 1.0),
        config={"target_indoor_temp": 21.0, "tolerance": 0.5, "latitude": 59.33},
        heat_pump_model=NibeF750Profile(),
    )
    state = NibeState(
        outdoor_temp=-5.0,
        indoor_temp=20.6,
        supply_temp=38.0,
        return_temp=33.0,
        degree_minutes=-300.0,
        current_offset=2.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc),
        power_kw=None,
    )

    decision = engine.calculate_decision(state, None, None, 0.0, 0.0)

    assert decision.layers, "the engine returned no layer votes"

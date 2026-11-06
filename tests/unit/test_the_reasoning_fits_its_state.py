"""The decision's reasoning fits Home Assistant's state by whole entries; its votes, one per line.

Cut at the limit mid-word, the reasoning lost its end in 2473 of production's 2877 decisions from
27 Oct to 6 Nov 2026 - the forecast's part with it.
"""

from unittest.mock import MagicMock, Mock

from homeassistant.const import MAX_LENGTH_STATE_STATE

from custom_components.effektguard.const import REASONING_SEPARATOR
from custom_components.effektguard.optimization.decision_engine import (
    LayerDecision,
    OptimizationDecision,
)
from custom_components.effektguard.sensor import SENSORS, EffektGuardSensor

# Production's 03:45 decision on 6 Nov 2026, as the reasoning now reads
ENTRIES = [
    "Immersion heat lowered +0.0 to -1.0 (DM -582, starts at -700)",
    "Compressor at max lowered +1.4 to +0.0",
    "Spot Price +3.5@0.80 (very cheap night → +1.7, price +73% in 4h → pre-heat +2.0, "
    "room 0.7°C over → -0.2)",
    "Comfort -2.1@0.36 (0.7°C over target)",
    "Math WC -0.3@0.23 (flow 31.2°C, wants 30.7°C, DM debt → weight cut)",
]


def _sensor(decision: OptimizationDecision) -> EffektGuardSensor:
    coordinator = MagicMock()
    coordinator.data = {"decision": decision}
    entry = Mock()
    entry.entry_id = "test-entry"
    description = next(s for s in SENSORS if s.key == "optimization_reasoning")
    return EffektGuardSensor(coordinator, entry, description)


def test_a_long_reasoning_keeps_whole_entries_from_the_front():
    reasoning = REASONING_SEPARATOR.join(ENTRIES)
    assert len(reasoning) > MAX_LENGTH_STATE_STATE, "precondition: too long for a state"

    state = _sensor(OptimizationDecision(offset=-1.0, reasoning=reasoning)).native_value

    assert len(state) <= MAX_LENGTH_STATE_STATE, state
    *kept, rest = state.split(REASONING_SEPARATOR)
    assert rest == "…" and kept == ENTRIES[: len(kept)], state


def test_the_votes_are_listed_one_per_line_heaviest_first():
    layers = [
        LayerDecision(name="Safety", offset=0.0, weight=0.0, reason="OK"),
        LayerDecision(name="Math WC", offset=-0.3, weight=0.23, reason="flow 31.2°C"),
        LayerDecision(name="Spot Price", offset=3.5, weight=0.8, reason="very cheap night"),
        LayerDecision(name="Comfort", offset=-2.1, weight=0.36, reason="0.7°C over target"),
    ]

    attributes = _sensor(
        OptimizationDecision(offset=-1.0, layers=layers, reasoning="")
    ).extra_state_attributes

    assert list(attributes["layers"]) == ["Spot Price", "Comfort", "Math WC"]
    assert attributes["layers"]["Comfort"] == "-2.1@0.36 (0.7°C over target)"

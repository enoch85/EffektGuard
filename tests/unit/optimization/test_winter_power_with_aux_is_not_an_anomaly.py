"""A winter reading with the elpatron running is normal, not an every-cycle warning.

typical_electrical_range_kw is the compressor draw alone (0.27-2.06 kW). The validator compared
the whole-machine reading against it and flagged "exceeds max" on every cold cycle where the
immersion heater was doing its job. The machine's plausible ceiling is compressor + immersion
heater: below it, aux-range draw is normal; above it, the reading is implausible for the
hardware and worth a warning.
"""

from unittest.mock import MagicMock

from custom_components.effektguard.models.nibe.f750 import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine


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

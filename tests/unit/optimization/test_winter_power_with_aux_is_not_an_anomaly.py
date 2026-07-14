"""A winter reading with the elpatron running is normal, not an every-cycle log line.

The F750's typical_electrical_range_kw was corrected to its rating-point compressor draw
(0.27-2.06 kW) - true, but the power validator compared the whole-machine reading against it
and logged "exceeds max (auxiliary heating active?)" on EVERY cold-weather cycle where the
immersion heater was doing exactly its job. A channel that cries wolf every five minutes all
January is a channel nobody reads in February.

The machine's plausible ceiling is compressor + immersion heater. Below it, aux-range draw is
silent normality; above it, the reading is implausible for the hardware and worth a warning.
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
    # Compressor max 2.06 + immersion 6.5 = 8.56; 12 kW is not this machine.
    result = _engine()._validate_power_consumption(12.0, outdoor_temp=-10.0)

    assert result["valid"] is False
    assert result["warning"] is not None

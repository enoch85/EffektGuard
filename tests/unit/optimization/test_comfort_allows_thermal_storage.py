"""The comfort layer must not fight a deliberate price-driven excursion inside the storage band.

The building fabric is the only battery EffektGuard has. Charging it means running the house warm
while power is cheap and coasting while it is dear - the house MUST be allowed to move.

The comfort layer prevented that. It applied a strong correction (weight 0.7) as soon as indoor
passed target + 0.5 C, so every charge was cancelled almost as soon as it began. The house swung
about 0.2 C and captured 0.7% of the spot bill, where a reference controller swinging the owner's
authorised 1.0 C captured 5.1% on the same day, plant and prices.

Comfort's job is to keep the house INSIDE the band, not pinned to the middle of it. Within the
band it is a weak spring - enough to return the house to target when prices are neutral, not
enough to overrule a price layer that has a reason to move it. Outside the band it is in charge
again, and nothing about the hard safety floor changes.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    LAYER_WEIGHT_COMFORT_HIGH,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    THERMAL_BATTERY_BAND,
)
from custom_components.effektguard.optimization.comfort_layer import ComfortLayer

TARGET = 22.0

# The price layer speaks at ~0.8. To be able to charge the fabric, comfort must be quieter than
# that inside the band, or the charge is simply averaged away.
QUIET_ENOUGH_TO_BE_OVERRULED = 0.5


def _state(indoor: float) -> NibeState:
    return NibeState(
        outdoor_temp=0.0,
        indoor_temp=indoor,
        supply_temp=40.0,
        return_temp=35.0,
        degree_minutes=-30.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 1, 15, 12, 0),
        compressor_hz=50,
        power_kw=2.0,
    )


@pytest.fixture
def comfort() -> ComfortLayer:
    return ComfortLayer(
        target_temp=TARGET,
        mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],
        tolerance_range=0.5,
    )


def _evaluate(comfort: ComfortLayer, indoor: float):
    return comfort.evaluate_layer(
        nibe_state=_state(indoor),
        weather_data=MagicMock(forecast_hours=[]),
        price_data=None,
    )


@pytest.mark.parametrize("charge", [0.4, 0.6, 0.8, 0.95])
def test_comfort_does_not_cancel_a_charge_inside_the_band(comfort, charge):
    """A house deliberately run warm on cheap power is doing its job, not misbehaving."""
    decision = _evaluate(comfort, TARGET + charge)

    assert decision.weight < QUIET_ENOUGH_TO_BE_OVERRULED, (
        f"Charged {charge:+.2f} C above target - inside the {THERMAL_BATTERY_BAND:.1f} C storage "
        f"band - and comfort answers with weight {decision.weight:.2f} and offset "
        f"{decision.offset:+.2f}. It will cancel the charge before the fabric holds any heat."
    )


@pytest.mark.parametrize("coast", [0.4, 0.6, 0.8, 0.95])
def test_comfort_does_not_cancel_a_coast_inside_the_band(comfort, coast):
    """Nor is coasting on dear power a fault, so long as the house stays in the band."""
    decision = _evaluate(comfort, TARGET - coast)

    assert decision.weight < QUIET_ENOUGH_TO_BE_OVERRULED, (
        f"Coasted {coast:.2f} C below target - inside the storage band - and comfort answers with "
        f"weight {decision.weight:.2f}."
    )


def test_comfort_still_pulls_back_toward_target_inside_the_band(comfort):
    """A weak spring, not an absence of one: neutral prices must return the house to target.

    Without this the optimiser could park at the cold edge of the band indefinitely and bank the
    shortfall as savings - the very trade this audit exists to stop.
    """
    warm = _evaluate(comfort, TARGET + 0.8)
    cold = _evaluate(comfort, TARGET - 0.8)

    assert warm.offset < 0, "warm house must be gently cooled, not left to drift"
    assert cold.offset > 0, "cool house must be gently warmed, not left to drift"
    assert warm.weight > 0, "a zero weight is no spring at all"
    assert cold.weight > 0


def test_comfort_takes_charge_again_outside_the_band(comfort):
    """The band is a limit, not a licence. Past it, comfort outranks any price signal."""
    for excursion in (THERMAL_BATTERY_BAND + 0.3, THERMAL_BATTERY_BAND + 1.0):
        warm = _evaluate(comfort, TARGET + excursion)
        cold = _evaluate(comfort, TARGET - excursion)

        assert warm.weight >= LAYER_WEIGHT_COMFORT_HIGH, (
            f"{excursion:+.1f} C above target is outside the {THERMAL_BATTERY_BAND:.1f} C band; "
            f"comfort must reassert itself (weight {warm.weight:.2f})"
        )
        assert cold.weight >= LAYER_WEIGHT_COMFORT_HIGH, (
            f"{excursion:.1f} C below target is outside the band; comfort must reassert itself "
            f"(weight {cold.weight:.2f})"
        )

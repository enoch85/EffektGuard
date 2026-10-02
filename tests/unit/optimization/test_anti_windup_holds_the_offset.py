"""Anti-windup holds the offset the pump is already running, and never blocks backing off.

When degree minutes are FALLING while the offset is already positive, the heat is in
transit through the thermal mass: raising S1 widens the (BT25 - S1) gap that DM integrates,
so asking for more makes the number worse for hours. The response is to stop escalating.

The cap was written as

    max(current_offset, current_offset * ANTI_WINDUP_OFFSET_CAP_MULTIPLIER)   # mult = 0.7

which is identically `current_offset` for every positive offset - and anti-windup only
triggers at an offset of at least ANTI_WINDUP_MIN_POSITIVE_OFFSET, so the multiplier never
changed an outcome at any reachable input while reading like a 30% safety reduction. It is
gone; holding is the behaviour and now also the stated behaviour. If a reduction is ever
wanted here it has to be deliberate - ACTIVE reduction is a separate path keyed on how bad
the spiral is (ANTI_WINDUP_REDUCTION_THRESHOLD and a rate-proportional reduction).

Driven through `evaluate_layer`, the way the coordinator drives it: the layer builds its own
DM history from successive calls, so a falling series with a positive offset is all it takes
to arm anti-windup. At -5 C with DM past the threshold the T1 tier asks for +4.0 C, which is
what makes both directions observable from outside - below 4.0 the request is an escalation
and gets held, above it the request is a reduction and must pass.
"""

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer

TZ = timezone(timedelta(hours=1))
TARGET = 21.0
OUTDOOR = -5.0
# What T1 asks for in this state, measured. Offsets below it are escalations, above it
# reductions - the two sides of the only behaviour this file pins.
TIER_REQUEST = 4.0


class _State:
    def __init__(self, dm: float, current_offset: float, timestamp: datetime):
        self.degree_minutes = dm
        self.outdoor_temp = OUTDOOR
        self.indoor_temp = TARGET - 0.5
        self.current_offset = current_offset
        self.timestamp = timestamp
        self.flow_temp = 35.0
        self.supply_temp = 35.0
        self.is_heating = True
        self.is_hot_water = False
        self.power_kw = 2.0


def _drive(current_offset: float, dm_start: float, dm_step: float, samples: int = 6):
    """Feed a DM series into a fresh layer and return its last decision.

    A fresh layer per call: the DM history is what arms anti-windup, so sharing one layer
    between probes would have it comparing unrelated samples.
    """
    layer = EmergencyLayer(climate_detector=ClimateZoneDetector(59.33), heating_type="radiator")
    start = datetime(2026, 1, 15, 12, 0, tzinfo=TZ)
    decision = None
    for i in range(samples):
        decision = layer.evaluate_layer(
            nibe_state=_State(
                dm_start + dm_step * i, current_offset, start + timedelta(minutes=5 * i)
            ),
            weather_data=None,
            price_data=None,
            target_temp=TARGET,
            tolerance_range=0.2,
        )
    return decision


def _falling(current_offset: float):
    return _drive(current_offset, dm_start=-520.0, dm_step=-25.0)


@pytest.mark.parametrize("current", [1.0, 3.0])
def test_an_escalation_is_held_at_the_offset_already_running(current):
    """A tier asking for more than the pump is already doing gets held, not raised.

    Held means held: exactly the current offset. Any fraction of it would be a reduction,
    which is a different mechanism and must not appear here by accident.
    """
    decision = _falling(current)

    assert (
        decision.anti_windup_active
    ), "precondition: falling DM at a positive offset arms anti-windup"
    assert (
        decision.offset == current
    ), f"anti-windup let the offset go from {current} to {decision.offset} while DM was falling"


@pytest.mark.parametrize("current", [6.0, 9.0])
def test_anti_windup_never_blocks_backing_off(current):
    """Anti-windup prevents escalation. It must not prevent a reduction."""
    decision = _falling(current)

    assert (
        decision.anti_windup_active
    ), "precondition: falling DM at a positive offset arms anti-windup"
    assert decision.offset == TIER_REQUEST < current, (
        f"anti-windup held the offset at {decision.offset} when the tier asked to come down "
        f"from {current} to {TIER_REQUEST}"
    )


@pytest.mark.parametrize("dm_step", [25.0, 0.0], ids=["recovering", "flat"])
def test_a_positive_offset_alone_does_not_arm_anti_windup(dm_step):
    """Heat in transit is the trigger, not a positive offset. DM must actually be falling."""
    decision = _drive(1.0, dm_start=-770.0 if dm_step else -645.0, dm_step=dm_step)

    assert not decision.anti_windup_active
    assert decision.offset == TIER_REQUEST, "a stood-down anti-windup must not cap anything"

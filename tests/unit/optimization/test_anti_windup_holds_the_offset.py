"""Anti-windup holds the offset, and says so with a number that means something.

When degree minutes are FALLING while the offset is already positive, the heat is in
transit through the thermal mass: raising S1 widens the (BT25 - S1) gap that DM integrates,
so asking for more makes the number worse for hours. The response is to stop escalating.

The cap was written as

    max(current_offset, current_offset * ANTI_WINDUP_OFFSET_CAP_MULTIPLIER)   # mult = 0.7

which is identically `current_offset` for every positive offset - and anti-windup only
triggers at an offset of at least ANTI_WINDUP_MIN_POSITIVE_OFFSET, so the multiplier never
changed an outcome at any reachable input while reading like a 30% safety reduction. It is
gone; holding is the behaviour and now also the stated behaviour.

Active reduction is a SEPARATE path, keyed on how bad the spiral is
(ANTI_WINDUP_REDUCTION_THRESHOLD and a rate-proportional reduction), and these tests pin
the boundary between the two so a future change cannot quietly merge them.
"""

import pytest

from custom_components.effektguard import const
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer


@pytest.fixture
def layer():
    return EmergencyLayer(climate_detector=ClimateZoneDetector(59.33), heating_type="radiator")


def test_the_dead_multiplier_is_gone():
    """A constant that cannot change an outcome must not sit in const.py looking load-bearing."""
    assert not hasattr(const, "ANTI_WINDUP_OFFSET_CAP_MULTIPLIER"), (
        "ANTI_WINDUP_OFFSET_CAP_MULTIPLIER is back. max(x, 0.7x) == x for every positive x, "
        "and anti-windup only fires above +0.5, so it can never apply. If a real reduction "
        "is wanted, use the ANTI_WINDUP_REDUCTION_* path, which is rate-proportional."
    )


@pytest.mark.parametrize("current", [0.5, 1.0, 2.0, 5.0, 9.0])
def test_an_escalation_is_held_at_the_current_offset(layer, current):
    """A tier asking for more than the pump is already doing gets held, not raised."""
    held, reason = layer._apply_anti_windup_cap(
        calculated_offset=current + 3.0,
        current_offset=current,
        anti_windup_active=True,
        tier_name="T2",
    )
    assert held == current, (
        f"anti-windup let the offset climb from {current} to {held} while DM was falling"
    )
    assert reason, "a held offset must report that it was held"
    assert f"{current:.1f}" in reason


@pytest.mark.parametrize("current", [2.0, 5.0])
def test_a_lower_request_passes_through_untouched(layer, current):
    """Anti-windup prevents escalation. It must not prevent backing off."""
    passed, reason = layer._apply_anti_windup_cap(
        calculated_offset=current - 1.5,
        current_offset=current,
        anti_windup_active=True,
        tier_name="T1",
    )
    assert passed == current - 1.5
    assert reason == ""


def test_inactive_anti_windup_changes_nothing(layer):
    for calculated in (-3.0, 0.0, 4.0, 8.5):
        out, reason = layer._apply_anti_windup_cap(
            calculated_offset=calculated,
            current_offset=1.0,
            anti_windup_active=False,
            tier_name="T3",
        )
        assert out == calculated and reason == ""


def test_holding_is_not_silently_a_reduction(layer):
    """The old name said 'cap at 70%'. If someone reintroduces a reduction here it must
    be deliberate, so pin that the held value equals the current offset exactly."""
    held, _ = layer._apply_anti_windup_cap(
        calculated_offset=10.0, current_offset=4.0, anti_windup_active=True, tier_name="T3"
    )
    assert held == 4.0
    assert held != pytest.approx(4.0 * 0.7), "a 30% reduction here would be a new behaviour"

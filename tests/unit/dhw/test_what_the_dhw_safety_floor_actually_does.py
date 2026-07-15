"""What DHW_SAFETY_CRITICAL (20 C) actually does, versus its old "always heat below this" comment.

Below 20 C the optimizer stops WAITING FOR A CHEAPER PRICE. It does NOT heat unconditionally, and
must not, because two things still outrank the hot water - both deliberate:

  * CRITICAL THERMAL DEBT - a DHW cycle takes the compressor from space heating; doing that in deep
    degree-minute debt turns a recoverable debt into an immersion-heater one.
  * THE HOUSE BELOW ITS OWN SAFETY FLOOR (MIN_TEMP_LIMIT) - "DHW wins, but never below safety."

The code was right; the comment was the lie. These tests pin the real behaviour.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custom_components.effektguard.const import (
    DHW_SAFETY_CRITICAL,
    DHW_SAFETY_MIN,
    MIN_TEMP_LIMIT,
)
from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler

NOW = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)

# Five degrees UNDER the "hard floor". Every case below uses it.
FREEZING_TANK = DHW_SAFETY_CRITICAL - 5.0


def _decide(dhw: float, dm: float, indoor: float):
    return IntelligentDHWScheduler().should_start_dhw(
        current_dhw_temp=dhw,
        space_heating_demand_kw=3.0,
        thermal_debt_dm=dm,
        indoor_temp=indoor,
        target_indoor_temp=21.0,
        outdoor_temp=-5.0,
        price_classification="normal",
        current_time=NOW,
        price_periods=[],
        hours_since_last_dhw=6.0,
    )


def test_the_tank_used_in_these_tests_really_is_below_the_floor():
    """The premise."""
    assert FREEZING_TANK < DHW_SAFETY_CRITICAL < DHW_SAFETY_MIN


def test_below_the_floor_price_stops_being_a_reason_to_wait():
    """What the constant DOES do. A healthy house heats its water, whatever the price is doing."""
    decision = _decide(dhw=FREEZING_TANK, dm=-150.0, indoor=21.0)

    assert decision.should_heat is True, (
        f"The tank is at {FREEZING_TANK} C, below DHW_SAFETY_CRITICAL ({DHW_SAFETY_CRITICAL}), the "
        f"house is warm and the degree minutes are healthy - and the optimizer still declined to "
        f"heat: {decision.priority_reason}."
    )


def test_a_house_in_deep_thermal_debt_still_outranks_the_hot_water():
    """NOT "always heat". Taking the compressor now is how a recoverable debt becomes aux heat."""
    decision = _decide(dhw=FREEZING_TANK, dm=-1400.0, indoor=21.0)

    assert decision.should_heat is False, (
        f"The house is in deep thermal debt (DM -1400) and the optimizer started a hot-water cycle "
        f"anyway, because the tank was below DHW_SAFETY_CRITICAL. That takes the compressor away "
        f"from space heating at the worst possible moment. The constant's old comment - 'Hard "
        f"floor, always heat below this' - says to do exactly this, and it is wrong."
    )
    assert "THERMAL_DEBT" in decision.priority_reason


def test_a_house_below_its_own_safety_floor_still_outranks_the_hot_water():
    """The owner's rule: DHW wins, but never below safety."""
    decision = _decide(dhw=FREEZING_TANK, dm=-150.0, indoor=MIN_TEMP_LIMIT - 1.0)

    assert decision.should_heat is False, (
        f"The house is at {MIN_TEMP_LIMIT - 1.0} C - below its {MIN_TEMP_LIMIT} C safety floor - "
        f"and the optimizer started a hot-water cycle because the tank was cold. Space heating "
        f"outranks hot water when the house itself is unsafe. Nobody wants a hot shower in a "
        f"freezing house."
    )
    assert "SPACE_HEATING" in decision.priority_reason


def test_an_adequate_tank_in_a_healthy_house_still_waits_for_a_better_price():
    """The regression guard: none of this may switch the optimisation off."""
    decision = _decide(dhw=45.0, dm=-150.0, indoor=21.0)

    assert decision.should_heat is False
    assert "ADEQUATE" in decision.priority_reason


@pytest.mark.parametrize("dm", [-1400.0, -2000.0])
def test_the_precedence_does_not_depend_on_how_cold_the_tank_is(dm):
    """A tank at 5 C does not buy its way past a house in danger either."""
    assert _decide(dhw=5.0, dm=dm, indoor=21.0).should_heat is False

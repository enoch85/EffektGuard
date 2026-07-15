"""Three invariants of the EN 442 emitter law, each guarding a real defect in the flow curve.

1. No STEP at the balance point: `return indoor_setpoint` above it leaves a 2.5 C jump (spread/2),
   and the shoulder season crosses the balance point (~17 C) repeatedly.
2. Both anchors see internal gains: `calculate_rated_output_flow_temp` is the PREFERRED anchor
   (confidence 0.95), so wiring gains only into the design-point anchor is a no-op for installers
   who configured their emitters, and the two anchors of one law then disagree.
3. Internal gains are WATTS over the house's own W/K, not a fixed offset in degrees - the balance
   point is derived (INTERNAL_GAINS_W / heat_loss_coefficient), bounded, and follows the setpoint.
"""

from __future__ import annotations

import pytest

from custom_components.effektguard.const import (
    BALANCE_POINT_MAX_OFFSET,
    BALANCE_POINT_MIN_OFFSET,
    DEFAULT_DESIGN_SPREAD,
    DEFAULT_HEAT_LOSS_COEFFICIENT,
    INTERNAL_GAINS_W,
)
from custom_components.effektguard.optimization.weather_layer import (
    WeatherCompensationCalculator,
)

TARGET = 21.0


def test_the_flow_curve_has_no_step_at_the_balance_point():
    """Sweep the curve across its own discontinuity and demand that there isn't one.

    Below the balance point the emitters need no excess over the room. The naive `return
    indoor_setpoint` puts a step of spread/2 (2.5 C on the defaults) right at the balance point
    (~17-18 C), because the other side tends to `indoor_setpoint + spread/2` as load goes to zero.
    The shoulder season crosses that boundary repeatedly, so a step there is the pump chattering.
    """
    calc = WeatherCompensationCalculator(heat_loss_coefficient=DEFAULT_HEAT_LOSS_COEFFICIENT)
    balance = calc.balance_point_temp(TARGET)
    cliff_if_broken = DEFAULT_DESIGN_SPREAD / 2.0  # 2.5 C - what the naive `return setpoint` costs

    # Straddle the balance point finely enough that a step cannot hide between samples.
    outdoors = [balance - 1.0 + i * 0.01 for i in range(201)]
    flows = [calc.calculate_design_point_flow_temp(TARGET, t) for t in outdoors]

    steps = [abs(b - a) for a, b in zip(flows, flows[1:])]
    worst = max(steps)
    where = outdoors[steps.index(worst)]

    # The law is continuous but STEEP at zero load: dT ~ phi^(1/n) has an infinite derivative at
    # phi = 0, so the curve genuinely does move a few hundredths over the last 0.01 C. That is the
    # emitter law, not a defect. A missing spread term is a 2.5 C JUMP - fifty times larger.
    assert worst < cliff_if_broken / 10.0, (
        f"The flow curve jumps {worst:.2f} C between {where:.2f} C and {where + 0.01:.2f} C "
        f"outdoor - a cliff at the balance point ({balance:.2f} C). The shoulder season sits on "
        f"top of this boundary and the outdoor temperature crosses it repeatedly, so the pump "
        f"would be commanded up and down by {worst:.2f} C all day. Returning a bare "
        f"`indoor_setpoint` above the balance point costs exactly {cliff_if_broken:.1f} C here."
    )


def test_the_curve_is_flat_and_continuous_above_the_balance_point():
    """Above the balance point the house needs no heat, and the two sides must meet."""
    calc = WeatherCompensationCalculator(heat_loss_coefficient=DEFAULT_HEAT_LOSS_COEFFICIENT)
    balance = calc.balance_point_temp(TARGET)
    no_heat_needed = TARGET + DEFAULT_DESIGN_SPREAD / 2.0

    just_above = calc.calculate_design_point_flow_temp(TARGET, balance + 0.5)
    far_above = calc.calculate_design_point_flow_temp(TARGET, balance + 10.0)

    assert just_above == pytest.approx(no_heat_needed, abs=0.01)
    assert far_above == pytest.approx(no_heat_needed, abs=0.01)

    # Approach the boundary from below. The excess over the room must tend to zero, so the two
    # sides meet - that is what makes the curve continuous rather than merely close.
    from_below = calc.calculate_design_point_flow_temp(TARGET, balance - 1e-9)
    assert from_below == pytest.approx(no_heat_needed, abs=0.01), (
        f"Approaching the balance point from below, the curve converges on {from_below:.3f} C but "
        f"holds {no_heat_needed:.3f} C above it. The two sides do not meet: there is a step of "
        f"{abs(from_below - no_heat_needed):.2f} C at the balance point."
    )


def test_the_preferred_anchor_is_not_left_out_of_the_gains_fix():
    """`calculate_rated_output_flow_temp` is chosen at confidence 0.95. It must see the gains too.

    The layer prefers the rated-output anchor whenever an installer supplies their emitters'
    nameplate figure, so gains wired into the design-point anchor only would do nothing for them.
    A gains-aware curve stops needing heat at the balance point, so at an outdoor temperature
    above the balance point but below the setpoint it is already flat while a gains-blind one
    still asks for heat.
    """
    calc = WeatherCompensationCalculator(
        heat_loss_coefficient=DEFAULT_HEAT_LOSS_COEFFICIENT,
        radiator_rated_output=9000.0,
    )
    balance = calc.balance_point_temp(TARGET)
    assert balance < TARGET - 1.0, "precondition: gains must move the balance point at all"

    # Between the balance point and the setpoint: no heat is needed, and both anchors must say so.
    outdoor = (balance + TARGET) / 2.0
    rated = calc.calculate_rated_output_flow_temp(TARGET, outdoor, DEFAULT_DESIGN_SPREAD)
    flat = TARGET + DEFAULT_DESIGN_SPREAD / 2.0

    assert rated == pytest.approx(flat, abs=0.01), (
        f"At {outdoor:.1f} C outdoor - above the {balance:.1f} C balance point - the house is "
        f"heating itself, yet the PREFERRED anchor still asks for {rated:.1f} C of flow. It is "
        f"computing its load as (setpoint - outdoor) and has never been told about internal gains. "
        f"Every installer who filled in their emitters' rated output gets this path."
    )


def test_both_anchors_agree_when_the_house_is_described_consistently():
    """One law, two anchors - so given a self-consistent house they must give the SAME curve.

    The five inputs (heat loss, design flow, design outdoor, spread, rated output) are
    over-determined: any four fix the fifth, but nothing enforces consistency and the layer
    silently prefers the rated-output anchor. This pins the invariant: when the inputs agree,
    the anchors agree exactly.
    """
    room, dot, spread, hlc = TARGET, -15.0, DEFAULT_DESIGN_SPREAD, DEFAULT_HEAT_LOSS_COEFFICIENT
    design_flow = 50.0

    probe = WeatherCompensationCalculator(heat_loss_coefficient=hlc)
    balance = probe.balance_point_temp(room)

    # The rated output this house's design point implies, by the same EN 442 law.
    design_load_w = hlc * (balance - dot)
    mean_dt = design_flow - spread / 2.0 - room
    consistent_rated = design_load_w / ((mean_dt / 50.0) ** 1.3)

    calc = WeatherCompensationCalculator(
        heat_loss_coefficient=hlc,
        radiator_rated_output=consistent_rated,
        design_outdoor_temp=dot,
        design_flow_temp=design_flow,
        design_spread=spread,
    )

    for outdoor in (-20.0, -15.0, -5.0, 0.0, 5.0, 10.0, 15.0):
        by_design = calc.calculate_design_point_flow_temp(room, outdoor)
        by_rating = calc.calculate_rated_output_flow_temp(room, outdoor, spread)
        assert by_rating == pytest.approx(by_design, abs=0.05), (
            f"At {outdoor:+.1f} C the two anchors of the same law disagree: design point says "
            f"{by_design:.2f} C, rated output says {by_rating:.2f} C. They were given a house whose "
            f"description is self-consistent, so they must produce the same curve."
        )


class TestGainsAreWattsNotDegrees:
    """The balance point must be DERIVED from the house, not stamped on as a constant."""

    def test_an_insulated_house_gets_more_degrees_from_the_same_free_heat(self):
        """600 W of bodies and appliances is worth more degrees in a house that loses heat slowly.

        This is the whole reason the constant is watts. A fixed offset in degrees would hand a
        draughty 300 W/K house the same 4 K of free heat as a 100 W/K passive house - crediting the
        leaky one with three times the internal gains it actually has.
        """
        leaky = WeatherCompensationCalculator(heat_loss_coefficient=300.0)
        typical = WeatherCompensationCalculator(heat_loss_coefficient=180.0)
        tight = WeatherCompensationCalculator(heat_loss_coefficient=100.0)

        leaky_offset = TARGET - leaky.balance_point_temp(TARGET)
        typical_offset = TARGET - typical.balance_point_temp(TARGET)
        tight_offset = TARGET - tight.balance_point_temp(TARGET)

        assert leaky_offset < typical_offset < tight_offset, (
            f"The balance-point offset must shrink as a house gets leakier: got {leaky_offset:.2f} "
            f"K at 300 W/K, {typical_offset:.2f} K at 180 W/K, {tight_offset:.2f} K at 100 W/K. If "
            f"these are equal, the gains have been re-frozen into a constant number of degrees and "
            f"the same fridge is heating a draughty house as much as a sealed one."
        )

    def test_the_offset_is_the_gains_divided_by_the_heat_loss(self):
        """Not approximately. Exactly - it is a definition, not a tuning."""
        for hlc in (120.0, 180.0, 250.0):
            calc = WeatherCompensationCalculator(heat_loss_coefficient=hlc)
            expected = TARGET - INTERNAL_GAINS_W / hlc
            assert calc.balance_point_temp(TARGET) == pytest.approx(expected, abs=0.001)

    def test_the_balance_point_follows_the_setpoint_the_owner_chose(self):
        """A 19 C house balances 2 C lower than a 21 C house. The gains do not change."""
        calc = WeatherCompensationCalculator(heat_loss_coefficient=DEFAULT_HEAT_LOSS_COEFFICIENT)
        assert calc.balance_point_temp(19.0) == pytest.approx(calc.balance_point_temp(21.0) - 2.0)

    def test_an_absurd_heat_loss_cannot_switch_the_heating_off(self):
        """A mis-typed 20 W/K would put the balance point 30 K below the setpoint.

        That is a house that never asks for heat. The bound is not cosmetic: `heat_loss_coefficient`
        is not validated anywhere in the config flow today, so it is exactly the kind of number that
        arrives wrong.
        """
        absurdly_tight = WeatherCompensationCalculator(heat_loss_coefficient=20.0)
        absurdly_leaky = WeatherCompensationCalculator(heat_loss_coefficient=5000.0)

        tight_offset = TARGET - absurdly_tight.balance_point_temp(TARGET)
        leaky_offset = TARGET - absurdly_leaky.balance_point_temp(TARGET)

        assert tight_offset == pytest.approx(BALANCE_POINT_MAX_OFFSET), (
            f"A 20 W/K heat loss puts the balance point {tight_offset:.1f} K below the setpoint. "
            f"The house would stop asking for heat at {TARGET - tight_offset:.1f} C outdoor."
        )
        assert leaky_offset == pytest.approx(BALANCE_POINT_MIN_OFFSET)

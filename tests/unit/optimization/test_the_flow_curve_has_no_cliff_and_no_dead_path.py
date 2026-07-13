"""Three fixes to the emitter law that no test could see.

Each of these was a real defect, each was fixed, and each mutation-survived a 793-test suite
afterwards - which means the fix was worth nothing: the next refactor would have silently undone it
and everything would still have been green.

  1. A 2.5 C STEP in the flow curve at the balance point.
  2. The balance point never reaching `calculate_rated_output_flow_temp` - the anchor the layer
     PREFERS (confidence 0.95). The gains fix was a no-op for exactly the installers who had
     configured their emitters properly.
  3. Internal gains as a fixed offset in DEGREES rather than watts over the house's own W/K, which
     credits a leaky house with more free heat than an insulated one from the same fridge.

They are grouped here because they share a cause. The balance point was introduced as a constant
fitted to a heating curve, and a fitted constant has no physical anchor to reason from - so nobody
asked what it did at its own boundary, whether it reached both call sites, or what it was a
proportion OF. Deriving it from watts answers all three questions at once.
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

    Below the balance point the house heats itself and the emitters need no excess over the room.
    The naive way to express that is `return indoor_setpoint` - and it puts a step of spread/2
    (2.5 C on the defaults) right at the balance point, because the expression on the other side
    tends to `indoor_setpoint + spread/2` as the load goes to zero, not to `indoor_setpoint`.

    The balance point is around 17 C. Swedish autumn crosses 17 C back and forth all day. A step
    there is not a rounding error, it is a control system chattering 2.5 C on a heat pump.
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

    The layer prefers the rated-output anchor whenever an installer has supplied their emitters'
    nameplate figure. When internal gains were added, they were wired into the design-point anchor
    only - so the fix did nothing at all for those users, and the two anchors of what the code
    calls "the same law" disagreed by up to 3.5 C.

    A curve that ignores internal gains asks for heat right up to room temperature. One that models
    them stops needing heat at the balance point. So: at an outdoor temperature ABOVE the balance
    point but BELOW the setpoint, the two are unmistakably different - the gains-aware curve is
    already flat.
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
    over-determined: any four fix the fifth. Nothing in the config flow enforces that, and the
    layer silently prefers the rated-output anchor - so an inconsistent set does not raise, it just
    quietly runs the pump on a different curve. This pins the invariant that makes such a check
    meaningful: when the inputs DO agree, the anchors agree exactly.
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

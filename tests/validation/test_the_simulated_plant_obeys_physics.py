"""The simulator is the instrument. An instrument that flatters the thing it measures is worse
than no instrument, because it produces numbers people quote.

Every simulation claim on this branch rests on `scripts/simulation/sim_harness.py`, and the harness
had no test of its own. It shipped three defects that this file now pins, all of which I introduced
or kept, and all of which it reported as PASS.

1. THE ENERGY "AUDITS" WERE ALGEBRAIC IDENTITIES.

       power_kw = q_comp/cop + aux + standby        (the plant)
       metered  = power_kw - aux - standby          (the "meter")
       owed     = q_comp/cop                        (the "independent" figure)

   Substitute the first into the second and you get the third: x - y + y = x. I called these "two
   different expressions of the same joules" in the code and in a test docstring. Doubling the
   compressor's COP - which halves the electricity bill, a catastrophic plant bug - left the audit
   reporting 0.00 % error and PASS on all five houses. The room-side "first law residual" is the
   same trick with the room ODE and I had already caught that one, then rebuilt it.

   There is no exact energy audit to be had inside a closed ODE plant. Every residual you can write
   is a rearrangement of the equations that produced it. What CAN fail is a statement about
   something the bookkeeping does not determine - a physical bound, or a leak - and those are what
   the harness asserts now, and what this file checks it still asserts.

2. THE PLANT DESTROYED ENERGY IT HAD CHARGED FOR. The water node's temperature was force-clamped to
   the pump's maximum AFTER the ODE integrated it, so joules vanished with no residual noticing:
   183 kWh in the F2040 cold snap, while every "audit" above read 0.00 %. The immersion heater was
   pouring 3 kW into a node already at its ceiling - 2.6 K of overshoot per five-minute step - and
   the clamp deleted it. Real immersion heaters have thermostats.

3. THE PLANT INTEGRATED DEGREE MINUTES AGAINST A SETPOINT THE PUMP WAS FORBIDDEN TO REACH. `flow`
   was clamped to max_flow_temp; `flow_target` was not. DM is the integral of (flow - flow_target),
   so in the F2040 cold snap DM fell at up to 4.1 per minute NO MATTER WHAT ANY CONTROLLER DID, ran
   to the integrator floor on its own, and the harness recorded 1134 `dm_runaway` violations and
   blamed the recovery ladder. A NIBE limits its calculated supply temperature to the configured
   maximum; it does not chase water it cannot make.

   This one matters beyond the harness: it inflated the evidence for F-124 by about three times.
   See test_a_saturated_compressor_is_a_positive_feedback_trap, where the honest numbers now live.
"""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import pathlib

import pytest

from custom_components.effektguard.const import MAX_OFFSET

_SPEC = importlib.util.spec_from_file_location(
    "sim_harness", pathlib.Path("scripts/simulation/sim_harness.py")
)
sim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sim)


@pytest.fixture(params=[h.name for h in sim.HOUSES])
def house(request):
    return next(h for h in sim.HOUSES if h.name == request.param)


@functools.lru_cache(maxsize=1)
def _weather_and_prices():
    """The harness's own two-day self-test data. Enough to exercise the plant loop, and fast.

    A plain cache rather than a module-scoped fixture: pytest-homeassistant-custom-component
    installs an autouse function-scoped event loop, and a module-scoped fixture in the same file
    drags every test in it into a scope mismatch.
    """
    times, temps, price_days, unit = sim.load_data(selftest=True)
    return times, temps, sim.PriceSource(price_days, unit)


_SATURATING_HOUSE = "airsource_f2040"


@functools.lru_cache(maxsize=1)
def _the_only_run_that_reaches_the_immersion_heater() -> dict:
    """A full cold-snap month on the F2040, cached: the only scenario that saturates a pump.

    It is an outdoor-air machine, so it is the only one whose capacity collapses as the weather
    does. The other four sail through a Swedish January without ever touching resistive heat, which
    means a leak test run on them proves nothing about a plant that mishandles the heater - and the
    first version of that test was run on exactly those, and passed on a broken plant.
    """
    times, temps, price_days, unit = sim.load_data(selftest=False)
    house = next(h for h in sim.HOUSES if h.name == _SATURATING_HOUSE)
    try:
        stats, _violations, _trace = sim.simulate(
            house,
            times,
            sim.apply_coldsnap(times, temps),
            sim.PriceSource(price_days, unit),
            days=sim.SIM_DAYS,
        )
    finally:
        asyncio.set_event_loop(asyncio.new_event_loop())
    return stats


def _short_run(house, coldsnap: bool = False) -> dict:
    """Drive the real plant for two days.

    The harness is a script: it drives the async engine with `asyncio.run`, which closes the loop
    and leaves the thread without one. pytest-homeassistant-custom-component has an autouse fixture
    that calls `asyncio.get_event_loop()`, so without putting a loop back every LATER test in the
    run errors out in setup. Hand it a fresh one.
    """
    times, temps, prices = _weather_and_prices()
    if coldsnap:
        temps = sim.apply_coldsnap(times, temps)
    try:
        stats, _violations, _trace = sim.simulate(house, times, temps, prices, days=2)
    finally:
        asyncio.set_event_loop(asyncio.new_event_loop())
    return stats


class TestTheCopModelIsBoundedByPhysicsAndByTheDatasheet:
    """The two statements about efficiency that are NOT rearrangements of the plant's own books."""

    @pytest.mark.parametrize("outdoor", [-30.0, -20.0, -10.0, 0.0, 7.0, 15.0])
    @pytest.mark.parametrize("flow", [25.0, 35.0, 45.0, 55.0, 65.0])
    def test_no_pump_beats_carnot(self, house, outdoor, flow):
        """The second law. An external bound, so it can disagree with the model - and it must not."""
        cop = house.cop_at(outdoor, flow)
        ceiling = house.carnot_cop(outdoor, flow)

        assert cop <= ceiling, (
            f"{house.name} at {outdoor:+.0f} C outdoor making {flow:.0f} C water has COP {cop:.2f}, "
            f"above the Carnot limit of {ceiling:.2f} between those temperatures. No machine can do "
            f"this, so the plant is inventing energy and every cost it reports is fiction."
        )

    def test_hotter_water_costs_efficiency(self, house):
        """The mechanism weather compensation exists to exploit. A flow-blind COP cannot see it."""
        assert house.cop_at(-5.0, 55.0) < house.cop_at(-5.0, 35.0), (
            f"{house.name} makes 55 C water as efficiently as 35 C water. Running cooler water IS "
            f"how weather compensation saves money - with a flow-blind COP the optimiser can only "
            f"ever look like a loss, and it duly did."
        )

    def test_the_datasheet_check_lives_where_the_datasheet_does(self):
        """Two tests used to live here, and BOTH rested on a COP model that was invented.

        They compared the plant against `profile.get_cop_at_temperature(outdoor)` - an outdoor-keyed
        curve which, for four of the five machines, described a heat source that does not exist. The
        F750's said COP 5.0 at +7 C outdoor. NIBE's datasheet has no such figure, and the outdoor
        air never touches that machine's evaporator: its rating points are A20(12), twenty-degree
        extract air from inside the house.

        They are replaced by tests/validation/test_the_pump_models_match_their_datasheets.py, which
        checks something strictly stronger, against real data: the model reproduces every published
        EN 14511 rating point to within 2 %, and PREDICTS the F2040's W45 rows - which the fit never
        saw - to within 8 %.

        What stays in this file is the part that is a property of the PLANT rather than of the pump:
        the second law, and the fact that hotter water costs efficiency.
        """
        source = pathlib.Path(
            "tests/validation/test_the_pump_models_match_their_datasheets.py"
        ).read_text(encoding="utf-8")

        assert "def test_it_reproduces_every_point_it_was_fitted_on" in source, (
            "the datasheet reproduction test is gone, and this file no longer checks the COP model "
            "against anything the manufacturer published"
        )
        assert "def test_it_predicts_the_points_it_never_saw" in source, (
            "the held-out prediction test is gone. Reproducing a fit is not evidence; predicting "
            "data the fit never saw is."
        )


class TestThePlantDoesNotDestroyEnergyItChargedFor:
    """The clamp overwrites a state variable after the ODE integrated it. Nothing else can leak.

    THE FIRST VERSION OF THIS CLASS COULD NOT FAIL, and I only found that by mutating the plant
    underneath it. Two ways, both worth naming, because they are the same two mistakes this whole
    branch keeps making:

      * the leak test ran on two days of MILD self-test weather, where no pump ever reaches its
        immersion heater. Nothing ran, so nothing leaked, so it passed - on a plant with the
        thermostat torn out. A test needs a PRECONDITION proving the mechanism it guards actually
        engaged, and it now has one.
      * the thermostat test recomputed the headroom formula inside the test and asserted the result
        equalled itself. Pure tautology. It now reads the real plant's output.
    """

    def test_the_pump_that_actually_reaches_its_immersion_heater_leaks_nothing(self):
        """The F2040 in a deep cold snap: the ONE case that pins the water node at its ceiling.

        This is where 183 kWh went missing while every energy audit in the harness read 0.00 %. It
        is an outdoor-air pump, so it is the only one whose capacity collapses with the weather,
        the only one that saturates, and the only one that falls back on resistive heat.
        """
        stats = _the_only_run_that_reaches_the_immersion_heater()

        assert stats["aux_kwh"] > 0, (
            "PRECONDITION FAILED, and this is the important half: if the immersion heater never "
            "ran, this test proves nothing about a plant that mishandles it. The first version of "
            "this test had no such check, ran on mild weather, and passed happily against a plant "
            "with the heater's thermostat removed."
        )
        assert abs(stats["water_node_leak_kwh"]) <= sim.WATER_NODE_LEAK_BUDGET_KWH, (
            f"The F2040 burned {stats['aux_kwh']:.1f} kWh of immersion heat and the flow clamp "
            f"destroyed {abs(stats['water_node_leak_kwh']):.1f} kWh of it: energy the meter charged "
            f"for and the room never received. No energy residual in this harness can see that, "
            f"because they are all rearrangements of the ODE that runs BEFORE the clamp - which is "
            f"exactly why they all read 0.00 % while 183 kWh went missing."
        )

    @pytest.mark.parametrize("coldsnap", [False, True], ids=["mild", "coldsnap"])
    def test_no_house_leaks_in_ordinary_operation(self, house, coldsnap):
        """The broad regression guard, across every pump. Cheap, and it covers the compressor side.

        It is NOT the test above: none of these runs reaches the immersion heater, which is why
        that one exists and says so.
        """
        stats = _short_run(house, coldsnap)

        assert abs(stats["water_node_leak_kwh"]) <= sim.WATER_NODE_LEAK_BUDGET_KWH, (
            f"{house.name} destroyed {abs(stats['water_node_leak_kwh']):.1f} kWh in the flow clamp "
            f"without even reaching its immersion heater."
        )


class TestThePumpIsNeverAskedForWaterItCannotMake:
    """The artifact that inflated F-124 by about three times.

    THE FIRST VERSION OF THIS TEST WAS A TAUTOLOGY. It computed `capped = min(uncapped, max_flow)`
    in the test body and then asserted `capped <= max_flow`. It never touched the plant, so
    unclamping the plant's own S1 - the actual bug - left it green. `min(x, m) <= m` is true of
    arithmetic, not of this codebase.

    It now reads `flow_target_max` off a real run: what the plant ACTUALLY asked the pump for.
    """

    def test_the_saturated_pump_is_never_asked_for_water_above_its_maximum(self):
        """The F2040 in a cold snap, where the curve plus a +10 emergency offset overshoots.

        Degree minutes are the integral of (BT25 - S1). `flow` was clamped to max_flow_temp and
        `flow_target` was not, so the plant integrated against a setpoint the pump was physically
        forbidden to reach: DM fell at up to 4.1 per minute regardless of what the controller did,
        hit the integrator floor unaided, and the harness called it a control failure 1134 times.
        """
        stats = _the_only_run_that_reaches_the_immersion_heater()
        house = next(h for h in sim.HOUSES if h.name == _SATURATING_HOUSE)
        max_flow = float(house.profile.max_flow_temp)

        assert stats["offset_max"] >= MAX_OFFSET, (
            "PRECONDITION: this only bites when the emergency tier commands its maximum offset on "
            "top of an already-steep curve. If the ladder never latched, the overshoot never "
            "happened and this test is not exercising anything."
        )
        assert stats["flow_target_max"] <= max_flow + 1e-6, (
            f"The plant asked the pump for {stats['flow_target_max']:.1f} C water, "
            f"{stats['flow_target_max'] - max_flow:.1f} C above the {max_flow:.0f} C maximum it is "
            f"allowed to make. Degree minutes integrate (BT25 - S1) and BT25 is capped, so DM then "
            f"falls at {stats['flow_target_max'] - max_flow:.1f} per minute FOREVER - no controller "
            f"can escape it, the integrator floors on its own, and the harness blames the recovery "
            f"ladder for a defect in the plant."
        )

    def test_degree_minutes_no_longer_run_away_on_their_own(self):
        """The consequence. `dm_runaway` was 1134 samples of plant artifact, and it is now zero.

        The trap underneath it is REAL and still fails the run - the house is still cooked, the
        immersion heater still burns. But it fails for the reason it actually fails for, at its
        actual size. See test_a_saturated_compressor_is_a_positive_feedback_trap.
        """
        stats = _the_only_run_that_reaches_the_immersion_heater()

        assert stats["dm_min"] > sim.DM_INTEGRATOR_FLOOR, (
            f"Degree minutes reached the integrator floor ({sim.DM_INTEGRATOR_FLOOR:.0f}). An "
            f"integrator that saturates has stopped measuring anything, and it got there because "
            f"the plant was chasing water the pump could not make."
        )


class TestTheHarnessCannotGoBackToBeingUnfalsifiable:
    """A guard on the guards. Every one of these was, at some point, a number nobody asserted."""

    def test_the_identity_audits_are_gone_and_stay_gone(self):
        """They reported 0.00 % error on a plant that had doubled its own COP."""
        source = pathlib.Path("scripts/simulation/sim_harness.py").read_text(encoding="utf-8")

        for banned in ("compressor_elec_metered_kwh", "compressor_elec_owed_kwh"):
            assert banned not in source, (
                f"`{banned}` is back. It is one half of `metered = power - aux - standby` against "
                f"`owed = q/cop`, where power was DEFINED as q/cop + aux + standby - an identity "
                f"dressed up as an audit. It cannot fail, so it cannot detect, and it spent several "
                f"commits being quoted as evidence that the plant was sound."
            )

    def test_the_checks_that_can_fail_are_all_asserted(self):
        """Counted-and-never-asserted is how this harness failed the first three times."""
        source = pathlib.Path("scripts/simulation/sim_harness.py").read_text(encoding="utf-8")
        checked = source.split("def check_invariants")[1]

        for metric in ("water_node_leak_kwh", "datasheet_cop", "aux_kwh", "comfort_minutes_above"):
            assert metric in checked, (
                f"`{metric}` is computed by the harness and never asserted in check_invariants. "
                f"A number that is tracked and ignored is decoration: aux_kwh and the comfort "
                f"minutes were both tracked and ignored while the optimiser cooked a house to "
                f"35 C and burned 266 kWh of resistive heat, and every run still printed PASS."
            )

    def test_carnot_is_asserted_during_the_run_not_merely_available(self):
        source = pathlib.Path("scripts/simulation/sim_harness.py").read_text(encoding="utf-8")

        assert "cop_beats_carnot" in source and "cop_beats_carnot" in str(
            sim.FATAL_VIOLATIONS
        ), "the Carnot bound must be a FATAL violation raised per step, not a helper nobody calls"

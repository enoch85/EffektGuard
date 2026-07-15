"""The heat-pump models must come from the datasheets, not from an invented curve.

Every profile in `models/nibe/` once carried an outdoor-keyed `cop_curve` whose docstring called it
"Real-world COP curve (tested and validated)" and sourced it to "NIBE F750 datasheet, Swedish NIBE
forum validation". It was neither: the F750 and F730 shipped BYTE-IDENTICAL curves despite different
published outputs, and the F750's said COP 5.0 at +7 C outdoor - a figure in no NIBE document, at a
condition an EXHAUST-AIR pump is never rated at (its points are A20(12), 20 C extract air; outdoor
air never touches its evaporator).

What the datasheets say, and what this file checks the model against:

    NIBE F750, "Output data according to EN 14 511", part no. 066 063:
        4.994 kW / COP 2.43   A20(12)W45, 252 m3/h, MAX compressor frequency
    F2040 (air source): capacity RISES as it cools - 3.86 -> 6.60 kW from +7 to -7 C - because an
        inverter throttles back at its mild rating point; the COP falls instead, 4.65 -> 2.68 at W35.

The old curve gave the F750 an 8 kW compressor (it makes 4.994), derated the F2040 the wrong way,
and dropped a ground-source F1155's COP because the outdoor AIR got cold - though its heat source is
0 C borehole brine. Each profile now carries its EN 14511 rating points VERBATIM, and the
simulator's COP is

    COP = eta_exergy(load, flow) x Carnot(source, flow)

with eta fitted to each machine's own published points - a claim that CAN be falsified, which the
curve it replaced could not be, and this file falsifies it or fails.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from custom_components.effektguard.models.nibe import (
    NibeF730Profile,
    NibeF750Profile,
    NibeF1155Profile,
    NibeF2040Profile,
    NibeS1155Profile,
)

_SPEC = importlib.util.spec_from_file_location(
    "sim_harness", pathlib.Path("scripts/simulation/sim_harness.py")
)
sim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sim)

PROFILES = [
    NibeF750Profile,
    NibeF730Profile,
    NibeF1155Profile,
    NibeS1155Profile,
    NibeF2040Profile,
]

# The model must reproduce every point it was fitted on to within this. Measured: 0.82 % worst.
DATASHEET_TOLERANCE_PCT = 2.0

# And it must predict points it was NEVER fitted on to within this. Measured: 5.7 % worst, on the
# F2040's W45 rows when the fit only ever saw W35. That is the number that makes this a model
# rather than a curve-fit, and it is why the tolerance here is looser and still meaningful.
HELD_OUT_TOLERANCE_PCT = 8.0


@pytest.fixture(params=PROFILES, ids=lambda p: p().model_name)
def profile(request):
    return request.param()


def _house_for(profile):
    """A HouseConfig wrapping this profile, so the real simulator physics is exercised."""
    return next(h for h in sim.HOUSES if h.profile.model_name == profile.model_name)


class TestEveryNumberHasASource:
    """No source, no number. That is the whole rule, and it was not being followed."""

    def test_the_profile_carries_its_datasheet(self, profile):
        assert profile.datasheet_points, (
            f"{profile.model_name} has no EN 14511 rating points. Every performance figure in this "
            f"package is now derived from the manufacturer's published measurements, because the "
            f"ones that were not turned out to be a template with the digits nudged."
        )
        assert profile.datasheet_source, (
            f"{profile.model_name} does not say where its numbers came from. The last time this "
            f"field said 'NIBE F750 datasheet, Swedish NIBE forum validation', the numbers were in "
            f"neither."
        )

    def test_every_rating_point_names_its_condition(self, profile):
        """`A20(12)W35, 252 m3/h, min compressor frequency` is the datasheet's own string.

        Without it a rating point is just four floats, and four floats are what got invented.
        """
        for point in profile.datasheet_points:
            assert len(point.condition) > 8 and any(
                c.isdigit() for c in point.condition
            ), f"{profile.model_name} has a rating point with no condition: {point.condition!r}"

    def test_the_two_exhaust_air_pumps_no_longer_share_one_curve(self):
        """The tell. Different machines, byte-identical COP curves, for a year."""
        f750, f730 = NibeF750Profile(), NibeF730Profile()

        assert f750.datasheet_points != f730.datasheet_points, (
            "The F750 and F730 carry identical performance data. They are different machines: "
            "NIBE publishes 4.994 kW / COP 2.43 for one and 5.35 kW / COP 2.43 for the other."
        )

    def test_the_f1155_is_not_set_slightly_below_the_s1155(self):
        """Its docstring said it was. The datasheets say they are the same machine."""
        assert NibeF1155Profile().datasheet_points == NibeS1155Profile().datasheet_points, (
            "The F1155 and S1155 publish IDENTICAL EN 14511 data at every size. The old profile "
            "'set' the F1155's COP curve slightly below the S1155's - which was not merely "
            "unsourced, it was wrong."
        )


class TestTheModelReproducesTheDatasheet:
    """The claim that can be falsified. It is the difference between a model and a decoration."""

    def test_it_reproduces_every_point_it_was_fitted_on(self, profile):
        house = _house_for(profile)
        rated_airflow = max(
            (p.airflow_m3h for p in profile.datasheet_points if p.airflow_m3h), default=None
        )

        for point in profile.datasheet_points:
            if rated_airflow is not None and point.airflow_m3h != rated_airflow:
                continue  # a different source condition - see exergy_fit

            load = point.heat_output_kw / profile.max_heat_output_kw
            eta = house.exergy_efficiency(load, point.flow_temp_c)
            modelled = eta * house.carnot_at(point.source_temp_c, point.flow_temp_c)
            error = abs(modelled - point.cop) / point.cop * 100

            assert error < DATASHEET_TOLERANCE_PCT, (
                f"{profile.model_name} at '{point.condition}': NIBE measured COP {point.cop:.2f}, "
                f"the model says {modelled:.2f} ({error:.1f}% out). The model exists to reproduce "
                f"this machine's own published measurements; if it cannot, it is not a model of "
                f"this machine."
            )

    def test_it_predicts_the_points_it_never_saw(self):
        """THE REAL TEST. Fit the F2040 on its W35 rows only, then predict its W45 rows.

        The F2040 is the only machine whose datasheet is rich enough to hold points back: five
        rating points, three at 35 C flow and two at 45 C. A curve can be drawn through anything.
        A model has to work on data it has not seen.
        """
        f2040 = NibeF2040Profile()
        house = _house_for(f2040)

        held_out = [p for p in f2040.datasheet_points if p.flow_temp_c == 45.0]
        assert len(held_out) == 2, "precondition: the F2040 must publish W45 rows to hold back"

        for point in held_out:
            load = point.heat_output_kw / f2040.max_heat_output_kw
            eta = house.exergy_efficiency(load, point.flow_temp_c)
            modelled = eta * house.carnot_at(point.source_temp_c, point.flow_temp_c)
            error = abs(modelled - point.cop) / point.cop * 100

            assert error < HELD_OUT_TOLERANCE_PCT, (
                f"F2040 at '{point.condition}': NIBE measured COP {point.cop:.2f}, the model "
                f"predicts {modelled:.2f} ({error:.1f}% out) from a fit that only ever saw 35 C "
                f"flow temperatures. Predicting held-out data is the only thing that separates "
                f"this from the invented curve it replaced."
            )


class TestThePhysicsIsTheRightWayUp:
    """A sign error here is invisible to the Carnot guard, so it is pinned directly."""

    def test_efficiency_falls_as_the_compressor_is_pushed(self, profile):
        """An inverter gets LESS efficient the harder it runs.

        A fit with efficiency RISING with load extrapolates to COP 9.86 at full load and 35 C flow,
        under Carnot's ceiling of 12.5 there - so the second-law guard cannot catch it. (The trap:
        the F750's two minimum-frequency points differ by AIRFLOW, 108 vs 252 m3/h, not by load;
        treating them as a load pair turns the physics upside down.)
        """
        _, load_slope, _ = _house_for(profile).exergy_fit

        assert load_slope < 0, (
            f"{profile.model_name}'s exergy efficiency RISES with compressor load "
            f"(slope {load_slope:+.3f}). A heat pump does not get more efficient by working "
            f"harder. This is the sign error that produced COP 9.86, and the Carnot guard cannot "
            f"catch it."
        )

    def test_hotter_water_costs_efficiency_beyond_carnot(self, profile):
        """A real machine loses MORE than Carnot predicts when you raise the flow temperature."""
        _, _, flow_slope = _house_for(profile).exergy_fit

        assert flow_slope < 0, (
            f"{profile.model_name}'s exergy efficiency RISES with flow temperature "
            f"(slope {flow_slope:+.4f}). Running hotter water is not free, and running COOLER "
            f"water is the entire mechanism by which weather compensation saves money."
        )

    def test_no_machine_beats_carnot_anywhere_the_simulator_goes(self, profile):
        house = _house_for(profile)

        for outdoor in (-20.0, -10.0, 0.0, 10.0):
            for flow in (25.0, 35.0, 45.0, 55.0):
                for load in (0.1, 0.5, 1.0):
                    cop = house.cop_at(outdoor, flow, load)
                    ceiling = house.carnot_cop(outdoor, flow)
                    assert cop <= ceiling, (
                        f"{profile.model_name} at {outdoor:+.0f} C, {flow:.0f} C flow, "
                        f"{load:.0%} load: COP {cop:.2f} beats the Carnot limit {ceiling:.2f}."
                    )


class TestTheHeatSourceIsNotTheWeather:
    """Four of these five machines do not know what the weather is doing, and now nor does the model."""

    @pytest.mark.parametrize("model", ["F750", "F730", "F1155", "S1155"])
    def test_a_pump_that_does_not_breathe_outdoor_air_has_a_flat_cop(self, model):
        """The one that mattered most. An F1155's COP fell from 5.3 to 3.3 because of the WEATHER.

        Its heat source is brine from a borehole. NIBE's capacity chart plots its output against
        "Incoming brine temp, C" and there is no air-temperature rating point in its datasheet at
        all. An exhaust-air pump breathes 20 C house air. Neither cares about the sky.
        """
        house = next(h for h in sim.HOUSES if h.profile.model_name == model)

        warm = house.cop_at(7.0, 40.0, 0.6)
        freezing = house.cop_at(-20.0, 40.0, 0.6)

        assert warm == pytest.approx(freezing), (
            f"{model}'s COP moves from {warm:.2f} to {freezing:.2f} when the outdoor air goes from "
            f"+7 C to -20 C, at the same flow temperature and the same load. Its heat source did "
            f"not move. The old curve did exactly this, and the simulator priced a month of "
            f"electricity with it."
        )

    def test_the_air_source_pump_is_the_only_one_that_does_care(self):
        """And for the F2040 it is real, measured, and in the datasheet: COP 4.65 -> 2.68."""
        house = next(h for h in sim.HOUSES if h.profile.model_name == "F2040")

        assert house.cop_at(7.0, 35.0, 0.6) > house.cop_at(-7.0, 35.0, 0.6) * 1.2, (
            "The F2040's source IS the outdoor air. Its COP must fall with the weather - NIBE "
            "publishes 4.65 at 7/35 and 2.68 at -7/35 - and it is the ONLY machine here for which "
            "an outdoor-keyed curve was ever meaningful."
        )


class TestCapacityComesFromTheDatasheetToo:
    """The 8 kW compressor that does not exist."""

    def test_no_machine_can_make_more_than_it_is_published_to_make(self, profile):
        house = _house_for(profile)

        for outdoor in (-20.0, -10.0, 0.0, 7.0, 15.0):
            capacity = house.capacity_kw_at(outdoor)
            assert capacity <= profile.max_heat_output_kw + 1e-9, (
                f"{profile.model_name} is modelled as making {capacity:.2f} kW at {outdoor:+.0f} C, "
                f"above its published maximum of {profile.max_heat_output_kw:.2f} kW. The F750 was "
                f"given 8.0 kW against a published 4.994, and it is the reason no exhaust-air pump "
                f"has ever saturated in this simulator."
            )

    def test_the_exhaust_air_pumps_are_bounded_by_the_air_they_breathe(self):
        """~5 kW, and it does not depend on the weather. It depends on the ventilation rate."""
        for model, published in (("F750", 4.994), ("F730", 5.35)):
            house = next(h for h in sim.HOUSES if h.profile.model_name == model)

            assert house.capacity_kw_at(-20.0) == pytest.approx(published), (
                f"{model} must make {published} kW whatever the weather - its evaporator is fed by "
                f"the house's own ventilation air at 20 C, and its output is set by the airflow."
            )

    def test_the_air_source_pumps_capacity_rises_as_it_gets_colder(self):
        """It does not derate. It ramps up - an inverter throttled back at its mild rating point."""
        house = next(h for h in sim.HOUSES if h.profile.model_name == "F2040")

        mild, cold = house.capacity_kw_at(7.0), house.capacity_kw_at(-7.0)

        assert cold > mild, (
            f"The F2040 is modelled as making {cold:.2f} kW at -7 C and {mild:.2f} kW at +7 C. "
            f"NIBE publishes 6.60 and 3.86: an inverter is throttled back at its mild rating point "
            f"and ramps UP as the weather cools. The old model derated it 2.5 %/C and blamed the "
            f"EN 14511 rating points, which say the opposite."
        )


class TestTheImmersionHeaterIsAlsoFromTheDatasheet:
    """It was ONE invented number, applied to five machines, matching none of them.

    The simulator gave every house the same `AUX_STEP_KW = 3.0`. NIBE ships the F750 and F730 with a
    6.5 kW heater set to 3.5 kW at delivery, the F1155-12 and S1155-12 with a 7 kW heater in seven
    automatic steps, and the F2040 with NO HEATER AT ALL - it is an outdoor monobloc, and the
    electric addition belongs to the indoor module it is paired with.
    """

    def test_each_machine_carries_its_own_published_heater(self, profile):
        published = {
            "F750": 3.5,  # "6.5 (3.5) kW" - max 6.5, delivery setting 3.5
            "F730": 3.5,
            "F1155": 7.0,  # additional power 1/2/3/4/5/6/7 kW
            "S1155": 7.0,
            "F2040": 0.0,  # it has none
        }

        assert profile.immersion_heater_kw == published[profile.model_name], (
            f"{profile.model_name}'s immersion heater is "
            f"{profile.immersion_heater_kw} kW; its datasheet says "
            f"{published[profile.model_name]} kW. One invented constant used to stand for all five."
        )

    def test_the_f2040_has_no_immersion_heater_at_all(self):
        """Not "0 kW by default". The machine physically does not have one."""
        f2040 = NibeF2040Profile()

        assert f2040.immersion_heater_kw == 0.0 and not f2040.supports_aux_heating, (
            "The F2040 is an outdoor monobloc. Its technical-specifications table has no "
            "immersion-heater row. The profile used to claim 'True # Larger immersion heaters'."
        )

    def test_the_simulator_falls_back_only_for_the_machine_that_has_none(self):
        """And it names that fallback an ASSUMPTION, because it is one."""
        for house in sim.HOUSES:
            published = house.profile.immersion_heater_kw
            if published > 0:
                assert house.immersion_heater_kw == published, (
                    f"{house.name} is simulated with a {house.immersion_heater_kw} kW heater while "
                    f"its datasheet publishes {published} kW."
                )
            else:
                assert house.immersion_heater_kw == sim.ASSUMED_INDOOR_MODULE_HEATER_KW, (
                    "the F2040's backup heat is an assumption about the paired indoor module, and "
                    "the constant that supplies it must say so in its name"
                )

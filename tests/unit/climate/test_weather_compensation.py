"""Tests for the EN 442 emitter law used by weather compensation.

Replaces the tests for Andre Kuehne's formula, Timbones' method as a separate "method", and the
UFH flow-temperature "adjustment" - all three are gone (audit F-119 / F-121). What remains is one
law with two anchors, so these tests check the law, its anchors, and the properties any heating
curve must have.

Two of the old tests are preserved deliberately, because they encode real external references:

  * Timbones' published spreadsheet example (18 000 W of emitters, 260 W/K, 19 C target, 0 C
    outdoor -> ~40 C flow). The rated-output anchor reproduces it to 0.01 C. It is now a
    validation of the EN 442 law rather than of a separate method.

  * The HeatpumpMonitor SPF-4.0 target (flow = outdoor + 27 C). This one was ENSHRINING THE BUG:
    it asserted the model must return 24-35 C at 0 C outdoor for a 150 W/K house at 20 C, and the
    emitter law says such a house needs 39.3 C. "Outdoor + 27" is an efficiency ASPIRATION,
    achievable only if the emitters are large enough to deliver the load at that temperature. It
    is not a temperature the house can be held at by decree. Asserting it as a requirement is
    exactly the efficiency-over-adequacy error that made EffektGuard under-heat: a flow
    temperature below what the emitter law demands does not save energy, it just fails to heat
    the house. The test now asserts adequacy, and records the aspiration as the comment it is.
"""

import pytest

from custom_components.effektguard.const import (
    DEFAULT_DESIGN_FLOW_TEMP_RADIATOR,
    DEFAULT_DESIGN_FLOW_TEMP_UFH,
    DEFAULT_DESIGN_OUTDOOR_TEMP,
    RADIATOR_POWER_COEFFICIENT,
    UFH_POWER_COEFFICIENT,
    WEATHER_COMP_MAX_OFFSET,
)
from custom_components.effektguard.optimization.weather_layer import (
    FlowTempCalculation,
    WeatherCompensationCalculator,
)


class TestEmitterLawAnchors:
    """One law, two anchors: the emitters' rated output, or the system's design point."""

    def test_design_point_anchor_reproduces_the_design_point(self):
        """At the design outdoor temperature the law must return the design flow temperature.

        This is what "anchored" means, and it is what makes the correction near zero on a
        correctly tuned pump. Kuehne returned 31.7 C here, where the house needs 50 C.
        """
        calc = WeatherCompensationCalculator(heat_loss_coefficient=150.0, heating_type="radiator")

        result = calc.calculate_optimal_flow_temp(
            indoor_setpoint=22.0,
            outdoor_temp=DEFAULT_DESIGN_OUTDOOR_TEMP,
        )

        assert result.method == "en442_design_point"
        assert result.flow_temp == pytest.approx(DEFAULT_DESIGN_FLOW_TEMP_RADIATOR, abs=0.01)

    def test_rated_output_anchor_is_preferred_when_configured(self):
        """A measured nameplate figure beats an assumed design point, so it wins."""
        calc = WeatherCompensationCalculator(
            heat_loss_coefficient=260.0,
            radiator_rated_output=18000.0,
        )

        result = calc.calculate_optimal_flow_temp(indoor_setpoint=19.0, outdoor_temp=0.0)

        assert result.method == "en442_rated_output"
        assert result.raw_rated_output is not None
        assert result.raw_design_point is not None  # both computed, for diagnostics

    def test_design_point_anchor_used_when_rated_output_unknown(self):
        """Nothing in the config flow asks for rated output, so this is the default path."""
        calc = WeatherCompensationCalculator(
            heat_loss_coefficient=180.0,
            radiator_rated_output=None,
        )

        result = calc.calculate_optimal_flow_temp(indoor_setpoint=20.0, outdoor_temp=0.0)

        assert result.method == "en442_design_point"
        assert result.raw_rated_output is None

    def test_timbones_published_example(self):
        """External reference: Timbones' spreadsheet, 18 kW emitters, 260 W/K, 19 C, 0 C outdoor.

        Published result ~40 C. The EN 442 rated-output anchor gives 39.99 C.
        """
        calc = WeatherCompensationCalculator(
            heat_loss_coefficient=260.0,
            radiator_rated_output=18000.0,
        )

        result = calc.calculate_optimal_flow_temp(indoor_setpoint=19.0, outdoor_temp=0.0)

        assert result.flow_temp == pytest.approx(40.0, abs=0.5)


class TestHeatingCurveProperties:
    """Properties every heating curve must have, whatever the anchor."""

    def test_flow_temp_rises_as_it_gets_colder(self):
        """Colder outside means hotter water. Kuehne's curve rose only 0.22 C per -1 C."""
        calc = WeatherCompensationCalculator(heat_loss_coefficient=150.0, heating_type="radiator")

        walk = [15.0, 10.0, 5.0, 0.0, -5.0, -10.0, -15.0, -20.0]
        flows = [calc.calculate_optimal_flow_temp(22.0, t).flow_temp for t in walk]

        for warm, cold, flow_warm, flow_cold in zip(walk, walk[1:], flows, flows[1:]):
            assert flow_cold > flow_warm, (
                f"{cold:+.0f} C asks for {flow_cold:.1f} C but the warmer {warm:+.0f} C asks for "
                f"{flow_warm:.1f} C - the curve slopes the wrong way."
            )

    def test_curve_slope_is_physically_plausible(self):
        """The curve must be steep enough to track the building's load.

        This house needs (50 - 22) / (22 - -15) = 0.757 C of supply per C of outdoor. Kuehne's
        0.22 was 3.5x too flat, which is why its shortfall grew as it got colder.
        """
        calc = WeatherCompensationCalculator(heat_loss_coefficient=150.0, heating_type="radiator")

        warm = calc.calculate_optimal_flow_temp(22.0, 10.0).flow_temp
        cold = calc.calculate_optimal_flow_temp(22.0, -20.0).flow_temp
        slope = (cold - warm) / 30.0

        assert 0.5 <= slope <= 1.0, (
            f"Curve slope {slope:.2f} C of supply per C of outdoor is not plausible for a "
            f"radiator system needing ~0.76."
        )

    def test_colder_than_design_asks_for_more_than_design_flow(self):
        """Below the design temperature the load exceeds design, so the flow must too.

        Clamping to the design flow here would silently under-heat in exactly the conditions the
        house is least able to tolerate it.
        """
        calc = WeatherCompensationCalculator(heat_loss_coefficient=150.0, heating_type="radiator")

        flow = calc.calculate_optimal_flow_temp(22.0, DEFAULT_DESIGN_OUTDOOR_TEMP - 10.0).flow_temp

        assert flow > DEFAULT_DESIGN_FLOW_TEMP_RADIATOR

    def test_no_heat_needed_when_outdoor_reaches_the_setpoint(self):
        """Water colder than the room would cool it."""
        calc = WeatherCompensationCalculator(heat_loss_coefficient=180.0)

        for outdoor in (20.0, 25.0, 30.0):
            result = calc.calculate_optimal_flow_temp(indoor_setpoint=20.0, outdoor_temp=outdoor)
            assert result.flow_temp == 20.0

    def test_a_leakier_house_needs_hotter_water(self):
        """Only via the rated-output anchor: the design-point anchor encodes sizing already."""
        tight = WeatherCompensationCalculator(
            heat_loss_coefficient=150.0, radiator_rated_output=12000.0
        )
        leaky = WeatherCompensationCalculator(
            heat_loss_coefficient=300.0, radiator_rated_output=12000.0
        )

        flow_tight = tight.calculate_optimal_flow_temp(21.0, 0.0).flow_temp
        flow_leaky = leaky.calculate_optimal_flow_temp(21.0, 0.0).flow_temp

        assert flow_leaky > flow_tight


class TestUnderfloorHeating:
    """UFH gets its own exponent and its own design point - not a radiator curve minus 8 C."""

    def test_underfloor_uses_its_own_emitter_exponent(self):
        """EN 1264 gives n ~ 1.1 for underfloor, not the radiator's 1.3."""
        ufh = WeatherCompensationCalculator(heating_type="concrete_ufh")
        rad = WeatherCompensationCalculator(heating_type="radiator")

        assert ufh.emitter_exponent == UFH_POWER_COEFFICIENT
        assert rad.emitter_exponent == RADIATOR_POWER_COEFFICIENT

    def test_underfloor_is_dimensioned_cooler_than_radiators(self):
        """NIBE: underfloor supply is normally set between 35 and 45 C."""
        ufh = WeatherCompensationCalculator(heating_type="concrete_ufh")

        assert ufh.design_flow_temp == DEFAULT_DESIGN_FLOW_TEMP_UFH
        assert ufh.design_flow_temp < DEFAULT_DESIGN_FLOW_TEMP_RADIATOR

    def test_underfloor_curve_is_not_flat(self):
        """The old model pinned concrete slabs to 25 C at EVERY outdoor temperature.

        Kuehne (fed a heat-loss coefficient) already produced a low-temperature curve; the code
        then subtracted a further 8 C and floored the result at UFH_MIN_FLOW_TEMP_CONCRETE = 25.
        The floor won across the whole Swedish winter, so weather compensation was completely
        INERT for a concrete-slab house - it targeted 25 C from +10 C down to -20 C.
        """
        calc = WeatherCompensationCalculator(heat_loss_coefficient=180.0, heating_type="concrete_ufh")

        mild = calc.calculate_optimal_flow_temp(21.0, 10.0).flow_temp
        cold = calc.calculate_optimal_flow_temp(21.0, -20.0).flow_temp

        assert cold > mild + 5.0, (
            f"Underfloor curve is flat: {mild:.1f} C at +10 C vs {cold:.1f} C at -20 C."
        )

    def test_underfloor_reaches_its_own_design_point(self):
        calc = WeatherCompensationCalculator(heating_type="timber_ufh")

        flow = calc.calculate_optimal_flow_temp(21.0, DEFAULT_DESIGN_OUTDOOR_TEMP).flow_temp

        assert flow == pytest.approx(DEFAULT_DESIGN_FLOW_TEMP_UFH, abs=0.01)


class TestOffsetCalculation:
    """Converting a flow-temperature target into a heating-curve offset."""

    def test_offset_calculation_basic(self):
        calc = WeatherCompensationCalculator()

        # +3 C of flow, at 1.5 C of flow per offset unit -> +2.0
        offset = calc.calculate_required_offset(
            optimal_flow_temp=40.0,
            current_flow_temp=37.0,
            curve_sensitivity=1.5,
        )

        assert offset == pytest.approx(2.0, abs=0.1)

    def test_offset_calculation_negative(self):
        calc = WeatherCompensationCalculator()

        offset = calc.calculate_required_offset(
            optimal_flow_temp=35.0,
            current_flow_temp=39.0,
            curve_sensitivity=2.0,
        )

        assert offset == pytest.approx(-2.0, abs=0.1)

    def test_offset_calculation_different_sensitivity(self):
        calc = WeatherCompensationCalculator()

        high = calc.calculate_required_offset(40.0, 35.0, curve_sensitivity=2.5)
        low = calc.calculate_required_offset(40.0, 35.0, curve_sensitivity=1.0)

        assert high < low  # a more sensitive curve needs a smaller offset

    def test_offset_is_bounded_in_both_directions(self):
        """The old implementation had no clamp and could return -11.4."""
        calc = WeatherCompensationCalculator()

        assert calc.calculate_required_offset(80.0, 20.0, 1.5) == WEATHER_COMP_MAX_OFFSET
        assert calc.calculate_required_offset(20.0, 80.0, 1.5) == -WEATHER_COMP_MAX_OFFSET


class TestRealWorldScenarios:
    """Whole-system checks in real Swedish conditions."""

    def test_house_that_needs_hot_water_gets_it(self):
        """Formerly `test_heatpumpmonitor_spf4_target`, which enshrined the bug.

        It asserted 24-35 C at 0 C outdoor for a 150 W/K house at 20 C indoor. The emitter law
        says that house needs 39.3 C. "Flow = outdoor + 27 for SPF 4.0" is an ASPIRATION that
        holds only when the emitters can deliver the load at that temperature; it is not a
        temperature you can simply choose. Demanding it of a system that cannot deliver it does
        not buy efficiency, it just leaves the house cold - which is precisely what EffektGuard
        was doing for 92% of a simulated month.
        """
        calc = WeatherCompensationCalculator(heat_loss_coefficient=150.0)

        result = calc.calculate_optimal_flow_temp(indoor_setpoint=20.0, outdoor_temp=0.0)

        # What the emitters actually need to carry a 3.0 kW load in this house.
        assert result.flow_temp == pytest.approx(39.3, abs=1.0)
        assert result.flow_temp > 20.0 + 27.0 - 10.0  # nowhere near the flat aspiration

    def test_swedish_winter_kiruna(self):
        """Extreme Swedish winter (-30 C), concrete slab."""
        calc = WeatherCompensationCalculator(
            heat_loss_coefficient=200.0,
            heating_type="concrete_ufh",
        )

        result = calc.calculate_optimal_flow_temp(indoor_setpoint=21.0, outdoor_temp=-30.0)

        assert result.heating_type == "concrete_ufh"
        # Colder than the design point, so it asks for MORE than the design flow - and stays
        # inside what a heat pump can physically produce.
        assert DEFAULT_DESIGN_FLOW_TEMP_UFH < result.flow_temp <= 65.0

    def test_swedish_mild_stockholm(self):
        """Typical Stockholm winter (-5 C), radiators."""
        calc = WeatherCompensationCalculator(
            heat_loss_coefficient=180.0,
            heating_type="radiator",
        )

        result = calc.calculate_optimal_flow_temp(indoor_setpoint=21.0, outdoor_temp=-5.0)

        assert 38.0 <= result.flow_temp <= 48.0
        assert isinstance(result, FlowTempCalculation)

    def test_reasoning_names_the_law_and_its_anchor(self):
        """The reasoning string is surfaced to the user; it must say what it actually did."""
        calc = WeatherCompensationCalculator(heat_loss_coefficient=180.0, heating_type="radiator")

        reasoning = calc.calculate_optimal_flow_temp(21.0, -5.0).reasoning

        assert "EN 442" in reasoning
        assert "design point" in reasoning
        assert "-5.0" in reasoning  # the outdoor temperature it reasoned from

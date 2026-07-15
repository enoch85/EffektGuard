"""Tests for the EN 442 emitter law used by weather compensation.

The old Kuehne formula, Timbones "method" and UFH flow "adjustment" are gone (F-119 / F-121):
weather compensation is now one law anchored on either the emitters' rated output or the system
design point. These tests check the law, its anchors, and the properties any heating curve must
have (monotonic, physically plausible slope, flat above the balance point, bounded offset).

Two external references are retained as validation of the law itself: Timbones' published
spreadsheet example (18 kW emitters, 260 W/K, 19 C, 0 C outdoor -> ~40 C flow), and the adequacy
a 150 W/K house needs at 0 C (39.3 C) - not the "outdoor + 27" efficiency aspiration that, asserted
as a requirement, made EffektGuard under-heat.
"""

import pytest

from custom_components.effektguard.const import (
    DEFAULT_DESIGN_FLOW_TEMP_RADIATOR,
    DEFAULT_DESIGN_FLOW_TEMP_UFH,
    DEFAULT_DESIGN_OUTDOOR_TEMP,
    DEFAULT_DESIGN_SPREAD,
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

        Published result ~40 C. Like OpenEnergyMonitor's WeatherComp, Timbones' spreadsheet models
        demand as linear in (room - outdoor) and carries NO internal-gains term - so the demand
        model is held identical on both sides here, and what is being checked is the EMITTER LAW,
        which is what the reference is authoritative about.
        """
        calc = WeatherCompensationCalculator(
            heat_loss_coefficient=260.0,
            radiator_rated_output=18000.0,
            internal_gains_w=0.0,  # match the reference's demand model, not our house
        )

        result = calc.calculate_optimal_flow_temp(indoor_setpoint=19.0, outdoor_temp=0.0)

        assert result.flow_temp == pytest.approx(40.0, abs=0.5)

    def test_internal_gains_are_what_move_us_off_the_uk_reference_tools(self):
        """Modelling internal gains asks for cooler water than the gains-free UK tools.

        The gains term must reach the (preferred) rated-output anchor; if these flows converge it
        has been switched off. A deliberate, evidenced departure (heatpumpmonitor.org fleet gains).
        """
        house = dict(heat_loss_coefficient=260.0, radiator_rated_output=18000.0)

        reference = WeatherCompensationCalculator(**house, internal_gains_w=0.0)
        ours = WeatherCompensationCalculator(**house)

        flow_reference = reference.calculate_optimal_flow_temp(19.0, 0.0).flow_temp
        flow_ours = ours.calculate_optimal_flow_temp(19.0, 0.0).flow_temp

        assert flow_ours < flow_reference - 1.0, (
            f"Modelling internal gains should ask for cooler water than the gains-free UK tools: "
            f"they want {flow_reference:.2f} C and we want {flow_ours:.2f} C. If these have "
            f"converged, the gains term is no longer reaching the rated-output anchor - which is "
            f"the anchor this layer PREFERS, and which has silently missed the gains before."
        )


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

    def test_no_heat_needed_above_the_balance_point(self):
        """Above the balance point the flow is FLAT at room + spread/2, with no cliff.

        Zero load means zero excess over the room, but the spread term does not vanish: like OEM's
        WeatherComp (flowT = MWT + systemDT/2), the curve converges to room + spread/2, not room.
        The flow must never fall below the setpoint (colder water would cool the room), and must not
        step at the boundary - a bare `indoor_setpoint` put a spread/2 cliff right in the shoulder
        season where the outdoor temperature crosses the balance point back and forth all day.
        """
        calc = WeatherCompensationCalculator(heat_loss_coefficient=180.0)
        balance = calc.balance_point_temp(20.0)
        no_load_flow = 20.0 + DEFAULT_DESIGN_SPREAD / 2.0

        for outdoor in (balance + 0.1, 20.0, 25.0, 30.0):
            result = calc.calculate_optimal_flow_temp(indoor_setpoint=20.0, outdoor_temp=outdoor)

            assert result.flow_temp >= 20.0, (
                f"At {outdoor:.1f} C outdoor the layer asks for {result.flow_temp:.2f} C of flow, "
                f"below the 20.0 C room. Water colder than the room removes heat from it."
            )
            assert result.flow_temp == pytest.approx(no_load_flow, abs=0.01), (
                f"Above the balance point ({balance:.1f} C) the house heats itself, so the curve "
                f"must be flat at room + spread/2 = {no_load_flow:.1f} C - the same value it "
                f"converges to from below. It returns {result.flow_temp:.2f} C at {outdoor:.1f} C."
            )

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
        calc = WeatherCompensationCalculator(
            heat_loss_coefficient=180.0, heating_type="concrete_ufh"
        )

        mild = calc.calculate_optimal_flow_temp(21.0, 10.0).flow_temp
        cold = calc.calculate_optimal_flow_temp(21.0, -20.0).flow_temp

        assert (
            cold > mild + 5.0
        ), f"Underfloor curve is flat: {mild:.1f} C at +10 C vs {cold:.1f} C at -20 C."

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
        """A 150 W/K house at 20 C, 0 C outdoor, needs 39.3 C - adequacy, not aspiration.

        The old test asserted 24-35 C here ("flow = outdoor + 27 for SPF 4.0"), an efficiency
        aspiration that holds only when the emitters can deliver the load at that temperature.
        Demanding it of a system that cannot just leaves the house cold.
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

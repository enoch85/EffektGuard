"""Maximum capacity is different from nominal COP ratings; running has a minimum output."""

from datetime import datetime

import pytest

from scripts.simulation import sim_harness as sim

F2040 = next(h for h in sim.HOUSES if h.profile.model_name == "F2040")
F1155 = next(h for h in sim.HOUSES if h.profile.model_name == "F1155")
# NIBE F2040-8 capacity/output sheet, compressor peak at W35, not its nominal COP test load.
PEAK_W35 = [(7.0, 9.2), (2.0, 6.8), (-7.0, 7.7)]


@pytest.mark.parametrize("outdoor,peak", PEAK_W35)
def test_maximum_output_comes_from_the_peak_capacity_sheet(outdoor, peak):
    assert F2040.capacity_kw_at(outdoor) == pytest.approx(peak)


@pytest.mark.parametrize("outdoor,flow,peak", [(7.0, 55.0, 8.6), (-7.0, 50.0, 7.3)])
def test_maximum_output_depends_on_water_temperature(outdoor, flow, peak):
    assert F2040.capacity_kw_at(outdoor, flow) == pytest.approx(peak)


@pytest.mark.parametrize("point", F2040.profile.datasheet_points)
def test_cop_at_a_nominal_output_keeps_the_nominal_rating(point):
    cop = F2040.cop_at_heat_output(point.source_temp_c, point.flow_temp_c, point.heat_output_kw)
    assert cop == pytest.approx(point.cop, rel=0.02)


def test_outside_the_peak_sheet_is_reported_as_an_assumption():
    assert not F2040.capacity_is_extrapolated(-7.0, 35.0)
    assert F2040.capacity_is_extrapolated(-10.0, 35.0)
    assert F2040.capacity_is_extrapolated(2.0, 58.0)
    assert F2040.capacity_kw_at(-20.1, 35.0) == 0.0


def test_a_pump_cannot_modulate_below_its_declared_minimum_in_a_mild_house():
    start = datetime(2026, 1, 15, tzinfo=sim.TZ)
    times, temperatures, raw, unit = sim._synthetic_days(start, 2)
    stats, violations, trace = sim.simulate(
        F1155, times, [15.0] * len(temperatures), sim.PriceSource(raw, unit), 2, baseline=True
    )
    assert not violations
    assert stats["compressor_starts"] > 0, "the declared minimum requires cycling below 3 kW demand"
    assert stats["compressor_below_declared_minimum_minutes"] == 0
    assert abs(stats["energy_balance_residual_kwh"]) < 0.1


def test_the_f2040_cannot_run_continuously_below_its_minimum_output():
    start = datetime(2026, 1, 15, tzinfo=sim.TZ)
    times, temperatures, raw, unit = sim._synthetic_days(start, 2)
    stats, violations, _ = sim.simulate(
        F2040, times, [15.0] * len(temperatures), sim.PriceSource(raw, unit), 2, baseline=True
    )
    assert not violations
    # NIBE's A7/W35 minimum is 2.5 kW; holding it at other conditions is a model assumption.
    assert stats["minimum_positive_compressor_heat_kw"] >= 2.5
    assert stats["compressor_starts"] > 0


F750 = next(h for h in sim.HOUSES if h.profile.model_name == "F750")
F730 = next(h for h in sim.HOUSES if h.profile.model_name == "F730")


# NIBE's minimum-frequency outputs at A20(12)W35: the F750 1.144 kW at 108 m3/h and 1.498 kW at
# 252 m3/h, the F730 1.27 kW at 90 m3/h and 1.53 kW at 252 m3/h. The maximum and the efficiency fit
# are both at 252 m3/h.
@pytest.mark.parametrize(
    ("house", "minimum"), [(F750, 1.498), (F730, 1.53), (F1155, 3.0)], ids=["F750", "F730", "F1155"]
)
def test_the_minimum_output_is_taken_at_the_airflow_of_the_maximum(house, minimum):
    assert house.minimum_output_kw == pytest.approx(minimum)


_NO_FLOW = object()


def _priced_run(monkeypatch, house, outdoor, days=2):
    """Two days on the baseline curve, recording each step's heat output, the compressor's
    electricity for it, the flow its capacity and its COP were read at, and the flow the emitters
    saw at the start of the step - the plant's own water temperature, read where the room's heat is
    computed rather than where the capacity or the COP is."""
    house_type = type(house)
    cop_for_output, capacity_at = house_type.cop_at_heat_output, house_type.capacity_kw_at
    emitted = house_type.heat_output_w
    run = {"heat_kw": [], "compressor_kw": [], "capacity_flows": [], "cop_flows": [], "water": []}

    def priced(self, outdoor_temp, flow, heat_kw):
        cop = cop_for_output(self, outdoor_temp, flow, heat_kw)
        run["heat_kw"].append(heat_kw)
        run["compressor_kw"].append(heat_kw / cop)
        run["cop_flows"].append(flow)
        return cop

    def bounded(self, outdoor_temp, flow=_NO_FLOW):
        if flow is _NO_FLOW:  # whether the compressor may run at all, not this step's output
            return capacity_at(self, outdoor_temp)
        run["capacity_flows"].append(flow)
        return capacity_at(self, outdoor_temp, flow)

    def emitting(self, flow, indoor):
        run["water"].append(flow)
        return emitted(self, flow, indoor)

    monkeypatch.setattr(house_type, "cop_at_heat_output", priced)
    monkeypatch.setattr(house_type, "capacity_kw_at", bounded)
    monkeypatch.setattr(house_type, "heat_output_w", emitting)
    start = datetime(2026, 1, 15, tzinfo=sim.TZ)
    times, temperatures, raw, unit = sim._synthetic_days(start, days)
    stats, violations, _ = sim.simulate(
        house, times, [outdoor] * len(temperatures), sim.PriceSource(raw, unit), days, baseline=True
    )
    assert not violations
    return stats, run


def test_an_exhaust_air_pump_cycles_rather_than_run_below_its_minimum(monkeypatch):
    stats, run = _priced_run(monkeypatch, F750, 15.0)
    assert stats["compressor_starts"] > 0
    assert min(kw for kw in run["heat_kw"] if kw > 0) >= 1.498 - 1e-9


def test_the_plant_buys_each_step_of_heat_at_the_cop_of_that_output(monkeypatch):
    stats, run = _priced_run(monkeypatch, F2040, 2.0)
    hours = sim.STEP_MIN / 60.0
    assert stats["energy_kwh"] == pytest.approx(
        sum(run["compressor_kw"]) * hours
        + stats["immersion_kwh"]
        + sim.STANDBY_KW * len(run["compressor_kw"]) * hours
    ), "every step's electricity is its heat over the COP of that heat output"


def test_the_capacity_and_the_cop_are_read_at_the_water_the_plant_has(monkeypatch):
    """The capacity bounds a step from the flow temperature it starts at, and the step's heat is
    priced at a flow temperature it passes through: not the curve's target, not the return."""
    _, run = _priced_run(monkeypatch, F2040, 2.0)
    water = run["water"]
    assert len(set(water)) > 1, "the run must move the flow temperature to tell anything apart"
    assert run["capacity_flows"] == water
    assert len(run["cop_flows"]) == len(water)
    for step, flow in enumerate(run["cop_flows"][:-1]):
        low, high = sorted((water[step], water[step + 1]))
        assert low - 1e-9 <= flow <= high + 1e-9, (
            f"step {step}: COP read at {flow:.2f} C while the water ran from {water[step]:.2f} to "
            f"{water[step + 1]:.2f} C"
        )

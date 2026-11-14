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

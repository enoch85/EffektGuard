"""Calibration pins and instantaneous deficits do not establish operating or causal bounds."""

import pytest
from scripts.simulation import sim_harness as sim


def _quiet_run():
    return dict(
        indoor_min=22.0,
        indoor_mean=22.0,
        exceptions=0,
        water_node_leak_kwh=0.0,
        immersion_kwh=0.0,
        instantaneous_capacity_deficit_kwh=0.0,
        comfort_minutes_below=0,
        comfort_minutes_above=0,
        comfort_band_low_c=21.5,
        mean_cop=3.0,
        datasheet_cop=4.72,
    )


def test_a_nominal_rating_does_not_bound_other_operating_conditions():
    house = sim.HOUSES[0]
    stats = _quiet_run()
    stats["mean_cop"] = house.cop_at(15.0, 25.0, 0.1)
    assert stats["mean_cop"] > stats["datasheet_cop"] * 1.15
    assert stats["mean_cop"] < house.carnot_cop(15.0, 25.0)
    assert not sim.check_invariants("outside-rated-conditions", stats, [], house)


def test_immersion_minus_instantaneous_deficit_does_not_identify_a_control_effect():
    stats = _quiet_run()
    stats["immersion_kwh"] = 20.0
    # These summaries contain no counterfactual run, storage trajectory or matched comfort.
    assert not sim.check_invariants("unattributed-backup", stats, [])


@pytest.mark.parametrize("violation", ["cop_beats_carnot", "dm_runaway"])
def test_removing_invalid_bounds_keeps_real_violation_checks(violation):
    failures = sim.check_invariants("physical-violation", _quiet_run(), [{"type": violation}])
    assert any("safety violation" in reason for reason in failures)

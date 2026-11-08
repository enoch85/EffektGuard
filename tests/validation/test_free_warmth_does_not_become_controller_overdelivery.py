"""Even minimum floor-maintaining heat can exceed net loss when free warmth charges the room."""

from scripts.simulation import sim_harness as sim
from tests.validation.test_simulation_diagnostics_are_not_causal_bounds import _quiet_run


def test_minimum_heat_with_a_warm_finish_is_not_evidence_of_overdelivery():
    house = next(h for h in sim.HOUSES if h.name == "apartment_f730")
    room = 22.0
    floor = 21.8
    heat = loss = 0.0
    # Greedy nonnegative heat keeps this first-order reference at the lowest allowed room.
    # It minimises both cumulative heat loss and final stored energy. The last warm day adds
    # free energy; none of the supplied heat can be removed while still holding the floor.
    for step in range(21 * 24 * 60 // sim.STEP_MIN):
        outside = 15.0 if step < 20 * 24 * 60 // sim.STEP_MIN else 17.0
        net_loss = house.hlc_w_per_k * (room - outside) - sim.INTERNAL_GAINS_W
        passive = room - net_loss * sim.STEP_MIN * 60 / house.capacity_j_per_k
        room = max(floor, passive)
        heat += (room - passive) * house.capacity_j_per_k / sim.J_PER_KWH
        loss += net_loss * sim.STEP_MIN * 60 / sim.J_PER_KWH
    stored = house.capacity_j_per_k * (room - 22.0) / sim.J_PER_KWH
    assert abs(heat - loss - stored) < 1e-6
    assert loss > 1.0 and heat / loss > 1.1
    assert floor <= room <= sim.TARGET_INDOOR + sim.OVERSHOOT_TOLERANCE
    stats = _quiet_run() | {"heat_kwh": heat, "loss_kwh": loss, "layer_votes": {}}
    failures = sim.check_scenario_expectations(sim.SCENARIOS["shoulder_may2024"], stats, house)
    assert not failures, (heat, loss, stored, failures)

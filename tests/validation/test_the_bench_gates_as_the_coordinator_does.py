"""The bench runs the coordinator's gate policy, not the bare volatility tracker.

The coordinator takes an emergency, an anti-windup decision and the wear guard's withdrawal of a
raise past the tracker; the harness ran every proposal through it, so the scenario gate measured
a pump that held exactly the cuts production lets through (external review R3).
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib

import pytest

from custom_components.effektguard.optimization.decision_engine import OptimizationDecision

_SPEC = importlib.util.spec_from_file_location(
    "sim_harness", pathlib.Path("scripts/simulation/sim_harness.py")
)
sim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sim)


@pytest.mark.parametrize("flag", ["is_emergency", "anti_windup_active", "saturation_capped"])
def test_a_cut_the_coordinator_lets_through_reaches_the_simulated_pump(monkeypatch, flag):
    """+3 C, then a cut to -1 C five minutes later: a volatile reversal, and one of the three
    decisions the coordinator never holds back."""
    real_build = sim.build_engine

    def scripted(*args, **kwargs):
        engine, effect = real_build(*args, **kwargs)
        first = iter([OptimizationDecision(offset=3.0)])
        cut = OptimizationDecision(offset=-1.0, **{flag: True})
        engine.calculate_decision = lambda **_: next(first, cut)
        return engine, effect

    monkeypatch.setattr(sim, "build_engine", scripted)
    times, temps, price_days, unit = sim.load_data(selftest=True)
    try:
        stats, _violations, _trace = sim.simulate(
            sim.HOUSES[0], times, temps, sim.PriceSource(price_days, unit), days=1
        )
    finally:
        asyncio.set_event_loop(asyncio.new_event_loop())

    assert stats["offset_reversals_blocked"] == 0, (
        f"the bench held a {flag} cut {stats['offset_reversals_blocked']} times; the coordinator "
        f"applies it at once"
    )

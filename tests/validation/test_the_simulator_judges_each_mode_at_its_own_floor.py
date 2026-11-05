"""The simulator judges a run's comfort against the band its mode keeps.

It judged every mode against the target less the configured tolerance. Savings mode is allowed a
degree below the target (MODE_CONFIGS, the owner's decision of 2026-10-29), so a savings run living
in its own allowance was reported as a starved house - the F2040's 31-day month in savings mode
spent 1605 minutes "below the band" doing what it was asked.
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib

from custom_components.effektguard.const import (
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    OPTIMIZATION_MODE_COMFORT,
    OPTIMIZATION_MODE_SAVINGS,
)

_SPEC = importlib.util.spec_from_file_location(
    "sim_harness", pathlib.Path("scripts/simulation/sim_harness.py")
)
sim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sim)

# A brine house on a pinned -2 cools through both bands within the self-test's two days.
_HOUSE = "concrete_f1155"
_PINNED_OFFSET = -2.0


def _quiet_month(mode: str, indoor_mean: float) -> dict:
    """A run's stats with nothing wrong in them but, perhaps, the room's mean (held flat)."""
    return {
        "indoor_min": indoor_mean,
        "indoor_mean": indoor_mean,
        "exceptions": 0,
        "water_node_leak_kwh": 0.0,
        "unavoidable_aux_kwh": 0.0,
        "aux_kwh": 0.0,
        "comfort_minutes_below": 0,
        "comfort_minutes_above": 0,
        "comfort_band_low_c": sim.comfort_band_low(mode),
    }


def test_savings_is_judged_at_its_floor_and_no_mode_tighter_than_the_tolerance():
    tolerance_band = sim.TARGET_INDOOR - sim.COMFORT_TOLERANCE
    savings_floor = (
        sim.TARGET_INDOOR - MODE_CONFIGS[OPTIMIZATION_MODE_SAVINGS].room_floor_below_target
    )

    assert sim.comfort_band_low(OPTIMIZATION_MODE_SAVINGS) == savings_floor < tolerance_band
    assert sim.comfort_band_low(OPTIMIZATION_MODE_BALANCED) == tolerance_band
    assert sim.comfort_band_low(OPTIMIZATION_MODE_COMFORT) == tolerance_band


def test_a_run_counts_its_cold_minutes_against_its_own_modes_band():
    house = next(h for h in sim.HOUSES if h.name == _HOUSE)
    times, temps, price_days, unit = sim.load_data(selftest=True)
    runs = {}
    for mode in (OPTIMIZATION_MODE_BALANCED, OPTIMIZATION_MODE_SAVINGS):
        try:
            runs[mode], _violations, _trace = sim.simulate(
                house,
                times,
                temps,
                sim.PriceSource(price_days, unit),
                days=2,
                mode=mode,
                fixed_offset=_PINNED_OFFSET,
            )
        finally:
            asyncio.set_event_loop(asyncio.new_event_loop())
    balanced, savings = runs[OPTIMIZATION_MODE_BALANCED], runs[OPTIMIZATION_MODE_SAVINGS]

    assert (balanced["indoor_min"], balanced["indoor_mean"]) == (
        savings["indoor_min"],
        savings["indoor_mean"],
    ), "PRECONDITION: the pinned offset gives both modes the same room"
    assert savings["comfort_band_low_c"] == sim.comfort_band_low(OPTIMIZATION_MODE_SAVINGS)
    assert balanced["comfort_band_low_c"] == sim.comfort_band_low(OPTIMIZATION_MODE_BALANCED)
    assert (
        0 < savings["comfort_minutes_below"] < balanced["comfort_minutes_below"]
    ), "the same room must count fewer cold minutes against savings' lower band, and some"


def test_a_savings_month_may_average_below_the_tolerance_band_but_not_below_its_floor():
    between = (
        sim.comfort_band_low(OPTIMIZATION_MODE_SAVINGS)
        + sim.comfort_band_low(OPTIMIZATION_MODE_BALANCED)
    ) / 2

    assert not sim.check_invariants("savings", _quiet_month(OPTIMIZATION_MODE_SAVINGS, between), [])
    assert sim.check_invariants("balanced", _quiet_month(OPTIMIZATION_MODE_BALANCED, between), [])
    below_floor = sim.comfort_band_low(OPTIMIZATION_MODE_SAVINGS) - sim.COMFORT_TOLERANCE
    assert sim.check_invariants("savings", _quiet_month(OPTIMIZATION_MODE_SAVINGS, below_floor), [])

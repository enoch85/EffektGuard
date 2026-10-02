"""Below -20 C outdoor, the F2040 does not run. NIBE's manual says so; the plant must too.

The F2040 installer manual publishes an operating range - "Min. / Max. air temp: -20 / 43 C" -
and the profile has carried that number (f2040.py, MIN_AIR_TEMP_C) since the datasheet audit.
It was referenced NOWHERE: the simulated plant held the compressor's capacity at its coldest
published point forever, so at Kiruna temperatures the model made phantom heat with a machine
that is switched off in reality.

Real January 2024 in Kiruna (Open-Meteo ERA5, scripts/simulation/data/weather_kiruna_jan2024.json)
spends 211 of 744 hours - 28% of the month - below that floor, with a minimum of -36.8 C. A plant
that keeps an F2040 running through that is not a model of the machine, it is a model of a wish.

The cutoff is STRICTLY below the floor and F2040-only:
  * At exactly -20.0 C the machine is inside its published range and the existing datasheet pins
    (capacity at -20, cop_at at -20, Carnot sweeps) must keep holding.
  * The other four machines do not have the outdoor air as their heat source. A brine pump's
    source sits at 0 C and an exhaust-air pump breathes 20 C house air whatever the weather does;
    NIBE publishes no outdoor operating floor for them, so the model imposes none.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from custom_components.effektguard.models.nibe.f2040 import MIN_AIR_TEMP_C

_SPEC = importlib.util.spec_from_file_location(
    "sim_harness", pathlib.Path("scripts/simulation/sim_harness.py")
)
sim_harness = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sim_harness)

HOUSES = {house.name: house for house in sim_harness.HOUSES}
F2040 = HOUSES["airsource_f2040"]
KIRUNA_MINIMUM_C = -36.8  # the real ERA5 minimum, January 2024


def test_the_f2040_stops_strictly_below_its_published_floor():
    """The manual's operating range is a hard edge, not a derating."""
    assert F2040.capacity_kw_at(MIN_AIR_TEMP_C) > 0.0, (
        f"at exactly {MIN_AIR_TEMP_C} C the F2040 is INSIDE its published operating range "
        f"('Min. / Max. air temp: -20 / 43 C') and must still deliver heat - the datasheet "
        f"pins at -20 depend on it."
    )
    assert F2040.capacity_kw_at(MIN_AIR_TEMP_C - 0.1) == 0.0, (
        "0.1 C below the published floor the machine does not operate. The manual gives a "
        "range, not a curve; outside it there is no compressor heat to model."
    )
    assert F2040.capacity_kw_at(KIRUNA_MINIMUM_C) == 0.0, (
        "at the real Kiruna January minimum (-36.8 C, ERA5) the F2040 is 16.8 C below its "
        "operating floor. Holding its -7 C capacity here is phantom heat."
    )


@pytest.mark.parametrize("name", ["wooden_f750", "apartment_f730", "concrete_f1155", "villa_s1155"])
def test_the_indoor_sourced_machines_run_through_the_arctic_night(name):
    """No invented floors. NIBE publishes no outdoor operating limit for these machines.

    Their heat sources are 20 C extract air and 0 C brine - the weather never touches them.
    An arctic cutoff applied to all five machines would be exactly the kind of unsourced
    physics this audit exists to remove.
    """
    house = HOUSES[name]
    assert house.capacity_kw_at(KIRUNA_MINIMUM_C) > 0.0, (
        f"{name} lost its capacity at -36.8 C outdoor. Its heat source is indoors (or in the "
        f"ground); NIBE publishes no outdoor floor for it, so the model must not invent one."
    )


def test_cop_stays_finite_at_the_floor_itself():
    """The Carnot sweeps and datasheet pins evaluate cop_at(-20.0); it must stay a real COP."""
    cop = F2040.cop_at(MIN_AIR_TEMP_C, 35.0)
    assert 1.0 <= cop < F2040.carnot_cop(MIN_AIR_TEMP_C, 35.0), (
        f"cop_at({MIN_AIR_TEMP_C}) returned {cop}. At the edge of the range the machine still "
        f"runs; the cutoff zeroes CAPACITY strictly below the floor, never the COP - a COP "
        f"sentinel would poison the mean-COP and Carnot accounting."
    )


def test_the_plant_does_not_lie_to_the_decision_engine_below_the_floor():
    """With the compressor physically stopped, the simulated NibeState must say so.

    Zeroing capacity alone leaves `compressor_on` True, so the plant would report
    compressor_hz > 0 and is_heating=True for a machine that is off - and the DecisionEngine
    under test would be optimising a lie. The harness exposes the availability rule so this
    test fails if the reported state is decoupled from the physics.
    """
    assert hasattr(sim_harness, "compressor_available"), (
        "sim_harness must expose compressor_available(house, outdoor_c) - the single rule that "
        "both the plant physics and the reported NibeState derive from."
    )
    assert sim_harness.compressor_available(F2040, MIN_AIR_TEMP_C) is True
    assert sim_harness.compressor_available(F2040, MIN_AIR_TEMP_C - 0.1) is False
    assert sim_harness.compressor_available(HOUSES["villa_s1155"], KIRUNA_MINIMUM_C) is True

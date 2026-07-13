"""The research must stay true, or it becomes what it replaced.

`docs/research/` exists because the code cited fifteen research documents as the authority for
safety-critical thresholds, and every one of them was absent from the repository. The rulebook's
binding rule - "never guess NIBE behaviour, verify against research" - could not be obeyed by
anyone who cloned this repo.

Replacing dangling citations with sourced ones only helps if the sourced ones stay true. A research
note that has drifted from the code is worse than no note at all: it looks settled. So the numbers
these documents quote are checked here, against the code they claim to justify.

These are not the derivations - those live in the documents, with their sources. This is the part a
machine can hold you to.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from custom_components.effektguard import const
from custom_components.effektguard.optimization.airflow_optimizer import calculate_net_thermal_gain
from custom_components.effektguard.utils.emitter import en442_flow_temp

RESEARCH = Path(__file__).resolve().parents[2] / "docs" / "research"

# Every constant the research documents quote a value for, and the value they quote.
QUOTED = {
    "DM_THRESHOLD_START": -60,  # 01: NIBE menu 4.9.3 "start compressor"
    "DM_THRESHOLD_AUX_LIMIT": -1500,  # 01: the absolute backstop
    "UFH_CONCRETE_PREDICTION_HORIZON": 24.0,  # 03: the slab's planning horizon
    "WEATHER_COMP_MAX_OFFSET": 3.0,  # 03: the bound on weather-driven offsets
    "RADIATOR_RATED_DT": 50.0,  # 02: EN 442-1 §3.23
    "RADIATOR_POWER_COEFFICIENT": 1.3,  # 02: EN 442 panel radiators
    "UFH_POWER_COEFFICIENT": 1.1,  # 02: EN 1264 - NOT 1.3
    "DEFAULT_CURVE_SENSITIVITY": 1.5,  # 03: used in the pre-heat sizing rule
    "AIRFLOW_COMPRESSOR_BASE_THRESHOLD": 61.0,  # 04: not the 50.0 the old docs printed
}


@pytest.mark.parametrize("name,quoted", sorted(QUOTED.items()))
def test_research_quotes_the_constant_the_code_actually_holds(name, quoted):
    """A citation that no longer matches the code is a citation that misleads."""
    actual = getattr(const, name)

    assert actual == quoted, (
        f"docs/research quotes {name} = {quoted!r}; const.py holds {actual!r}. Either the constant "
        f"was retuned without revisiting the evidence for it, or the note is wrong. Both matter: "
        f"this directory exists so that these numbers can be checked."
    )


@pytest.mark.parametrize(
    "outdoor,quoted_gain",
    [(10, 0.03), (5, -0.14), (0, -0.31), (-5, -0.48), (-10, -0.65), (-15, -0.82)],
)
def test_the_airflow_gain_table_is_what_the_code_computes(outdoor, quoted_gain):
    """04_exhaust_air_recovery.md prints a net-gain table. It must be the real one.

    The whole point of that page is that the gain is NEGATIVE once the double-counted COP term is
    removed. If someone restores the COP term, this table goes positive and the page becomes a lie
    that argues for a feature that loses heat.
    """
    gain = calculate_net_thermal_gain(
        const.AIRFLOW_DEFAULT_STANDARD, const.AIRFLOW_DEFAULT_ENHANCED, 21.0, float(outdoor)
    )

    assert gain == pytest.approx(quoted_gain, abs=0.005), (
        f"docs/research/04 says enhanced airflow nets {quoted_gain:+.2f} kW at {outdoor}°C; "
        f"calculate_net_thermal_gain gives {gain:+.2f} kW."
    )


def test_enhanced_airflow_still_loses_heat_in_the_cold():
    """The claim the page is actually making, stated as a property rather than a table."""
    for outdoor in (5, 0, -5, -10, -15):
        gain = calculate_net_thermal_gain(
            const.AIRFLOW_DEFAULT_STANDARD, const.AIRFLOW_DEFAULT_ENHANCED, 21.0, float(outdoor)
        )
        assert gain < 0, (
            f"Enhanced airflow shows a POSITIVE net gain of {gain:+.2f} kW at {outdoor}°C. The "
            f"research (docs/research/04) says it cannot: extracting more heat from more air and "
            f"'improving the COP' are the same joules, and NIBE's own S735 data confirms it. If "
            f"this now passes, someone has re-added the double-counted term."
        )


def test_the_en442_worked_example_in_the_docs_reproduces():
    """02_emitter_law.md shows a code block and prints its result. Run it.

    This is the validation that anchors the whole flow-temperature model: NIBE's published curve 9
    reads 41.0 °C at 0 °C outdoor, and the EN 442 emitter law lands within a fifth of a degree of it
    where a straight line is out by more than two.
    """
    flow = en442_flow_temp(
        indoor_setpoint=20.0,
        outdoor_temp=0.0,
        design_outdoor_temp=-15.0,
        design_flow_temp=52.6,
        design_spread=10.0,
        emitter_exponent=1.3,
    )

    doc = (RESEARCH / "02_emitter_law.md").read_text(encoding="utf-8")
    quoted = float(re.search(r"\)\s*#\s*->\s*([\d.]+)", doc).group(1))

    assert flow == pytest.approx(quoted, abs=0.01), (
        f"The worked example in 02_emitter_law.md says this call returns {quoted}; it returns "
        f"{flow:.2f}. A research note whose own code block does not run is exactly the kind of "
        f"citation this directory was created to replace."
    )

    nibe_published = 41.0
    linear = 20.0 + (52.6 - 20.0) * (20.0 - 0.0) / (20.0 - (-15.0))

    assert abs(flow - nibe_published) < abs(linear - nibe_published), (
        "The EN 442 emitter law must track NIBE's own published curve more closely than a straight "
        "line does. That curvature is the whole justification for the exponent."
    )


def test_every_research_note_is_indexed():
    """A note nobody can find is a note nobody will maintain."""
    index = (RESEARCH / "README.md").read_text(encoding="utf-8")
    notes = sorted(p.name for p in RESEARCH.glob("*.md") if p.name != "README.md")

    missing = [n for n in notes if n not in index]

    assert not missing, f"docs/research/README.md does not link: {', '.join(missing)}"

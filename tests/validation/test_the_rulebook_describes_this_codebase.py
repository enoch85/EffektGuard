"""The document every contributor is told to read first must not describe a codebase that is gone.

`CLAUDE.md` sends every contributor to `.github/copilot-instructions.md` as "the single source of
truth ... to be read at the start of every session", so a false claim there is an instruction, not
a documentation nit. This test reads the rulebook and holds it to the code:

  - it must not teach the removed Kuhne flow-temperature formula (F-119/F-121), nor a second, linear
    flow rule, as live models - both were replaced by the EN 442 emitter law;
  - every climate DM table and UFH prediction horizon it prints must be what const.py computes (the
    table appears more than once, and an earlier fix corrected only one copy);
  - every module it tells you to import must exist, and every research document it cites must be in
    the repository (`docs/research/`), not one of the gitignored, absent ones (F-106).

The docs here drifted because no test ever read one. Now one does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from custom_components.effektguard import const
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector

ROOT = Path(__file__).resolve().parents[2]
RULEBOOK = ROOT / ".github" / "copilot-instructions.md"
DOC = RULEBOOK.read_text(encoding="utf-8")

# What the rulebook ASSERTS, as opposed to what it warns you against. A document that says
# "do not reintroduce X" necessarily contains X, and must not trip the test that forbids X - the
# same trap that made an earlier pass of this file flag its own corrections.
DENIALS = (
    "has never existed",
    "neither of which exists",
    "There is **no",
    "does not exist",
    "does not have",
    "not sourced",
    # "used to X" - any past-tense correction; matching the phrase "used to " covers every verb, so
    # a new correction ("used to offer") does not trip its own test.
    "used to ",
    "Do not reintroduce",
    "no longer",
)


def _claims() -> str:
    """What the rulebook ASSERTS, minus the paragraphs that warn you against something removed.

    PARAGRAPH-wise, not line-wise: a warning spans several lines ("This example used to show X ...
    Do not reintroduce it.") and only one carries the marker, so a line filter would keep the rest
    and trip the test on the very correction it is meant to protect.
    """
    kept = [p for p in DOC.split("\n\n") if not any(d in p for d in DENIALS)]
    return _strip_corrections("\n\n".join(kept))


def _strip_corrections(text: str) -> str:
    """ "(NOT -700)" is a correction, not a claim that the number is -700."""
    return re.sub(r"\(NOT\s*-?\d+\)", "", text)


# For "this thing was REMOVED, do not bring it back" checks. A warning must be allowed to name the
# thing it forbids, so the paragraphs that carry a denial are dropped.
CLAIMS = _claims()

# For NUMBERS. Nothing is dropped, because a number is never legitimately wrong - not even inside a
# warning. Filtering denials here would hide a drifting second copy of the degree-minute table
# whose paragraph happens to contain "not sourced" (about DM -1500): the filter that protects the
# Kuhne check must not blind the climate check.
EVERY_WORD = _strip_corrections(DOC)


def test_the_rulebook_does_not_teach_a_formula_that_was_removed():
    """Kühne drove the flow temperature of a real heat pump, and was taken out for being wrong.

    Checked against everything the document ASSERTS - prose as much as code. Naming the formula in
    a warning ("do not reintroduce this") is exactly what the file should do; crediting it under
    "Research-Based", or copying it into a "✅ Do this" example, is what it must not.
    """
    assert "Kühne" not in CLAIMS and "Kuhne" not in CLAIMS, (
        "The rulebook still credits André Kühne's flow-temperature formula. It appears ZERO times "
        "in the codebase: it was removed (F-119/F-121) and replaced by the EN 442 emitter law, "
        "because it was fed a heat-loss coefficient where the derivation needs a dimensionless "
        "relative load. A contributor following the rulebook reintroduces it. "
        "See docs/research/02_emitter_law.md."
    )
    assert "2.55" not in CLAIMS, (
        "The Kühne coefficient 2.55 is still asserted somewhere in the rulebook. The flow "
        "temperature comes from the EN 442 emitter law now - see utils/emitter.py and "
        "docs/research/02_emitter_law.md."
    )


def test_the_rulebook_does_not_offer_a_second_flow_temperature_model():
    """A fixed "Flow = Outdoor + 27 °C" rule must not sit beside the emitter law as advice.

    It is offered in the rulebook as "OEM Research", in the document that tells contributors how to
    implement - inviting someone to build a model this project does not have (there are no
    OPTIMAL_FLOW_DELTA_SPF_* constants). The flow temperature comes from the EN 442 emitter law,
    anchored on the house's own design point, not a fixed offset from the outdoor temperature.
    """
    assert "Flow = Outdoor +" not in CLAIMS, (
        "The rulebook offers a linear flow-temperature rule (Flow = Outdoor + 27 °C) as OEM "
        "research. The flow temperature comes from the EN 442 emitter law, anchored on the house's "
        "own design point. There are no OPTIMAL_FLOW_DELTA_SPF_* constants; this describes a model "
        "the code does not have."
    )


@pytest.mark.parametrize(
    "city,latitude,outdoor",
    [("Stockholm", 59.33, -10.0), ("Kiruna", 67.86, -30.0), ("Paris", 48.86, 5.0)],
)
def test_every_climate_number_in_the_rulebook_is_the_number_the_code_computes(
    city, latitude, outdoor
):
    """The same table appears twice in this file. An earlier fix corrected only one copy."""
    dm_range = ClimateZoneDetector(latitude=latitude).get_expected_dm_range(outdoor)
    real = {round(v) for v in dm_range.values()}

    # Every degree-minute figure the rulebook prints on a line that names this city, wherever in
    # the file that line appears. All of them must be numbers the code actually produces.
    quoted = {
        int(n)
        for line in EVERY_WORD.splitlines()
        if city in line
        for n in re.findall(r"(-\d{3,4})\b", line)
    }

    assert quoted, f"the rulebook no longer quotes a DM threshold for {city} at all"

    invented = quoted - real
    assert not invented, (
        f"On a line naming {city}, the rulebook prints {sorted(invented)}. At {outdoor:.0f}°C the "
        f"code produces {sorted(real)} (normal_min, normal_max, warning, critical). These are the "
        f"numbers a maintainer reads to decide whether a degree-minute reading is safe - and this "
        f"table appears more than once in the file, so correct EVERY copy."
    )


@pytest.mark.parametrize(
    "emitter,constant",
    [
        ("Concrete slab", "UFH_CONCRETE_PREDICTION_HORIZON"),
        ("Timber", "UFH_TIMBER_PREDICTION_HORIZON"),
        ("Radiators", "UFH_RADIATOR_PREDICTION_HORIZON"),
    ],
)
def test_the_prediction_horizons_match_the_constants(emitter, constant):
    """A slab plans over 24 hours, not 12. Six hours is its LAG, not its horizon."""
    real = int(getattr(const, constant))

    line = next((ln for ln in EVERY_WORD.splitlines() if f"**{emitter}**" in ln), None)
    assert line, f"the rulebook no longer describes {emitter}"

    quoted = re.findall(r"\*{0,2}(\d+)h\*{0,2} prediction horizon", line)
    assert quoted, f"no prediction horizon quoted for {emitter}: {line.strip()!r}"

    assert int(quoted[0]) == real, (
        f"The rulebook says {emitter} uses a {quoted[0]}h prediction horizon; {constant} is "
        f"{real}.0. For a concrete slab this is the difference between seeing a two-day cold slide "
        f"and being blind to it (F-130)."
    )


def test_every_module_the_rulebook_tells_you_to_import_exists():
    """The "verify your work" snippet imports a module that has never existed."""
    imports = re.findall(r"from (custom_components\.effektguard[\w.]*) import", CLAIMS)
    imports += re.findall(r"import (custom_components\.effektguard[\w.]*)", CLAIMS)

    missing = []
    for dotted in set(imports):
        path = ROOT / (dotted.replace(".", "/") + ".py")
        if not path.exists() and not (ROOT / dotted.replace(".", "/")).is_dir():
            missing.append(dotted)

    assert not missing, (
        f"The rulebook tells you to import {', '.join(sorted(missing))}, which does not exist. "
        f"The thermal model lives in `optimization/thermal_layer.py` - every module in that "
        f"package is `*_layer.py`."
    )


def test_the_research_pointers_point_at_research_that_is_in_the_repository():
    """ "Never guess NIBE behaviour, verify with research docs" - and then names absent documents."""
    absent = [
        name
        for name in re.findall(r"`?([\w/]+\.md)`?", CLAIMS)
        if "IMPLEMENTATION_PLAN" in name or "COMPLETED" in name
    ]
    absent += [
        name
        for name in (
            "Forum_Summary.md",
            "Swedish_NIBE_Forum_Findings.md",
            "Setpoint_Optimizing_Algorithm.md",
            "MyUplink_Complete_Guide.md",
            "Mathematical_Enhancement_Summary.md",
            "Enhancement_Proposals.md",
        )
        if name in CLAIMS and not list(ROOT.rglob(name))
    ]

    assert not absent, (
        f"The rulebook's binding rule is 'never guess NIBE behaviour, verify with research docs', "
        f"and it then cites {', '.join(sorted(set(absent)))} - none of which is in this repository "
        f"(they are gitignored; audit F-106). The rule cannot be obeyed. `docs/research/` holds "
        f"the sourced evidence: point at that."
    )


def test_the_rulebook_sends_you_to_the_research_that_does_exist():
    """Having removed the dangling citations, it has to name the real ones."""
    assert (ROOT / "docs" / "research").is_dir(), "docs/research/ is missing"

    assert "docs/research" in DOC, (
        "docs/research/ holds the sourced evidence for the safety limits - EN 442-1, EN 1264, the "
        "F750 manual's menu 4.9.3, NIBE's own S735 tables - and the rulebook does not mention it. "
        "That directory exists precisely so the 'verify with research' rule can be obeyed."
    )

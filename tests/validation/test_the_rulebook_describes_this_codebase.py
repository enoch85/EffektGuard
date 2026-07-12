"""The document every contributor is told to read first must not teach a removed model.

`CLAUDE.md` sends every contributor - human or agent - to `.github/copilot-instructions.md`, and
calls it "the single source of truth for this repository's rules, architecture, and implementation
guidelines", to be read at the start of every session. So a false claim in that file is not a
documentation nit. It is an instruction.

The worst of them was in the section that teaches you how to write a good docstring:

    \"\"\"Calculate optimal flow temperature using André Kühne's formula.
    ...
    Formula: TFlow = 2.55 × (HC × (Tset - Tout))^0.78 + Tset
    \"\"\"

**Kühne appears zero times in the codebase.** It was removed (audit F-119/F-121) and replaced by
the EN 442 emitter law, because it was being fed a heat-loss coefficient where the derivation
requires a dimensionless relative load - a dimensionally inconsistent input to a structurally
correct law, which produces numbers that look plausible and are not. It drove the flow temperature
of a real heat pump. The rulebook was still holding it up as the example to copy.

The rest was the ordinary rot that nobody checks for, because nothing has ever checked:

  * the SAME wrong climate table appeared TWICE, and an earlier fix corrected only one copy
    (Stockholm -700 where the code gives -740; Kiruna -1200 vs -1400; Paris -350 vs -250);
  * all three UFH prediction horizons were wrong (12/6/2 h against the constants' 24/12/6);
  * the "verify your work" snippet imports `optimization.thermal_model`, which does not exist;
  * "Always Check Research Before Implementing" points at four documents that are gitignored and
    absent from the repository, while `docs/research/` - which exists, and holds the sourced
    evidence - goes unmentioned.

This test is the point. The docs in this repository drifted to ~55-65% wrong because no test ever
read one. Now one does.
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
    "used to show",
    "used to name",
    "Do not reintroduce",
    "no longer",
)


def _claims() -> str:
    kept = [line for line in DOC.splitlines() if not any(d in line for d in DENIALS)]
    # "(NOT -700)" is a correction, not a claim that the number is -700.
    return re.sub(r"\(NOT\s*-?\d+\)", "", "\n".join(kept))


CLAIMS = _claims()

# Only the fenced python blocks - the part a contributor COPIES.
CODE_BLOCKS = "\n".join(re.findall(r"```python\n(.*?)```", DOC, re.S))


def test_the_rulebook_does_not_teach_a_formula_that_was_removed():
    """Kühne drove the flow temperature of a real heat pump, and was taken out for being wrong.

    Checked against the fenced code blocks - the part a contributor copies. Naming the formula in a
    warning ("do not reintroduce this") is exactly what the file SHOULD do; putting it in an example
    labelled "✅ Do this" is what it must not.
    """
    assert "Kühne" not in CODE_BLOCKS and "Kuhne" not in CODE_BLOCKS, (
        "The rulebook still teaches André Kühne's flow-temperature formula - as its worked example "
        "of a GOOD docstring. It appears zero times in the codebase: it was removed (F-119/F-121) "
        "and replaced by the EN 442 emitter law, because it was fed a heat-loss coefficient where "
        "the derivation needs a dimensionless relative load. A contributor following the rulebook "
        "reintroduces it. See docs/research/02_emitter_law.md."
    )
    assert "2.55" not in CODE_BLOCKS, (
        "The Kühne coefficient 2.55 is still inside a code example in the rulebook. The flow "
        "temperature comes from the EN 442 emitter law now - see utils/emitter.py and "
        "docs/research/02_emitter_law.md."
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
        for line in CLAIMS.splitlines()
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

    line = next((ln for ln in CLAIMS.splitlines() if f"**{emitter}**" in ln), None)
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

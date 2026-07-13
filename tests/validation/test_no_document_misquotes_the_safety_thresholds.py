"""One test for every document, because the wrong number was in five of them.

`docs/CLIMATE_ZONES.md` was corrected and given a test. The test parses its TABLE rows. So the
prose in the other documents went on being wrong, and the same figures kept turning up:

    docs/architecture/00_overview.md          Stockholm (-10°C): Expects DM -450 to -700
    docs/architecture/02_emergency_thermal_debt.md   Cold Zone at -10°C: -450 to -700
    docs/architecture/10_adaptive_climate_zones.md   "Winter avg: -10.0°C"

The trap is that **-450 to -700 is a real Stockholm range**. It is what the code produces at
**-8 °C** - the Cold zone's actual winter average. Every one of those documents asserts it at
**-10 °C**, where the code gives **-490 to -740**. You cannot catch that by looking for a bad
number, because it is not a bad number; it is a good number attached to the wrong temperature.

The root of it is one constant. The Cold zone's `winter_avg_low` is **-8.0**, and four documents
still say -10.0, so every threshold they derive from it is off by 40 degree-minutes.

So this checks the claim, not the digits: wherever a document names a climate zone or a city, gives
an outdoor temperature, and prints a degree-minute range, that range must be the one
`ClimateZoneDetector` actually computes at that temperature. It reads every markdown file in the
repository - which is the P3 recommendation the audit made and nobody implemented: generate the
numbers from `const.py`, or at least refuse to let them drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from custom_components.effektguard.optimization.climate_zones import (
    HEATING_CLIMATE_ZONES,
    ClimateZoneDetector,
)

ROOT = Path(__file__).resolve().parents[2]

# A latitude squarely inside each zone, and the names a document might use for it.
ZONES = {
    "extreme_cold": (67.86, ("Extreme Cold", "Kiruna", "Tromsø", "Tromso")),
    "very_cold": (65.58, ("Very Cold", "Luleå", "Lulea", "Umeå", "Umea")),
    "cold": (59.33, ("Cold", "Stockholm", "Oslo", "Göteborg", "Goteborg", "Helsinki")),
    "moderate_cold": (55.60, ("Moderate Cold", "Malmö", "Malmo", "Copenhagen")),
    "standard": (48.86, ("Standard", "Paris", "London", "Berlin")),
}

# A degree-minute range: "-450 to -700".
DM_RANGE = re.compile(r"(-\d{2,4})\s*(?:to|–|-)\s*(-\d{2,4})")
# An outdoor temperature: "-10°C", "-10.0°C", "at -10 C".
OUTDOOR = re.compile(r"(-?\d{1,2}(?:\.\d)?)\s*°?\s*C\b")
# Every way this repository writes a zone's winter average:
#   "Winter avg: -10.0°C"     prose and mermaid labels
#   "Average winter low: -8°C"
#   "winter_avg_low: -10.0°C"  the CONSTANT's own name, quoted in code blocks
#
# The underscore form was missed at first, and a mutation test caught it: drifting
# `winter_avg_low: -8.0` back to -10.0 in docs/architecture/10 passed cleanly. A guard that only
# reads prose does not guard the code blocks people actually copy.
WINTER_AVG = re.compile(
    r"[Ww]inter[\s_](?:avg|average)(?:[\s_]low)?[:\s]+(-?\d{1,2}(?:\.\d)?)"
    r"|[Aa]verage\s+winter\s+low[:\s]+(-?\d{1,2}(?:\.\d)?)"
)


def _markdown_files() -> list[Path]:
    files = [ROOT / "README.md"]
    files += sorted((ROOT / "docs").rglob("*.md"))
    files += sorted((ROOT / ".github").rglob("*.md"))
    return [f for f in files if f.exists()]


def _zone_named_in(line: str) -> str | None:
    """Which climate zone, if any, this line is talking about.

    The most specific match wins: a line naming "Extreme Cold" is not a "Cold" line.
    """
    best: tuple[int, str] | None = None
    for key, (_lat, names) in ZONES.items():
        for name in names:
            if re.search(rf"\b{re.escape(name)}\b", line):
                if best is None or len(name) > best[0]:
                    best = (len(name), key)
    return best[1] if best else None


def _claims() -> list[tuple[Path, int, str, str, float, int, int]]:
    """Every (file, line, zone, outdoor_temp, dm_low, dm_high) a document asserts."""
    found = []
    for path in _markdown_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            dm = DM_RANGE.search(line)
            if not dm:
                continue
            zone = _zone_named_in(line)
            if zone is None:
                continue
            temps = [float(t) for t in OUTDOOR.findall(line)]
            # The outdoor temperature is the one that is not a degree-minute figure.
            temps = [t for t in temps if -40.0 <= t <= 20.0]
            if not temps:
                continue
            found.append(
                (path, lineno, zone, line.strip(), temps[0], int(dm.group(1)), int(dm.group(2)))
            )
    return found


def test_the_scanner_actually_finds_the_claims_it_is_checking():
    """A parser that silently matches nothing makes every assertion below vacuous."""
    claims = _claims()

    assert len(claims) >= 5, (
        f"Only {len(claims)} degree-minute claims were found across every markdown file in the "
        f"repository. The scanner has stopped matching, and this test now proves nothing."
    )


@pytest.mark.parametrize(
    "path,lineno,zone,line,outdoor,low,high",
    _claims(),
    ids=lambda v: f"{v.name}" if isinstance(v, Path) else str(v),
)
def test_every_documented_dm_range_is_what_the_code_computes(
    path, lineno, zone, line, outdoor, low, high
):
    """A good number attached to the wrong temperature is still a wrong claim."""
    latitude = ZONES[zone][0]
    actual = ClimateZoneDetector(latitude=latitude).get_expected_dm_range(outdoor)
    expected = (round(actual["normal_min"]), round(actual["normal_max"]))

    assert (low, high) == expected, (
        f"{path.relative_to(ROOT)}:{lineno} says the {zone} zone at {outdoor:g}°C expects DM "
        f"{low} to {high}. ClimateZoneDetector computes {expected[0]} to {expected[1]}.\n"
        f"    {line}\n"
        f"Note {low} to {high} may well be a REAL range for this zone - at a different outdoor "
        f"temperature. The Cold zone's winter average is "
        f"{HEATING_CLIMATE_ZONES['cold']['winter_avg_low']}°C, not -10°C, and four documents "
        f"derive their thresholds from the wrong one."
    )


@pytest.mark.parametrize("zone_key", sorted(ZONES))
def test_no_document_misstates_a_zones_winter_average(zone_key):
    """One constant, wrong in four places, and every threshold derived from it is wrong."""
    real = float(HEATING_CLIMATE_ZONES[zone_key]["winter_avg_low"])
    names = ZONES[zone_key][1]

    wrong = []
    for path in _markdown_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, 1):
            match = WINTER_AVG.search(line)
            if not match:
                continue
            # Attribute the claim to a zone named on this line, or on the nearest heading above it.
            zone = _zone_named_in(line)
            if zone is None:
                context = "\n".join(lines[max(0, lineno - 8) : lineno])
                zone = _zone_named_in(context)
            if zone != zone_key:
                continue
            claimed = match.group(1) or match.group(2)
            if float(claimed) != real:
                wrong.append(f"{path.relative_to(ROOT)}:{lineno} says {claimed} — {line.strip()}")

    assert not wrong, (
        f"The {zone_key} zone's winter average is {real}°C in const.py. These documents say "
        f"otherwise, and every degree-minute threshold they derive from it is wrong:\n  "
        + "\n  ".join(wrong)
    )


# ── The removed flow-temperature model, across EVERY document ────────────────────────────────
#
# The rulebook was cleaned of Kühne and given a test. The test read the rulebook. So the README
# went on advertising "André Kühne + Timbones formulas" to users, docs/architecture/10 went on
# deriving four worked examples from it, and docs/CLIMATE_ZONES went on naming it as the weather
# compensation model. Three documents, teaching a model that appears ZERO times in the codebase.
#
# A guard scoped to one file is a guard with a hole the shape of every other file.

DENIALS = (
    "used to ",
    "no longer",
    "was removed",
    "Do not reintroduce",
    "does not exist",
    "not sourced",
    "has never existed",
)


def _paragraphs_that_assert(path: Path) -> str:
    """A document's claims, minus the paragraphs that exist to warn you off something.

    Whitespace is normalised BEFORE the markers are looked for. Markdown wraps prose, so a denial
    reads "**was\nremoved**" in the file and a naive substring check for "was removed" misses it -
    which it duly did, on the one document whose whole purpose is to explain what was removed. Four
    separate holes in these guards have now come from testing a marker against un-normalised text.
    """
    paragraphs = path.read_text(encoding="utf-8").split("\n\n")
    return "\n\n".join(p for p in paragraphs if not any(d in " ".join(p.split()) for d in DENIALS))


@pytest.mark.parametrize("path", _markdown_files(), ids=lambda p: str(p.name))
def test_no_document_teaches_the_flow_temperature_model_that_was_removed(path):
    """Kühne drove the flow temperature of a real heat pump, and was taken out for being wrong.

    It was fed a heat-loss coefficient where the derivation requires a dimensionless relative load
    (audit F-119/F-121), and it is gone: the flow temperature comes from the EN 442 emitter law in
    `utils/emitter.py`.

    A document may explain what Kühne WAS, why it went, or use its curve as a REFERENCE - it is a
    pure power law with provably zero internal gains, which makes it the cleanest way to demonstrate
    that a balance point cannot be fitted to a heating curve. `docs/research/02_emitter_law.md` does
    exactly that, and that is the point of it. What a document may not do is present Kühne's formula
    as the model this project uses to set a flow temperature.
    """
    claims = _paragraphs_that_assert(path)

    assert "TFlow = 2.55" not in claims and "2.55 * (HC" not in claims, (
        f"{path.relative_to(ROOT)} presents André Kühne's flow-temperature formula as a live model. "
        f"It appears ZERO times in the codebase - it was replaced by the EN 442 emitter law. A "
        f"reader following this document builds the model this project deliberately removed. "
        f"See docs/research/02_emitter_law.md."
    )


@pytest.mark.parametrize("path", _markdown_files(), ids=lambda p: str(p.name))
def test_no_document_teaches_the_scaled_spread(path):
    """The bug the docs kept teaching for a whole commit after the code stopped doing it.

    `utils/emitter.py` holds the flow-return spread CONSTANT, because a heat pump modulates its
    circulator to maintain the commissioned spread and varies the flow rate. Scaling the spread by
    load - `spread_design * phi` - models a fixed-speed pump on a wet boiler.

    The commit that fixed the code left `docs/research/02_emitter_law.md` printing the scaled form
    in its HEADLINE equation, so anyone implementing from the research note would have rebuilt the
    bug on the spot. The error is invisible at the design point and grows in both directions from
    it, which is exactly why it needs a guard rather than a careful reader.
    """
    claims = " ".join(_paragraphs_that_assert(path).split())

    for scaled in ("spread_design · φ", "spread_design * phi", "systemDT * phi", "spread * phi"):
        assert scaled not in claims, (
            f"{path.relative_to(ROOT)} still teaches the SCALED spread ('{scaled}'). The code holds "
            f"the spread constant - a heat pump modulates its circulator. Scaling it makes the flow "
            f"temperature too cool in mild weather and too hot in cold, pivoting invisibly on the "
            f"design point. See utils/emitter.py."
        )

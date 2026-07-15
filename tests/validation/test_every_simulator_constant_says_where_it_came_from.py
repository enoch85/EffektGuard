"""Every number in the plant model must say where it came from.

The pump profiles once carried an outdoor-keyed COP curve labelled "Real-world COP curve (tested
and validated)" and sourced to "NIBE F750 datasheet, Swedish NIBE forum validation" - a template
with the digits nudged, whose numbers were in neither. A plain number with a confident comment is
indistinguishable from a measurement until someone checks, and for a year nobody did.

So every physical constant in the harness is declared as exactly one of two things:

    SOURCED   - a document, quoted, that a reader can go and open.
    ASSUMED   - no published source exists; then the sensitivity MUST be measured and stated. If
                the conclusions move when the number moves, the number is load-bearing and the
                conclusions are not trustworthy.

An ASSUMED constant is not a sin. An UNDECLARED one is. Loop counters, unit conversions and the
harness's own reporting budgets are not physical claims and are listed in NOT_A_PHYSICAL_CLAIM.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

HARNESS = pathlib.Path("scripts/simulation/sim_harness.py")

# Names that are not physical claims: loop counters, unit conversions, and the harness's own
# reporting budgets. They do not describe a heat pump, a house or a tariff, so there is nothing to
# source. Anything else must be in PROVENANCE.
NOT_A_PHYSICAL_CLAIM = frozenset(
    {
        "STEP_MIN",
        "SIM_DAYS",
        "DST_SIM_DAYS",
        "QUARTER_MINUTES",
        "J_PER_KWH",
        "KELVIN",
        "ORE_PER_KWH_FROM_SEK_PER_MWH",
        "EXERGY_FIT_PARAMETERS",
        # The harness's own pass/fail budgets. They are what the SIMULATION demands of the
        # controller, not claims about hardware, and each is argued where it is defined.
        "WATER_NODE_LEAK_BUDGET_KWH",
        "COP_ENVELOPE_TOLERANCE",
        "AUX_OVER_PHYSICS_TOLERANCE",
        "AUX_SLACK_KWH",
        "DM_AUX_MARGIN",
        "MAX_COMFORT_MINUTES_BELOW",
        "MAX_COMFORT_MINUTES_ABOVE",
        "INDOOR_CEILING",
        "COMFORT_TOLERANCE",
        "OVERSHOOT_TOLERANCE",
        "DM_INTEGRATOR_FLOOR",
        "DM_INTEGRATOR_CEILING",
        "MIN_EXERGY_EFFICIENCY",
        "MAX_EXERGY_EFFICIENCY",
        "MIN_LIFT_K",
        # The reference battery controller: a comparison strategy, not a model of anything.
        "BATTERY_BAND",
        "BATTERY_CHARGE_OFFSET",
        "BATTERY_COAST_OFFSET",
        "BATTERY_CHEAP_PERCENTILE",
        "BATTERY_DEAR_PERCENTILE",
        "TARGET_INDOOR",
        "TOMORROW_VISIBLE_HOUR",
    }
)


def _module_constants() -> dict[str, float]:
    """Every module-level numeric constant the harness defines."""
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    found: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.isupper():
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            found[target.id] = float(value.value)
        elif (
            isinstance(value, ast.UnaryOp)
            and isinstance(value.op, ast.USub)
            and isinstance(value.operand, ast.Constant)
        ):
            found[target.id] = -float(value.operand.value)
    return found


def _provenance() -> dict[str, str]:
    """The PROVENANCE table the harness declares."""
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    for node in tree.body:
        # `PROVENANCE: dict[str, str] = {...}` is an AnnAssign, not an Assign - handle both, or a
        # walker that looks only for Assign finds nothing and reports every constant as undeclared.
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "PROVENANCE":
            target = node.value
        elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "PROVENANCE":
            target = node.value
        else:
            continue
        if isinstance(target, ast.Dict):
            return {
                key.value: value.value
                for key, value in zip(target.keys, target.values)
                if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
            }
    return {}


def test_the_harness_declares_a_provenance_table():
    assert _provenance(), (
        "scripts/simulation/sim_harness.py has no PROVENANCE table. Every number that describes a "
        "heat pump, a house or a tariff must say where it came from - a document, or an explicit "
        "admission that there is none and a measurement of what the answer costs if it is wrong."
    )


@pytest.mark.parametrize("name", sorted(set(_module_constants()) - NOT_A_PHYSICAL_CLAIM))
def test_every_physical_constant_says_where_it_came_from(name):
    """A number with a confident comment and no source is indistinguishable from a measurement."""
    provenance = _provenance()

    assert name in provenance, (
        f"{name} is a physical claim in the plant model and it does not say where it came from. "
        f"Add it to PROVENANCE with either a document you can quote, or the word ASSUMED and the "
        f"measured sensitivity of the conclusions to it. The last time a number like this went "
        f"unchecked, the simulator derated a heat pump in the wrong direction and cited EN 14511 "
        f"for it, and every finding built on that was wrong. If {name} is not a physical claim, "
        f"say so by listing it in NOT_A_PHYSICAL_CLAIM - deliberately, in a diff someone reviews."
    )


@pytest.mark.parametrize("name", sorted(_provenance()))
def test_a_sourced_constant_quotes_a_document_and_an_assumed_one_admits_it(name):
    """The two are not interchangeable, and the difference is the whole point of the table."""
    claim = _provenance()[name]

    if claim.startswith("ASSUMED"):
        assert "sensitivity" in claim.lower(), (
            f"{name} is ASSUMED, which is allowed - not every number has a published source. But "
            f"then the conclusions must be shown NOT to depend on it: state the measured "
            f"sensitivity. An unsourced number that moves the answer is a finding about the "
            f"modeller, not about the heat pump."
        )
        return

    assert claim.startswith("SOURCED"), (
        f"{name}'s provenance reads {claim!r}. It must begin with SOURCED (and quote the document) "
        f"or ASSUMED (and state the measured sensitivity). There is no third kind."
    )
    # A SOURCED claim must name a REFERENCE, not merely use the word "datasheet" - a bare word can
    # sit in a sentence that says the opposite ("No datasheet publishes it"). A reference is a URL,
    # a numbered standard, a NIBE document code, a part number, or a docs/research file.
    references = (
        r"https?://",
        r"\bEN \d{3,5}\b",  # EN 442, EN 1264, EN 14511, EN 14825
        r"\bISO \d{3,5}\b",
        r"\b(IHB|UHB)\b",  # NIBE installer / user handbook codes
        r"part no",
        r"docs/research/",
    )

    assert any(re.search(pattern, claim) for pattern in references), (
        f"{name} claims to be SOURCED but names no reference: {claim!r}. A reference is a URL, a "
        f"numbered standard, a NIBE document code, a part number, or a docs/research note. "
        f"'Swedish NIBE forum validation' was the last thing that passed for a source here, and "
        f"the numbers it justified were in no forum and no datasheet."
    )


def test_no_constant_is_declared_that_does_not_exist():
    """A provenance table that outlives its constants is a table nobody is reading."""
    stale = sorted(set(_provenance()) - set(_module_constants()))

    assert not stale, (
        f"PROVENANCE declares {stale}, which the harness no longer defines. A stale entry is worse "
        f"than none: it says a number was checked when the number is gone."
    )

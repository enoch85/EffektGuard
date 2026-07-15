"""The research must stay true, or it becomes what it replaced.

`docs/research/` exists because the code cited fifteen research documents that were all absent from
the repository, so the binding rule "never guess NIBE behaviour, verify against research" could not
be obeyed by anyone who cloned it. Sourced citations only help if they stay true: a research note
that has drifted from the code is worse than none, because it looks settled.

So the documents are PARSED, not remembered. `NAME = value` in the prose, the net-gain table in 04,
the worked example in 02 - all read out of the markdown and checked against the code that runs. A
digit changed in either place fails here, which is the only arrangement under which "the research
still holds" means anything. These are not the derivations - those live in the documents, with
their sources; this is the part a machine can hold you to.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from custom_components.effektguard import const
from custom_components.effektguard.const import DEFAULT_HEAT_LOSS_COEFFICIENT, INTERNAL_GAINS_W
from custom_components.effektguard.optimization.airflow_optimizer import calculate_net_thermal_gain
from custom_components.effektguard.utils.emitter import en442_flow_temp

RESEARCH = Path(__file__).resolve().parents[2] / "docs" / "research"

# The documents are typeset, not code: they use the Unicode MINUS SIGN and bold the numbers they
# want you to look at. A parser that does not know that reads "−0.31" as a string and quietly
# matches nothing, which is the same failure as not reading them at all.
MINUS_SIGNS = str.maketrans({"−": "-", "–": "-", "—": "-"})


def _text(name: str) -> str:
    return (RESEARCH / name).read_text(encoding="utf-8").translate(MINUS_SIGNS)


def _all_research_text() -> list[tuple[str, str]]:
    return [(p.name, _text(p.name)) for p in sorted(RESEARCH.glob("*.md"))]


def _constants_cited_in_the_research() -> list[tuple[str, str, float]]:
    """Every `NAME = value` in the prose, where NAME is a real constant. Read, not remembered."""
    cited = []
    for filename, text in _all_research_text():
        for name, value in re.findall(
            r"\b([A-Z][A-Z0-9_]{3,})\s*=\s*(-?[0-9]+(?:\.[0-9]+)?)", text
        ):
            if hasattr(const, name):
                cited.append((filename, name, float(value)))
    return sorted(set(cited))


def _constants_declared_in_code_fences() -> list[tuple[str, str, float]]:
    """Every `NAME = value` inside a fenced code block - a DECLARATION, not a mention.

    Prose may lawfully name a constant that no longer exists ("DEFAULT_BALANCE_POINT_OFFSET ... is
    gone"). A fenced ```python block reads as the code the document is deriving, so a name there
    that const.py does not have is a promise the codebase is not keeping - and the hasattr filter
    used elsewhere would silently skip it.
    """
    declared = []
    for filename, text in _all_research_text():
        for fence in re.findall(r"```[a-z]*\n(.*?)```", text, flags=re.DOTALL):
            for name, value in re.findall(
                r"^\s*([A-Z][A-Z0-9_]{3,})\s*=\s*(-?[0-9]+(?:\.[0-9]+)?)", fence, flags=re.M
            ):
                declared.append((filename, name, float(value)))
    return sorted(set(declared))


def test_no_fenced_declaration_names_a_constant_the_code_does_not_have():
    declared = _constants_declared_in_code_fences()
    assert declared, "the fence parser matched nothing - it can no longer catch anything either"

    phantoms = [(f, n, v) for f, n, v in declared if not hasattr(const, n)]
    assert not phantoms, (
        f"docs/research declares constants the code does not have: {phantoms}. A reader takes a "
        f"fenced declaration as fact; if the constant was renamed or the change never landed, "
        f"the document must say so in prose instead of declaring it."
    )


def _net_gain_table() -> list[tuple[float, float]]:
    """The `| outdoor | net gain |` table in 04. `| +10 °C | **+0.03 kW** |` -> (10.0, 0.03)."""
    rows = re.findall(
        r"\|\s*\*{0,2}([+-]?[0-9.]+)\s*°C\*{0,2}\s*\|\s*\*{0,2}([+-]?[0-9.]+)\s*kW\*{0,2}\s*\|",
        _text("04_exhaust_air_recovery.md"),
    )
    return [(float(outdoor), float(gain)) for outdoor, gain in rows]


class TestTheDocumentsAreActuallyRead:
    """A parser that matches nothing is indistinguishable from the dict it replaced."""

    def test_the_research_really_does_cite_constants_by_name(self):
        cited = _constants_cited_in_the_research()

        assert len(cited) >= 6, (
            f"Only {len(cited)} constants were parsed out of docs/research/: "
            f"{[c[1] for c in cited]}. Every test below is parametrised over this list, so if the "
            f"parser stops matching, the whole file silently passes and checks nothing."
        )

    def test_the_net_gain_table_is_really_parsed(self):
        """The net-gain table uses a Unicode minus AND a leading `+` on positive rows.

        A regex that handles neither reads only a subset - and the rows it drops are the positive
        ones, the only rows where the feature looks GOOD - so the row count is asserted, not assumed.
        """
        table = _net_gain_table()

        assert len(table) == 6, (
            f"Parsed {len(table)} rows from the net-gain table in 04: {table}. The document prints "
            f"six. A parser that quietly matches a subset is the same failure as not reading the "
            f"document at all."
        )
        assert any(gain > 0 for _, gain in table), "the +10 C row is the one that shows a gain"
        assert sum(1 for _, gain in table if gain < 0) == 5, "and five rows show a LOSS"

    def test_a_corrupted_document_would_be_caught(self, tmp_path):
        """The guard on the guard. Prove the parser can see a wrong number, on a fake document."""
        doc = tmp_path / "fake.md"
        doc.write_text("`DM_THRESHOLD_START = -99` is this number.\n", encoding="utf-8")

        found = re.findall(
            r"\b([A-Z][A-Z0-9_]{3,})\s*=\s*(-?[0-9]+(?:\.[0-9]+)?)",
            doc.read_text(encoding="utf-8"),
        )

        assert found == [("DM_THRESHOLD_START", "-99")]
        assert (
            float(found[0][1]) != const.DM_THRESHOLD_START
        ), "a document quoting the wrong value must not compare equal to the code"


@pytest.mark.parametrize(
    "filename,name,quoted",
    _constants_cited_in_the_research(),
    ids=lambda v: str(v) if not isinstance(v, float) else f"{v:g}",
)
def test_research_quotes_the_constant_the_code_actually_holds(filename, name, quoted):
    """A citation that no longer matches the code is a citation that misleads.

    The value is read from the markdown. Change the digit in the document OR retune the constant
    without revisiting the evidence, and this fails - which is the whole point of the directory.
    """
    actual = getattr(const, name)

    assert float(actual) == quoted, (
        f"docs/research/{filename} quotes {name} = {quoted!r}; const.py holds {actual!r}. Either "
        f"the constant was retuned without revisiting the evidence for it, or the note is wrong. "
        f"Both matter: this directory exists so that these numbers can be checked."
    )


@pytest.mark.parametrize("outdoor,quoted_gain", _net_gain_table())
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
    """02_emitter_law.md shows a code block and prints its result. Run it, against ITS number.

    This anchors the whole flow-temperature model: NIBE's published curve 9 reads 41.0 C at 0 C
    outdoor, our law lands ~0.64 C above it, and that gap is the TRIM, not an error - NIBE
    interpolates its curves linearly, we follow EN 442. The expected value is read OUT of the doc's
    comparison table rather than copied from it.
    """
    table = _text("02_emitter_law.md")
    row = re.search(r"\|\s*EN 442[^|]*\|\s*([0-9.]+)\s*°C\s*\|", table)

    assert row, (
        "02_emitter_law.md no longer prints an 'EN 442 + derived gains' row in its comparison "
        "table. This test reads its expected value from that row, so without it the test is "
        "checking nothing."
    )
    doc_says = float(row.group(1))

    flow = en442_flow_temp(
        indoor_setpoint=21.0,
        outdoor_temp=0.0,
        design_outdoor_temp=-15.0,
        design_flow_temp=52.6,
        design_spread=5.0,
        emitter_exponent=1.3,
        balance_point_temp=21.0 - INTERNAL_GAINS_W / DEFAULT_HEAT_LOSS_COEFFICIENT,
    )

    assert flow == pytest.approx(doc_says, abs=0.01), (
        f"The worked example in 02_emitter_law.md says this call returns {doc_says}; it returns "
        f"{flow:.2f}. A research note whose own code block does not run is exactly the kind of "
        f"citation this directory was created to replace."
    )
    assert abs(flow - 41.0) < 1.0, (
        f"The emitter law gives {flow:.2f} C where NIBE's own published curve 9 gives 41.0 C. We "
        f"TRIM that curve, so a gap is expected - but a large one would mean the design point, the "
        f"spread or the gains are misconfigured, and every offset we emit would be a correction "
        f"toward our own error."
    )


def test_every_research_note_is_indexed():
    """A note nobody can find is a note nobody will maintain."""
    index = (RESEARCH / "README.md").read_text(encoding="utf-8")
    notes = sorted(p.name for p in RESEARCH.glob("*.md") if p.name != "README.md")

    missing = [n for n in notes if n not in index]

    assert not missing, f"docs/research/README.md does not link: {', '.join(missing)}"

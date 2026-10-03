"""The document a maintainer opens to ask "what DM is normal here?" must match the code.

`docs/CLIMATE_ZONES.md` is the reference for the most safety-critical question in the project: at a
given zone and outdoor temperature, what degree-minute range is normal. Every one of its DM rows
must be exactly what ClimateZoneDetector computes - a good number attached to the wrong outdoor
temperature is still a wrong claim, and that is how the tables drifted (a Cold-zone winter average
of -8.0 C, once quoted as -10.0, moves every threshold derived from it).

So this parses the DM tables straight out of the markdown and asks the real detector what it would
say. A maintainer who tunes a threshold in const.py and leaves the document behind gets a failing
test naming the row.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector

DOC = Path(__file__).resolve().parents[2] / "docs" / "CLIMATE_ZONES.md"

# A latitude that lands squarely inside each zone, to ask the detector with.
ZONE_LATITUDE = {
    "Extreme Cold": 67.86,  # Kiruna
    "Very Cold": 65.58,  # Luleå
    "Cold": 59.33,  # Stockholm
    "Moderate Cold": 55.60,  # Malmö
    "Standard": 48.86,  # Paris
}

ROW = re.compile(r"^\|\s*(-?\d+)°C\s*\|\s*(-?\d+)\s+to\s+(-?\d+)\s*\|\s*(-?\d+)\s*\|", re.M)


def _documented_rows() -> list[tuple[str, int, int, int, int]]:
    """Every DM row in the document, tagged with the zone whose section it sits in."""
    text = DOC.read_text(encoding="utf-8")
    rows: list[tuple[str, int, int, int, int]] = []
    zone: str | None = None

    for line in text.splitlines():
        heading = re.match(r"^###\s+\S*\s*(.+?)\s+Zone\b", line)
        if heading:
            zone = heading.group(1).strip()
            continue
        match = ROW.match(line)
        if match and zone in ZONE_LATITUDE:
            outdoor, low, high, warning = (int(g) for g in match.groups())
            rows.append((zone, outdoor, low, high, warning))

    return rows


def test_the_document_actually_has_tables_to_check():
    """A parser that silently matches nothing would make every assertion below vacuous."""
    rows = _documented_rows()

    assert len(rows) >= 17, (
        f"Only {len(rows)} DM rows were parsed out of {DOC.name}. The tables were reformatted or "
        f"removed, and this test has quietly stopped checking anything."
    )
    assert {zone for zone, *_ in rows} == set(
        ZONE_LATITUDE
    ), "Every climate zone must have a DM table in the document."


@pytest.mark.parametrize("zone,outdoor,low,high,warning", _documented_rows())
def test_each_documented_dm_row_is_what_the_code_computes(zone, outdoor, low, high, warning):
    """The number a maintainer reads must be the number the heat pump gets."""
    detector = ClimateZoneDetector(latitude=ZONE_LATITUDE[zone])
    actual = detector.get_expected_dm_range(float(outdoor))

    documented = (low, high, warning)
    computed = (
        round(actual["normal_min"]),
        round(actual["normal_max"]),
        round(actual["warning"]),
    )

    assert documented == computed, (
        f"{zone} at {outdoor}°C: the document says normal {low} to {high}, warning {warning}. "
        f"ClimateZoneDetector actually gives normal {computed[0]} to {computed[1]}, warning "
        f"{computed[2]}. This is the table a maintainer consults to decide whether a degree-minute "
        f"reading is safe."
    )


def test_the_adjustment_formula_is_stated_with_the_right_sign():
    """The document must state the adjustment in the direction the code computes it.

    `adjustment = (outdoor_temp - zone_avg_winter_low) x 20`: colder than the zone average is a
    NEGATIVE delta and a DEEPER threshold. The inverted form yields the opposite sign.
    """
    text = DOC.read_text(encoding="utf-8")

    assert "(outdoor_temp - zone_avg_winter_low)" in text, (
        "CLIMATE_ZONES.md must state the adjustment formula in the direction the code computes it: "
        "adjustment = (outdoor_temp - zone_avg_winter_low) × 20. Colder than the zone average is a "
        "NEGATIVE delta and a DEEPER threshold."
    )
    assert (
        "(zone_avg_winter_low - outdoor_temp)" not in text
    ), "The inverted form of the formula is back in the document."

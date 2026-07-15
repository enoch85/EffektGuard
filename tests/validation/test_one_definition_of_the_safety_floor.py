"""One definition of the most safety-critical number in the project.

DM -1500 is the absolute degree-minute floor: the reading at which an absolute emergency is
declared. It must have a SINGLE source, or its copies drift apart and disagree about when the
house is in danger. This guard holds four things together:

  - const.py defines DM_THRESHOLD_AUX_LIMIT = -1500 exactly once (the only literal permitted);
  - climate_zones publishes it as `critical`, and get_expected_dm_range()["critical"] must be the
    SAME object as the emergency tier's DM_THRESHOLD_AUX_LIMIT, not merely equal to it;
  - the simulator reads the aux limit from the pump profile, so the profile must REFERENCE the
    constant, not restate a literal - else a change to the constant leaves the plant validating
    against the old threshold;
  - there is one latitude-to-climate classification, not two.

The number itself may yet change - F-112 is open with the owner: on an F750 the pump's own "start
addition" fires at -700 and works DM back up, so -1500 describes a regime a healthy pump never
enters. When it changes, everything above must move with it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from custom_components.effektguard import const
from custom_components.effektguard.optimization import climate_zones

COMPONENT = Path(__file__).resolve().parents[2] / "custom_components" / "effektguard"

# A literal -1500 assigned to a name, anywhere in production code.
LITERAL = re.compile(r"^\s*(\w+)\s*(?::[^=]+)?=\s*-1500\b", re.M)


def _definitions() -> list[tuple[Path, str]]:
    found = []
    for path in sorted(COMPONENT.rglob("*.py")):
        for name in LITERAL.findall(path.read_text(encoding="utf-8")):
            found.append((path, name))
    return found


def test_the_absolute_degree_minute_floor_is_defined_exactly_once():
    """Two live definitions of the same number cannot be kept equal by hoping."""
    definitions = _definitions()

    assert len(definitions) == 1, (
        "The absolute degree-minute floor (-1500) is defined "
        f"{len(definitions)} times:\n  "
        + "\n  ".join(f"{p.relative_to(COMPONENT)}: {name} = -1500" for p, name in definitions)
        + "\n\nIt is one physical quantity: the DM at which an absolute emergency is declared. "
        "thermal_layer tests against DM_THRESHOLD_AUX_LIMIT; get_expected_dm_range() publishes "
        "DM_ABSOLUTE_MAXIMUM as `critical`. Change one - as F-112 may require - and the other "
        "silently disagrees about when the house is in danger."
    )


def test_the_published_critical_threshold_is_the_emergency_trigger_itself():
    """Not merely equal today. The same object.

    `get_expected_dm_range()` publishes a `critical` threshold to every consumer, and
    `thermal_layer` fires the EMERGENCY tier on `DM_THRESHOLD_AUX_LIMIT`. These are one quantity.
    Asserting identity, not equality, is the point: two constants holding -1500 are equal today and
    that is exactly the state this test exists to forbid.
    """
    published = climate_zones.ClimateZoneDetector(latitude=59.33).get_expected_dm_range(-10.0)

    assert published["critical"] is const.DM_THRESHOLD_AUX_LIMIT, (
        f"get_expected_dm_range() publishes critical={published['critical']!r}, which is not the "
        f"same object as const.DM_THRESHOLD_AUX_LIMIT={const.DM_THRESHOLD_AUX_LIMIT!r}. The "
        f"emergency tier and the published critical threshold must move together, or they will "
        f"disagree about when the house is in danger."
    )


def test_the_simulator_validates_against_the_threshold_production_actually_uses():
    """The simulator reads the profile. The profile must not restate the number.

    This is the one that would bite hardest. The simulator is what validates a change to the aux
    limit - and it takes the limit from the heat-pump profile, deliberately, so that "the plant
    model tracks whatever the integration believes". If the profile carries its own literal, the
    plant does NOT track the integration: change the constant, and the simulator goes on modelling
    the old threshold and pronounces the new behaviour safe against a plant that never sees it.
    """
    from custom_components.effektguard.models.nibe import NibeF750Profile

    profile = NibeF750Profile()

    # Value equality is NOT the assertion. Both are -1500 today, and a test that checks only that
    # passes by coincidence - which is the entire defect. It has to REFERENCE the constant.
    for module in ("models/base.py", "models/nibe/f750.py"):
        source = (COMPONENT / module).read_text(encoding="utf-8")
        declaration = next(
            (ln for ln in source.splitlines() if "dm_threshold_aux_swedish" in ln and "=" in ln),
            None,
        )
        if declaration is None:
            continue

        assert "DM_THRESHOLD_AUX_LIMIT" in declaration, (
            f"{module} declares dm_threshold_aux_swedish with a literal:\n"
            f"    {declaration.strip()}\n"
            f"The simulator reads this field so the plant tracks what the integration believes. "
            f"A literal cannot track anything. It must reference DM_THRESHOLD_AUX_LIMIT."
        )

    assert profile.dm_threshold_aux_swedish == const.DM_THRESHOLD_AUX_LIMIT, (
        f"The F750 profile's aux threshold ({profile.dm_threshold_aux_swedish}) is not "
        f"DM_THRESHOLD_AUX_LIMIT ({const.DM_THRESHOLD_AUX_LIMIT})."
    )


def test_there_is_one_latitude_to_climate_classification_not_two():
    """The coordinator has its own latitude bands, and nothing reads the result.

    `_detect_climate_region()` maps latitude to CLIMATE_SOUTHERN_SWEDEN / CENTRAL / MID_NORTHERN /
    NORTHERN / LAPLAND on boundaries of 58 / 62 / 65 / 67. `ClimateZoneDetector` maps the SAME
    latitude to a climate zone on boundaries of 54.5 / 56 / 60.5 / 66.5, and that one actually
    drives the degree-minute thresholds.

    Two answers to "what climate is this house in", from one latitude, with different boundaries -
    and the dead one has eleven tests, which test only each other.
    """
    coordinator_source = (COMPONENT / "coordinator.py").read_text(encoding="utf-8")

    assert "_detect_climate_region" not in coordinator_source, (
        "coordinator._detect_climate_region() is a SECOND latitude-to-climate classification, with "
        "different boundaries from ClimateZoneDetector, and its result (self.climate_region) is "
        "read by nothing in production. A maintainer could wire it up believing it is the real "
        "one. There must be one answer to what climate a house is in."
    )

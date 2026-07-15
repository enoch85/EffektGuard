"""The simulated pump engages additive heat where the REAL pump does - not at -1500.

NIBE ships every supported machine with its additive heat armed far above EffektGuard's
absolute floor: the F750/F730 "start addition" defaults to -700 (IHB GB 1301-1, menu 4.9.3),
the S1155/F1155 controllers to about -460, the VVM 320 that pairs with an F2040 to about
-760. On a healthy pump DM asymptotes AT the start-addition value, because the elpatron
engages there and works it back up.

Waiting for EffektGuard's own -1500 floor instead under-fires the elpatron - 800 degree-minutes
late for an F750 - so cold-snap aux and overshoot get computed against a machine no factory ships.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "simulation"))

from sim_harness import HOUSES  # noqa: E402

from custom_components.effektguard.const import DM_THRESHOLD_AUX_LIMIT  # noqa: E402


def test_every_house_fires_aux_at_its_pumps_own_start_addition():
    for house in HOUSES:
        assert house.aux_start_dm == house.profile.aux_start_dm, house.name


def test_the_hardware_start_addition_is_not_effektguards_floor():
    """The two numbers are different FACTS: confusing them is audit finding F-112."""
    for house in HOUSES:
        assert house.aux_start_dm > DM_THRESHOLD_AUX_LIMIT, (
            f"{house.name}: the plant arms additive heat at {house.aux_start_dm}, at or below "
            f"EffektGuard's absolute floor ({DM_THRESHOLD_AUX_LIMIT}). No factory-default NIBE "
            f"waits that long - the elpatron is part of the machine being simulated."
        )


def test_the_factory_defaults_match_the_installer_manuals():
    expected = {
        "wooden_f750": -700.0,
        "apartment_f730": -700.0,
        "concrete_f1155": -460.0,
        "villa_s1155": -460.0,
        "airsource_f2040": -760.0,
    }
    for house in HOUSES:
        assert house.aux_start_dm == expected[house.name], house.name

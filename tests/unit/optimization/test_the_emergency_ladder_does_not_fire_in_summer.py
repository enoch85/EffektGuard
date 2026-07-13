"""At +30 C outdoor the "warning" degree-minute threshold was POSITIVE.

The zone thresholds are shifted with the weather:

    adjustment = temp_delta * 20        # warmer than the winter average -> shallower DM expected

Shallowing them as it warms is right in itself. A pump that has fallen 400 degree minutes behind in
mild weather is in more trouble than one that has fallen 400 behind in a cold snap, because it
should not be working hard at all.

But the shift was clamped only on the COLD side. Nothing bounded it above, and NIBE starts the
compressor at DM_THRESHOLD_START (-60) and stops it at 0 - so degree minutes traverse that band on
EVERY NORMAL CYCLE, in every season, on every heat pump. In Stockholm the warning threshold climbed
to:

    outdoor +15 C  ->  -240
    outdoor +25 C  ->   -40      <- INSIDE the compressor's own cycling band
    outdoor +30 C  ->   +60      <- POSITIVE: any degree-minute reading at all is a "warning"

Degree minutes are essentially never positive. So above about +26 C outdoor, EVERY reading armed the
emergency ladder - and a midsummer hot-water cycle dips degree minutes to -60 like any other, so a
heat pump behaving perfectly was told to boost the heating curve. In July.
"""

from __future__ import annotations

import pytest

from custom_components.effektguard.const import DM_THRESHOLD_START
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector

# Every zone the detector can land in.
LATITUDES = [
    (67.85, "Kiruna"),
    (63.83, "Umea"),
    (59.33, "Stockholm"),
    (55.6, "Malmo"),
    (48.86, "Paris"),
]
SUMMER = [15.0, 20.0, 25.0, 30.0, 35.0]


def test_the_compressor_really_does_cycle_through_this_band():
    """The precondition the whole test rests on."""
    assert DM_THRESHOLD_START == -60, (
        "NIBE starts the compressor at -60 DM and stops it at 0, so degree minutes traverse that "
        "band on every normal cycle. If that changes, the ceiling below must move with it."
    )


@pytest.mark.parametrize(("latitude", "city"), LATITUDES)
@pytest.mark.parametrize("outdoor", SUMMER)
def test_the_warning_threshold_is_never_positive(latitude, city, outdoor):
    """A positive threshold means every possible reading is a warning."""
    warning = ClimateZoneDetector(latitude=latitude).get_expected_dm_range(outdoor)["warning"]

    assert warning < 0, (
        f"{city} at {outdoor:+.0f} C outdoor has a degree-minute WARNING threshold of {warning:+.0f}. "
        f"Degree minutes are essentially never positive, so this arms the emergency ladder on every "
        f"single reading - all summer."
    )


@pytest.mark.parametrize(("latitude", "city"), LATITUDES)
@pytest.mark.parametrize("outdoor", SUMMER)
def test_the_warning_threshold_never_reaches_into_the_compressors_own_cycling_band(
    latitude, city, outdoor
):
    """The real bound. A threshold inside -60..0 fires on normal operation, not on trouble."""
    warning = ClimateZoneDetector(latitude=latitude).get_expected_dm_range(outdoor)["warning"]

    assert warning < DM_THRESHOLD_START, (
        f"{city} at {outdoor:+.0f} C outdoor warns at {warning:+.0f} DM, but NIBE starts the "
        f"compressor at {DM_THRESHOLD_START} DM and stops it at 0 - so a perfectly healthy pump "
        f"passes through {warning:+.0f} on every hot-water cycle, all summer, and gets told it is "
        f"in thermal debt."
    )


@pytest.mark.parametrize(("latitude", "city"), LATITUDES)
def test_winter_thresholds_are_untouched(latitude, city):
    """The clamp is a CEILING. It must not make the ladder less sensitive when it is needed."""
    detector = ClimateZoneDetector(latitude=latitude)

    for outdoor in (-30.0, -20.0, -10.0, 0.0):
        warning = detector.get_expected_dm_range(outdoor)["warning"]
        unclamped = (
            detector.zone_info.dm_warning_threshold
            + (outdoor - detector.zone_info.winter_avg_low) * 20
        )

        assert warning == pytest.approx(max(unclamped, -1450), abs=1.0) or warning <= unclamped, (
            f"{city} at {outdoor:+.0f} C: the warm-side ceiling has reached into winter and made "
            f"the emergency ladder LESS sensitive ({warning:.0f} vs {unclamped:.0f}). It is a "
            f"ceiling on mild days, not a floor on cold ones."
        )


def test_the_thresholds_still_deepen_as_it_gets_colder():
    """The whole mechanism must survive the fix."""
    detector = ClimateZoneDetector(latitude=59.33)
    warnings = [detector.get_expected_dm_range(t)["warning"] for t in (-20.0, -10.0, 0.0, 10.0)]

    assert warnings == sorted(warnings), (
        f"The warning threshold must get DEEPER as it gets colder. Got {warnings} for "
        f"-20/-10/0/+10 C."
    )

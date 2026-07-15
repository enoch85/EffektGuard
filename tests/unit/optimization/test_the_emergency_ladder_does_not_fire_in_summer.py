"""No degree-minute warning threshold may reach into the compressor's own cycling band.

The zone thresholds are shifted shallower as it warms (`adjustment = temp_delta * 20`). NIBE starts
the compressor at DM_THRESHOLD_START (-60) and stops it at 0, so degree minutes traverse that band on
every normal cycle. Unbounded above, the Stockholm warning threshold climbed to -40 at +25 C and +60
at +30 C - so in summer a healthy pump's ordinary compressor start armed the emergency ladder.

The clamp must hold on the number the layers actually READ - AFTER `apply_thermal_mass_buffer`, which
divides by up to 1.3 (a clamp at -110 becomes -85, back inside the band). So these tests drive the
real layers with every heating_type, and check the warm-side ceiling never touches a winter threshold.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from custom_components.effektguard.const import (
    DM_THERMAL_MASS_BUFFER_CONCRETE,
    DM_THERMAL_MASS_BUFFER_RADIATOR,
    DM_THERMAL_MASS_BUFFER_TIMBER,
    DM_THRESHOLD_START,
    DM_WARNING_BUFFER,
)
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.thermal_layer import (
    EmergencyLayer,
    apply_thermal_mass_buffer,
)

# Every zone the detector can land in.
LATITUDES = [
    (67.85, "Kiruna"),
    (63.83, "Umea"),
    (59.33, "Stockholm"),
    (55.6, "Malmo"),
    (48.86, "Paris"),
]
SUMMER = [15.0, 20.0, 25.0, 30.0, 35.0]

# Every emitter the buffer knows about. The multiplier is what makes them differ, and it is the
# multiplier that undid the clamp - so a test that does not vary this cannot see the bug.
MULTIPLIERS = {
    "radiator": DM_THERMAL_MASS_BUFFER_RADIATOR,
    "concrete_ufh": DM_THERMAL_MASS_BUFFER_CONCRETE,
    "concrete_slab": DM_THERMAL_MASS_BUFFER_CONCRETE,
    "timber": DM_THERMAL_MASS_BUFFER_TIMBER,
    "timber_ufh": DM_THERMAL_MASS_BUFFER_TIMBER,
}
HEATING_TYPES = list(MULTIPLIERS)

TARGET = 22.0
TOLERANCE = 1.0


class _HealthyPumpOnASummerMorning:
    """Nothing wrong here. The compressor has just started, so DM has dipped past its start point.

    Indoor is a fraction under target - which is ordinary, and is what stops the layer abstaining
    outright - and the pump is answering it. This is the state that must NOT be called an emergency.
    """

    supply_temp = 30.0
    return_temp = 27.0
    current_offset = 0.0
    is_heating = True
    is_hot_water = False
    compressor_frequency = 40.0
    hot_water_temp = 50.0

    def __init__(self, outdoor: float, degree_minutes: float):
        self.outdoor_temp = outdoor
        self.indoor_temp = TARGET - 0.2
        self.degree_minutes = degree_minutes


def _thresholds_the_layers_actually_read(latitude: float, outdoor: float, heating_type: str):
    """The full production path: zone -> weather shift -> clamp -> thermal-mass buffer."""
    base = ClimateZoneDetector(latitude=latitude).get_expected_dm_range(outdoor)
    return apply_thermal_mass_buffer(base, heating_type)


def test_the_compressor_really_does_cycle_through_this_band():
    """The precondition the whole file rests on."""
    assert DM_THRESHOLD_START == -60, (
        "NIBE starts the compressor at -60 DM and stops it at 0, so degree minutes traverse that "
        "band on every normal cycle. If that changes, the ceiling below must move with it."
    )


@pytest.mark.parametrize(("latitude", "city"), LATITUDES)
@pytest.mark.parametrize("outdoor", SUMMER)
@pytest.mark.parametrize("heating_type", HEATING_TYPES)
def test_the_warning_threshold_never_reaches_into_the_compressors_own_cycling_band(
    latitude, city, outdoor, heating_type
):
    """The real bound, on the real number. A threshold inside -60..0 fires on normal operation.

    This is asserted AFTER the thermal-mass buffer, because that is the last thing that changes it
    and it is what every layer reads. Asserting it before the divide is what let a concrete slab
    warn at -85 while this file was green.
    """
    warning = _thresholds_the_layers_actually_read(latitude, outdoor, heating_type)["warning"]

    assert warning <= DM_THRESHOLD_START - DM_WARNING_BUFFER, (
        f"{city}, {heating_type}, at {outdoor:+.0f} C outdoor warns at {warning:+.0f} DM. NIBE "
        f"starts the compressor at {DM_THRESHOLD_START} DM and stops it at 0, and degree minutes "
        f"undershoot the start point while the pump ramps - so a perfectly healthy pump passes "
        f"through {warning:+.0f} on every cycle, all summer, and is told it is in thermal debt."
    )


@pytest.mark.parametrize(("latitude", "city"), LATITUDES)
@pytest.mark.parametrize("outdoor", SUMMER)
@pytest.mark.parametrize("heating_type", HEATING_TYPES)
def test_the_normal_band_does_not_end_inside_it_either(latitude, city, outdoor, heating_type):
    """`normal_max` is the deep end of "normal", and the proactive tiers trigger off it too."""
    normal_max = _thresholds_the_layers_actually_read(latitude, outdoor, heating_type)["normal_max"]

    assert normal_max <= DM_THRESHOLD_START - DM_WARNING_BUFFER, (
        f"{city}, {heating_type}, at {outdoor:+.0f} C outdoor calls DM {normal_max:+.0f} the deep "
        f"end of normal, which is inside the band the compressor cycles through by itself."
    )


@pytest.mark.parametrize("heating_type", HEATING_TYPES)
def test_a_healthy_pump_in_july_is_not_given_a_curve_boost(heating_type):
    """A healthy summer compressor start must draw no emergency curve boost from any emitter.

    The layer is driven for real, with the heating_type set, at the degree minutes an ordinary
    summer compressor start produces.
    """
    layer = EmergencyLayer(
        climate_detector=ClimateZoneDetector(latitude=59.33), heating_type=heating_type
    )
    pump = _HealthyPumpOnASummerMorning(outdoor=25.0, degree_minutes=-85.0)
    now = datetime(2026, 7, 13, 6, 0, tzinfo=timezone.utc)

    decision = layer.evaluate_layer(pump, None, None, TARGET, TOLERANCE, lambda: now, False)

    assert decision.weight == 0.0 and decision.offset == 0.0, (
        f"A {heating_type} house at {pump.indoor_temp} C ({TARGET - pump.indoor_temp:.1f} C under "
        f"target) on a +{pump.outdoor_temp:.0f} C July morning, with degree minutes at "
        f"{pump.degree_minutes:+.0f} because the compressor has just started, is commanded "
        f"{decision.offset:+.1f} C of curve offset at weight {decision.weight:.2f}. Reason: "
        f"{decision.reason!r}. There is nothing wrong with this heat pump."
    )


@pytest.mark.parametrize("heating_type", HEATING_TYPES)
def test_a_pump_in_real_thermal_debt_in_winter_still_gets_help(heating_type):
    """The regression guard, and the more important half. The ceiling must not sedate the ladder.

    -30 C, the house losing ground, degree minutes far past anything the zone calls normal. Every
    emitter must still answer, or the clamp has traded a July false alarm for a January failure.
    """
    layer = EmergencyLayer(
        climate_detector=ClimateZoneDetector(latitude=59.33), heating_type=heating_type
    )
    pump = _HealthyPumpOnASummerMorning(outdoor=-30.0, degree_minutes=-1300.0)
    pump.indoor_temp = 19.5  # well below the comfort band
    pump.supply_temp = 55.0
    now = datetime(2026, 1, 13, 6, 0, tzinfo=timezone.utc)

    decision = layer.evaluate_layer(pump, None, None, TARGET, TOLERANCE, lambda: now, False)

    assert decision.offset > 0 and decision.weight > 0, (
        f"A {heating_type} house at 19.5 C in a -30 C snap, {pump.degree_minutes:+.0f} degree "
        f"minutes in debt, is offered {decision.offset:+.1f} C at weight {decision.weight:.2f}. "
        f"The warm-side ceiling is a ceiling on mild days, never a floor on cold ones."
    )


@pytest.mark.parametrize(("latitude", "city"), LATITUDES)
@pytest.mark.parametrize("heating_type", HEATING_TYPES)
def test_the_ceiling_is_inert_in_winter(latitude, city, heating_type):
    """The warm-side ceiling must not touch a single winter threshold.

    In winter the final warning must be exactly base / multiplier, unclamped - the ceiling is a
    no-op there.
    """
    detector = ClimateZoneDetector(latitude=latitude)
    multiplier = MULTIPLIERS[heating_type]

    for outdoor in (-30.0, -20.0, -10.0, 0.0):
        base = detector.get_expected_dm_range(outdoor)
        buffered = apply_thermal_mass_buffer(base, heating_type)

        assert buffered["warning"] == pytest.approx(base["warning"] / multiplier), (
            f"{city}, {heating_type}, at {outdoor:+.0f} C: the warm-side ceiling has reached into "
            f"winter. The warning threshold should be {base['warning'] / multiplier:.0f} "
            f"(base {base['warning']:.0f} / {multiplier}), but the ceiling pulled it to "
            f"{buffered['warning']:.0f} and made the emergency ladder less sensitive in a cold "
            f"snap. It is a ceiling on mild days, never a floor on cold ones."
        )


@pytest.mark.parametrize("heating_type", HEATING_TYPES)
def test_the_thresholds_still_deepen_as_it_gets_colder(heating_type):
    """The whole mechanism must survive the fix."""
    detector = ClimateZoneDetector(latitude=59.33)
    warnings = [
        apply_thermal_mass_buffer(detector.get_expected_dm_range(t), heating_type)["warning"]
        for t in (-20.0, -10.0, 0.0, 10.0)
    ]

    assert warnings == sorted(warnings), (
        f"The warning threshold for a {heating_type} house must get DEEPER as it gets colder. Got "
        f"{[round(w) for w in warnings]} for -20/-10/0/+10 C."
    )


def test_a_slow_house_still_reacts_sooner_than_a_fast_one():
    """The buffer's actual purpose, which the clamp must not flatten.

    In winter - where the buffer is meant to act - a concrete slab must still warn EARLIER (at a
    shallower DM) than a radiator system, because heat put into a slab arrives hours later. If the
    ceiling made every emitter equal, it would have deleted the feature instead of bounding it.
    """
    base = ClimateZoneDetector(latitude=59.33).get_expected_dm_range(-10.0)

    radiator = apply_thermal_mass_buffer(base, "radiator")["warning"]
    slab = apply_thermal_mass_buffer(base, "concrete_slab")["warning"]

    assert slab > radiator, (
        f"At -10 C a concrete slab warns at {slab:.0f} DM and a radiator at {radiator:.0f}. The "
        f"slab must warn SOONER (shallower), or the thermal-mass buffer is doing nothing."
    )

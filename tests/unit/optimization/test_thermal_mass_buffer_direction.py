"""A slab that takes six hours to respond must be helped SOONER, not later.

Degree-minute thresholds are NEGATIVE. `_get_thermal_mass_adjusted_thresholds` multiplied them by
a buffer above 1.0 for high-mass systems:

    warning = -540 * 1.3 = -702

which does not tighten the threshold, it deepens it. The concrete slab - the system whose own
docstring says it needs to act earlier, because "current DM doesn't immediately affect indoor
temperature" and the lag is six hours or more - was made the LAST to intervene, and a radiator
system, which can recover in under an hour, the first.

The buffer must DIVIDE:

    warning = -540 / 1.3 = -415        (fires earlier, as intended)

The direction is not a matter of taste. Heat put into a concrete slab arrives in the room hours
later, so a slab must start recovering while the debt is still shallow; by the time it reaches a
radiator system's threshold, the slab has hours of unrecoverable deficit already committed.
"""

import pytest

from custom_components.effektguard.const import (
    DM_THERMAL_MASS_BUFFER_CONCRETE,
    DM_THERMAL_MASS_BUFFER_RADIATOR,
    DM_THERMAL_MASS_BUFFER_TIMBER,
)
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer

STOCKHOLM = 59.33
OUTDOOR = 0.0


def _thresholds(heating_type: str) -> dict:
    detector = ClimateZoneDetector(latitude=STOCKHOLM)
    layer = EmergencyLayer(climate_detector=detector, heating_type=heating_type)
    base = detector.get_expected_dm_range(OUTDOOR)
    return layer._get_thermal_mass_adjusted_thresholds(base)


def test_the_buffers_are_ordered_by_thermal_lag():
    """Sanity: the constants themselves say concrete lags most."""
    assert DM_THERMAL_MASS_BUFFER_CONCRETE > DM_THERMAL_MASS_BUFFER_TIMBER
    assert DM_THERMAL_MASS_BUFFER_TIMBER > DM_THERMAL_MASS_BUFFER_RADIATOR


def test_concrete_intervenes_earlier_than_a_radiator():
    """A six-hour lag must start recovering while the debt is still shallow."""
    concrete = _thresholds("concrete_ufh")["warning"]
    radiator = _thresholds("radiator")["warning"]

    assert concrete > radiator, (
        f"Concrete warns at DM {concrete:.0f} and a radiator system at DM {radiator:.0f}. "
        f"Degree minutes are NEGATIVE, so the concrete slab - six hours of thermal lag - is being "
        f"made to wait {abs(concrete - radiator):.0f} DM LONGER for help than a radiator system "
        f"that recovers in under an hour."
    )


def test_timber_sits_between_them():
    """Timber lags 2-4 hours: later than concrete, earlier than radiators."""
    concrete = _thresholds("concrete_ufh")["warning"]
    timber = _thresholds("timber")["warning"]
    radiator = _thresholds("radiator")["warning"]

    assert concrete > timber > radiator, (
        f"Ordered by lag, the warning thresholds must be concrete > timber > radiator. "
        f"Got concrete {concrete:.0f}, timber {timber:.0f}, radiator {radiator:.0f}."
    )


def test_a_radiator_system_is_left_exactly_where_it_was():
    """The radiator buffer is 1.0: it must be the unmodified baseline, whatever the operation."""
    detector = ClimateZoneDetector(latitude=STOCKHOLM)
    base = detector.get_expected_dm_range(OUTDOOR)

    adjusted = _thresholds("radiator")

    assert adjusted["warning"] == pytest.approx(base["warning"])
    assert adjusted["normal_min"] == pytest.approx(base["normal_min"])


def test_the_absolute_maximum_is_never_buffered():
    """The aux limit is hardware, not a tuning knob. It is the same for every emitter."""
    concrete = _thresholds("concrete_ufh")["critical"]
    radiator = _thresholds("radiator")["critical"]

    assert concrete == radiator

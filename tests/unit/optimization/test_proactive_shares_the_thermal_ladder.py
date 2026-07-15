"""Both thermal-debt layers must read the same (thermal-mass-buffered) warning threshold.

EmergencyLayer applies the thermal-mass buffer; ProactiveLayer must too. When it read the raw range
instead, the two layers used different thresholds for the same house, leaving a band of degree
minutes where the proactive layer had handed over but the emergency layer had not yet picked up -
worst for the concrete slab, the house that can least afford to fall behind.

Invariant: for every heating type the two layers warn at the same DM, and the slab has no silent band.
"""

import pytest

from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.thermal_layer import (
    EmergencyLayer,
    ProactiveLayer,
)

STOCKHOLM = 59.33
OUTDOOR = 0.0

HEATING_TYPES = ["concrete_ufh", "timber", "radiator"]


@pytest.fixture
def detector() -> ClimateZoneDetector:
    return ClimateZoneDetector(latitude=STOCKHOLM)


@pytest.mark.parametrize("heating_type", HEATING_TYPES)
def test_both_layers_use_the_same_warning_threshold(detector, heating_type):
    """A threshold is a property of the house, not of the layer that happens to read it."""
    emergency = EmergencyLayer(climate_detector=detector, heating_type=heating_type)
    proactive = ProactiveLayer(climate_detector=detector, heating_type=heating_type)

    emergency_warning = emergency._get_thermal_mass_adjusted_thresholds(
        detector.get_expected_dm_range(OUTDOOR)
    )["warning"]
    proactive_warning = proactive._calculate_expected_dm_for_temperature(OUTDOOR)["warning"]

    assert proactive_warning == pytest.approx(emergency_warning), (
        f"For {heating_type!r} the proactive layer warns at DM {proactive_warning:.0f} while the "
        f"emergency layer warns at DM {emergency_warning:.0f}. Between the two lies a band in "
        f"which neither layer responds."
    )


def test_the_concrete_slab_has_no_silent_band(detector):
    """No degree-minute value may leave both layers idle while the debt is real.

    The slab is the case that matters: its debt does not reach the room for hours, so a band where
    nothing acts is a band of deficit that can never be recovered.
    """
    emergency = EmergencyLayer(climate_detector=detector, heating_type="concrete_ufh")
    proactive = ProactiveLayer(climate_detector=detector, heating_type="concrete_ufh")

    emergency_warning = emergency._get_thermal_mass_adjusted_thresholds(
        detector.get_expected_dm_range(OUTDOOR)
    )["warning"]
    proactive_warning = proactive._calculate_expected_dm_for_temperature(OUTDOOR)["warning"]

    # The proactive layer must not hand over LATER than the emergency layer picks up.
    assert proactive_warning >= emergency_warning, (
        f"The proactive layer stays silent until DM {proactive_warning:.0f}, but the emergency "
        f"layer does not engage until DM {emergency_warning:.0f}. Every degree minute between "
        f"them is unattended."
    )

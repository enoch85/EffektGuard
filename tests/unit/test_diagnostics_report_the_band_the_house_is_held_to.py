"""Diagnostics must report the DM band production ENFORCES, not the raw zone table.

The production path runs every zone range through apply_thermal_mass_buffer - a concrete slab
is helped ~1.3x sooner - so a diagnostics dump quoting the unadjusted range told a
slab-house owner they were being held to -414 while the code was actually intervening at
-318. A diagnostics file that disagrees with the decision it was downloaded to explain is
worse than none: it sends the reader hunting for a discrepancy that is the dump's own.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from custom_components.effektguard.diagnostics import _dm_thresholds
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.thermal_layer import apply_thermal_mass_buffer


def _coordinator(heating_type: str) -> MagicMock:
    coordinator = MagicMock()
    coordinator.engine.climate_detector = ClimateZoneDetector(latitude=59.33)
    coordinator.engine.emergency_layer.heating_type = heating_type
    return coordinator


def test_the_reported_range_is_the_thermal_mass_adjusted_one():
    coordinator = _coordinator("concrete_ufh")
    nibe = SimpleNamespace(outdoor_temp=0.0)

    report = _dm_thresholds(coordinator, nibe)

    detector = coordinator.engine.climate_detector
    enforced = apply_thermal_mass_buffer(detector.get_expected_dm_range(0.0), "concrete_ufh")
    assert report["range"] == enforced, (
        f"Diagnostics report {report['range']} but production holds this house to {enforced}. "
        f"The dump exists to explain the decision; it must quote the band the decision used."
    )
    assert report["heating_type"] == "concrete_ufh"


def test_a_radiator_house_is_unchanged_by_the_adjustment():
    coordinator = _coordinator("radiator")
    nibe = SimpleNamespace(outdoor_temp=0.0)

    report = _dm_thresholds(coordinator, nibe)

    detector = coordinator.engine.climate_detector
    assert report["range"] == apply_thermal_mass_buffer(
        detector.get_expected_dm_range(0.0), "radiator"
    )

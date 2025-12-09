"""Tests for climate zone detection and DM threshold calculations.

Tests the new dedicated climate_zones.py module that provides climate-aware
DM (Degree Minutes) thresholds for heat pump optimization.
"""

from custom_components.effektguard.const import (
    CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG,
    CLIMATE_ZONE_VERY_COLD_WINTER_AVG,
    CLIMATE_ZONE_COLD_WINTER_AVG,
    CLIMATE_ZONE_MODERATE_COLD_WINTER_AVG,
    CLIMATE_ZONE_STANDARD_WINTER_AVG,
)
from custom_components.effektguard.optimization.climate_zones import (
    ClimateZoneDetector,
    ClimateZoneInfo,
    HEATING_CLIMATE_ZONES,
    ZONE_ORDER,
    DM_ABSOLUTE_MAXIMUM,
)


class TestClimateZoneDetection:
    """Test climate zone detection based on latitude."""

    def test_extreme_cold_zone_kiruna(self):
        """Test Kiruna (Arctic) detection."""
        detector = ClimateZoneDetector(67.85)
        assert detector.zone_key == "extreme_cold"
        assert detector.zone_info.name == "Extreme Cold"
        assert detector.zone_info.winter_avg_low == CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG

    def test_very_cold_zone_lulea(self):
        """Test Luleå (Northern Sweden) detection."""
        detector = ClimateZoneDetector(65.58)
        assert detector.zone_key == "very_cold"
        assert detector.zone_info.name == "Very Cold"
        assert detector.zone_info.winter_avg_low == CLIMATE_ZONE_VERY_COLD_WINTER_AVG

    def test_very_cold_zone_umea(self):
        """Test Umeå (Northern Sweden) detection - should be in very_cold zone."""
        detector = ClimateZoneDetector(63.83)
        assert detector.zone_key == "very_cold"
        assert detector.zone_info.name == "Very Cold"

    def test_cold_zone_stockholm(self):
        """Test Stockholm (Southern Nordics) detection."""
        detector = ClimateZoneDetector(59.33)
        assert detector.zone_key == "cold"
        assert detector.zone_info.name == "Cold"
        assert detector.zone_info.winter_avg_low == CLIMATE_ZONE_COLD_WINTER_AVG

    def test_cold_zone_helsinki(self):
        """Test Helsinki (Finland, but southern climate) detection."""
        detector = ClimateZoneDetector(60.17)
        assert detector.zone_key == "cold"
        assert detector.zone_info.name == "Cold"

    def test_moderate_cold_zone_copenhagen(self):
        """Test Copenhagen (Denmark) detection."""
        detector = ClimateZoneDetector(55.68)
        assert detector.zone_key == "moderate_cold"
        assert detector.zone_info.name == "Moderate Cold"
        assert detector.zone_info.winter_avg_low == CLIMATE_ZONE_MODERATE_COLD_WINTER_AVG

    def test_moderate_cold_zone_malmo(self):
        """Test Malmö (Southern Sweden, Øresund region) detection."""
        detector = ClimateZoneDetector(55.60)
        assert detector.zone_key == "moderate_cold"
        assert detector.zone_info.name == "Moderate Cold"

    def test_standard_zone_paris(self):
        """Test Paris (Standard zone) detection."""
        detector = ClimateZoneDetector(48.86)
        assert detector.zone_key == "standard"
        assert detector.zone_info.name == "Standard"
        assert detector.zone_info.winter_avg_low == CLIMATE_ZONE_STANDARD_WINTER_AVG

    def test_southern_hemisphere(self):
        """Test southern hemisphere uses absolute latitude."""
        detector = ClimateZoneDetector(-59.33)  # Negative latitude
        assert detector.zone_key == "cold"
        assert detector.latitude == 59.33  # Absolute value used


class TestDMRangeCalculations:
    """Test DM range calculations for different zones and temperatures."""

    def test_extreme_cold_at_average(self):
        """Test Kiruna at winter average temperature."""
        detector = ClimateZoneDetector(67.85)
        # Test at the zone's actual winter average (from constants)
        dm_range = detector.get_expected_dm_range(CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG)

        # At zone's winter average, should return base DM range
        assert dm_range["normal_min"] == -800
        assert dm_range["normal_max"] == -1200
        assert dm_range["warning"] == -1200
        assert dm_range["critical"] == DM_ABSOLUTE_MAXIMUM

    def test_cold_at_average(self):
        """Test Stockholm at winter average temperature."""
        detector = ClimateZoneDetector(59.33)
        # Test at the zone's actual winter average (from constants)
        dm_range = detector.get_expected_dm_range(CLIMATE_ZONE_COLD_WINTER_AVG)

        assert dm_range["normal_min"] == -450
        assert dm_range["normal_max"] == -700
        assert dm_range["warning"] == -700

    def test_moderate_cold_at_average(self):
        """Test Copenhagen at winter average temperature."""
        detector = ClimateZoneDetector(55.68)
        # Test at the zone's actual winter average (from constants)
        dm_range = detector.get_expected_dm_range(CLIMATE_ZONE_MODERATE_COLD_WINTER_AVG)

        assert dm_range["normal_min"] == -300
        assert dm_range["normal_max"] == -500
        assert dm_range["warning"] == -500

    def test_temperature_adjustment_warmer(self):
        """Test DM expectations adjust when warmer than zone average."""
        detector = ClimateZoneDetector(59.33)  # Stockholm (Cold zone)

        # At zone's winter average (from constants)
        dm_at_avg = detector.get_expected_dm_range(CLIMATE_ZONE_COLD_WINTER_AVG)

        # 10°C warmer than average
        dm_warmer = detector.get_expected_dm_range(CLIMATE_ZONE_COLD_WINTER_AVG + 10.0)

        # Should be +200 DM (shallower) = -20 DM per degree warmer
        adjustment = dm_warmer["normal_min"] - dm_at_avg["normal_min"]
        assert adjustment == 200  # Shallower when warmer

    def test_temperature_adjustment_colder(self):
        """Test DM expectations adjust when colder than zone average."""
        detector = ClimateZoneDetector(59.33)  # Stockholm (Cold zone)

        # At zone's winter average (from constants)
        dm_at_avg = detector.get_expected_dm_range(CLIMATE_ZONE_COLD_WINTER_AVG)

        # 10°C colder than average
        dm_colder = detector.get_expected_dm_range(CLIMATE_ZONE_COLD_WINTER_AVG - 10.0)

        # Should be -200 DM (deeper) = -20 DM per degree colder
        adjustment = dm_colder["normal_min"] - dm_at_avg["normal_min"]
        assert adjustment == -200  # Deeper when colder

    def test_extreme_cold_adjustment(self):
        """Test Kiruna at extremely cold temperature."""
        detector = ClimateZoneDetector(67.85)  # Kiruna (Extreme Cold zone)

        # At zone's winter average (from constants)
        dm_at_avg = detector.get_expected_dm_range(CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG)

        # At 10°C colder than average
        dm_extreme = detector.get_expected_dm_range(CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG - 10.0)

        # Should allow deeper DM
        assert dm_extreme["normal_max"] < dm_at_avg["normal_max"]

    def test_never_exceeds_absolute_maximum(self):
        """Test that DM thresholds never exceed absolute maximum."""
        detector = ClimateZoneDetector(67.85)  # Extreme cold

        # Even at extreme temperatures (30°C colder than zone average)
        dm_range = detector.get_expected_dm_range(CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG - 30.0)

        # Should stay above absolute maximum (less negative)
        assert dm_range["normal_min"] > DM_ABSOLUTE_MAXIMUM
        assert dm_range["normal_max"] > DM_ABSOLUTE_MAXIMUM
        assert dm_range["warning"] > DM_ABSOLUTE_MAXIMUM
        assert dm_range["critical"] == DM_ABSOLUTE_MAXIMUM


class TestSafetyMargins:
    """Test safety margin calculations for different climate zones."""

    def test_extreme_cold_margin(self):
        """Test Extreme Cold zone has highest safety margin."""
        detector = ClimateZoneDetector(67.85)
        assert detector.get_safety_margin() == 2.5

    def test_very_cold_margin(self):
        """Test Very Cold zone has high safety margin."""
        detector = ClimateZoneDetector(65.58)
        assert detector.get_safety_margin() == 1.5

    def test_cold_margin(self):
        """Test Cold zone has moderate safety margin."""
        detector = ClimateZoneDetector(59.33)
        assert detector.get_safety_margin() == 1.0

    def test_moderate_cold_margin(self):
        """Test Moderate Cold zone has low safety margin."""
        detector = ClimateZoneDetector(55.68)
        assert detector.get_safety_margin() == 0.5

    def test_standard_margin(self):
        """Test Standard zone has no safety margin."""
        detector = ClimateZoneDetector(48.86)
        assert detector.get_safety_margin() == 0.0


class TestZoneBoundaries:
    """Test zone boundary detection."""

    def test_boundary_arctic_circle(self):
        """Test Arctic Circle boundary at 66.5°N."""
        just_below = ClimateZoneDetector(66.4)
        at_boundary = ClimateZoneDetector(66.5)
        just_above = ClimateZoneDetector(66.6)

        assert just_below.zone_key == "very_cold"
        assert at_boundary.zone_key == "extreme_cold"
        assert just_above.zone_key == "extreme_cold"

    def test_boundary_northern_southern_nordics(self):
        """Test boundary between Very Cold and Cold zones at 60.5°N."""
        just_below = ClimateZoneDetector(60.4)
        at_boundary = ClimateZoneDetector(60.5)
        just_above = ClimateZoneDetector(60.6)

        assert just_below.zone_key == "cold"
        assert at_boundary.zone_key == "very_cold"
        assert just_above.zone_key == "very_cold"

    def test_boundary_oresund_region(self):
        """Test Øresund region boundary at 56.0°N (Copenhagen/Malmö vs Stockholm)."""
        just_below = ClimateZoneDetector(55.9)
        malmo = ClimateZoneDetector(55.60)
        copenhagen = ClimateZoneDetector(55.68)
        at_boundary = ClimateZoneDetector(56.0)
        just_above = ClimateZoneDetector(56.1)

        assert just_below.zone_key == "moderate_cold"
        assert malmo.zone_key == "moderate_cold"
        assert copenhagen.zone_key == "moderate_cold"
        assert at_boundary.zone_key == "cold"
        assert just_above.zone_key == "cold"


class TestConstants:
    """Test module constants are properly defined."""

    def test_zone_order_complete(self):
        """Test all zones are in ZONE_ORDER."""
        assert len(ZONE_ORDER) == 5
        assert "extreme_cold" in ZONE_ORDER
        assert "very_cold" in ZONE_ORDER
        assert "cold" in ZONE_ORDER
        assert "moderate_cold" in ZONE_ORDER
        assert "standard" in ZONE_ORDER

    def test_heating_climate_zones_complete(self):
        """Test all zones are defined in HEATING_CLIMATE_ZONES."""
        assert len(HEATING_CLIMATE_ZONES) == 5
        for zone_key in ZONE_ORDER:
            assert zone_key in HEATING_CLIMATE_ZONES

    def test_absolute_maximum_constant(self):
        """Test DM_ABSOLUTE_MAXIMUM is defined correctly."""
        assert DM_ABSOLUTE_MAXIMUM == -1500

    def test_zone_data_structure(self):
        """Test each zone has required fields."""
        required_fields = {
            "name",
            "description",
            "latitude_range",
            "winter_avg_low",
            "dm_normal_range",
            "dm_warning_threshold",
            "safety_margin_base",
            "examples",
        }

        for zone_key, zone_data in HEATING_CLIMATE_ZONES.items():
            assert set(zone_data.keys()) == required_fields


class TestClimateZoneInfo:
    """Test ClimateZoneInfo dataclass."""

    def test_zone_info_creation(self):
        """Test ClimateZoneInfo is properly populated."""
        detector = ClimateZoneDetector(59.33)
        info = detector.zone_info

        assert isinstance(info, ClimateZoneInfo)
        assert info.zone_key == "cold"
        assert info.name == "Cold"
        assert info.winter_avg_low == CLIMATE_ZONE_COLD_WINTER_AVG
        assert info.dm_normal_min == -450
        assert info.dm_normal_max == -700
        assert info.dm_warning_threshold == -700
        assert info.safety_margin_base == 1.0
        assert len(info.examples) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_equator_defaults_to_standard(self):
        """Test equatorial location defaults to Standard zone."""
        detector = ClimateZoneDetector(0.0)
        assert detector.zone_key == "standard"

    def test_extreme_north_in_range(self):
        """Test extreme north pole latitude."""
        detector = ClimateZoneDetector(85.0)
        assert detector.zone_key == "extreme_cold"

    def test_extreme_south_in_range(self):
        """Test extreme south pole latitude."""
        detector = ClimateZoneDetector(-85.0)
        assert detector.zone_key == "extreme_cold"
        assert detector.latitude == 85.0  # Absolute value

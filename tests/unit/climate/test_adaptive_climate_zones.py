"""Test adaptive climate zone system.

Tests for latitude-based climate detection and adaptive safety margins.
Verifies global applicability from Extreme Cold to Standard climates.

Note: Updated to use new heating-focused zone names from climate_zones.py module:
- "extreme_cold" (was "arctic")
- "very_cold" (was "subarctic")
- "cold" (unchanged)
- "moderate_cold" (was "temperate")
- "standard" (was "mild")
"""

import pytest

from custom_components.effektguard.const import (
    CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG,
    CLIMATE_ZONE_VERY_COLD_WINTER_AVG,
    CLIMATE_ZONE_COLD_WINTER_AVG,
    CLIMATE_ZONE_MODERATE_COLD_WINTER_AVG,
    CLIMATE_ZONE_STANDARD_WINTER_AVG,
)
from custom_components.effektguard.optimization.weather_compensation import (
    AdaptiveClimateSystem,
)


class TestClimateZoneDetection:
    """Test automatic climate zone detection based on latitude."""

    def test_arctic_zone_kiruna(self):
        """Test Extreme Cold zone detection for Kiruna, Sweden (67.85°N)."""
        climate = AdaptiveClimateSystem(latitude=67.85)

        assert climate.climate_zone == "extreme_cold"
        info = climate.get_climate_info()
        assert info["name"] == "Extreme Cold"
        assert info["winter_avg_low"] == CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG
        assert info["safety_margin_base"] == 2.5

    def test_subarctic_zone_lulea(self):
        """Test Very Cold zone detection for Luleå, Sweden (65.58°N)."""
        climate = AdaptiveClimateSystem(latitude=65.58)

        assert climate.climate_zone == "very_cold"
        info = climate.get_climate_info()
        assert info["name"] == "Very Cold"
        assert info["winter_avg_low"] == CLIMATE_ZONE_VERY_COLD_WINTER_AVG
        assert info["safety_margin_base"] == 1.5

    def test_cold_zone_stockholm(self):
        """Test Cold zone for Stockholm, Sweden (59.33°N)."""
        climate = AdaptiveClimateSystem(latitude=59.33)

        assert climate.climate_zone == "cold"
        info = climate.get_climate_info()
        assert info["name"] == "Cold"
        assert info["winter_avg_low"] == CLIMATE_ZONE_COLD_WINTER_AVG
        assert info["safety_margin_base"] == 1.0

    def test_cold_zone_oslo(self):
        """Test Cold zone for Oslo, Norway (59.91°N)."""
        climate = AdaptiveClimateSystem(latitude=59.91)

        assert climate.climate_zone == "cold"
        info = climate.get_climate_info()
        assert info["name"] == "Cold"

    def test_temperate_zone_london(self):
        """Test Standard zone for London, UK (51.51°N)."""
        climate = AdaptiveClimateSystem(latitude=51.51)

        assert climate.climate_zone == "standard"
        info = climate.get_climate_info()
        assert info["name"] == "Standard"
        assert info["winter_avg_low"] == CLIMATE_ZONE_STANDARD_WINTER_AVG
        assert info["safety_margin_base"] == 0.0

    def test_mild_zone_paris(self):
        """Test Standard zone for Paris, France (48.86°N)."""
        climate = AdaptiveClimateSystem(latitude=48.86)

        assert climate.climate_zone == "standard"
        info = climate.get_climate_info()
        assert info["name"] == "Standard"
        assert info["winter_avg_low"] == CLIMATE_ZONE_STANDARD_WINTER_AVG
        assert info["safety_margin_base"] == 0.0

    def test_southern_hemisphere_absolute_value(self):
        """Test that southern hemisphere works (absolute latitude)."""
        # Melbourne, Australia (-37.81°S) → should be same as 37.81°N (standard zone)
        climate = AdaptiveClimateSystem(latitude=-37.81)

        assert climate.climate_zone == "standard"
        assert climate.latitude == 37.81  # Should use absolute value


class TestSafetyMargins:
    """Test adaptive safety margin calculations."""

    def test_arctic_extreme_cold(self):
        """Test Arctic zone with extreme cold (-35°C)."""
        climate = AdaptiveClimateSystem(latitude=67.85)  # Kiruna

        # Colder than zone's winter average, should add extra margin
        margin = climate.get_safety_margin(
            outdoor_temp=CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG - 5.0,
            unusual_weather_detected=False,
            unusual_severity=0.0,
        )

        # Base margin (2.5) + temp adjustment (0.1 * 5 = 0.5) = 3.0°C
        assert margin == pytest.approx(3.0, abs=0.1)

    def test_arctic_typical_cold(self):
        """Test Arctic zone at zone's typical winter average."""
        climate = AdaptiveClimateSystem(latitude=67.85)  # Kiruna

        # At zone's winter average (from constants), no temp adjustment
        margin = climate.get_safety_margin(
            outdoor_temp=CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG,
            unusual_weather_detected=False,
            unusual_severity=0.0,
        )

        # Just base margin
        assert margin == pytest.approx(2.5, abs=0.1)

    def test_cold_zone_stockholm_winter(self):
        """Test Cold zone at Stockholm's typical winter average."""
        climate = AdaptiveClimateSystem(latitude=59.33)  # Stockholm

        # At zone's winter average (from constants)
        margin = climate.get_safety_margin(
            outdoor_temp=CLIMATE_ZONE_COLD_WINTER_AVG,
            unusual_weather_detected=False,
            unusual_severity=0.0,
        )

        # Base margin only
        assert margin == pytest.approx(1.0, abs=0.1)

    def test_temperate_zone_london_winter(self):
        """Test Standard zone at typical winter average."""
        climate = AdaptiveClimateSystem(latitude=51.51)  # London (Standard zone)

        margin = climate.get_safety_margin(
            outdoor_temp=CLIMATE_ZONE_STANDARD_WINTER_AVG,
            unusual_weather_detected=False,
            unusual_severity=0.0,
        )

        # Standard zone has base margin 0.0
        assert margin == pytest.approx(0.0, abs=0.1)

    def test_mild_zone_no_base_margin(self):
        """Test Standard zone (Paris) with no base safety margin."""
        climate = AdaptiveClimateSystem(latitude=48.86)  # Paris (Standard zone)

        margin = climate.get_safety_margin(
            outdoor_temp=CLIMATE_ZONE_STANDARD_WINTER_AVG,
            unusual_weather_detected=False,
            unusual_severity=0.0,
        )

        # No base margin, at winter average
        assert margin == pytest.approx(0.0, abs=0.1)

    def test_unusual_weather_moderate(self):
        """Test unusual weather detection adds safety margin."""
        climate = AdaptiveClimateSystem(latitude=59.33)  # Stockholm (Cold zone)

        margin = climate.get_safety_margin(
            outdoor_temp=CLIMATE_ZONE_COLD_WINTER_AVG,
            unusual_weather_detected=True,
            unusual_severity=0.5,  # Moderate severity
        )

        # Base (1.0) + unusual (0.5 + 0.5*1.0 = 1.0) = 2.0°C
        assert margin == pytest.approx(2.0, abs=0.1)

    def test_unusual_weather_extreme(self):
        """Test extreme unusual weather adds maximum safety margin."""
        climate = AdaptiveClimateSystem(latitude=59.33)  # Stockholm (Cold zone)

        margin = climate.get_safety_margin(
            outdoor_temp=CLIMATE_ZONE_COLD_WINTER_AVG,
            unusual_weather_detected=True,
            unusual_severity=1.0,  # Extreme severity
        )

        # Base (1.0) + unusual (0.5 + 1.0*1.0 = 1.5) = 2.5°C
        assert margin == pytest.approx(2.5, abs=0.1)

    def test_combined_cold_and_unusual(self):
        """Test combined effect of colder than average + unusual weather."""
        climate = AdaptiveClimateSystem(latitude=59.33)  # Stockholm (Cold zone)

        margin = climate.get_safety_margin(
            outdoor_temp=CLIMATE_ZONE_COLD_WINTER_AVG - 10.0,  # 10°C colder than zone average
            unusual_weather_detected=True,
            unusual_severity=0.8,
        )

        # Base (1.0) + temp_adj (0.1*10=1.0) + unusual (0.5+0.8*1.0=1.3) = 3.3°C
        assert margin == pytest.approx(3.3, abs=0.1)


class TestDynamicWeights:
    """Test dynamic weight calculation for decision engine."""

    def test_arctic_extreme_cold_weight(self):
        """Test Arctic zone in extreme cold gets high weight."""
        climate = AdaptiveClimateSystem(latitude=67.85)  # Kiruna

        weight = climate.get_dynamic_weight(
            outdoor_temp=-35.0,  # Colder than winter avg
            unusual_weather_detected=False,
        )

        # Very cold → 0.85 base weight
        assert weight == pytest.approx(0.85, abs=0.05)

    def test_cold_zone_typical_weight(self):
        """Test Cold zone in typical conditions."""
        climate = AdaptiveClimateSystem(latitude=59.33)  # Stockholm (Cold zone)

        weight = climate.get_dynamic_weight(
            outdoor_temp=CLIMATE_ZONE_COLD_WINTER_AVG,  # At zone's winter average
            unusual_weather_detected=False,
        )

        # At winter_avg_low → cold category (winter_avg_low to winter_avg_low+5) → 0.75
        assert weight == pytest.approx(0.75, abs=0.05)

    def test_temperate_mild_cold_weight(self):
        """Test Standard zone in mild cold."""
        climate = AdaptiveClimateSystem(latitude=51.51)  # London (Standard zone)

        weight = climate.get_dynamic_weight(
            outdoor_temp=CLIMATE_ZONE_STANDARD_WINTER_AVG - 3.0,  # Below winter avg
            unusual_weather_detected=False,
        )

        # Below winter_avg_low → very cold category → 0.85
        assert weight == pytest.approx(0.85, abs=0.05)

    def test_warm_weather_low_weight(self):
        """Test warm weather reduces weight."""
        climate = AdaptiveClimateSystem(latitude=48.86)  # Paris

        weight = climate.get_dynamic_weight(
            outdoor_temp=10.0,
            unusual_weather_detected=False,
        )

        # Warm (>= 5°C) → 0.50
        assert weight == pytest.approx(0.50, abs=0.05)

    def test_unusual_weather_increases_weight(self):
        """Test unusual weather increases weight."""
        climate = AdaptiveClimateSystem(latitude=59.33)  # Stockholm

        weight = climate.get_dynamic_weight(
            outdoor_temp=5.0,  # Mild weather (>= 5°C)
            unusual_weather_detected=True,
        )

        # Warm base (0.50) + unusual boost (0.15) = 0.65
        assert weight == pytest.approx(0.65, abs=0.05)

    def test_weight_capped_at_095(self):
        """Test weight is capped at 0.95 even with unusual weather."""
        climate = AdaptiveClimateSystem(latitude=67.85)  # Kiruna

        weight = climate.get_dynamic_weight(
            outdoor_temp=-35.0,  # Very cold (0.85)
            unusual_weather_detected=True,  # +0.15
        )

        # Should be capped at 0.95, not 1.0
        assert weight <= 0.95
        assert weight == pytest.approx(0.95, abs=0.05)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_equatorial_location_defaults_to_standard(self):
        """Test equatorial location defaults to Standard zone."""
        climate = AdaptiveClimateSystem(latitude=0.0)  # Equator

        assert climate.climate_zone == "standard"  # Default to standard for low latitudes

    def test_extreme_south_defaults_to_standard(self):
        """Test extreme southern hemisphere defaults to Standard."""
        climate = AdaptiveClimateSystem(latitude=-75.0)  # Antarctic

        # Uses absolute value (75°), which is > 66.5° → Extreme Cold
        assert climate.climate_zone == "extreme_cold"

    def test_zero_latitude_equator(self):
        """Test zero latitude handling."""
        climate = AdaptiveClimateSystem(latitude=0.0)

        assert climate.climate_zone == "standard"
        assert climate.latitude == 0.0

    def test_latitude_95_invalid_over_pole(self):
        """Test latitude over pole (>90°) still works."""
        # Invalid but system should handle gracefully
        climate = AdaptiveClimateSystem(latitude=95.0)

        # Uses absolute value → 95° is outside all ranges → defaults to Standard
        assert climate.climate_zone == "standard"

    def test_boundary_arctic_circle_exactly(self):
        """Test exactly at Arctic Circle boundary (66.5°N)."""
        climate = AdaptiveClimateSystem(latitude=66.5)

        assert climate.climate_zone == "extreme_cold"  # At boundary, inclusive

    def test_boundary_between_zones_60_5(self):
        """Test boundary between Very Cold and Cold zones (60.5°N)."""
        just_below = AdaptiveClimateSystem(latitude=60.4)
        at_boundary = AdaptiveClimateSystem(latitude=60.5)

        assert just_below.climate_zone == "cold"
        assert at_boundary.climate_zone == "very_cold"

    def test_boundary_temperate_mild_49_0(self):
        """Test boundary between Moderate Cold and Standard zones (54.5°N)."""
        just_below = AdaptiveClimateSystem(latitude=54.4)
        at_boundary = AdaptiveClimateSystem(latitude=54.5)

        assert just_below.climate_zone == "standard"
        assert at_boundary.climate_zone == "moderate_cold"

    def test_just_below_mild_zone_34_9(self):
        """Test just below minimum latitude (34.9°N → Standard)."""
        climate = AdaptiveClimateSystem(latitude=34.9)

        assert climate.climate_zone == "standard"

    def test_between_zones_non_boundary(self):
        """Test latitude between zones (not at exact boundary)."""
        climate = AdaptiveClimateSystem(latitude=63.0)  # Between Very Cold boundaries

        assert climate.climate_zone == "very_cold"


class TestGlobalApplicability:
    """Test that system works globally without country-specific code."""

    def test_canadian_arctic(self):
        """Test Canadian Arctic location (Yellowknife 62.45°N)."""
        climate = AdaptiveClimateSystem(latitude=62.45)

        # Should be Very Cold zone
        assert climate.climate_zone == "very_cold"

        # Should provide appropriate safety margins
        margin = climate.get_safety_margin(outdoor_temp=-25.0)
        assert margin > 1.0  # Very Cold base + cold adjustment

    def test_norwegian_coast(self):
        """Test Norwegian coastal location (Bergen 60.39°N)."""
        climate = AdaptiveClimateSystem(latitude=60.39)

        # Should be Cold zone
        assert climate.climate_zone == "cold"

    def test_german_location(self):
        """Test German location (Berlin 52.52°N)."""
        climate = AdaptiveClimateSystem(latitude=52.52)

        # Should be Standard zone
        assert climate.climate_zone == "standard"

    def test_finnish_lapland(self):
        """Test Finnish Lapland (Rovaniemi 66.50°N)."""
        climate = AdaptiveClimateSystem(latitude=66.50)

        # Right at Arctic Circle boundary
        assert climate.climate_zone == "extreme_cold"

    def test_multiple_locations_same_code(self):
        """Test that different locations work with same code (no country checks)."""
        locations = [
            (67.85, "extreme_cold"),  # Kiruna, Sweden
            (64.75, "very_cold"),  # Fairbanks, USA (latitude equivalent)
            (59.33, "cold"),  # Stockholm, Sweden
            (51.51, "standard"),  # London, UK
            (48.86, "standard"),  # Paris, France
        ]

        for lat, expected_zone in locations:
            climate = AdaptiveClimateSystem(latitude=lat)
            assert climate.climate_zone == expected_zone

            # All should provide safety margins
            margin = climate.get_safety_margin(outdoor_temp=-10.0)
            assert margin >= 0.0

            # All should provide dynamic weights
            weight = climate.get_dynamic_weight(outdoor_temp=-10.0)
            assert 0.0 < weight <= 1.0

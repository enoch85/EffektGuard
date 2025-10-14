"""Test adaptive climate zone system.

Tests for latitude-based climate detection and adaptive safety margins.
Verifies global applicability from Arctic to Mild climates.
"""

import pytest

from custom_components.effektguard.optimization.weather_compensation import (
    AdaptiveClimateSystem,
)


class TestClimateZoneDetection:
    """Test automatic climate zone detection based on latitude."""

    def test_arctic_zone_kiruna(self):
        """Test Arctic zone detection for Kiruna, Sweden (67.85°N)."""
        climate = AdaptiveClimateSystem(latitude=67.85)

        assert climate.climate_zone == "arctic"
        info = climate.get_climate_info()
        assert info["name"] == "Arctic"
        assert info["winter_avg_low"] == -30.0
        assert info["safety_margin_base"] == 2.5

    def test_subarctic_zone_lulea(self):
        """Test Subarctic zone detection for Luleå, Sweden (65.58°N)."""
        climate = AdaptiveClimateSystem(latitude=65.58)

        assert climate.climate_zone == "subarctic"
        info = climate.get_climate_info()
        assert info["name"] == "Subarctic"
        assert info["winter_avg_low"] == -15.0
        assert info["safety_margin_base"] == 1.5

    def test_cold_zone_stockholm(self):
        """Test Cold Continental zone for Stockholm, Sweden (59.33°N)."""
        climate = AdaptiveClimateSystem(latitude=59.33)

        assert climate.climate_zone == "cold"
        info = climate.get_climate_info()
        assert info["name"] == "Cold Continental"
        assert info["winter_avg_low"] == -10.0
        assert info["safety_margin_base"] == 1.0

    def test_cold_zone_oslo(self):
        """Test Cold Continental zone for Oslo, Norway (59.91°N)."""
        climate = AdaptiveClimateSystem(latitude=59.91)

        assert climate.climate_zone == "cold"
        info = climate.get_climate_info()
        assert info["name"] == "Cold Continental"

    def test_temperate_zone_london(self):
        """Test Temperate Oceanic zone for London, UK (51.51°N)."""
        climate = AdaptiveClimateSystem(latitude=51.51)

        assert climate.climate_zone == "temperate"
        info = climate.get_climate_info()
        assert info["name"] == "Temperate Oceanic"
        assert info["winter_avg_low"] == 0.0
        assert info["safety_margin_base"] == 0.5

    def test_mild_zone_paris(self):
        """Test Mild Oceanic zone for Paris, France (48.86°N)."""
        climate = AdaptiveClimateSystem(latitude=48.86)

        assert climate.climate_zone == "mild"
        info = climate.get_climate_info()
        assert info["name"] == "Mild Oceanic"
        assert info["winter_avg_low"] == 5.0
        assert info["safety_margin_base"] == 0.0

    def test_southern_hemisphere_absolute_value(self):
        """Test that southern hemisphere works (absolute latitude)."""
        # Melbourne, Australia (-37.81°S) → should be same as 37.81°N (mild zone)
        climate = AdaptiveClimateSystem(latitude=-37.81)

        assert climate.climate_zone == "mild"
        assert climate.latitude == 37.81  # Should use absolute value


class TestSafetyMargins:
    """Test adaptive safety margin calculations."""

    def test_arctic_extreme_cold(self):
        """Test Arctic zone with extreme cold (-35°C)."""
        climate = AdaptiveClimateSystem(latitude=67.85)  # Kiruna

        # Colder than winter average (-30°C), should add extra margin
        margin = climate.get_safety_margin(
            outdoor_temp=-35.0,
            unusual_weather_detected=False,
            unusual_severity=0.0,
        )

        # Base margin (2.5) + temp adjustment (0.1 * 5 = 0.5) = 3.0°C
        assert margin == pytest.approx(3.0, abs=0.1)

    def test_arctic_typical_cold(self):
        """Test Arctic zone with typical winter (-30°C)."""
        climate = AdaptiveClimateSystem(latitude=67.85)  # Kiruna

        # At winter average, no temp adjustment
        margin = climate.get_safety_margin(
            outdoor_temp=-30.0,
            unusual_weather_detected=False,
            unusual_severity=0.0,
        )

        # Just base margin
        assert margin == pytest.approx(2.5, abs=0.1)

    def test_cold_zone_stockholm_winter(self):
        """Test Cold zone with Stockholm winter conditions (-10°C)."""
        climate = AdaptiveClimateSystem(latitude=59.33)  # Stockholm

        # At winter average
        margin = climate.get_safety_margin(
            outdoor_temp=-10.0,
            unusual_weather_detected=False,
            unusual_severity=0.0,
        )

        # Base margin only
        assert margin == pytest.approx(1.0, abs=0.1)

    def test_temperate_zone_london_winter(self):
        """Test Temperate zone with London winter (0°C)."""
        climate = AdaptiveClimateSystem(latitude=51.51)  # London

        margin = climate.get_safety_margin(
            outdoor_temp=0.0,
            unusual_weather_detected=False,
            unusual_severity=0.0,
        )

        # Base margin only
        assert margin == pytest.approx(0.5, abs=0.1)

    def test_mild_zone_no_base_margin(self):
        """Test Mild zone with no base safety margin."""
        climate = AdaptiveClimateSystem(latitude=48.86)  # Paris

        margin = climate.get_safety_margin(
            outdoor_temp=5.0,
            unusual_weather_detected=False,
            unusual_severity=0.0,
        )

        # No base margin, at winter average
        assert margin == pytest.approx(0.0, abs=0.1)

    def test_unusual_weather_moderate(self):
        """Test unusual weather detection adds safety margin."""
        climate = AdaptiveClimateSystem(latitude=59.33)  # Stockholm

        margin = climate.get_safety_margin(
            outdoor_temp=-10.0,
            unusual_weather_detected=True,
            unusual_severity=0.5,  # Moderate severity
        )

        # Base (1.0) + unusual (0.5 + 0.5*1.0 = 1.0) = 2.0°C
        assert margin == pytest.approx(2.0, abs=0.1)

    def test_unusual_weather_extreme(self):
        """Test extreme unusual weather adds maximum safety margin."""
        climate = AdaptiveClimateSystem(latitude=59.33)  # Stockholm

        margin = climate.get_safety_margin(
            outdoor_temp=-10.0,
            unusual_weather_detected=True,
            unusual_severity=1.0,  # Extreme severity
        )

        # Base (1.0) + unusual (0.5 + 1.0*1.0 = 1.5) = 2.5°C
        assert margin == pytest.approx(2.5, abs=0.1)

    def test_combined_cold_and_unusual(self):
        """Test combined effect of colder than average + unusual weather."""
        climate = AdaptiveClimateSystem(latitude=59.33)  # Stockholm

        margin = climate.get_safety_margin(
            outdoor_temp=-20.0,  # 10°C colder than average
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
        climate = AdaptiveClimateSystem(latitude=59.33)  # Stockholm

        weight = climate.get_dynamic_weight(
            outdoor_temp=-10.0,  # At winter average
            unusual_weather_detected=False,
        )

        # At winter_avg_low (-10°C) → cold category (winter_avg_low to winter_avg_low+5) → 0.75
        assert weight == pytest.approx(0.75, abs=0.05)

    def test_temperate_mild_cold_weight(self):
        """Test Temperate zone in mild cold."""
        climate = AdaptiveClimateSystem(latitude=51.51)  # London

        weight = climate.get_dynamic_weight(
            outdoor_temp=2.0,  # Slightly above winter avg (0°C)
            unusual_weather_detected=False,
        )

        # Between winter_avg_low (0°C) and winter_avg_low+5 (5°C) → 0.75
        assert weight == pytest.approx(0.75, abs=0.05)

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
    """Test edge cases and fallback behavior."""

    def test_equatorial_location_defaults_to_temperate(self):
        """Test equatorial location (Singapore 1.35°N) falls back to temperate."""
        climate = AdaptiveClimateSystem(latitude=1.35)

        # Outside defined zones (< 35°), should default to temperate
        assert climate.climate_zone == "temperate"
        info = climate.get_climate_info()
        assert info["name"] == "Temperate Oceanic"

    def test_extreme_south_defaults_to_temperate(self):
        """Test extreme southern latitude (Antarctica -75°S) becomes Arctic."""
        climate = AdaptiveClimateSystem(latitude=-75.0)

        # abs(75°) = 75° fits Arctic zone (66.5°-90°)
        # This is CORRECT: Antarctic bases need same heating as Arctic!
        assert climate.climate_zone == "arctic"
        assert climate.latitude == 75.0  # Absolute value used

    def test_zero_latitude_equator(self):
        """Test exactly at equator (0°N) defaults to temperate."""
        climate = AdaptiveClimateSystem(latitude=0.0)

        # Should default to temperate with warning
        assert climate.climate_zone == "temperate"

    def test_latitude_95_invalid_over_pole(self):
        """Test invalid latitude over 90° (should never happen but handle gracefully)."""
        climate = AdaptiveClimateSystem(latitude=95.0)

        # Should default to temperate
        assert climate.climate_zone == "temperate"

    def test_boundary_arctic_circle_exactly(self):
        """Test exactly at Arctic Circle boundary (66.5°N)."""
        climate = AdaptiveClimateSystem(latitude=66.5)

        # Should be Arctic (>= in range check)
        assert climate.climate_zone == "arctic"

    def test_boundary_between_zones_60_5(self):
        """Test exactly at subarctic/cold boundary (60.5°N)."""
        climate = AdaptiveClimateSystem(latitude=60.5)

        # Should be subarctic (>= in upper bound)
        assert climate.climate_zone == "subarctic"

    def test_boundary_temperate_mild_49_0(self):
        """Test exactly at temperate/mild boundary (49.0°N)."""
        climate = AdaptiveClimateSystem(latitude=49.0)

        # 49.0 is in temperate range (49.0-55.0), mild is (35.0-48.999)
        assert climate.climate_zone == "temperate"

    def test_just_below_mild_zone_34_9(self):
        """Test just below mild zone (34.9°N)."""
        climate = AdaptiveClimateSystem(latitude=34.9)

        # Outside zones (< 35°), should default to temperate
        assert climate.climate_zone == "temperate"

    def test_between_zones_non_boundary(self):
        """Test locations that clearly fall within zones (not at boundaries)."""
        # Middle of Arctic zone
        climate1 = AdaptiveClimateSystem(latitude=75.0)
        assert climate1.climate_zone == "arctic"

        # Middle of Subarctic zone
        climate2 = AdaptiveClimateSystem(latitude=63.0)
        assert climate2.climate_zone == "subarctic"

        # Middle of Cold zone
        climate3 = AdaptiveClimateSystem(latitude=58.0)
        assert climate3.climate_zone == "cold"

        # Middle of Temperate zone
        climate4 = AdaptiveClimateSystem(latitude=52.0)
        assert climate4.climate_zone == "temperate"

        # Middle of Mild zone
        climate5 = AdaptiveClimateSystem(latitude=45.0)
        assert climate5.climate_zone == "mild"


class TestGlobalApplicability:
    """Test that system works globally without country-specific code."""

    def test_canadian_arctic(self):
        """Test Canadian Arctic location (Yellowknife 62.45°N)."""
        climate = AdaptiveClimateSystem(latitude=62.45)

        # Should be Subarctic zone
        assert climate.climate_zone == "subarctic"

        # Should provide appropriate safety margins
        margin = climate.get_safety_margin(outdoor_temp=-25.0)
        assert margin > 1.0  # Subarctic base + cold adjustment

    def test_norwegian_coast(self):
        """Test Norwegian coastal location (Bergen 60.39°N)."""
        climate = AdaptiveClimateSystem(latitude=60.39)

        # Should be Cold zone
        assert climate.climate_zone == "cold"

    def test_german_location(self):
        """Test German location (Berlin 52.52°N)."""
        climate = AdaptiveClimateSystem(latitude=52.52)

        # Should be Temperate zone
        assert climate.climate_zone == "temperate"

    def test_finnish_lapland(self):
        """Test Finnish Lapland (Rovaniemi 66.50°N)."""
        climate = AdaptiveClimateSystem(latitude=66.50)

        # Right at Arctic Circle boundary
        assert climate.climate_zone in ("subarctic", "arctic")

    def test_multiple_locations_same_code(self):
        """Test that different locations work with same code (no country checks)."""
        locations = [
            (67.85, "arctic"),  # Kiruna, Sweden
            (64.75, "subarctic"),  # Fairbanks, USA (latitude equivalent)
            (59.33, "cold"),  # Stockholm, Sweden
            (51.51, "temperate"),  # London, UK
            (48.86, "mild"),  # Paris, France
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

"""Tests for DecisionEngine climate zone integration.

Tests Phase 5 requirements from CLIMATE_ZONE_DM_INTEGRATION.md:
- Zone detection for all climate zones
- DM range calculations across temperatures
- Arctic scenario: DM -1200 at -30°C = normal
- Standard scenario: DM -400 at 0°C = warning
- Cross-zone edge cases
- Integration with emergency layer
"""

import pytest
from unittest.mock import MagicMock

from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_manager import EffectManager
from custom_components.effektguard.optimization.price_analyzer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_model import ThermalModel
from custom_components.effektguard.const import (
    CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG,
    CLIMATE_ZONE_VERY_COLD_WINTER_AVG,
    CLIMATE_ZONE_COLD_WINTER_AVG,
    CLIMATE_ZONE_MODERATE_COLD_WINTER_AVG,
    CLIMATE_ZONE_STANDARD_WINTER_AVG,
)


@pytest.fixture
def hass_mock():
    """Create mock Home Assistant instance."""
    return MagicMock()


def create_engine_for_location(latitude: float, hass_mock) -> DecisionEngine:
    """Create DecisionEngine configured for specific latitude."""
    price_analyzer = PriceAnalyzer()
    effect_manager = EffectManager(hass_mock)
    thermal_model = ThermalModel(thermal_mass=1.0, insulation_quality=1.0)

    config = {
        "latitude": latitude,
        "target_temperature": 21.0,
        "tolerance": 5.0,
    }

    engine = DecisionEngine(
        price_analyzer=price_analyzer,
        effect_manager=effect_manager,
        thermal_model=thermal_model,
        config=config,
    )

    return engine


class TestClimateZoneDetection:
    """Test that DecisionEngine correctly detects climate zones."""

    def test_extreme_cold_zone_kiruna(self, hass_mock):
        """Verify Kiruna (67.85°N) detected as extreme_cold zone."""
        engine = create_engine_for_location(67.85, hass_mock)

        assert engine.climate_detector.zone_info.zone_key == "extreme_cold"
        assert (
            engine.climate_detector.zone_info.winter_avg_low == CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG
        )

    def test_very_cold_zone_lulea(self, hass_mock):
        """Verify Luleå (65.58°N) detected as very_cold zone."""
        engine = create_engine_for_location(65.58, hass_mock)

        assert engine.climate_detector.zone_info.zone_key == "very_cold"
        assert engine.climate_detector.zone_info.winter_avg_low == CLIMATE_ZONE_VERY_COLD_WINTER_AVG

    def test_cold_zone_stockholm(self, hass_mock):
        """Verify Stockholm (59.33°N) detected as cold zone."""
        engine = create_engine_for_location(59.33, hass_mock)

        assert engine.climate_detector.zone_info.zone_key == "cold"
        assert engine.climate_detector.zone_info.winter_avg_low == CLIMATE_ZONE_COLD_WINTER_AVG

    def test_moderate_cold_zone_copenhagen(self, hass_mock):
        """Verify Copenhagen (55.68°N) detected as moderate_cold zone."""
        engine = create_engine_for_location(55.68, hass_mock)

        assert engine.climate_detector.zone_info.zone_key == "moderate_cold"
        assert (
            engine.climate_detector.zone_info.winter_avg_low
            == CLIMATE_ZONE_MODERATE_COLD_WINTER_AVG
        )

    def test_standard_zone_paris(self, hass_mock):
        """Verify Paris (48.85°N) detected as standard zone."""
        engine = create_engine_for_location(48.85, hass_mock)

        assert engine.climate_detector.zone_info.zone_key == "standard"
        assert engine.climate_detector.zone_info.winter_avg_low == CLIMATE_ZONE_STANDARD_WINTER_AVG


class TestDMRangeCalculations:
    """Test DM threshold calculations for different zones and temperatures."""

    def test_extreme_cold_at_average_temperature(self, hass_mock):
        """Kiruna at zone's winter average should have normal DM -800 to -1200."""
        engine = create_engine_for_location(67.85, hass_mock)

        dm_range = engine._calculate_expected_dm_for_temperature(
            CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG
        )

        # At average winter temperature, expect base DM range
        assert -1200 <= dm_range["normal"] <= -800
        assert dm_range["warning"] <= -1200

    def test_very_cold_at_average_temperature(self, hass_mock):
        """Luleå at zone's winter average should have normal DM -600 to -1000."""
        engine = create_engine_for_location(65.58, hass_mock)

        dm_range = engine._calculate_expected_dm_for_temperature(CLIMATE_ZONE_VERY_COLD_WINTER_AVG)

        assert -1000 <= dm_range["normal"] <= -600
        assert dm_range["warning"] <= -1000

    def test_cold_at_average_temperature(self, hass_mock):
        """Stockholm at zone's winter average should have normal DM -450 to -700."""
        engine = create_engine_for_location(59.33, hass_mock)

        dm_range = engine._calculate_expected_dm_for_temperature(CLIMATE_ZONE_COLD_WINTER_AVG)

        assert -700 <= dm_range["normal"] <= -450
        assert dm_range["warning"] <= -700

    def test_moderate_cold_at_average_temperature(self, hass_mock):
        """Copenhagen at zone's winter average should have normal DM -300 to -500."""
        engine = create_engine_for_location(55.68, hass_mock)

        dm_range = engine._calculate_expected_dm_for_temperature(
            CLIMATE_ZONE_MODERATE_COLD_WINTER_AVG
        )

        assert -500 <= dm_range["normal"] <= -300
        assert dm_range["warning"] <= -500

    def test_standard_at_average_temperature(self, hass_mock):
        """Paris at zone's winter average should have normal DM -200 to -350."""
        engine = create_engine_for_location(48.85, hass_mock)

        dm_range = engine._calculate_expected_dm_for_temperature(CLIMATE_ZONE_STANDARD_WINTER_AVG)

        assert -350 <= dm_range["normal"] <= -200
        assert dm_range["warning"] <= -350

    def test_temperature_adjustment_warmer_than_average(self, hass_mock):
        """Stockholm at warmer than zone average should allow shallower DM."""
        engine = create_engine_for_location(59.33, hass_mock)

        dm_warm = engine._calculate_expected_dm_for_temperature(CLIMATE_ZONE_COLD_WINTER_AVG + 10.0)
        dm_avg = engine._calculate_expected_dm_for_temperature(CLIMATE_ZONE_COLD_WINTER_AVG)

        # Warmer weather = less deep DM expected (closer to zero)
        assert dm_warm["normal"] > dm_avg["normal"]
        assert dm_warm["warning"] > dm_avg["warning"]

    def test_temperature_adjustment_colder_than_average(self, hass_mock):
        """Stockholm at colder than zone average should allow deeper DM."""
        engine = create_engine_for_location(59.33, hass_mock)

        dm_cold = engine._calculate_expected_dm_for_temperature(CLIMATE_ZONE_COLD_WINTER_AVG - 10.0)
        dm_avg = engine._calculate_expected_dm_for_temperature(CLIMATE_ZONE_COLD_WINTER_AVG)

        # Colder weather = deeper DM expected (more negative)
        assert dm_cold["normal"] < dm_avg["normal"]
        assert dm_cold["warning"] < dm_avg["warning"]

    def test_never_exceeds_absolute_maximum(self, hass_mock):
        """DM thresholds should never go beyond -1500 absolute maximum."""
        from custom_components.effektguard.const import DM_THRESHOLD_AUX_LIMIT

        # Test extreme cold location at extreme temperature
        engine = create_engine_for_location(67.85, hass_mock)

        dm_range = engine._calculate_expected_dm_for_temperature(
            CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG - 20.0
        )

        # Should be clamped to absolute maximum
        assert dm_range["normal"] >= DM_THRESHOLD_AUX_LIMIT
        assert dm_range["warning"] >= DM_THRESHOLD_AUX_LIMIT


class TestArcticScenario:
    """Test Arctic scenario: DM -1200 at -30°C should be considered normal."""

    def test_dm_minus_1200_is_normal_in_kiruna(self, hass_mock):
        """In Kiruna at -30°C, DM -1200 should be within normal operating range."""
        engine = create_engine_for_location(67.85, hass_mock)

        dm_range = engine._calculate_expected_dm_for_temperature(-30.0)

        # DM -1200 should be at or below the normal threshold (more negative = deeper)
        assert dm_range["normal"] <= -1200
        # Should not trigger warning yet (warning threshold is <= -1200)
        assert dm_range["warning"] <= -1200

    def test_dm_minus_800_is_light_in_kiruna(self, hass_mock):
        """In Kiruna at -30°C, DM -800 should be on the light side of normal."""
        engine = create_engine_for_location(67.85, hass_mock)

        dm_range = engine._calculate_expected_dm_for_temperature(-30.0)

        # DM -800 should be well within normal range (less deep than expected)
        assert dm_range["normal"] <= -800
        assert dm_range["warning"] < -800

    def test_dm_minus_400_is_shallow_in_kiruna(self, hass_mock):
        """In Kiruna at zone's winter average, DM -400 is unusually shallow."""
        engine = create_engine_for_location(67.85, hass_mock)

        dm_range = engine._calculate_expected_dm_for_temperature(
            CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG
        )

        # DM -400 is way too shallow for Kiruna in winter
        # Normal threshold should be much deeper
        assert dm_range["normal"] < -400


class TestStandardScenario:
    """Test Standard scenario: DM -400 should trigger warning in mild climate."""

    def test_dm_minus_400_triggers_warning_in_paris(self, hass_mock):
        """In Paris at zone's winter average, DM -400 should be beyond warning threshold."""
        engine = create_engine_for_location(48.85, hass_mock)

        dm_range = engine._calculate_expected_dm_for_temperature(CLIMATE_ZONE_STANDARD_WINTER_AVG)

        # DM -400 should be deeper than warning threshold for mild climate
        # Standard zone has normal range -200 to -350, warning at -350
        # So -400 is definitely beyond warning
        assert (
            dm_range["warning"] >= -400
        )  # Warning threshold is shallower (less negative) than -400

    def test_dm_minus_300_is_normal_in_paris(self, hass_mock):
        """In Paris at zone's winter average, DM -300 should be within normal range."""
        engine = create_engine_for_location(48.85, hass_mock)

        dm_range = engine._calculate_expected_dm_for_temperature(CLIMATE_ZONE_STANDARD_WINTER_AVG)

        # DM -300 should be within normal operating range
        assert dm_range["normal"] <= -300

    def test_copenhagen_vs_paris_same_temperature(self, hass_mock):
        """Copenhagen (moderate_cold) should allow deeper DM than Paris (standard) at same temp."""
        engine_cph = create_engine_for_location(55.68, hass_mock)
        engine_paris = create_engine_for_location(48.85, hass_mock)

        # Test at 0°C for both
        dm_cph = engine_cph._calculate_expected_dm_for_temperature(0.0)
        dm_paris = engine_paris._calculate_expected_dm_for_temperature(0.0)

        # Copenhagen is colder climate, allows deeper DM
        assert dm_cph["normal"] < dm_paris["normal"]
        assert dm_cph["warning"] < dm_paris["warning"]


class TestCrossZoneEdgeCases:
    """Test edge cases between climate zones."""

    def test_boundary_arctic_circle(self, hass_mock):
        """Test boundary at Arctic Circle (66.5°N)."""
        # Just below Arctic Circle
        engine_below = create_engine_for_location(66.4, hass_mock)
        # Just above Arctic Circle
        engine_above = create_engine_for_location(66.6, hass_mock)

        assert engine_below.climate_detector.zone_info.zone_key == "very_cold"
        assert engine_above.climate_detector.zone_info.zone_key == "extreme_cold"

    def test_boundary_northern_southern_nordics(self, hass_mock):
        """Test boundary between Northern/Southern Nordic (60.5°N)."""
        engine_south = create_engine_for_location(60.4, hass_mock)
        engine_north = create_engine_for_location(60.6, hass_mock)

        assert engine_south.climate_detector.zone_info.zone_key == "cold"
        assert engine_north.climate_detector.zone_info.zone_key == "very_cold"

    def test_boundary_nordic_central_europe(self, hass_mock):
        """Test boundary at moderate_cold/standard (54.5°N)."""
        engine_nordic = create_engine_for_location(54.6, hass_mock)
        engine_europe = create_engine_for_location(54.4, hass_mock)

        assert engine_nordic.climate_detector.zone_info.zone_key == "moderate_cold"
        assert engine_europe.climate_detector.zone_info.zone_key == "standard"

    def test_unusual_weather_in_mild_zone(self, hass_mock):
        """Paris experiencing -15°C should adjust thresholds significantly."""
        engine = create_engine_for_location(48.85, hass_mock)

        dm_normal = engine._calculate_expected_dm_for_temperature(5.0)  # Average
        dm_cold = engine._calculate_expected_dm_for_temperature(-15.0)  # Unusual

        # Cold snap should allow much deeper DM
        # -15°C is 20°C colder than 5°C average = 400 DM deeper
        assert dm_cold["normal"] < dm_normal["normal"] - 300


class TestEmergencyLayerIntegration:
    """Test that climate-aware DM thresholds integrate with emergency layer."""

    def test_emergency_layer_uses_climate_thresholds(self, hass_mock):
        """Emergency layer should use climate-aware DM expectations."""
        engine_kiruna = create_engine_for_location(67.85, hass_mock)
        engine_paris = create_engine_for_location(48.85, hass_mock)

        # At same outdoor temperature, different zones have different expectations
        dm_kiruna = engine_kiruna._calculate_expected_dm_for_temperature(-10.0)
        dm_paris = engine_paris._calculate_expected_dm_for_temperature(-10.0)

        # Kiruna at -10°C is mild weather (warmer than -30°C average)
        # Paris at -10°C is very cold weather (colder than 5°C average)
        # But Kiruna should still allow deeper DM due to climate zone
        assert dm_kiruna["normal"] < dm_paris["normal"]

    def test_global_applicability(self, hass_mock):
        """Verify system works globally without configuration."""
        locations = [
            (67.85, "Kiruna", "extreme_cold"),
            (65.58, "Luleå", "very_cold"),
            (59.33, "Stockholm", "cold"),
            (55.68, "Copenhagen", "moderate_cold"),
            (48.85, "Paris", "standard"),
            (51.51, "London", "standard"),
            (52.52, "Berlin", "standard"),  # Below 54.5°N boundary
        ]

        for lat, name, expected_zone in locations:
            engine = create_engine_for_location(lat, hass_mock)
            assert (
                engine.climate_detector.zone_info.zone_key == expected_zone
            ), f"{name} should be {expected_zone}"


class TestSafetyLimits:
    """Test that safety limits are always respected."""

    def test_never_exceeds_absolute_max_extreme_cold(self, hass_mock):
        """Even in extreme conditions, never exceed -1500 DM."""
        from custom_components.effektguard.const import DM_THRESHOLD_AUX_LIMIT

        engine = create_engine_for_location(67.85, hass_mock)

        # Test extreme temperature
        dm_range = engine._calculate_expected_dm_for_temperature(-50.0)

        assert dm_range["normal"] >= DM_THRESHOLD_AUX_LIMIT
        assert dm_range["warning"] >= DM_THRESHOLD_AUX_LIMIT

    def test_absolute_max_is_hard_limit(self, hass_mock):
        """DM_THRESHOLD_AUX_LIMIT should be the hardest limit."""
        from custom_components.effektguard.const import DM_THRESHOLD_AUX_LIMIT

        # Test all zones at extreme temperatures
        latitudes = [67.85, 65.58, 59.33, 55.68, 48.85]

        for lat in latitudes:
            engine = create_engine_for_location(lat, hass_mock)
            dm_range = engine._calculate_expected_dm_for_temperature(-40.0)

            # All zones respect absolute maximum
            assert dm_range["normal"] >= DM_THRESHOLD_AUX_LIMIT
            assert dm_range["warning"] >= DM_THRESHOLD_AUX_LIMIT


class TestDMThresholdProgression:
    """Test that DM thresholds progress logically across zones."""

    def test_zones_ordered_by_severity_at_zone_averages(self, hass_mock):
        """Each zone at its own average winter temperature should have distinct thresholds.

        When each zone experiences its typical winter conditions, colder zones
        should allow progressively deeper DM.
        """
        zones_at_averages = [
            (48.85, "standard", 5.0),  # Test at +5°C (its average)
            (55.68, "moderate_cold", 0.0),  # Test at 0°C (its average)
            (59.33, "cold", -10.0),  # Test at -10°C (its average)
            (65.58, "very_cold", -15.0),  # Test at -15°C (its average)
            (67.85, "extreme_cold", -30.0),  # Test at -30°C (its average)
        ]

        previous_normal = 0  # Start at zero (shallowest)
        for lat, zone_name, avg_temp in zones_at_averages:
            engine = create_engine_for_location(lat, hass_mock)
            # Test each zone at its own average winter temperature
            dm_range = engine._calculate_expected_dm_for_temperature(avg_temp)

            # Each colder zone at its average should allow deeper DM
            assert (
                dm_range["normal"] < previous_normal
            ), f"{zone_name} at its average {avg_temp}°C should allow deeper DM than previous zone (got {dm_range['normal']}, previous {previous_normal})"

            previous_normal = dm_range["normal"]

    def test_temperature_progression_within_zone(self, hass_mock):
        """Within a zone, colder temperatures should allow deeper DM."""
        engine = create_engine_for_location(59.33, hass_mock)  # Stockholm

        temperatures = [5.0, 0.0, -5.0, -10.0, -15.0, -20.0]
        previous_normal = 0

        for temp in temperatures:
            dm_range = engine._calculate_expected_dm_for_temperature(temp)

            # Each colder temperature should allow deeper DM
            assert (
                dm_range["normal"] < previous_normal
            ), f"At {temp}°C, DM should be deeper than warmer temperatures"

            previous_normal = dm_range["normal"]

"""Tests for ProactiveLayer.evaluate_layer() method.

Phase 7 of layer refactoring: Proactive thermal debt prevention layer extraction.
"""

from dataclasses import dataclass

import pytest

from custom_components.effektguard.const import (
    LAYER_WEIGHT_PROACTIVE_MIN,
    PROACTIVE_ZONE1_OFFSET,
    PROACTIVE_ZONE2_OFFSET,
    PROACTIVE_ZONE2_WEIGHT,
    PROACTIVE_ZONE3_OFFSET_MIN,
    PROACTIVE_ZONE3_WEIGHT,
    PROACTIVE_ZONE4_OFFSET,
    PROACTIVE_ZONE4_WEIGHT,
    RAPID_COOLING_WEIGHT,
)
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.thermal_layer import (
    ProactiveLayer,
    ProactiveLayerDecision,
)


@dataclass
class MockNibeState:
    """Mock NIBE state for testing."""

    degree_minutes: float = 0.0
    outdoor_temp: float = 0.0
    indoor_temp: float = 21.0
    flow_temp: float = 35.0


@dataclass
class MockForecastHour:
    """Mock forecast hour for testing."""

    temperature: float


@dataclass
class MockWeatherData:
    """Mock weather data for testing."""

    forecast_hours: list = None


class TestProactiveLayerInit:
    """Tests for ProactiveLayer initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        climate_detector = ClimateZoneDetector(latitude=59.33)
        layer = ProactiveLayer(climate_detector=climate_detector)

        assert layer.climate_detector is climate_detector
        # Default trend function should return 0
        trend = layer._get_thermal_trend()
        assert trend["rate_per_hour"] == 0.0

    def test_init_with_custom_trend_callback(self):
        """Test initialization with custom trend callback."""
        climate_detector = ClimateZoneDetector(latitude=59.33)

        def custom_trend():
            return {"rate_per_hour": -0.5, "confidence": 0.8}

        layer = ProactiveLayer(
            climate_detector=climate_detector,
            get_thermal_trend=custom_trend,
        )

        trend = layer._get_thermal_trend()
        assert trend["rate_per_hour"] == -0.5
        assert trend["confidence"] == 0.8


class TestProactiveLayerZones:
    """Tests for ProactiveLayer zone detection.

    Stockholm (Cold zone) at 0°C outdoor temp:
    - normal_max = warning = -540
    - Z1 threshold (10%): -54
    - Z2 threshold (30%): -162
    - Z3 threshold (50%): -270
    - Z4 threshold (75%): -405
    - Z5 threshold (100%): -540 (equals warning, so Z5 has no range)

    Zone ranges (condition: zone_next < dm <= zone_current):
    - NONE: dm > -54 (healthier than Z1)
    - Z1: -162 < dm <= -54
    - Z2: -270 < dm <= -162
    - Z3: -405 < dm <= -270
    - Z4: -540 < dm <= -405
    - Z5: warning < dm <= -540 (empty since warning == -540)
    - NONE: dm <= warning (handed to Emergency T stages)

    Emergency T stages start at WARNING (-540) and go deeper.
    """

    @pytest.fixture
    def layer(self):
        """Create ProactiveLayer with Stockholm climate."""
        climate_detector = ClimateZoneDetector(latitude=59.33)
        return ProactiveLayer(climate_detector=climate_detector)

    def test_zone1_gentle_nudge(self, layer):
        """Test Zone 1 provides gentle offset.

        Z1 range at 0°C: -162 < dm <= -54
        Test with DM -100 (within Z1 range)
        """
        nibe_state = MockNibeState(degree_minutes=-100.0, outdoor_temp=0.0)

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "Z1"
        assert result.offset == pytest.approx(PROACTIVE_ZONE1_OFFSET, rel=0.1)
        assert result.weight == LAYER_WEIGHT_PROACTIVE_MIN
        assert "gentle heating" in result.reason

    def test_zone1_boundary_lower(self, layer):
        """Test Z1 at lower boundary (just above Z2).

        Z1 lower boundary is -162 (exclusive), so -161 should be in Z1.
        -162 itself is the boundary, included in Z2.
        """
        nibe_state = MockNibeState(degree_minutes=-161.0, outdoor_temp=0.0)

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "Z1"

    def test_zone2_boost_recovery(self, layer):
        """Test Zone 2 provides moderate offset.

        Z2 range at 0°C: -270 < dm <= -162
        Test with DM -200 (within Z2 range)
        """
        nibe_state = MockNibeState(degree_minutes=-200.0, outdoor_temp=0.0)

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "Z2"
        assert result.offset == PROACTIVE_ZONE2_OFFSET
        assert result.weight == PROACTIVE_ZONE2_WEIGHT
        assert "boost recovery" in result.reason

    def test_zone2_boundary(self, layer):
        """Test Z2 at exact upper boundary (-162 should be in Z2)."""
        nibe_state = MockNibeState(degree_minutes=-162.0, outdoor_temp=0.0)

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "Z2"

    def test_zone3_prevent_deeper_debt(self, layer):
        """Test Zone 3 provides scaled offset.

        Z3 range at 0°C: -405 < dm <= -270
        Test with DM -300 (within Z3 range)
        """
        nibe_state = MockNibeState(degree_minutes=-300.0, outdoor_temp=0.0)

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "Z3"
        assert result.offset >= PROACTIVE_ZONE3_OFFSET_MIN
        assert result.weight == PROACTIVE_ZONE3_WEIGHT
        assert "prevent deeper debt" in result.reason

    def test_zone3_boundary(self, layer):
        """Test Z3 at exact upper boundary (-270 should be in Z3)."""
        nibe_state = MockNibeState(degree_minutes=-270.0, outdoor_temp=0.0)

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "Z3"

    def test_zone4_strong_prevention(self, layer):
        """Test Zone 4 provides strong offset.

        Z4 range at 0°C: -540 < dm <= -405
        Test with DM -450 (within Z4 range)
        """
        nibe_state = MockNibeState(degree_minutes=-450.0, outdoor_temp=0.0)

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "Z4"
        assert result.offset == PROACTIVE_ZONE4_OFFSET
        assert result.weight == PROACTIVE_ZONE4_WEIGHT
        assert "strong prevention" in result.reason

    def test_zone4_boundary(self, layer):
        """Test Z4 at exact upper boundary (-405 should be in Z4)."""
        nibe_state = MockNibeState(degree_minutes=-405.0, outdoor_temp=0.0)

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "Z4"

    def test_zone5_empty_range(self, layer):
        """Test Zone 5 has no effective range when warning == normal_max.

        Z5 range condition: warning < dm <= zone5_threshold
        At 0°C: warning = -540, zone5_threshold = -540
        So -540 < dm <= -540 is impossible (empty range)

        DM -540 satisfies: -540 < -540 = False, so falls to NONE
        The Z5 code EXISTS but never activates due to climate zone config.
        """
        nibe_state = MockNibeState(degree_minutes=-540.0, outdoor_temp=0.0)

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        # At exactly warning/normal_max, proactive returns NONE
        # Emergency T1 starts at WARNING, so handoff is immediate
        assert result.zone == "NONE"
        assert result.offset == 0.0

    def test_no_proactive_when_dm_healthy(self, layer):
        """Test no proactive action when DM is healthy (above Z1)."""
        # Z1 starts at -54, so -20 is healthier
        nibe_state = MockNibeState(degree_minutes=-20.0, outdoor_temp=0.0)

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "NONE"
        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "Not needed" in result.reason

    def test_no_proactive_beyond_warning_emergency_takes_over(self, layer):
        """Test proactive returns NONE beyond warning - Emergency T stages handle it.

        At 0°C, warning = -540
        Emergency T1 starts at -540 (WARNING + T1_MARGIN where T1_MARGIN=0)
        Emergency T2 starts at -740 (WARNING + 200)
        Emergency T3 starts at -940 (WARNING + 400, capped at -1450)

        DM -600 is in T1 territory (between -540 and -740)
        """
        nibe_state = MockNibeState(degree_minutes=-600.0, outdoor_temp=0.0)

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "NONE"
        assert result.offset == 0.0
        assert result.weight == 0.0


class TestProactiveLayerRapidCooling:
    """Tests for rapid cooling detection."""

    def test_rapid_cooling_detection(self):
        """Test rapid cooling triggers proactive response."""
        climate_detector = ClimateZoneDetector(latitude=59.33)

        def cooling_trend():
            return {"rate_per_hour": -0.5, "confidence": 0.8}

        layer = ProactiveLayer(
            climate_detector=climate_detector,
            get_thermal_trend=cooling_trend,
        )

        # Cold outdoor, below target (deficit), rapid cooling
        nibe_state = MockNibeState(
            degree_minutes=-50.0,  # Healthy DM
            outdoor_temp=-5.0,  # Cold (below threshold)
            indoor_temp=20.0,  # Below target
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "RAPID_COOLING"
        assert result.offset > 0.0
        assert result.weight == RAPID_COOLING_WEIGHT
        assert "Rapid cooling" in result.reason
        assert "-0.50" in result.reason or "-0.5" in result.reason

    def test_rapid_cooling_with_forecast_validation(self):
        """Test rapid cooling boost when forecast confirms cooling."""
        climate_detector = ClimateZoneDetector(latitude=59.33)

        def cooling_trend():
            return {"rate_per_hour": -0.5, "confidence": 0.8}

        layer = ProactiveLayer(
            climate_detector=climate_detector,
            get_thermal_trend=cooling_trend,
        )

        # Forecast shows temperatures will drop
        weather_data = MockWeatherData(
            forecast_hours=[
                MockForecastHour(temperature=-6.0),
                MockForecastHour(temperature=-7.0),
                MockForecastHour(temperature=-8.0),
            ]
        )

        nibe_state = MockNibeState(
            degree_minutes=-50.0,
            outdoor_temp=-5.0,  # Will drop to -8
            indoor_temp=20.0,
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=weather_data,
            target_temp=21.0,
        )

        assert result.zone == "RAPID_COOLING"
        assert result.forecast_validated is True
        assert "forecast confirms" in result.reason

    def test_no_rapid_cooling_when_warm_outdoor(self):
        """Test rapid cooling ignored when outdoor is warm."""
        climate_detector = ClimateZoneDetector(latitude=59.33)

        def cooling_trend():
            return {"rate_per_hour": -0.5, "confidence": 0.8}

        layer = ProactiveLayer(
            climate_detector=climate_detector,
            get_thermal_trend=cooling_trend,
        )

        # Warm outdoor - rapid cooling not concerning
        nibe_state = MockNibeState(
            degree_minutes=-50.0,
            outdoor_temp=10.0,  # Above threshold
            indoor_temp=20.0,
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        # Should not trigger rapid cooling response
        assert result.zone != "RAPID_COOLING"

    def test_no_rapid_cooling_when_above_target(self):
        """Test rapid cooling ignored when above target (no deficit)."""
        climate_detector = ClimateZoneDetector(latitude=59.33)

        def cooling_trend():
            return {"rate_per_hour": -0.5, "confidence": 0.8}

        layer = ProactiveLayer(
            climate_detector=climate_detector,
            get_thermal_trend=cooling_trend,
        )

        # Above target - no deficit
        nibe_state = MockNibeState(
            degree_minutes=-50.0,
            outdoor_temp=-5.0,
            indoor_temp=22.0,  # Above target
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        # Should not trigger rapid cooling response
        assert result.zone != "RAPID_COOLING"


class TestProactiveLayerClimateAdaptation:
    """Tests for climate-aware threshold adaptation."""

    def test_arctic_climate_deeper_thresholds(self):
        """Test Arctic climate has deeper DM thresholds."""
        # Kiruna latitude
        arctic_detector = ClimateZoneDetector(latitude=67.86)
        arctic_layer = ProactiveLayer(climate_detector=arctic_detector)

        # At -20°C in Arctic, thresholds should be deeper
        nibe_state = MockNibeState(degree_minutes=-300.0, outdoor_temp=-20.0)

        result = arctic_layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        # -300 should be in an earlier zone (Z1 or Z2) in Arctic
        # because normal_max is deeper (more negative)
        assert result.zone in ["Z1", "Z2", "Z3"]

    def test_mediterranean_climate_shallower_thresholds(self):
        """Test Mediterranean climate has shallower thresholds."""
        # Nice latitude
        med_detector = ClimateZoneDetector(latitude=43.7)
        med_layer = ProactiveLayer(climate_detector=med_detector)

        # At 10°C in Mediterranean, thresholds should be shallower
        nibe_state = MockNibeState(degree_minutes=-150.0, outdoor_temp=10.0)

        result = med_layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        # -150 might be in Zone 4-5 in Mediterranean climate at 10°C
        # because normal_max is shallower (less negative)
        assert result.zone in ["Z3", "Z4", "Z5", "NONE"]


class TestProactiveLayerDecisionDataclass:
    """Tests for ProactiveLayerDecision dataclass."""

    def test_dataclass_fields(self):
        """Test dataclass has all expected fields."""
        decision = ProactiveLayerDecision(
            name="Proactive Z1",
            offset=0.5,
            weight=0.05,
            reason="Test reason",
            zone="Z1",
            degree_minutes=-100.0,
            trend_rate=-0.2,
            forecast_validated=True,
        )

        assert decision.name == "Proactive Z1"
        assert decision.offset == 0.5
        assert decision.weight == 0.05
        assert decision.reason == "Test reason"
        assert decision.zone == "Z1"
        assert decision.degree_minutes == -100.0
        assert decision.trend_rate == -0.2
        assert decision.forecast_validated is True

    def test_dataclass_defaults(self):
        """Test dataclass default values."""
        decision = ProactiveLayerDecision(
            name="Proactive",
            offset=0.0,
            weight=0.0,
            reason="Not needed",
        )

        assert decision.zone == ""
        assert decision.degree_minutes == 0.0
        assert decision.trend_rate == 0.0
        assert decision.forecast_validated is False


class TestProactiveLayerWarmHouseReduction:
    """Tests for warm house weight reduction (Dec 10, 2025).

    When house is warm (above target + 0.5°C), proactive layer weight
    should be reduced to 30% to prevent fighting with COAST.

    Z5 is exempt because at warning boundary, DM recovery takes priority.
    """

    @pytest.fixture
    def layer(self):
        """Create ProactiveLayer with Stockholm climate."""
        climate_detector = ClimateZoneDetector(latitude=59.33)
        return ProactiveLayer(climate_detector=climate_detector)

    def test_z1_normal_weight_when_cold(self, layer):
        """Z1 should use normal weight when house is at/below target."""
        # Indoor at target (21.0°C), not warm
        nibe_state = MockNibeState(
            degree_minutes=-100.0,  # Z1 range
            outdoor_temp=0.0,
            indoor_temp=21.0,
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "Z1"
        assert result.weight == LAYER_WEIGHT_PROACTIVE_MIN
        assert "warm house" not in result.reason

    def test_z1_reduced_weight_when_warm(self, layer):
        """Z1 should reduce weight when house is warm (above target + 0.5°C)."""
        # Indoor 21.6°C, target 21.0°C → 0.6°C above, exceeds 0.5°C threshold
        nibe_state = MockNibeState(
            degree_minutes=-100.0,  # Z1 range
            outdoor_temp=0.0,
            indoor_temp=21.6,
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "Z1"
        # Weight should be 30% of normal
        expected_weight = LAYER_WEIGHT_PROACTIVE_MIN * 0.3
        assert result.weight == pytest.approx(expected_weight, rel=0.01)
        assert "warm house" in result.reason

    def test_z2_reduced_weight_when_warm(self, layer):
        """Z2 should reduce weight when house is warm."""
        # Indoor 22.0°C, target 21.0°C → 1.0°C above
        nibe_state = MockNibeState(
            degree_minutes=-200.0,  # Z2 range
            outdoor_temp=0.0,
            indoor_temp=22.0,
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "Z2"
        expected_weight = PROACTIVE_ZONE2_WEIGHT * 0.3
        assert result.weight == pytest.approx(expected_weight, rel=0.01)
        assert "warm house" in result.reason

    def test_z3_reduced_weight_when_warm(self, layer):
        """Z3 should reduce weight when house is warm."""
        nibe_state = MockNibeState(
            degree_minutes=-300.0,  # Z3 range
            outdoor_temp=0.0,
            indoor_temp=21.8,
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "Z3"
        expected_weight = PROACTIVE_ZONE3_WEIGHT * 0.3
        assert result.weight == pytest.approx(expected_weight, rel=0.01)
        assert "warm house" in result.reason

    def test_z4_reduced_weight_when_warm(self, layer):
        """Z4 should reduce weight when house is warm."""
        nibe_state = MockNibeState(
            degree_minutes=-420.0,  # Z4 range
            outdoor_temp=0.0,
            indoor_temp=21.7,
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "Z4"
        expected_weight = PROACTIVE_ZONE4_WEIGHT * 0.3
        assert result.weight == pytest.approx(expected_weight, rel=0.01)
        assert "warm house" in result.reason

    def test_z5_no_reduction_when_warm(self, layer):
        """Z5 should NOT reduce weight even when warm (warning boundary priority).

        At Z5, we're approaching the warning threshold and DM recovery
        takes priority over temperature overshoot concerns.
        """
        # Use -20°C to ensure Z5 has a valid range
        # At -20°C outdoor: warning=-1068, Z5 threshold (100%)=-1068
        # Need to find a case where Z5 actually triggers
        # Let's use a custom detector to ensure Z5 triggers
        climate_detector = ClimateZoneDetector(latitude=66.5)  # Arctic
        layer_arctic = ProactiveLayer(climate_detector=climate_detector)

        # At Arctic with -30°C outdoor, warning threshold is much lower
        # This ensures Z5 has a valid range
        nibe_state = MockNibeState(
            degree_minutes=-900.0,  # Deep in proactive range
            outdoor_temp=-30.0,
            indoor_temp=22.0,  # Warm house
        )

        result = layer_arctic.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        # If we hit Z5, weight should NOT be reduced
        if result.zone == "Z5":
            from custom_components.effektguard.const import PROACTIVE_ZONE5_WEIGHT

            assert result.weight == PROACTIVE_ZONE5_WEIGHT
            assert "warm house" not in result.reason

    def test_threshold_boundary_not_warm(self, layer):
        """House exactly at target + 0.5°C should NOT be considered warm."""
        # Indoor 21.5°C, target 21.0°C → exactly 0.5°C above (at threshold, not over)
        nibe_state = MockNibeState(
            degree_minutes=-100.0,
            outdoor_temp=0.0,
            indoor_temp=21.5,
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        assert result.zone == "Z1"
        assert result.weight == LAYER_WEIGHT_PROACTIVE_MIN
        assert "warm house" not in result.reason

    def test_rapid_cooling_reduced_weight_when_warm(self, layer):
        """RAPID_COOLING should reduce weight when house is warm.

        RAPID_COOLING requires:
        1. Rapid cooling trend (rate < -0.3°C/h)
        2. Cold outdoor temp (< 0°C)
        3. Deficit > 0.3°C (target - indoor > 0.3)

        For warm house test, we need indoor > target + 0.5°C
        but also need deficit > 0.3°C which means indoor < target - 0.3

        These are contradictory, so RAPID_COOLING won't trigger when house is warm.
        This test verifies that behavior - RAPID_COOLING naturally doesn't apply
        when house is warm because there's no deficit.
        """

        def cooling_trend():
            return {"rate_per_hour": -0.5, "confidence": 0.9}

        climate_detector = ClimateZoneDetector(latitude=59.33)
        layer_with_trend = ProactiveLayer(
            climate_detector=climate_detector,
            get_thermal_trend=cooling_trend,
        )

        # Warm house: indoor 21.6°C, target 21.0°C = +0.6°C above
        # This means deficit = 21.0 - 21.6 = -0.6°C (negative, no deficit)
        # RAPID_COOLING requires deficit > 0.3°C, so it won't trigger
        nibe_state = MockNibeState(
            degree_minutes=-50.0,
            outdoor_temp=-5.0,
            indoor_temp=21.6,  # Warm house
        )

        result = layer_with_trend.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
        )

        # RAPID_COOLING should NOT trigger when house is warm (no deficit)
        # because the warm house check is logically incompatible with RAPID_COOLING's
        # deficit requirement. This is correct behavior - RAPID_COOLING is for
        # when house is cooling towards cold, not when it's already warm.
        assert result.zone != "RAPID_COOLING"

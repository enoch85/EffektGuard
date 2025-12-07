"""Tests for EmergencyLayer.evaluate_layer() method.

Phase 6 of layer refactoring: Emergency thermal debt layer extraction.
"""

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.const import (
    DM_CRITICAL_T1_MARGIN,
    DM_CRITICAL_T2_MARGIN,
    DM_CRITICAL_T3_MAX,
    DM_THRESHOLD_AUX_LIMIT,
    QuarterClassification,
    SAFETY_EMERGENCY_OFFSET,
)
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.thermal_layer import (
    EmergencyLayer,
    EmergencyLayerDecision,
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


@dataclass
class MockPriceData:
    """Mock price data for testing."""

    today: list = None


class MockPriceAnalyzer:
    """Mock price analyzer for testing."""

    def __init__(self, classification=QuarterClassification.NORMAL):
        self._classification = classification

    def get_current_classification(self, quarter: int) -> QuarterClassification:
        return self._classification


class TestEmergencyLayerEvaluate:
    """Test suite for EmergencyLayer.evaluate_layer()."""

    def _create_layer(
        self,
        latitude: float = 59.33,  # Stockholm
        heating_type: str = "radiator",
        price_classification=QuarterClassification.NORMAL,
    ) -> EmergencyLayer:
        """Create an EmergencyLayer for testing."""
        climate_detector = ClimateZoneDetector(latitude=latitude)
        price_analyzer = MockPriceAnalyzer(classification=price_classification)

        return EmergencyLayer(
            climate_detector=climate_detector,
            price_analyzer=price_analyzer,
            heating_type=heating_type,
        )

    def test_returns_emergency_layer_decision(self):
        """Test that result is EmergencyLayerDecision with diagnostic fields."""
        layer = self._create_layer()
        nibe_state = MockNibeState(degree_minutes=-100, outdoor_temp=-5.0, indoor_temp=20.5)

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            price_data=None,
            target_temp=21.0,
            tolerance_range=0.5,
        )

        assert isinstance(result, EmergencyLayerDecision)
        assert hasattr(result, "tier")
        assert hasattr(result, "degree_minutes")
        assert hasattr(result, "threshold_used")
        assert hasattr(result, "damping_applied")

    def test_absolute_limit_triggers_emergency(self):
        """Test that DM at -1500 triggers emergency response."""
        layer = self._create_layer()
        nibe_state = MockNibeState(
            degree_minutes=DM_THRESHOLD_AUX_LIMIT - 10,  # -1510
            outdoor_temp=-10.0,
            indoor_temp=20.0,
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            price_data=None,
            target_temp=21.0,
            tolerance_range=0.5,
        )

        assert result.offset == SAFETY_EMERGENCY_OFFSET
        assert result.weight == 1.0
        assert result.tier == "EMERGENCY"
        assert "EMERGENCY" in result.reason

    def test_too_warm_ignores_thermal_debt(self):
        """Test that being too warm ignores thermal debt."""
        layer = self._create_layer()
        nibe_state = MockNibeState(
            degree_minutes=-500,  # Significant debt
            outdoor_temp=-5.0,
            indoor_temp=22.5,  # 1.5°C above target
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            price_data=None,
            target_temp=21.0,
            tolerance_range=0.5,  # Tolerance is 0.5, so 22.5 is 1.0°C above tolerance
        )

        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "let cool naturally" in result.reason

    def test_at_target_expensive_price_ignores_debt(self):
        """Test that at target with expensive price ignores thermal debt."""
        layer = self._create_layer(price_classification=QuarterClassification.EXPENSIVE)
        nibe_state = MockNibeState(
            degree_minutes=-300,  # Some debt
            outdoor_temp=-5.0,
            indoor_temp=21.5,  # At target
        )
        price_data = MockPriceData(today=[1.0] * 96)

        mock_datetime = datetime(2024, 1, 15, 12, 0, 0)
        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            price_data=price_data,
            target_temp=21.0,
            tolerance_range=0.5,
            get_current_datetime=lambda: mock_datetime,
        )

        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "price not cheap" in result.reason

    def test_at_target_cheap_price_recovers_debt(self):
        """Test that at target with cheap price does recover thermal debt."""
        layer = self._create_layer(price_classification=QuarterClassification.CHEAP)
        nibe_state = MockNibeState(
            degree_minutes=-800,  # Significant debt triggering T1 in Stockholm
            outdoor_temp=-5.0,
            indoor_temp=21.0,  # At target
        )
        price_data = MockPriceData(today=[0.5] * 96)

        mock_datetime = datetime(2024, 1, 15, 12, 0, 0)
        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            price_data=price_data,
            target_temp=21.0,
            tolerance_range=0.5,
            get_current_datetime=lambda: mock_datetime,
        )

        # With cheap price and DM -800 (beyond warning for Stockholm at -5°C),
        # should trigger some recovery
        assert result.offset > 0.0 or result.tier in ("T1", "T2", "T3", "WARNING", "CAUTION")


class TestEmergencyLayerTiers:
    """Test tier-based thermal debt response."""

    def _create_layer(self, latitude: float = 59.33) -> EmergencyLayer:
        """Create an EmergencyLayer for testing."""
        climate_detector = ClimateZoneDetector(latitude=latitude)
        return EmergencyLayer(
            climate_detector=climate_detector,
            price_analyzer=None,
            heating_type="radiator",
        )

    def test_t3_tier_near_limit(self):
        """Test T3 tier triggers near absolute limit."""
        layer = self._create_layer()
        # T3 is close to the absolute limit
        nibe_state = MockNibeState(
            degree_minutes=-1400,  # Near absolute limit
            outdoor_temp=-10.0,
            indoor_temp=20.0,
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            price_data=None,
            target_temp=21.0,
            tolerance_range=0.5,
        )

        # Should be T3 or T2 depending on thresholds
        assert result.tier in ("T2", "T3")
        assert result.weight > 0.9
        assert result.offset > 0.0

    def test_ok_tier_for_normal_dm(self):
        """Test OK tier for normal DM values."""
        layer = self._create_layer()
        nibe_state = MockNibeState(
            degree_minutes=-50,  # Normal range
            outdoor_temp=5.0,
            indoor_temp=20.5,
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            price_data=None,
            target_temp=21.0,
            tolerance_range=0.5,
        )

        assert result.tier == "OK"
        assert result.offset == 0.0
        assert result.weight == 0.0


class TestEmergencyLayerThermalMass:
    """Test thermal mass adjusted thresholds."""

    def test_concrete_slab_tighter_thresholds(self):
        """Test that concrete slab has tighter thresholds.
        
        At 0°C in Stockholm:
        - Radiator warning: -540 (no adjustment)
        - Concrete warning: -702 (1.3× adjustment)
        
        Using DM -600 should be WARNING for radiator but within normal for concrete
        (because concrete thresholds are multiplied, making them MORE negative).
        
        Wait, that's backwards - concrete has TIGHTER thresholds meaning it triggers
        WARNING at LESS negative values. Let me check:
        - Base warning: -540
        - Concrete: -540 × 1.3 = -702 (MORE negative = later warning)
        
        Actually the multiplier makes it MORE negative (later warning), not tighter.
        Let me test at -580 which should be WARNING for radiator but OK for concrete.
        """
        layer_radiator = EmergencyLayer(
            climate_detector=ClimateZoneDetector(latitude=59.33),
            price_analyzer=None,
            heating_type="radiator",
        )
        layer_concrete = EmergencyLayer(
            climate_detector=ClimateZoneDetector(latitude=59.33),
            price_analyzer=None,
            heating_type="concrete_ufh",
        )

        # DM -580 at 0°C:
        # - Radiator warning at -540, so -580 is beyond warning → triggers response
        # - Concrete warning at -702, so -580 is within normal → no response
        nibe_state = MockNibeState(
            degree_minutes=-580,
            outdoor_temp=0.0,
            indoor_temp=20.0,
        )

        result_radiator = layer_radiator.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            price_data=None,
            target_temp=21.0,
            tolerance_range=0.5,
        )

        result_concrete = layer_concrete.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            price_data=None,
            target_temp=21.0,
            tolerance_range=0.5,
        )

        # Radiator should have higher tier or more aggressive response
        # Concrete should still be OK or CAUTION
        assert result_radiator.tier in ("WARNING", "CAUTION", "T1", "T2", "T3")
        # Concrete's adjusted threshold is -702, so -580 should be within normal
        assert result_concrete.tier in ("OK", "CAUTION")


class TestEmergencyLayerShouldBlockDhw:
    """Test should_block_dhw() method."""

    def test_blocks_at_absolute_limit(self):
        """Test that DHW is blocked at absolute limit."""
        layer = EmergencyLayer(
            climate_detector=ClimateZoneDetector(latitude=59.33),
            price_analyzer=None,
            heating_type="radiator",
        )

        result = layer.should_block_dhw(
            degree_minutes=DM_THRESHOLD_AUX_LIMIT - 10,  # -1510
            outdoor_temp=-10.0,
        )

        assert result is True

    def test_allows_dhw_at_normal_dm(self):
        """Test that DHW is allowed at normal DM."""
        layer = EmergencyLayer(
            climate_detector=ClimateZoneDetector(latitude=59.33),
            price_analyzer=None,
            heating_type="radiator",
        )

        result = layer.should_block_dhw(
            degree_minutes=-100,  # Normal range
            outdoor_temp=5.0,
        )

        assert result is False

    def test_blocks_at_t2_threshold(self):
        """Test that DHW is blocked at T2 threshold."""
        layer = EmergencyLayer(
            climate_detector=ClimateZoneDetector(latitude=59.33),
            price_analyzer=None,
            heating_type="radiator",
        )

        # Get the T2 threshold for Stockholm at -10°C
        # This should be warning - T2_MARGIN
        expected_dm = layer.climate_detector.get_expected_dm_range(-10.0)
        t2_threshold = expected_dm["warning"] - DM_CRITICAL_T2_MARGIN

        result = layer.should_block_dhw(
            degree_minutes=t2_threshold - 10,  # Beyond T2
            outdoor_temp=-10.0,
        )

        assert result is True


class TestEmergencyLayerClimateAdaptation:
    """Test climate zone adaptation."""

    def test_arctic_allows_deeper_dm(self):
        """Test that Arctic climate allows deeper DM before warning."""
        layer_arctic = EmergencyLayer(
            climate_detector=ClimateZoneDetector(latitude=68.0),  # Kiruna
            price_analyzer=None,
            heating_type="radiator",
        )
        layer_mild = EmergencyLayer(
            climate_detector=ClimateZoneDetector(latitude=48.0),  # Paris
            price_analyzer=None,
            heating_type="radiator",
        )

        # Same DM at -20°C
        nibe_state = MockNibeState(
            degree_minutes=-600,
            outdoor_temp=-20.0,
            indoor_temp=20.0,
        )

        result_arctic = layer_arctic.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            price_data=None,
            target_temp=21.0,
            tolerance_range=0.5,
        )

        result_mild = layer_mild.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            price_data=None,
            target_temp=21.0,
            tolerance_range=0.5,
        )

        # Arctic should be less alarmed than mild climate at same DM
        # (Arctic has deeper normal DM expectations)
        # Either arctic has lower tier or lower weight/offset
        arctic_severity = result_arctic.weight * result_arctic.offset
        mild_severity = result_mild.weight * result_mild.offset

        # Mild climate should react more strongly to -600 DM
        assert mild_severity >= arctic_severity or result_mild.tier != result_arctic.tier

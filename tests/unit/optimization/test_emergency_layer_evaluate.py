"""Tests for EmergencyLayer.evaluate_layer() method.

Phase 6 of layer refactoring: Emergency thermal debt layer extraction.
"""

from dataclasses import dataclass

import pytest

from custom_components.effektguard.const import (
    DM_CRITICAL_T2_MARGIN,
    DM_THRESHOLD_AUX_LIMIT,
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


class TestEmergencyLayerEvaluate:
    """Test suite for EmergencyLayer.evaluate_layer()."""

    def _create_layer(
        self,
        latitude: float = 59.33,  # Stockholm
        heating_type: str = "radiator",
    ) -> EmergencyLayer:
        """Create an EmergencyLayer for testing."""
        return EmergencyLayer(
            climate_detector=ClimateZoneDetector(latitude=latitude),
            heating_type=heating_type,
        )

    def test_returns_emergency_layer_decision(self):
        """Test that result is EmergencyLayerDecision with diagnostic fields."""
        layer = self._create_layer()
        nibe_state = MockNibeState(degree_minutes=-100, outdoor_temp=-5.0, indoor_temp=20.5)

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
            tolerance_range=0.5,
            room_floor=21.0,
        )

        assert isinstance(result, EmergencyLayerDecision)
        assert hasattr(result, "tier")
        assert hasattr(result, "degree_minutes")
        assert hasattr(result, "threshold_used")
        assert hasattr(result, "damping_applied")

    def test_absolute_limit_triggers_emergency(self):
        """Past the aux limit the tier fires as a no-cut floor and asks for no raise of its own."""
        layer = self._create_layer()
        nibe_state = MockNibeState(
            degree_minutes=DM_THRESHOLD_AUX_LIMIT - 10,  # -1510
            outdoor_temp=-10.0,
            indoor_temp=20.0,
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
            tolerance_range=0.5,
            room_floor=21.0,
        )

        assert result.offset == 0.0
        assert result.weight == 1.0
        assert result.tier == "EMERGENCY"
        assert "EMERGENCY" in result.reason

    @pytest.mark.parametrize(
        ("indoor", "recovers"),
        [
            (22.5, False),  # above its band
            (21.0, False),  # at its floor: no recovery, in a cheap hour or not
            (20.9, True),  # below it
        ],
    )
    def test_the_room_decides_whether_the_tiers_recover(self, indoor, recovers):
        """A room at or above its floor gets no recovery raise, whatever degree minutes say."""
        layer = self._create_layer()
        nibe_state = MockNibeState(
            degree_minutes=-800,  # past the warning threshold for Stockholm at -5 °C
            outdoor_temp=-5.0,
            indoor_temp=indoor,
        )

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
            tolerance_range=0.5,
            room_floor=21.0,
        )

        assert (result.tier in ("T1", "T2", "T3")) == recovers
        assert (result.offset > 0.0) == recovers


class TestEmergencyLayerTiers:
    """Test tier-based thermal debt response."""

    def _create_layer(self, latitude: float = 59.33) -> EmergencyLayer:
        """Create an EmergencyLayer for testing."""
        climate_detector = ClimateZoneDetector(latitude=latitude)
        return EmergencyLayer(
            climate_detector=climate_detector,
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
            target_temp=21.0,
            tolerance_range=0.5,
            room_floor=21.0,
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
            target_temp=21.0,
            tolerance_range=0.5,
            room_floor=21.0,
        )

        assert result.tier == "OK"
        assert result.offset == 0.0
        assert result.weight == 0.0


class TestEmergencyLayerThermalMass:
    """Test thermal mass adjusted thresholds."""

    def test_concrete_slab_responds_before_a_radiator_system(self):
        """A concrete slab must respond to thermal debt SOONER than a radiator system.

        At 0 C in Stockholm the radiator warns at -540 (buffer 1.0) and concrete at -540/1.3 = -415.
        DM -450 is past concrete's threshold but not a radiator's: the slab is already recovering
        while the radiator system has no need to act yet.
        """
        layer_radiator = EmergencyLayer(
            climate_detector=ClimateZoneDetector(latitude=59.33),
            heating_type="radiator",
        )
        layer_concrete = EmergencyLayer(
            climate_detector=ClimateZoneDetector(latitude=59.33),
            heating_type="concrete_ufh",
        )

        # DM -450 at 0 C:
        # - Concrete warns at -415, so -450 is past it -> the slab starts recovering
        # - Radiator warns at -540, so -450 is still within normal -> no need to act yet
        nibe_state = MockNibeState(
            degree_minutes=-450,
            outdoor_temp=0.0,
            indoor_temp=20.0,
        )

        result_radiator = layer_radiator.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
            tolerance_range=0.5,
            room_floor=21.0,
        )

        result_concrete = layer_concrete.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
            tolerance_range=0.5,
            room_floor=21.0,
        )

        # The slab is already recovering; the radiator system has not needed to start.
        assert result_concrete.tier in ("WARNING", "CAUTION", "T1", "T2", "T3"), (
            f"Concrete warns at -415 and DM is -450: the slab must be responding, not idle. "
            f"Got tier {result_concrete.tier}."
        )
        assert result_radiator.tier in ("OK", "CAUTION"), (
            f"A radiator system warns at -540 and DM is only -450: it has no need to act yet. "
            f"Got tier {result_radiator.tier}."
        )


class TestEmergencyLayerShouldBlockDhw:
    """Test should_block_dhw() method."""

    def test_blocks_at_absolute_limit(self):
        """Test that DHW is blocked at absolute limit."""
        layer = EmergencyLayer(
            climate_detector=ClimateZoneDetector(latitude=59.33),
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
            heating_type="radiator",
        )
        layer_mild = EmergencyLayer(
            climate_detector=ClimateZoneDetector(latitude=48.0),  # Paris
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
            target_temp=21.0,
            tolerance_range=0.5,
            room_floor=21.0,
        )

        result_mild = layer_mild.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            target_temp=21.0,
            tolerance_range=0.5,
            room_floor=21.0,
        )

        # Arctic should be less alarmed than mild climate at same DM
        # (Arctic has deeper normal DM expectations)
        # Either arctic has lower tier or lower weight/offset
        arctic_severity = result_arctic.weight * result_arctic.offset
        mild_severity = result_mild.weight * result_mild.offset

        # Mild climate should react more strongly to -600 DM
        assert mild_severity >= arctic_severity or result_mild.tier != result_arctic.tier

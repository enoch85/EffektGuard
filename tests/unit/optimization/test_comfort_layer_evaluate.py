"""Tests for ComfortLayer.evaluate_layer() method.

Phase 8 of layer refactoring: Comfort layer extraction.
"""

from dataclasses import dataclass

import pytest

from custom_components.effektguard.const import (
    COMFORT_CORRECTION_MULT,
    LAYER_WEIGHT_COMFORT_MAX,
    LAYER_WEIGHT_COMFORT_MIN,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    OPTIMIZATION_MODE_COMFORT,
    OPTIMIZATION_MODE_SAVINGS,
)
from custom_components.effektguard.optimization.comfort_layer import ComfortLayer


@dataclass
class MockNibeState:
    """Mock NIBE state for testing."""

    indoor_temp: float = 21.0
    outdoor_temp: float = 0.0
    degree_minutes: float = 0.0
    flow_temp: float = 35.0


class TestComfortLayerInit:
    """Tests for ComfortLayer initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        layer = ComfortLayer()

        assert layer.tolerance_range == 0.5
        assert layer.target_temp == 21.0

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        layer = ComfortLayer(
            tolerance_range=1.0,
            target_temp=22.0,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_COMFORT],
        )

        assert layer.tolerance_range == 1.0
        assert layer.target_temp == 22.0
        assert layer.mode_config.dead_zone == 0.1  # Comfort mode


class TestComfortLayerAtTarget:
    """Tests for comfort layer when at target."""

    def test_at_target_no_action(self):
        """Test no action when exactly at target."""
        layer = ComfortLayer(target_temp=21.0, tolerance_range=0.5)
        nibe_state = MockNibeState(indoor_temp=21.0)

        result = layer.evaluate_layer(nibe_state=nibe_state)

        assert result.offset == 0.0
        assert result.weight == 0.0
        assert result.reason == "At target"

    def test_within_dead_zone_no_action(self):
        """Test no action when within dead zone."""
        layer = ComfortLayer(
            target_temp=21.0,
            tolerance_range=0.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],  # 0.2°C dead zone
        )
        nibe_state = MockNibeState(indoor_temp=21.1)  # 0.1°C above, within 0.2 dead zone

        result = layer.evaluate_layer(nibe_state=nibe_state)

        assert result.offset == 0.0
        assert result.weight == 0.0


class TestComfortLayerWithinTolerance:
    """Tests for comfort layer when within tolerance but outside dead zone."""

    def test_slightly_warm_gentle_reduce(self):
        """Test gentle reduction when slightly warm."""
        layer = ComfortLayer(
            target_temp=21.0,
            tolerance_range=0.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],
        )
        nibe_state = MockNibeState(indoor_temp=21.3)  # 0.3°C above, beyond 0.2 dead zone

        result = layer.evaluate_layer(nibe_state=nibe_state)

        assert result.offset < 0.0  # Should reduce heating
        expected_offset = -0.3 * COMFORT_CORRECTION_MULT
        assert result.offset == pytest.approx(expected_offset, rel=0.01)
        assert "Slightly warm" in result.reason

    def test_slightly_cool_gentle_boost(self):
        """Test gentle boost when slightly cool."""
        layer = ComfortLayer(
            target_temp=21.0,
            tolerance_range=0.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],
        )
        nibe_state = MockNibeState(indoor_temp=20.7)  # 0.3°C below, beyond 0.2 dead zone

        result = layer.evaluate_layer(nibe_state=nibe_state)

        assert result.offset > 0.0  # Should boost heating
        expected_offset = 0.3 * COMFORT_CORRECTION_MULT
        assert result.offset == pytest.approx(expected_offset, rel=0.01)
        assert "Slightly cool" in result.reason


class TestComfortLayerTooCold:
    """Tests for comfort layer when too cold."""

    def test_too_cold_increase_heating(self):
        """Test strong heating increase when too cold."""
        layer = ComfortLayer(target_temp=21.0, tolerance_range=0.5)
        # 1.0°C below tolerance = 1.5°C below target
        nibe_state = MockNibeState(indoor_temp=19.5)

        result = layer.evaluate_layer(nibe_state=nibe_state)

        assert result.offset > 0.0  # Should increase heating
        assert result.weight == LAYER_WEIGHT_COMFORT_MAX
        assert "Too cold" in result.reason


class TestComfortLayerModeAdjustment:
    """Tests for mode-based weight adjustments."""

    def test_comfort_mode_higher_weight(self):
        """Test comfort mode has higher weight multiplier."""
        layer = ComfortLayer(
            target_temp=21.0,
            tolerance_range=0.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_COMFORT],
        )
        nibe_state = MockNibeState(indoor_temp=21.3)

        result = layer.evaluate_layer(nibe_state=nibe_state)

        # Weight should be multiplied by 1.3 in comfort mode
        expected_weight = LAYER_WEIGHT_COMFORT_MIN * 1.3
        assert result.weight == pytest.approx(expected_weight, rel=0.01)

    def test_savings_mode_lower_weight(self):
        """Test savings mode has lower weight multiplier."""
        layer = ComfortLayer(
            target_temp=21.0,
            tolerance_range=0.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_SAVINGS],
        )
        nibe_state = MockNibeState(indoor_temp=21.4)  # Outside 0.3 savings dead zone

        result = layer.evaluate_layer(nibe_state=nibe_state)

        # Weight should be multiplied by 0.7 in savings mode
        expected_weight = LAYER_WEIGHT_COMFORT_MIN * 0.7
        assert result.weight == pytest.approx(expected_weight, rel=0.01)

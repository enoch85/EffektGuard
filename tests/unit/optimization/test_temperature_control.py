"""Temperature control validation tests.

Tests for comfort layer overshoot protection with coast offsets.

Validates that:
1. Comfort layer uses graduated coast offsets (-7 to -10°C) for overshoot
2. System prevents prolonged overshoots using strong negative offsets
3. Upper limit is dynamic (based on user target, not fixed 24°C)
4. Overshoot protection uses OVERSHOOT_PROTECTION_START (0.6°C) and FULL (1.5°C)

Dec 2, 2025: Simplified overshoot protection - moved from proactive to comfort layer.
Uses coast offsets (-7 to -10°C) instead of multiplier-based corrections.
"""

import pytest
from unittest.mock import MagicMock

from custom_components.effektguard.const import (
    TOLERANCE_RANGE_MULTIPLIER,
    LAYER_WEIGHT_COMFORT_HIGH,
    LAYER_WEIGHT_COMFORT_CRITICAL,
    LAYER_WEIGHT_COMFORT_MIN,
    LAYER_WEIGHT_COMFORT_MAX,
    COMFORT_CORRECTION_MILD,
    COMFORT_CORRECTION_MULT,
    COMFORT_DEAD_ZONE,
    OVERSHOOT_PROTECTION_START,
    OVERSHOOT_PROTECTION_FULL,
    OVERSHOOT_PROTECTION_OFFSET_MIN,
    OVERSHOOT_PROTECTION_OFFSET_MAX,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
)
from custom_components.effektguard.optimization.comfort_layer import ComfortLayer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel


@pytest.fixture
def hass_mock():
    """Create mock Home Assistant instance."""
    return MagicMock()


# Calculate tolerance value that gives tolerance_range = 1.0°C
# tolerance_range = tolerance * TOLERANCE_RANGE_MULTIPLIER
# 1.0 = tolerance * 0.4
# tolerance = 1.0 / 0.4 = 2.5
TEST_TOLERANCE_FOR_1C_RANGE = 1.0 / TOLERANCE_RANGE_MULTIPLIER  # = 2.5


def create_comfort_layer(
    target_temp: float = 21.0, tolerance: float = TEST_TOLERANCE_FOR_1C_RANGE
) -> ComfortLayer:
    """Create comfort layer with specified settings."""
    thermal_model = ThermalModel(thermal_mass=1.0, insulation_quality=1.0)
    
    # Mock get_thermal_trend
    get_thermal_trend = MagicMock(return_value={"rate_per_hour": 0.0, "confidence": 0.8})
    
    mode_config = MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED]
    tolerance_range = tolerance * TOLERANCE_RANGE_MULTIPLIER

    return ComfortLayer(
        get_thermal_trend=get_thermal_trend,
        thermal_model=thermal_model,
        mode_config=mode_config,
        tolerance_range=tolerance_range,
        target_temp=target_temp,
    )


def create_mock_nibe_state(
    indoor_temp: float,
    outdoor_temp: float = 5.0,
    degree_minutes: float = -200.0,
    curve_offset: float = 0.0,
):
    """Create mock NIBE state for testing."""
    state = MagicMock()
    state.indoor_temp = indoor_temp
    state.outdoor_temp = outdoor_temp
    state.degree_minutes = degree_minutes
    state.curve_offset = curve_offset
    state.supply_temp = 35.0
    state.return_temp = 30.0
    state.flow_rate = 15.0
    return state


def test_comfort_layer_at_overshoot_start_threshold():
    """Test comfort layer at exactly OVERSHOOT_PROTECTION_START (0.6°C) above tolerance.

    At tolerance + 0.6°C, should start coast mode with -7°C offset.
    """
    target = 21.0
    # Use tolerance=0.5 (tolerance_range=0.2)
    tolerance = 0.5

    layer = create_comfort_layer(target_temp=target, tolerance=tolerance)

    # Indoor: 21.0 + 0.6 = 21.6°C
    # temp_deviation = 0.6°C > tolerance (0.2°C)
    # overshoot = 0.6°C >= OVERSHOOT_PROTECTION_START (0.6°C) → coast mode
    nibe_state = create_mock_nibe_state(indoor_temp=21.6)

    decision = layer.evaluate_layer(nibe_state)

    # Should trigger coast mode at START threshold
    assert (
        decision.weight >= LAYER_WEIGHT_COMFORT_HIGH
    ), f"Expected weight >= {LAYER_WEIGHT_COMFORT_HIGH} at START threshold, got {decision.weight}"
    # Offset should be at minimum coast offset (-7°C) or close to it
    assert (
        decision.offset <= OVERSHOOT_PROTECTION_OFFSET_MIN
    ), f"Expected offset <= {OVERSHOOT_PROTECTION_OFFSET_MIN} in coast mode, got {decision.offset}"
    assert "COAST" in decision.reason or "Overshoot" in decision.reason


def test_comfort_layer_at_overshoot_full_threshold():
    """Test comfort layer at OVERSHOOT_PROTECTION_FULL (1.5°C).

    At 1.5°C above target, should use full coast with -10°C offset and weight 1.0.
    """
    target = 21.0
    tolerance = 0.5

    layer = create_comfort_layer(target_temp=target, tolerance=tolerance)

    # Indoor: 21.0 + 1.5 = 22.5°C
    # temp_deviation = 1.5°C
    # overshoot = 1.5°C >= OVERSHOOT_PROTECTION_FULL (1.5°C)
    nibe_state = create_mock_nibe_state(indoor_temp=22.5)

    decision = layer.evaluate_layer(nibe_state)

    # Should use full coast offset and critical weight
    assert (
        decision.weight == LAYER_WEIGHT_COMFORT_CRITICAL
    ), f"Expected weight {LAYER_WEIGHT_COMFORT_CRITICAL} at FULL threshold, got {decision.weight}"
    assert (
        decision.offset == OVERSHOOT_PROTECTION_OFFSET_MAX
    ), f"Expected offset {OVERSHOOT_PROTECTION_OFFSET_MAX} at FULL, got {decision.offset}"
    assert "COAST" in decision.reason


def test_comfort_layer_above_full_threshold():
    """Test comfort layer well above OVERSHOOT_PROTECTION_FULL.

    Should still use maximum coast offset and critical weight.
    """
    target = 21.0
    tolerance = 0.5

    layer = create_comfort_layer(target_temp=target, tolerance=tolerance)

    # Indoor: 23.7°C, Target: 21.0°C
    # temp_deviation = 2.7°C > tolerance (0.2)
    # overshoot = 2.7 > OVERSHOOT_PROTECTION_FULL (1.5)
    nibe_state = create_mock_nibe_state(indoor_temp=23.7)

    decision = layer.evaluate_layer(nibe_state)

    # Should use maximum response
    assert decision.weight == LAYER_WEIGHT_COMFORT_CRITICAL
    assert decision.offset == OVERSHOOT_PROTECTION_OFFSET_MAX  # -10°C
    assert "COAST" in decision.reason


def test_comfort_layer_mild_overshoot_before_coast():
    """Test comfort layer with mild overshoot below OVERSHOOT_PROTECTION_START.

    Between tolerance and 0.6°C, should use gentle correction, not coast.
    """
    target = 21.0
    tolerance = 0.5

    layer = create_comfort_layer(target_temp=target, tolerance=tolerance)

    # Indoor: 21.55°C
    # temp_deviation = 0.55°C > tolerance (0.2°C)
    # overshoot = 0.55°C < OVERSHOOT_PROTECTION_START (0.6°C)
    nibe_state = create_mock_nibe_state(indoor_temp=21.55)

    decision = layer.evaluate_layer(nibe_state)

    # Should use gentle correction, NOT coast
    assert decision.weight == LAYER_WEIGHT_COMFORT_HIGH
    assert "COAST" not in decision.reason
    # Offset should be milder than coast offsets
    assert decision.offset > OVERSHOOT_PROTECTION_OFFSET_MIN


def test_graduated_offsets_scale_with_overshoot():
    """Test that offsets scale linearly between START and FULL thresholds."""
    target = 21.0
    tolerance = 0.5

    layer = create_comfort_layer(target_temp=target, tolerance=tolerance)

    # Test point halfway between START (0.6) and FULL (1.5)
    # Range is 0.9°C. Halfway is 0.6 + 0.45 = 1.05°C overshoot
    # Indoor = Target + Overshoot
    # Indoor = 21.0 + 1.05 = 22.05°C
    nibe_state = create_mock_nibe_state(indoor_temp=22.05)

    decision = layer.evaluate_layer(nibe_state)

    # Should be roughly halfway between MIN (-7) and MAX (-10)
    # Expected: -8.5°C
    assert -9.0 <= decision.offset <= -8.0
    assert "COAST" in decision.reason


def test_graduated_weights_increase_with_overshoot():
    """Test that weights increase as overshoot gets worse."""
    target = 21.0
    tolerance = 0.5

    layer = create_comfort_layer(target_temp=target, tolerance=tolerance)

    # Case 1: Just entered coast mode (START threshold)
    # Indoor = 21.0 + 0.6 = 21.6
    state1 = create_mock_nibe_state(indoor_temp=21.6)
    decision1 = layer.evaluate_layer(state1)

    # Case 2: Halfway through coast zone
    # Indoor = 21.0 + 1.05 = 22.05
    state2 = create_mock_nibe_state(indoor_temp=22.05)
    decision2 = layer.evaluate_layer(state2)

    # Case 3: Full coast mode (FULL threshold)
    # Indoor = 21.0 + 1.5 = 22.5
    state3 = create_mock_nibe_state(indoor_temp=22.5)
    decision3 = layer.evaluate_layer(state3)

    # Weights should increase: START < Halfway < FULL
    assert decision1.weight < decision2.weight
    assert decision2.weight < decision3.weight
    assert decision3.weight == LAYER_WEIGHT_COMFORT_CRITICAL


def test_comfort_layer_preserves_gentle_correction_within_tolerance():
    """Test that gentle corrections are used within tolerance range."""
    target = 21.0
    # Use default tolerance (2.5) -> tolerance_range = 1.0
    # So tolerance in evaluate_layer is 1.0
    layer = create_comfort_layer(target_temp=target)

    # Indoor: 21.25°C (0.25°C above target, within 1.0°C tolerance)
    nibe_state = create_mock_nibe_state(indoor_temp=21.25)

    decision = layer.evaluate_layer(nibe_state)

    # Should use gentle correction
    # Offset = -deviation * COMFORT_CORRECTION_MULT
    # Offset = -0.25 * 0.3 = -0.075
    expected_offset = -0.25 * COMFORT_CORRECTION_MULT
    assert decision.offset == pytest.approx(expected_offset)
    assert decision.weight == LAYER_WEIGHT_COMFORT_MIN
    assert "gentle reduce" in decision.reason


def test_comfort_layer_dead_zone():
    """Test that no action is taken within dead zone."""
    target = 21.0
    tolerance = 0.5

    layer = create_comfort_layer(target_temp=target, tolerance=tolerance)

    # Indoor: 21.05°C (within dead zone, usually 0.1°C)
    nibe_state = create_mock_nibe_state(indoor_temp=21.05)

    decision = layer.evaluate_layer(nibe_state)

    assert decision.offset == 0.0
    assert decision.weight == 0.0
    assert "At target" in decision.reason


def test_comfort_layer_too_cold():
    """Test comfort layer response when too cold (below tolerance)."""
    target = 21.0
    tolerance = 0.5

    layer = create_comfort_layer(target_temp=target, tolerance=tolerance)

    # Indoor: 19.5°C (1.5°C below target, outside 0.2°C tolerance)
    nibe_state = create_mock_nibe_state(indoor_temp=19.5)

    decision = layer.evaluate_layer(nibe_state)

    # Should boost heating
    assert decision.offset > 0
    # Current implementation uses LAYER_WEIGHT_COMFORT_MAX (0.5)
    assert decision.weight >= LAYER_WEIGHT_COMFORT_MAX
    assert "Too cold" in decision.reason

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

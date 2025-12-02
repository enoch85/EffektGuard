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
    COMFORT_DEAD_ZONE,
    OVERSHOOT_PROTECTION_START,
    OVERSHOOT_PROTECTION_FULL,
    OVERSHOOT_PROTECTION_OFFSET_MIN,
    OVERSHOOT_PROTECTION_OFFSET_MAX,
)
from custom_components.effektguard.optimization.decision_engine import (
    DecisionEngine,
    LayerDecision,
)
from custom_components.effektguard.optimization.price_analyzer import PriceAnalyzer
from custom_components.effektguard.optimization.effect_manager import EffectManager
from custom_components.effektguard.optimization.thermal_model import ThermalModel


@pytest.fixture
def hass_mock():
    """Create mock Home Assistant instance."""
    return MagicMock()


# Calculate tolerance value that gives tolerance_range = 1.0°C
# tolerance_range = tolerance * TOLERANCE_RANGE_MULTIPLIER
# 1.0 = tolerance * 0.4
# tolerance = 1.0 / 0.4 = 2.5
TEST_TOLERANCE_FOR_1C_RANGE = 1.0 / TOLERANCE_RANGE_MULTIPLIER  # = 2.5


def create_engine(
    target_temp: float = 21.0, tolerance: float = TEST_TOLERANCE_FOR_1C_RANGE, hass_mock=None
) -> DecisionEngine:
    """Create decision engine with specified settings."""
    if hass_mock is None:
        hass_mock = MagicMock()

    price_analyzer = PriceAnalyzer()
    effect_manager = EffectManager(hass_mock)
    thermal_model = ThermalModel(thermal_mass=1.0, insulation_quality=1.0)

    config = {
        "latitude": 59.33,  # Stockholm
        "target_indoor_temp": target_temp,
        "tolerance": tolerance,
    }

    return DecisionEngine(
        price_analyzer=price_analyzer,
        effect_manager=effect_manager,
        thermal_model=thermal_model,
        config=config,
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
    With tolerance_range=1.0°C and target=21.0°C:
    - tolerance boundary: 22.0°C
    - OVERSHOOT_PROTECTION_START (0.6°C above tolerance): 22.6°C
    """
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER

    engine = create_engine(target_temp=target, tolerance=tolerance)

    # Indoor: 22.6°C, Target: 21.0°C, tolerance_range: 1.0°C
    # temp_deviation = 1.6°C > tolerance (1.0°C) → enters overshoot handling
    # overshoot = 1.6°C >= OVERSHOOT_PROTECTION_START (0.6°C) → coast mode
    nibe_state = create_mock_nibe_state(indoor_temp=22.6)

    decision = engine._comfort_layer(nibe_state)

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
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER

    engine = create_engine(target_temp=target, tolerance=tolerance)

    # Indoor: 22.5°C, Target: 21.0°C
    # temp_deviation = 1.5°C = exactly OVERSHOOT_PROTECTION_FULL
    nibe_state = create_mock_nibe_state(indoor_temp=22.5)

    decision = engine._comfort_layer(nibe_state)

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
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER

    engine = create_engine(target_temp=target, tolerance=tolerance)

    # Indoor: 23.7°C, Target: 21.0°C (October 27 scenario)
    # temp_deviation = 2.7°C > OVERSHOOT_PROTECTION_FULL (1.5°C)
    nibe_state = create_mock_nibe_state(indoor_temp=23.7)

    decision = engine._comfort_layer(nibe_state)

    # Should use maximum response
    assert decision.weight == LAYER_WEIGHT_COMFORT_CRITICAL
    assert decision.offset == OVERSHOOT_PROTECTION_OFFSET_MAX  # -10°C
    assert "COAST" in decision.reason


def test_comfort_layer_mild_overshoot_before_coast():
    """Test comfort layer with mild overshoot below OVERSHOOT_PROTECTION_START.

    Between tolerance and 0.6°C, should use gentle correction, not coast.
    """
    target = 21.0
    # Use smaller tolerance so we can test the zone between tolerance and 0.6°C
    tolerance_range = 0.4
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER

    engine = create_engine(target_temp=target, tolerance=tolerance)

    # Indoor: 21.5°C, Target: 21.0°C, tolerance_range: 0.4°C
    # temp_deviation = 0.5°C > tolerance (0.4°C) → overshoot handling
    # But 0.5°C < OVERSHOOT_PROTECTION_START (0.6°C) → gentle correction, not coast
    nibe_state = create_mock_nibe_state(indoor_temp=21.5)

    decision = engine._comfort_layer(nibe_state)

    # Should use gentle correction, NOT coast
    assert decision.weight == LAYER_WEIGHT_COMFORT_HIGH
    assert "COAST" not in decision.reason
    # Offset should be milder than coast offsets
    assert decision.offset > OVERSHOOT_PROTECTION_OFFSET_MIN  # Less aggressive than -7°C


def test_no_fixed_max_temp_limit():
    """Verify that MAX_TEMP_LIMIT is no longer used in safety layer.

    Old behavior: Safety layer blocks at 24°C regardless of user target
    New behavior: Comfort layer handles dynamically based on user target
    """
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    engine = create_engine(target_temp=target, tolerance=tolerance)

    # Test at old MAX_TEMP_LIMIT (24°C)
    nibe_state = create_mock_nibe_state(indoor_temp=24.0)

    safety_decision = engine._safety_layer(nibe_state)

    # Safety layer should NOT block at 24°C anymore
    assert (
        safety_decision.weight == 0.0
    ), "Safety layer should not activate at 24°C - upper limit now handled by comfort layer"
    assert safety_decision.offset == 0.0
    assert "Safety OK" in safety_decision.reason

    # But comfort layer SHOULD handle it with coast offsets
    comfort_decision = engine._comfort_layer(nibe_state)
    assert comfort_decision.weight == LAYER_WEIGHT_COMFORT_CRITICAL
    assert comfort_decision.offset == OVERSHOOT_PROTECTION_OFFSET_MAX  # -10°C


def test_dynamic_upper_limit_adapts_to_target():
    """Verify overshoot protection adapts to user's target temperature.

    User with target 22°C at 24°C indoor: 2.0°C overshoot → full coast
    User with target 19°C at 21°C indoor: 2.0°C overshoot → full coast
    """
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER

    # User 1: Target 22°C, indoor 24°C
    engine1 = create_engine(target_temp=22.0, tolerance=tolerance)
    nibe_state1 = create_mock_nibe_state(indoor_temp=24.0)
    decision1 = engine1._comfort_layer(nibe_state1)

    # 24.0 - 22.0 = 2.0°C > OVERSHOOT_PROTECTION_FULL (1.5°C)
    assert decision1.weight == LAYER_WEIGHT_COMFORT_CRITICAL
    assert decision1.offset == OVERSHOOT_PROTECTION_OFFSET_MAX

    # User 2: Target 19°C, indoor 21°C
    engine2 = create_engine(target_temp=19.0, tolerance=tolerance)
    nibe_state2 = create_mock_nibe_state(indoor_temp=21.0)
    decision2 = engine2._comfort_layer(nibe_state2)

    # 21.0 - 19.0 = 2.0°C > OVERSHOOT_PROTECTION_FULL (1.5°C)
    assert decision2.weight == LAYER_WEIGHT_COMFORT_CRITICAL
    assert decision2.offset == OVERSHOOT_PROTECTION_OFFSET_MAX


def test_graduated_weights_increase_with_overshoot():
    """Verify weights increase as overshoot worsens.

    Weight scales from 0.7 at START threshold to 1.0 at FULL threshold.
    With tolerance_range=1.0°C:
    - 22.6°C = 1.6°C temp_deviation = 0.6°C overshoot = START
    - 23.5°C = 2.5°C temp_deviation = 1.5°C overshoot = FULL
    """
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    engine = create_engine(target_temp=target, tolerance=tolerance)

    # Test at different overshoot levels (above tolerance)
    test_cases = [
        (22.6, LAYER_WEIGHT_COMFORT_HIGH),  # 1.6°C temp_deviation, 0.6°C overshoot = START
        (23.5, LAYER_WEIGHT_COMFORT_CRITICAL),  # 2.5°C temp_deviation, 1.5°C overshoot = FULL
        (24.0, LAYER_WEIGHT_COMFORT_CRITICAL),  # 3.0°C temp_deviation, above FULL
    ]

    prev_weight = 0.0
    for indoor_temp, min_expected_weight in test_cases:
        nibe_state = create_mock_nibe_state(indoor_temp=indoor_temp)
        decision = engine._comfort_layer(nibe_state)

        assert (
            decision.weight >= min_expected_weight
        ), f"At {indoor_temp}°C, expected weight >= {min_expected_weight}, got {decision.weight}"

        # Weight should increase or stay the same with higher overshoot
        assert decision.weight >= prev_weight, (
            f"Weight should increase with overshoot. At {indoor_temp}°C got {decision.weight}, "
            f"previous was {prev_weight}"
        )
        prev_weight = decision.weight

        # Offset should always be negative (cooling) for coast mode
        assert (
            decision.offset < 0
        ), f"At {indoor_temp}°C, offset should be negative (cooling), got {decision.offset}"


def test_graduated_offsets_scale_with_overshoot():
    """Verify coast offsets scale from -7°C to -10°C.

    With tolerance_range=1.0°C:
    - At 22.6°C (1.6°C temp_deviation = START threshold): offset ~-7°C
    - At 23.5°C (2.5°C temp_deviation = FULL threshold): offset = -10°C
    """
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    engine = create_engine(target_temp=target, tolerance=tolerance)

    # At START threshold (22.6°C = 1.6°C temp_deviation): offset should be ~-7°C
    nibe_start = create_mock_nibe_state(indoor_temp=22.6)
    decision_start = engine._comfort_layer(nibe_start)
    assert (
        decision_start.offset <= OVERSHOOT_PROTECTION_OFFSET_MIN
    ), f"At START, expected offset <= {OVERSHOOT_PROTECTION_OFFSET_MIN}, got {decision_start.offset}"

    # At FULL threshold (23.5°C = 2.5°C temp_deviation): offset should be -10°C
    nibe_full = create_mock_nibe_state(indoor_temp=23.5)
    decision_full = engine._comfort_layer(nibe_full)
    assert (
        decision_full.offset == OVERSHOOT_PROTECTION_OFFSET_MAX
    ), f"At FULL, expected offset {OVERSHOOT_PROTECTION_OFFSET_MAX}, got {decision_full.offset}"

    # Mid-point (23.0°C): offset should be between -7 and -10
    nibe_mid = create_mock_nibe_state(indoor_temp=23.0)
    decision_mid = engine._comfort_layer(nibe_mid)
    assert (
        OVERSHOOT_PROTECTION_OFFSET_MAX <= decision_mid.offset <= OVERSHOOT_PROTECTION_OFFSET_MIN
    ), (
        f"At mid-point, expected offset between {OVERSHOOT_PROTECTION_OFFSET_MAX} and "
        f"{OVERSHOOT_PROTECTION_OFFSET_MIN}, got {decision_mid.offset}"
    )


def test_comfort_layer_preserves_gentle_correction_within_tolerance():
    """Verify gentle correction still works within tolerance."""
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    engine = create_engine(target_temp=target, tolerance=tolerance)

    # Slightly warm but within tolerance: 21.5°C (0.5°C over target, within 1.0°C tolerance)
    nibe_state = create_mock_nibe_state(indoor_temp=21.5)
    decision = engine._comfort_layer(nibe_state)

    # Should use minimum weight (advisory only)
    assert decision.weight == LAYER_WEIGHT_COMFORT_MIN, "Within tolerance should use minimum weight"

    # Should have gentle correction (not coast offsets)
    assert abs(decision.offset) < 1.0, "Within tolerance correction should be gentle"

    # Should mention "slightly warm"
    assert "Slightly warm" in decision.reason or "slightly" in decision.reason.lower()


def test_comfort_layer_dead_zone():
    """Verify dead zone still works (unchanged behavior)."""
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    engine = create_engine(target_temp=target, tolerance=tolerance)

    # Right at target
    nibe_state = create_mock_nibe_state(indoor_temp=21.0)
    decision = engine._comfort_layer(nibe_state)

    assert decision.weight == 0.0, "At target should have zero weight"
    assert decision.offset == 0.0, "At target should have zero offset"
    assert "at target" in decision.reason.lower()


def test_comfort_layer_too_cold():
    """Verify cold correction still works."""
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    engine = create_engine(target_temp=target, tolerance=tolerance)

    # Too cold: 19.0°C (2°C under target, outside tolerance)
    nibe_state = create_mock_nibe_state(indoor_temp=19.0)
    decision = engine._comfort_layer(nibe_state)

    # Should use high weight
    assert decision.weight == LAYER_WEIGHT_COMFORT_MAX, "Too cold should use max weight"

    # Should have positive offset (heating)
    assert decision.offset > 0, "Too cold should have positive offset"

    # Should mention too cold
    assert "cold" in decision.reason.lower()

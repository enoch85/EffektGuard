"""Temperature control validation tests (Phase 2).

Tests for graduated comfort layer response and MAX_TEMP_LIMIT removal.

Validates that:
1. Comfort layer provides graduated response (0.7, 0.9, 1.0 weights)
2. System prevents prolonged overshoots (2-3°C)
3. Upper limit is dynamic (based on user target, not fixed 24°C)
4. Critical overshoot (2°C+) forces cooling with weight 1.0
"""

import pytest
from unittest.mock import MagicMock

from custom_components.effektguard.const import (
    DEFAULT_TARGET_TEMP,
    DEFAULT_TOLERANCE,
    TOLERANCE_RANGE_MULTIPLIER,
    LAYER_WEIGHT_COMFORT_HIGH,
    LAYER_WEIGHT_COMFORT_SEVERE,
    LAYER_WEIGHT_COMFORT_CRITICAL,
    LAYER_WEIGHT_COMFORT_MIN,
    COMFORT_CORRECTION_MILD,
    COMFORT_CORRECTION_STRONG,
    COMFORT_CORRECTION_CRITICAL,
    COMFORT_DEAD_ZONE,
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


def create_engine(target_temp: float = 21.0, tolerance: float = TEST_TOLERANCE_FOR_1C_RANGE, hass_mock=None) -> DecisionEngine:
    """Create decision engine with specified settings."""
    if hass_mock is None:
        hass_mock = MagicMock()
    
    price_analyzer = PriceAnalyzer()
    effect_manager = EffectManager(hass_mock)
    thermal_model = ThermalModel(thermal_mass=1.0, insulation_quality=1.0)

    config = {
        "latitude": 59.33,  # Stockholm
        "target_indoor_temp": target_temp,  # Fixed: use correct config key
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


def test_comfort_layer_mild_overshoot():
    """Test comfort layer with 0.5°C overshoot (within 1°C over tolerance).
    
    Should use LAYER_WEIGHT_COMFORT_HIGH and COMFORT_CORRECTION_MILD.
    """
    target = 21.0
    tolerance_range = 1.0  # Desired tolerance_range for testing
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER  # Calculate actual tolerance value
    
    engine = create_engine(target_temp=target, tolerance=tolerance)
    
    # Indoor: 22.5°C, Target: 21.0°C
    # temp_error = 22.5 - 21.0 = 1.5°C
    # overshoot = 1.5 - 1.0 = 0.5°C (mild: 0-1°C range)
    nibe_state = create_mock_nibe_state(indoor_temp=22.5)
    
    decision = engine._comfort_layer(nibe_state)
    
    # Should use high priority weight
    assert decision.weight == LAYER_WEIGHT_COMFORT_HIGH, (
        f"Expected weight {LAYER_WEIGHT_COMFORT_HIGH} for mild overshoot, "
        f"got {decision.weight}"
    )
    
    # Should use mild correction multiplier
    overshoot = 0.5
    expected_correction = -overshoot * COMFORT_CORRECTION_MILD
    assert abs(decision.offset - expected_correction) < 0.01, (
        f"Expected offset {expected_correction}, got {decision.offset}"
    )
    
    # Should have appropriate reason
    assert "Too warm" in decision.reason
    assert "22.5" in decision.reason or "1.5" in decision.reason


def test_comfort_layer_severe_overshoot():
    """Test comfort layer with 1.5°C overshoot (1-2°C over tolerance).
    
    Should use LAYER_WEIGHT_COMFORT_SEVERE and COMFORT_CORRECTION_STRONG.
    """
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    
    engine = create_engine(target_temp=target, tolerance=tolerance)
    
    # Indoor: 23.5°C, Target: 21.0°C
    # temp_error = 23.5 - 21.0 = 2.5°C
    # overshoot = 2.5 - 1.0 = 1.5°C (severe: 1-2°C range)
    nibe_state = create_mock_nibe_state(indoor_temp=23.5)
    
    decision = engine._comfort_layer(nibe_state)
    
    # Should use very high priority weight
    assert decision.weight == LAYER_WEIGHT_COMFORT_SEVERE, (
        f"Expected weight {LAYER_WEIGHT_COMFORT_SEVERE} for severe overshoot, "
        f"got {decision.weight}"
    )
    
    # Should use strong correction multiplier
    overshoot = 1.5
    expected_correction = -overshoot * COMFORT_CORRECTION_STRONG
    assert abs(decision.offset - expected_correction) < 0.01, (
        f"Expected offset {expected_correction}, got {decision.offset}"
    )
    
    # Should have appropriate reason
    assert "Severe overheat" in decision.reason


def test_comfort_layer_critical_overshoot():
    """Test comfort layer with 2.0°C overshoot (2°C+ over tolerance).
    
    Should use LAYER_WEIGHT_COMFORT_CRITICAL and COMFORT_CORRECTION_CRITICAL.
    This is the October 27-28 case: target 21°C, actual 23.7°C = 2.7°C overshoot.
    """
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    
    engine = create_engine(target_temp=target, tolerance=tolerance)
    
    # Indoor: 24.0°C, Target: 21.0°C
    # temp_error = 24.0 - 21.0 = 3.0°C
    # overshoot = 3.0 - 1.0 = 2.0°C (critical: 2°C+ range)
    nibe_state = create_mock_nibe_state(indoor_temp=24.0)
    
    decision = engine._comfort_layer(nibe_state)
    
    # Should use CRITICAL weight - same as safety layer
    assert decision.weight == LAYER_WEIGHT_COMFORT_CRITICAL, (
        f"Expected weight {LAYER_WEIGHT_COMFORT_CRITICAL} for critical overshoot, "
        f"got {decision.weight}"
    )
    
    # Should use critical correction multiplier
    overshoot = 2.0
    expected_correction = -overshoot * COMFORT_CORRECTION_CRITICAL
    assert abs(decision.offset - expected_correction) < 0.01, (
        f"Expected offset {expected_correction}, got {decision.offset}"
    )
    
    # Should have CRITICAL in reason
    assert "CRITICAL overheat" in decision.reason
    assert "emergency cooling" in decision.reason


def test_comfort_layer_october_27_case():
    """Test the actual October 27-28 scenario: 23.7°C actual vs 21.0°C target.
    
    This was the bug that triggered Phase 2:
    - Target: 21.0°C
    - Actual: 23.7°C
    - Overshoot: 1.7°C over tolerance
    - Old behavior: weight 0.5, correction -0.85°C → other layers override
    - New behavior: weight 0.9, correction -2.04°C → strong cooling
    """
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    engine = create_engine(target_temp=target, tolerance=tolerance)
    
    # Exact scenario from logs
    nibe_state = create_mock_nibe_state(indoor_temp=23.7)
    
    decision = engine._comfort_layer(nibe_state)
    
    # Should use severe weight (0.9) since overshoot is 1.7°C (in 1-2°C range)
    assert decision.weight == LAYER_WEIGHT_COMFORT_SEVERE, (
        f"October 27 case should use severe weight {LAYER_WEIGHT_COMFORT_SEVERE}, "
        f"got {decision.weight}"
    )
    
    # Should provide strong cooling correction
    # temp_error = 23.7 - 21.0 = 2.7°C
    # overshoot = 2.7 - 1.0 = 1.7°C
    # correction = -1.7 * 1.2 = -2.04°C
    expected_correction = -1.7 * COMFORT_CORRECTION_STRONG
    assert abs(decision.offset - expected_correction) < 0.01, (
        f"Expected strong correction {expected_correction}, got {decision.offset}"
    )
    
    # With weight 0.9 and offset -2.04°C, this should dominate most other layers
    # (only safety layer with weight 1.0 can override)


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
    assert safety_decision.weight == 0.0, (
        "Safety layer should not activate at 24°C - "
        "upper limit now handled by comfort layer"
    )
    assert safety_decision.offset == 0.0
    assert "Safety OK" in safety_decision.reason
    
    # But comfort layer SHOULD handle it (critical overshoot)
    comfort_decision = engine._comfort_layer(nibe_state)
    assert comfort_decision.weight == LAYER_WEIGHT_COMFORT_CRITICAL
    assert comfort_decision.offset < 0  # Cooling


def test_dynamic_upper_limit_adapts_to_target():
    """Verify upper limit adapts to user's target temperature.
    
    User with target 22°C should get critical cooling at 25°C (22 + 1 + 2 = 25).
    User with target 19°C should get critical cooling at 22°C (19 + 1 + 2 = 22).
    """
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    
    # User 1: Target 22°C
    engine1 = create_engine(target_temp=22.0, tolerance=tolerance)
    nibe_state1 = create_mock_nibe_state(indoor_temp=25.0)
    decision1 = engine1._comfort_layer(nibe_state1)
    
    # 25.0 - 22.0 = 3.0°C error, overshoot = 3.0 - 1.0 = 2.0°C (critical)
    assert decision1.weight == LAYER_WEIGHT_COMFORT_CRITICAL
    assert "CRITICAL" in decision1.reason
    
    # User 2: Target 19°C
    engine2 = create_engine(target_temp=19.0, tolerance=tolerance)
    nibe_state2 = create_mock_nibe_state(indoor_temp=22.0)
    decision2 = engine2._comfort_layer(nibe_state2)
    
    # 22.0 - 19.0 = 3.0°C error, overshoot = 3.0 - 1.0 = 2.0°C (critical)
    assert decision2.weight == LAYER_WEIGHT_COMFORT_CRITICAL
    assert "CRITICAL" in decision2.reason
    
    # Same temperature (22°C) triggers different responses based on user target!


def test_graduated_weights_increase_with_severity():
    """Verify weights increase as overshoot worsens."""
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    engine = create_engine(target_temp=target, tolerance=tolerance)
    
    # Test at different overshoot levels
    # Boundaries: <1.0°C = HIGH (0.7), >=1.0 to <2.0 = SEVERE (0.9), >=2.0 = CRITICAL (1.0)
    test_cases = [
        (22.5, LAYER_WEIGHT_COMFORT_HIGH),     # 0.5°C overshoot → HIGH (0.7)
        (22.9, LAYER_WEIGHT_COMFORT_HIGH),     # 0.9°C overshoot → HIGH (0.7) 
        (23.0, LAYER_WEIGHT_COMFORT_SEVERE),   # 1.0°C overshoot → SEVERE (0.9) - boundary
        (23.5, LAYER_WEIGHT_COMFORT_SEVERE),   # 1.5°C overshoot → SEVERE (0.9)
        (24.0, LAYER_WEIGHT_COMFORT_CRITICAL), # 2.0°C overshoot → CRITICAL (1.0) - boundary
        (24.5, LAYER_WEIGHT_COMFORT_CRITICAL), # 2.5°C overshoot → CRITICAL (1.0)
    ]
    
    for indoor_temp, expected_weight in test_cases:
        nibe_state = create_mock_nibe_state(indoor_temp=indoor_temp)
        decision = engine._comfort_layer(nibe_state)
        
        assert decision.weight == expected_weight, (
            f"At {indoor_temp}°C, expected weight {expected_weight}, "
            f"got {decision.weight}"
        )
        
        # Offset should always be negative (cooling)
        assert decision.offset < 0, (
            f"At {indoor_temp}°C, offset should be negative (cooling), "
            f"got {decision.offset}"
        )


def test_correction_multipliers_increase_with_severity():
    """Verify correction multipliers scale properly."""
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    engine = create_engine(target_temp=target, tolerance=tolerance)
    
    # Test same overshoot amount but different severity levels
    # We'll use 0.5°C overshoot in each bracket
    
    # Mild: 22.5°C (0.5°C overshoot)
    nibe_mild = create_mock_nibe_state(indoor_temp=22.5)
    decision_mild = engine._comfort_layer(nibe_mild)
    expected_mild = -0.5 * COMFORT_CORRECTION_MILD  # -0.5
    
    # Severe: 23.5°C (1.5°C overshoot, but compare on same 0.5°C increment)
    # We'll compare the multiplier effect
    nibe_severe = create_mock_nibe_state(indoor_temp=23.5)
    decision_severe = engine._comfort_layer(nibe_severe)
    expected_severe = -1.5 * COMFORT_CORRECTION_STRONG  # -1.8
    
    # Critical: 24.5°C (2.5°C overshoot)
    nibe_critical = create_mock_nibe_state(indoor_temp=24.5)
    decision_critical = engine._comfort_layer(nibe_critical)
    expected_critical = -2.5 * COMFORT_CORRECTION_CRITICAL  # -3.75
    
    # Verify corrections get stronger
    assert decision_mild.offset > decision_severe.offset, (
        "Severe correction should be stronger (more negative) than mild"
    )
    assert decision_severe.offset > decision_critical.offset, (
        "Critical correction should be stronger (more negative) than severe"
    )
    
    # Verify they match expected values
    assert abs(decision_mild.offset - expected_mild) < 0.01
    assert abs(decision_severe.offset - expected_severe) < 0.01
    assert abs(decision_critical.offset - expected_critical) < 0.01


def test_comfort_layer_preserves_gentle_correction_within_tolerance():
    """Verify gentle correction still works within tolerance (unchanged behavior)."""
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    engine = create_engine(target_temp=target, tolerance=tolerance)
    
    # Slightly warm but within tolerance: 21.5°C (0.5°C over target, within 1.0°C tolerance)
    nibe_state = create_mock_nibe_state(indoor_temp=21.5)
    decision = engine._comfort_layer(nibe_state)
    
    # Should use minimum weight (advisory only)
    assert decision.weight == LAYER_WEIGHT_COMFORT_MIN, (
        "Within tolerance should use minimum weight"
    )
    
    # Should have gentle correction
    assert abs(decision.offset) < 1.0, (
        "Within tolerance correction should be gentle"
    )
    
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

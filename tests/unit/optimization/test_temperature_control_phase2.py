"""Temperature control validation tests (Phase 2).

Tests for graduated comfort layer response and MAX_TEMP_LIMIT removal.

Validates that:
1. Comfort layer provides graduated response (0.7, 0.9, 1.0 weights)
2. System prevents prolonged overshoots (1-2°C)
3. Upper limit is dynamic (based on user target, not fixed 24°C)
4. Critical overshoot (1°C+) forces cooling with weight 1.0

Dec 2, 2025: Lowered thresholds from (1.0°C, 2.0°C) to (0.5°C, 1.0°C)
- Severe starts at 0.5°C overshoot
- Critical starts at 1.0°C overshoot
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
    COMFORT_OVERSHOOT_SEVERE,
    COMFORT_OVERSHOOT_CRITICAL,
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
    """Test comfort layer with 0.25°C overshoot (within 0-0.5°C mild range).

    Should use LAYER_WEIGHT_COMFORT_HIGH and COMFORT_CORRECTION_MILD.
    
    Dec 2, 2025: Updated thresholds - mild is now 0-0.5°C (was 0-1°C).
    """
    target = 21.0
    tolerance_range = 1.0  # Desired tolerance_range for testing
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER  # Calculate actual tolerance value

    engine = create_engine(target_temp=target, tolerance=tolerance)

    # Indoor: 22.25°C, Target: 21.0°C
    # temp_error = 22.25 - 21.0 = 1.25°C
    # overshoot = 1.25 - 1.0 = 0.25°C (mild: 0-0.5°C range)
    nibe_state = create_mock_nibe_state(indoor_temp=22.25)

    decision = engine._comfort_layer(nibe_state)

    # Should use high priority weight
    assert decision.weight == LAYER_WEIGHT_COMFORT_HIGH, (
        f"Expected weight {LAYER_WEIGHT_COMFORT_HIGH} for mild overshoot, " f"got {decision.weight}"
    )

    # Should use mild correction multiplier
    overshoot = COMFORT_OVERSHOOT_SEVERE / 2  # 0.25 (halfway to severe threshold)
    expected_correction = -overshoot * COMFORT_CORRECTION_MILD
    assert (
        abs(decision.offset - expected_correction) < 0.01
    ), f"Expected offset {expected_correction}, got {decision.offset}"

    # Should have appropriate reason
    assert "Too warm" in decision.reason


def test_comfort_layer_severe_overshoot():
    """Test comfort layer with 0.7°C overshoot (0.5-1.0°C severe range).

    Should use LAYER_WEIGHT_COMFORT_SEVERE and COMFORT_CORRECTION_STRONG.
    
    Dec 2, 2025: Updated thresholds - severe is now 0.5-1.0°C (was 1-2°C).
    """
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER

    engine = create_engine(target_temp=target, tolerance=tolerance)

    # Indoor: 22.7°C, Target: 21.0°C
    # temp_error = 22.7 - 21.0 = 1.7°C
    # overshoot = 1.7 - 1.0 = 0.7°C (severe: 0.5-1.0°C range)
    nibe_state = create_mock_nibe_state(indoor_temp=22.7)

    decision = engine._comfort_layer(nibe_state)

    # Should use very high priority weight
    assert decision.weight == LAYER_WEIGHT_COMFORT_SEVERE, (
        f"Expected weight {LAYER_WEIGHT_COMFORT_SEVERE} for severe overshoot, "
        f"got {decision.weight}"
    )

    # Should use strong correction multiplier
    overshoot = COMFORT_OVERSHOOT_SEVERE + 0.2  # 0.7 (in severe range)
    expected_correction = -overshoot * COMFORT_CORRECTION_STRONG
    assert (
        abs(decision.offset - expected_correction) < 0.01
    ), f"Expected offset {expected_correction}, got {decision.offset}"

    # Should have appropriate reason
    assert "Severe overheat" in decision.reason


def test_comfort_layer_critical_overshoot():
    """Test comfort layer with 1.0°C overshoot (1.0°C+ critical range).

    Should use LAYER_WEIGHT_COMFORT_CRITICAL and COMFORT_CORRECTION_CRITICAL.
    
    Dec 2, 2025: Updated thresholds - critical is now 1.0°C+ (was 2.0°C+).
    This means the October 27-28 case (1.7°C overshoot) is now CRITICAL.
    """
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER

    engine = create_engine(target_temp=target, tolerance=tolerance)

    # Indoor: 23.0°C, Target: 21.0°C
    # temp_error = 23.0 - 21.0 = 2.0°C
    # overshoot = 2.0 - 1.0 = 1.0°C (critical: 1.0°C+ range)
    nibe_state = create_mock_nibe_state(indoor_temp=23.0)

    decision = engine._comfort_layer(nibe_state)

    # Should use CRITICAL weight - same as safety layer
    assert decision.weight == LAYER_WEIGHT_COMFORT_CRITICAL, (
        f"Expected weight {LAYER_WEIGHT_COMFORT_CRITICAL} for critical overshoot, "
        f"got {decision.weight}"
    )

    # Should use critical correction multiplier
    overshoot = COMFORT_OVERSHOOT_CRITICAL  # 1.0
    expected_correction = -overshoot * COMFORT_CORRECTION_CRITICAL
    assert (
        abs(decision.offset - expected_correction) < 0.01
    ), f"Expected offset {expected_correction}, got {decision.offset}"

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
    - New behavior (Dec 2): weight 1.0 CRITICAL, correction -2.55°C → emergency cooling
    
    Dec 2, 2025: With lowered thresholds (critical at 1.0°C), this 1.7°C overshoot
    is now CRITICAL, not SEVERE. This is the correct aggressive response.
    """
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    engine = create_engine(target_temp=target, tolerance=tolerance)

    # Exact scenario from logs
    nibe_state = create_mock_nibe_state(indoor_temp=23.7)

    decision = engine._comfort_layer(nibe_state)

    # With new thresholds, 1.7°C overshoot is now CRITICAL (>=1.0°C)
    assert decision.weight == LAYER_WEIGHT_COMFORT_CRITICAL, (
        f"October 27 case should now use CRITICAL weight {LAYER_WEIGHT_COMFORT_CRITICAL}, "
        f"got {decision.weight}"
    )

    # Should provide CRITICAL cooling correction
    # temp_error = 23.7 - 21.0 = 2.7°C
    # overshoot = 2.7 - 1.0 = 1.7°C
    # correction = -1.7 * 1.5 = -2.55°C (CRITICAL multiplier)
    expected_correction = -1.7 * COMFORT_CORRECTION_CRITICAL
    assert (
        abs(decision.offset - expected_correction) < 0.01
    ), f"Expected critical correction {expected_correction}, got {decision.offset}"

    # With weight 1.0 and offset -2.55°C, this forces immediate cooling
    # No other layer can override this critical response


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
        "Safety layer should not activate at 24°C - " "upper limit now handled by comfort layer"
    )
    assert safety_decision.offset == 0.0
    assert "Safety OK" in safety_decision.reason

    # But comfort layer SHOULD handle it (critical overshoot)
    comfort_decision = engine._comfort_layer(nibe_state)
    assert comfort_decision.weight == LAYER_WEIGHT_COMFORT_CRITICAL
    assert comfort_decision.offset < 0  # Cooling


def test_dynamic_upper_limit_adapts_to_target():
    """Verify upper limit adapts to user's target temperature.

    User with target 22°C should get critical cooling at 24°C (22 + 1 + 1 = 24).
    User with target 19°C should get critical cooling at 21°C (19 + 1 + 1 = 21).
    
    Dec 2, 2025: Updated for new thresholds (critical at 1.0°C overshoot).
    """
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER

    # User 1: Target 22°C
    engine1 = create_engine(target_temp=22.0, tolerance=tolerance)
    nibe_state1 = create_mock_nibe_state(indoor_temp=24.0)
    decision1 = engine1._comfort_layer(nibe_state1)

    # 24.0 - 22.0 = 2.0°C error, overshoot = 2.0 - 1.0 = 1.0°C (critical)
    assert decision1.weight == LAYER_WEIGHT_COMFORT_CRITICAL
    assert "CRITICAL" in decision1.reason

    # User 2: Target 19°C
    engine2 = create_engine(target_temp=19.0, tolerance=tolerance)
    nibe_state2 = create_mock_nibe_state(indoor_temp=21.0)
    decision2 = engine2._comfort_layer(nibe_state2)

    # 21.0 - 19.0 = 2.0°C error, overshoot = 2.0 - 1.0 = 1.0°C (critical)
    assert decision2.weight == LAYER_WEIGHT_COMFORT_CRITICAL
    assert "CRITICAL" in decision2.reason

    # Same temperature (21°C) triggers different responses based on user target!


def test_graduated_weights_increase_with_severity():
    """Verify weights increase as overshoot worsens.
    
    Dec 2, 2025: Updated for new thresholds:
    - Mild: 0-0.5°C overshoot
    - Severe: 0.5-1.0°C overshoot
    - Critical: 1.0°C+ overshoot
    """
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    engine = create_engine(target_temp=target, tolerance=tolerance)

    # Test at different overshoot levels
    # Boundaries: <0.5°C = HIGH (0.7), >=0.5 to <1.0 = SEVERE (0.9), >=1.0 = CRITICAL (1.0)
    test_cases = [
        (22.25, LAYER_WEIGHT_COMFORT_HIGH),  # 0.25°C overshoot → HIGH (0.7)
        (22.4, LAYER_WEIGHT_COMFORT_HIGH),  # 0.4°C overshoot → HIGH (0.7)
        (22.5, LAYER_WEIGHT_COMFORT_SEVERE),  # 0.5°C overshoot → SEVERE (0.9) - boundary
        (22.75, LAYER_WEIGHT_COMFORT_SEVERE),  # 0.75°C overshoot → SEVERE (0.9)
        (23.0, LAYER_WEIGHT_COMFORT_CRITICAL),  # 1.0°C overshoot → CRITICAL (1.0) - boundary
        (23.5, LAYER_WEIGHT_COMFORT_CRITICAL),  # 1.5°C overshoot → CRITICAL (1.0)
    ]

    for indoor_temp, expected_weight in test_cases:
        nibe_state = create_mock_nibe_state(indoor_temp=indoor_temp)
        decision = engine._comfort_layer(nibe_state)

        assert decision.weight == expected_weight, (
            f"At {indoor_temp}°C, expected weight {expected_weight}, " f"got {decision.weight}"
        )

        # Offset should always be negative (cooling)
        assert decision.offset < 0, (
            f"At {indoor_temp}°C, offset should be negative (cooling), " f"got {decision.offset}"
        )


def test_correction_multipliers_increase_with_severity():
    """Verify correction multipliers scale properly.
    
    Dec 2, 2025: Updated for new thresholds (0.5°C severe, 1.0°C critical).
    """
    target = 21.0
    tolerance_range = 1.0
    tolerance = tolerance_range / TOLERANCE_RANGE_MULTIPLIER
    engine = create_engine(target_temp=target, tolerance=tolerance)

    # Test different severity levels at meaningful overshoot values

    # Mild: 22.25°C (0.25°C overshoot, in <0.5 range)
    nibe_mild = create_mock_nibe_state(indoor_temp=22.25)
    decision_mild = engine._comfort_layer(nibe_mild)
    expected_mild = -0.25 * COMFORT_CORRECTION_MILD  # -0.25

    # Severe: 22.7°C (0.7°C overshoot, in 0.5-1.0 range)
    nibe_severe = create_mock_nibe_state(indoor_temp=22.7)
    decision_severe = engine._comfort_layer(nibe_severe)
    expected_severe = -0.7 * COMFORT_CORRECTION_STRONG  # -0.84

    # Critical: 23.5°C (1.5°C overshoot, in >=1.0 range)
    nibe_critical = create_mock_nibe_state(indoor_temp=23.5)
    decision_critical = engine._comfort_layer(nibe_critical)
    expected_critical = -1.5 * COMFORT_CORRECTION_CRITICAL  # -2.25

    # Verify corrections get stronger
    assert (
        decision_mild.offset > decision_severe.offset
    ), "Severe correction should be stronger (more negative) than mild"
    assert (
        decision_severe.offset > decision_critical.offset
    ), "Critical correction should be stronger (more negative) than severe"

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
    assert decision.weight == LAYER_WEIGHT_COMFORT_MIN, "Within tolerance should use minimum weight"

    # Should have gentle correction
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

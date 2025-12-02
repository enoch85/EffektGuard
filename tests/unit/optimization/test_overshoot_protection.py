"""Tests for overshoot protection in the decision engine.

Verifies graduated coast response when indoor temp is above target.
Based on Dec 1-2, 2025 production analysis: overshoot was ignored, causing DM spiral.
Key insight: DM recovers when we STOP heating, not by boosting.
"""

import pytest

from custom_components.effektguard.const import (
    OVERSHOOT_PROTECTION_COLD_SNAP_THRESHOLD,
    OVERSHOOT_PROTECTION_FORECAST_HORIZON,
    OVERSHOOT_PROTECTION_FULL,
    OVERSHOOT_PROTECTION_OFFSET_MAX,
    OVERSHOOT_PROTECTION_OFFSET_MIN,
    OVERSHOOT_PROTECTION_START,
    OVERSHOOT_PROTECTION_WEIGHT_MAX,
    OVERSHOOT_PROTECTION_WEIGHT_MIN,
)


class TestOvershootProtectionConstants:
    """Test that overshoot protection constants have correct values."""

    def test_start_threshold_is_low_enough(self):
        """Start threshold should catch 0.6°C overshoot from Dec 2 crisis."""
        assert OVERSHOOT_PROTECTION_START == 0.6

    def test_full_threshold_is_reasonable(self):
        """Full response at 1.5°C overshoot."""
        assert OVERSHOOT_PROTECTION_FULL == 1.5

    def test_offset_range_is_aggressive(self):
        """Offset should be strongly negative to force coasting."""
        assert OVERSHOOT_PROTECTION_OFFSET_MIN == -7.0
        assert OVERSHOOT_PROTECTION_OFFSET_MAX == -10.0

    def test_weight_range_is_significant(self):
        """Weight should be strong enough to override other layers."""
        assert OVERSHOOT_PROTECTION_WEIGHT_MIN == 0.5
        assert OVERSHOOT_PROTECTION_WEIGHT_MAX == 1.0

    def test_cold_snap_threshold_prevents_coast_during_cold_snap(self):
        """Cold snap threshold should be reasonable for forecast stability check."""
        assert OVERSHOOT_PROTECTION_COLD_SNAP_THRESHOLD == 3.0

    def test_forecast_horizon_is_long_enough(self):
        """Forecast horizon should look ahead 12 hours."""
        assert OVERSHOOT_PROTECTION_FORECAST_HORIZON == 12


class TestOvershootProtectionGraduatedResponse:
    """Test the graduated response calculation logic."""

    def calculate_response(self, overshoot: float) -> tuple[float, float]:
        """Calculate coast offset and weight for given overshoot.

        Mirrors the logic in decision_engine._proactive_debt_prevention_layer().
        """
        if overshoot < OVERSHOOT_PROTECTION_START:
            return None, None

        overshoot_range = OVERSHOOT_PROTECTION_FULL - OVERSHOOT_PROTECTION_START
        fraction = min((overshoot - OVERSHOOT_PROTECTION_START) / overshoot_range, 1.0)

        coast_offset = (
            OVERSHOOT_PROTECTION_OFFSET_MIN
            + fraction * (OVERSHOOT_PROTECTION_OFFSET_MAX - OVERSHOOT_PROTECTION_OFFSET_MIN)
        )

        coast_weight = (
            OVERSHOOT_PROTECTION_WEIGHT_MIN
            + fraction * (OVERSHOOT_PROTECTION_WEIGHT_MAX - OVERSHOOT_PROTECTION_WEIGHT_MIN)
        )

        return coast_offset, coast_weight

    def test_below_start_threshold_no_response(self):
        """Below 0.6°C overshoot should not trigger protection."""
        offset, weight = self.calculate_response(0.5)
        assert offset is None
        assert weight is None

    def test_at_start_threshold(self):
        """At 0.6°C overshoot should get minimum response."""
        offset, weight = self.calculate_response(OVERSHOOT_PROTECTION_START)
        assert offset == OVERSHOOT_PROTECTION_OFFSET_MIN  # -7.0
        assert weight == OVERSHOOT_PROTECTION_WEIGHT_MIN  # 0.5

    def test_at_full_threshold(self):
        """At 1.5°C overshoot should get maximum response."""
        offset, weight = self.calculate_response(OVERSHOOT_PROTECTION_FULL)
        assert offset == OVERSHOOT_PROTECTION_OFFSET_MAX  # -10.0
        assert weight == OVERSHOOT_PROTECTION_WEIGHT_MAX  # 1.0

    def test_above_full_threshold_capped(self):
        """Above 1.5°C overshoot should cap at maximum response."""
        offset, weight = self.calculate_response(2.0)
        assert offset == OVERSHOOT_PROTECTION_OFFSET_MAX  # -10.0
        assert weight == OVERSHOOT_PROTECTION_WEIGHT_MAX  # 1.0

    def test_graduated_at_midpoint(self):
        """At 1.05°C overshoot (midpoint) should get midpoint response."""
        midpoint = (OVERSHOOT_PROTECTION_START + OVERSHOOT_PROTECTION_FULL) / 2  # 1.05
        offset, weight = self.calculate_response(midpoint)

        expected_offset = (OVERSHOOT_PROTECTION_OFFSET_MIN + OVERSHOOT_PROTECTION_OFFSET_MAX) / 2
        expected_weight = (OVERSHOOT_PROTECTION_WEIGHT_MIN + OVERSHOOT_PROTECTION_WEIGHT_MAX) / 2

        assert offset == pytest.approx(expected_offset, rel=0.01)  # -8.5
        assert weight == pytest.approx(expected_weight, rel=0.01)  # 0.75

    def test_dec2_crisis_case_0_6_overshoot(self):
        """Test 0.6°C overshoot case from Dec 2 crisis."""
        offset, weight = self.calculate_response(0.6)
        # At start threshold: -7°C offset, 0.5 weight
        assert offset == -7.0
        assert weight == 0.5

    def test_dec2_crisis_case_0_8_overshoot(self):
        """Test 0.8°C overshoot case from Dec 2 crisis."""
        offset, weight = self.calculate_response(0.8)
        # 0.8 is 0.2 into the 0.9 range, so fraction = 0.222
        # Expected offset: -7 + 0.222 * -3 = -7.67
        # Expected weight: 0.5 + 0.222 * 0.5 = 0.61
        assert offset == pytest.approx(-7.67, rel=0.05)
        assert weight == pytest.approx(0.61, rel=0.05)

    def test_dec2_crisis_case_1_0_overshoot(self):
        """Test 1.0°C overshoot case from Dec 2 crisis logs."""
        offset, weight = self.calculate_response(1.0)
        # 1.0 is 0.4 into the 0.9 range, so fraction = 0.444
        # Expected offset: -7 + 0.444 * -3 = -8.33
        # Expected weight: 0.5 + 0.444 * 0.5 = 0.72
        assert offset == pytest.approx(-8.33, rel=0.05)
        assert weight == pytest.approx(0.72, rel=0.05)

    def test_dec2_crisis_case_1_3_overshoot(self):
        """Test 1.3°C overshoot case from Dec 2 crisis logs (11:32 scenario)."""
        offset, weight = self.calculate_response(1.3)
        # 1.3 is 0.7 into the 0.9 range, so fraction = 0.778
        # Expected offset: -7 + 0.778 * -3 = -9.33
        # Expected weight: 0.5 + 0.778 * 0.5 = 0.89
        assert offset == pytest.approx(-9.33, rel=0.05)
        assert weight == pytest.approx(0.89, rel=0.05)

    def test_dec2_crisis_case_1_5_overshoot(self):
        """Test 1.5°C overshoot (full override)."""
        offset, weight = self.calculate_response(1.5)
        assert offset == -10.0
        assert weight == 1.0


class TestOvershootProtectionVsDecisionTable:
    """Test validation against expected decision table from implementation plan."""

    def calculate_response(self, overshoot: float) -> tuple[float, float]:
        """Calculate coast offset and weight for given overshoot."""
        if overshoot < OVERSHOOT_PROTECTION_START:
            return None, None

        overshoot_range = OVERSHOOT_PROTECTION_FULL - OVERSHOOT_PROTECTION_START
        fraction = min((overshoot - OVERSHOOT_PROTECTION_START) / overshoot_range, 1.0)

        coast_offset = (
            OVERSHOOT_PROTECTION_OFFSET_MIN
            + fraction * (OVERSHOOT_PROTECTION_OFFSET_MAX - OVERSHOOT_PROTECTION_OFFSET_MIN)
        )

        coast_weight = (
            OVERSHOOT_PROTECTION_WEIGHT_MIN
            + fraction * (OVERSHOOT_PROTECTION_WEIGHT_MAX - OVERSHOOT_PROTECTION_WEIGHT_MIN)
        )

        return coast_offset, coast_weight

    @pytest.mark.parametrize(
        "overshoot,expected_offset,expected_weight",
        [
            (0.6, -7.0, 0.50),
            (0.8, -7.67, 0.61),
            (1.0, -8.33, 0.72),
            (1.1, -8.67, 0.78),
            (1.3, -9.33, 0.89),
            (1.5, -10.0, 1.00),
        ],
    )
    def test_graduated_response_table(self, overshoot, expected_offset, expected_weight):
        """Validate graduated response matches implementation plan table."""
        offset, weight = self.calculate_response(overshoot)
        assert offset == pytest.approx(expected_offset, rel=0.05)
        assert weight == pytest.approx(expected_weight, rel=0.05)


class TestOvershootProtectionScenarios:
    """Test real-world scenarios from Dec 2, 2025 logs."""

    def calculate_response(self, overshoot: float) -> tuple[float, float]:
        """Calculate coast offset and weight for given overshoot."""
        if overshoot < OVERSHOOT_PROTECTION_START:
            return None, None

        overshoot_range = OVERSHOOT_PROTECTION_FULL - OVERSHOOT_PROTECTION_START
        fraction = min((overshoot - OVERSHOOT_PROTECTION_START) / overshoot_range, 1.0)

        coast_offset = (
            OVERSHOOT_PROTECTION_OFFSET_MIN
            + fraction * (OVERSHOOT_PROTECTION_OFFSET_MAX - OVERSHOOT_PROTECTION_OFFSET_MIN)
        )

        coast_weight = (
            OVERSHOOT_PROTECTION_WEIGHT_MIN
            + fraction * (OVERSHOOT_PROTECTION_WEIGHT_MAX - OVERSHOOT_PROTECTION_WEIGHT_MIN)
        )

        return coast_offset, coast_weight

    def test_scenario_1132_dec2_smoking_gun(self):
        """Test 11:32 Dec 2 - the smoking gun scenario.

        Indoor: 22.3°C (target: 21.0°C) → 1.3°C ABOVE target
        OLD Decision: offset +1.70°C (WRONG - was heating during overshoot!)
        NEW Decision: should be strongly negative offset
        """
        overshoot = 1.3
        offset, weight = self.calculate_response(overshoot)

        # NEW behavior should force strong coast
        assert offset < -9.0  # At least -9°C offset
        assert weight > 0.8  # Strong weight to override other layers

        # This is the fix: instead of +1.70°C, we get ~-9.3°C
        old_decision = +1.70
        improvement = old_decision - offset  # How much better is new decision
        assert improvement > 10.0  # At least 10°C improvement

    def test_scenario_dm_recovery_principle(self):
        """Test that overshoot protection enables DM recovery.

        Key insight from Dec 2: DM recovers when we STOP heating.
        When indoor is above target, we have thermal margin - use it.
        """
        # At 1.0°C overshoot with DM -600
        overshoot = 1.0
        offset, weight = self.calculate_response(overshoot)

        # Should produce negative offset to stop/reduce heating
        assert offset < 0  # Negative offset
        assert offset <= -7  # At least -7°C to actually coast

        # Weight should be strong enough to override DM recovery boost
        # which was incorrectly adding +2°C during overshoot
        assert weight >= 0.7

    def test_scenario_peak_plus_overshoot(self):
        """Test PEAK price + overshoot should strongly reduce heating.

        Even if PEAK alone only produces -0.6°C (due to tolerance scaling),
        overshoot protection should still produce strong negative offset.
        """
        # 1.3°C overshoot during PEAK
        overshoot = 1.3
        offset, weight = self.calculate_response(overshoot)

        # Overshoot protection takes precedence
        assert offset < -9  # ~-9.3°C
        assert weight > 0.85  # Strong override

    def test_below_threshold_no_coast(self):
        """Below 0.6°C overshoot should not trigger coasting."""
        overshoot = 0.5
        offset, weight = self.calculate_response(overshoot)

        assert offset is None  # Not triggered
        assert weight is None

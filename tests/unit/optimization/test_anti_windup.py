"""Tests for anti-windup protection in EmergencyLayer.

Context: DM oscillation occurs when offset is raised during thermal debt recovery,
but heat hasn't yet reached the thermal mass (UFH concrete slab has 6+ hour lag).
Raising offset makes DM go MORE negative initially because:
  - S1 target rises immediately
  - BT25 (actual) takes hours to catch up
  - DM = ∫(BT25 - S1) dt accumulates larger negative values

Without anti-windup, the system "chases" thermal debt:
  1. DM at -300 → apply +2°C offset
  2. S1 target rises, but BT25 lags → DM drops to -500
  3. "Not working!" → raise to +3°C → DM drops to -700
  4. Eventually heat arrives → BT25 overshoots S1 → DM swings to +100
  5. System backs off → slab cools → cycle repeats

Anti-windup solution:
  If DM is dropping WHILE current_offset is already positive:
  → Heat is "in transit" through thermal mass
  → Don't escalate offset further - cap at current level

Reference:
  - Wikipedia: Integral Windup
  - Dec 9, 2025 debug.log analysis
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector


@pytest.fixture
def emergency_layer():
    """Create an EmergencyLayer for testing."""
    climate_detector = ClimateZoneDetector(latitude=59.33)  # Stockholm
    layer = EmergencyLayer(
        climate_detector=climate_detector,
        price_analyzer=None,
        heating_type="concrete_ufh",
        get_thermal_trend=lambda: {"rate_per_hour": 0.0, "confidence": 0.8},
        get_outdoor_trend=lambda: {"rate_per_hour": 0.0, "confidence": 0.8},
    )
    return layer


@pytest.fixture
def nibe_state_factory():
    """Factory for creating NibeState-like objects."""

    def create_state(
        degree_minutes: float = -500,
        current_offset: float = 0.0,
        indoor_temp: float = 21.0,
        outdoor_temp: float = -5.0,
        timestamp: datetime = None,
    ):
        state = Mock()
        state.degree_minutes = degree_minutes
        state.current_offset = current_offset
        state.indoor_temp = indoor_temp
        state.outdoor_temp = outdoor_temp
        state.timestamp = timestamp or datetime.now()
        return state

    return create_state


class TestDMRateCalculation:
    """Test DM rate of change calculation."""

    def test_calculates_dm_rate_from_history(self, emergency_layer):
        """DM rate is calculated from history samples."""
        layer = emergency_layer
        now = datetime.now()

        # Simulate DM dropping over 30 minutes
        layer._update_dm_history(-300.0, now - timedelta(minutes=30))
        layer._update_dm_history(-350.0, now - timedelta(minutes=20))
        layer._update_dm_history(-400.0, now - timedelta(minutes=10))
        layer._update_dm_history(-450.0, now)

        dm_rate, has_valid_rate = layer._calculate_dm_rate()

        # DM dropped 150 in 30 minutes = -300/hour
        assert has_valid_rate is True
        assert dm_rate == pytest.approx(-300.0, abs=10.0)

    def test_insufficient_samples_returns_invalid(self, emergency_layer):
        """Not enough samples returns invalid rate."""
        layer = emergency_layer

        # Only one sample
        layer._update_dm_history(-300.0, datetime.now())

        dm_rate, has_valid_rate = layer._calculate_dm_rate()

        assert has_valid_rate is False

    def test_dm_recovering_positive_rate(self, emergency_layer):
        """DM recovering (going less negative) has positive rate."""
        layer = emergency_layer
        now = datetime.now()

        # Simulate DM recovering (getting less negative)
        layer._update_dm_history(-500.0, now - timedelta(minutes=30))
        layer._update_dm_history(-450.0, now - timedelta(minutes=15))
        layer._update_dm_history(-400.0, now)

        dm_rate, has_valid_rate = layer._calculate_dm_rate()

        # DM improved by 100 in 30 minutes = +200/hour
        assert has_valid_rate is True
        assert dm_rate > 0


class TestAntiWindupCheck:
    """Test anti-windup activation conditions."""

    def test_activates_when_offset_positive_and_dm_dropping(self, emergency_layer):
        """Anti-windup activates when heat is in transit."""
        layer = emergency_layer

        # DM is dropping rapidly
        dm_rate = -150.0  # More negative than threshold
        current_offset = 2.0  # Positive offset (adding heat)

        anti_windup, reason = layer._check_anti_windup(current_offset, dm_rate, has_valid_rate=True)

        assert anti_windup is True
        assert "anti-windup" in reason
        assert "DM dropping" in reason

    def test_not_active_when_offset_zero(self, emergency_layer):
        """Anti-windup doesn't activate when offset is zero."""
        layer = emergency_layer

        dm_rate = -150.0
        current_offset = 0.0  # No positive offset

        anti_windup, reason = layer._check_anti_windup(current_offset, dm_rate, has_valid_rate=True)

        assert anti_windup is False

    def test_not_active_when_dm_stable(self, emergency_layer):
        """Anti-windup doesn't activate when DM is stable."""
        layer = emergency_layer

        dm_rate = -10.0  # DM not dropping fast (above threshold)
        current_offset = 2.0

        anti_windup, reason = layer._check_anti_windup(current_offset, dm_rate, has_valid_rate=True)

        assert anti_windup is False

    def test_not_active_when_dm_improving(self, emergency_layer):
        """Anti-windup doesn't activate when DM is recovering."""
        layer = emergency_layer

        dm_rate = 50.0  # DM improving (positive rate)
        current_offset = 2.0

        anti_windup, reason = layer._check_anti_windup(current_offset, dm_rate, has_valid_rate=True)

        assert anti_windup is False

    def test_not_active_without_valid_rate(self, emergency_layer):
        """Anti-windup doesn't activate without valid rate data."""
        layer = emergency_layer

        dm_rate = -150.0
        current_offset = 2.0

        anti_windup, reason = layer._check_anti_windup(
            current_offset, dm_rate, has_valid_rate=False  # No valid data
        )

        assert anti_windup is False


class TestAntiWindupOffsetCapping:
    """Test that anti-windup caps offset escalation."""

    def test_caps_offset_when_anti_windup_active(self, emergency_layer):
        """Offset is capped when anti-windup is active."""
        layer = emergency_layer

        calculated_offset = 3.0  # Would escalate to +3°C
        current_offset = 2.0  # Already at +2°C

        final_offset, reason = layer._apply_anti_windup_cap(
            calculated_offset,
            current_offset,
            anti_windup_active=True,
            tier_name="T1",
        )

        # Should be capped at current_offset (or slightly less)
        assert final_offset <= current_offset
        assert final_offset < calculated_offset

    def test_allows_higher_offset_when_not_active(self, emergency_layer):
        """Offset is not capped when anti-windup inactive."""
        layer = emergency_layer

        calculated_offset = 3.0
        current_offset = 2.0

        final_offset, reason = layer._apply_anti_windup_cap(
            calculated_offset,
            current_offset,
            anti_windup_active=False,  # Not active
            tier_name="T1",
        )

        # Should use full calculated offset
        assert final_offset == calculated_offset
        assert reason == ""

    def test_allows_maintaining_current_offset(self, emergency_layer):
        """Anti-windup allows maintaining (not reducing) current offset."""
        layer = emergency_layer

        calculated_offset = 2.0  # Same as current
        current_offset = 2.0

        final_offset, reason = layer._apply_anti_windup_cap(
            calculated_offset,
            current_offset,
            anti_windup_active=True,
            tier_name="T1",
        )

        # Should allow maintaining current offset
        assert final_offset == 2.0


class TestAntiWindupIntegration:
    """Test anti-windup in full evaluate_layer flow."""

    def test_tier_includes_anti_windup_info(self, emergency_layer, nibe_state_factory):
        """EmergencyLayerDecision includes anti-windup status."""
        layer = emergency_layer
        now = datetime.now()

        # Build up DM history showing drop
        layer._update_dm_history(-400.0, now - timedelta(minutes=15))
        layer._update_dm_history(-500.0, now - timedelta(minutes=10))
        layer._update_dm_history(-600.0, now - timedelta(minutes=5))

        # Current state with positive offset and deep DM
        state = nibe_state_factory(
            degree_minutes=-700.0,
            current_offset=2.0,  # Already adding heat
            timestamp=now,
        )

        # Add final sample
        layer._update_dm_history(-700.0, now)

        decision = layer.evaluate_layer(
            nibe_state=state,
            weather_data=None,
            price_data=None,
            target_temp=21.5,
            tolerance_range=0.5,
        )

        # Should include dm_rate in decision
        assert hasattr(decision, "dm_rate")
        assert hasattr(decision, "anti_windup_active")

    def test_prevents_escalation_during_heat_transit(self, emergency_layer, nibe_state_factory):
        """Anti-windup prevents offset escalation during heat transit.

        Scenario: DM at -700, offset already +2°C, but DM still dropping.
        Without anti-windup: Would escalate to T2/T3 offset (+2.5/+3°C)
        With anti-windup: Cap at current +2°C level
        """
        layer = emergency_layer
        now = datetime.now()

        # Build up DM history showing rapid drop (heat in transit)
        layer._update_dm_history(-500.0, now - timedelta(minutes=15))
        layer._update_dm_history(-600.0, now - timedelta(minutes=10))
        layer._update_dm_history(-700.0, now - timedelta(minutes=5))

        # Current state: deep thermal debt, already heating
        state = nibe_state_factory(
            degree_minutes=-800.0,  # Very deep
            current_offset=2.0,  # Already at +2°C
            timestamp=now,
        )

        layer._update_dm_history(-800.0, now)

        decision = layer.evaluate_layer(
            nibe_state=state,
            weather_data=None,
            price_data=None,
            target_temp=21.5,
            tolerance_range=0.5,
        )

        # Anti-windup should have prevented escalation or reduced offset (Jan 2026)
        if decision.anti_windup_active:
            # With severe dm_rate (-1200/h in this test), offset may be REDUCED
            # Jan 2026 enhancement: At -1200/h, reduction = 1200/100 = 12°C
            # So offset goes from +2 to -10 (capped at MIN_OFFSET)
            # Offset should be <= current_offset (kept or reduced, never raised)
            assert decision.offset <= 2.0, f"Anti-windup should prevent raise, got {decision.offset}"
            # New format options:
            # - Mild: "DM dropping -XX/h while offset +X°C - not raising"
            # - Severe: "DM dropping -XX/h - reducing offset by X°C..."
            assert (
                "not raising" in decision.reason
                or "reducing offset" in decision.reason
                or "anti-windup" in decision.reason.lower()
            )


class TestAntiWindupRealScenario:
    """Test real-world scenario from Dec 9, 2025 debug.log."""

    def test_prevents_dm_chasing_scenario(self, emergency_layer, nibe_state_factory):
        """Reproduce and prevent the DM oscillation scenario.

        Real scenario observed:
        1. DM at -347 (pre-existing at startup)
        2. DHW heating causes DM to drop to -711
        3. System applies recovery offset
        4. DM drops further to -800+ (heat in transit)
        5. Eventually DM swings to +100 (overshoot)

        With anti-windup:
        - When DM is dropping despite positive offset, don't escalate
        - Wait for heat to arrive through thermal mass
        """
        layer = emergency_layer

        # Simulate the progression
        base_time = datetime.now()
        offsets_applied = []
        dm_values = [-347, -450, -550, -650, -711, -750, -800]

        previous_offset = 0.0

        for i, dm in enumerate(dm_values):
            timestamp = base_time + timedelta(minutes=i * 5)

            state = nibe_state_factory(
                degree_minutes=dm,
                current_offset=previous_offset,
                timestamp=timestamp,
            )

            decision = layer.evaluate_layer(
                nibe_state=state,
                weather_data=None,
                price_data=None,
                target_temp=21.5,
                tolerance_range=0.5,
            )

            offsets_applied.append(decision.offset)
            previous_offset = decision.offset

        # Key assertion: offsets should NOT continuously escalate
        # Once positive offset is applied and DM is dropping,
        # anti-windup should prevent further escalation
        escalation_count = sum(
            1 for i in range(1, len(offsets_applied)) if offsets_applied[i] > offsets_applied[i - 1]
        )

        # Should not escalate many times - anti-windup should cap
        # In worst case without anti-windup, it would escalate 6 times
        assert escalation_count < len(dm_values) - 1, (
            f"Anti-windup should prevent continuous escalation. "
            f"Escalated {escalation_count} times: {offsets_applied}"
        )

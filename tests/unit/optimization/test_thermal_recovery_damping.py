"""Comprehensive tests for thermal recovery damping (T1, T2, T3).

Tests the general thermal recovery damping system that prevents concrete slab
overshoot during emergency recovery periods when house warming naturally.

Damping Conditions (ALL must be met):
1. Indoor warming >= 0.3°C/h (solar gain)
2. Outdoor stable >= -0.5°C/h (not fighting cold spell)
3. Trend confidence >= 0.4 (~1 hour data)
4. No ≥2°C temperature drop forecast within 6 hours

Test Coverage:
- Damping applied when conditions met (moderate and rapid warming)
- Damping blocked when safety conditions fail
- Tier-specific minimums respected (T1: 1.0°C, T2: 1.5°C, T3: 2.0°C)
- Forecast check blocks damping when cold weather incoming
- Real-world scenario: Night recovery → Morning solar gain
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.price_analyzer import PriceAnalyzer
from custom_components.effektguard.optimization.effect_manager import EffectManager
from custom_components.effektguard.optimization.thermal_model import ThermalModel
from custom_components.effektguard.const import (
    DM_CRITICAL_T1_OFFSET,
    DM_CRITICAL_T2_OFFSET,
    DM_CRITICAL_T3_OFFSET,
    THERMAL_RECOVERY_DAMPING_FACTOR,
    THERMAL_RECOVERY_RAPID_FACTOR,
    THERMAL_RECOVERY_T1_MIN_OFFSET,
    THERMAL_RECOVERY_T2_MIN_OFFSET,
    THERMAL_RECOVERY_T3_MIN_OFFSET,
    THERMAL_RECOVERY_WARMING_THRESHOLD,
)


@pytest.fixture
def decision_engine():
    """Create decision engine with mock predictor."""
    hass = MagicMock()
    hass.config.latitude = 59.33  # Stockholm

    engine = DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(hass),
        thermal_model=ThermalModel(thermal_mass=1.0, insulation_quality=1.0),
        config={"latitude": 59.33, "target_indoor_temp": 22.0, "tolerance": 1.0},
    )

    # Mock predictor with sufficient history
    engine.predictor = MagicMock()
    engine.predictor.state_history = [None] * 16  # 4 hours of 15-min samples
    engine.predictor.__len__ = MagicMock(return_value=16)

    return engine


@pytest.fixture
def stable_outdoor_trend():
    """Standard stable outdoor trend (not dropping)."""
    return {"trend": "stable", "rate_per_hour": 0.0, "confidence": 0.5}


@pytest.fixture
def stable_weather_forecast():
    """Mock weather forecast showing stable conditions (no cold incoming)."""
    forecast = MagicMock()
    forecast.forecast_hours = [
        MagicMock(temperature=-5.0) for _ in range(12)  # Stable temps for 12 hours
    ]
    return forecast


class TestDampingConditions:
    """Test damping activation/blocking based on conditions."""

    def test_moderate_warming_applies_standard_damping(
        self, decision_engine, stable_outdoor_trend, stable_weather_forecast
    ):
        """Warming 0.3-0.5°C/h applies 60% damping."""
        decision_engine.predictor.get_current_trend.return_value = {
            "trend": "rising",
            "rate_per_hour": 0.4,  # Moderate warming
            "confidence": 0.5,
            "samples": 16,
        }
        decision_engine.predictor.get_outdoor_trend.return_value = stable_outdoor_trend

        state = NibeState(
            outdoor_temp=-5.0,
            indoor_temp=21.5,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-850,  # T2 level for Stockholm
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
        )

        decision = decision_engine._emergency_layer(state, stable_weather_forecast)

        # Should apply standard damping (0.6)
        expected = DM_CRITICAL_T2_OFFSET * THERMAL_RECOVERY_DAMPING_FACTOR  # 2.5 * 0.6 = 1.5
        assert decision.offset == pytest.approx(expected, abs=0.01)
        assert "warming" in decision.reason.lower()

    def test_rapid_warming_applies_stronger_damping(
        self, decision_engine, stable_outdoor_trend, stable_weather_forecast
    ):
        """Warming >0.5°C/h applies 40% damping."""
        decision_engine.predictor.get_current_trend.return_value = {
            "trend": "rising",
            "rate_per_hour": 0.6,  # Rapid warming
            "confidence": 0.6,
            "samples": 16,
        }
        decision_engine.predictor.get_outdoor_trend.return_value = stable_outdoor_trend

        state = NibeState(
            outdoor_temp=-5.0,
            indoor_temp=21.5,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-850,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
        )

        decision = decision_engine._emergency_layer(state, stable_weather_forecast)

        # DM_CRITICAL_T2_OFFSET * THERMAL_RECOVERY_RAPID_FACTOR = 7.0 * 0.4 = 2.8
        # Clamped to max(2.8, THERMAL_RECOVERY_T2_MIN_OFFSET=1.5) = 2.8
        expected_offset = max(
            DM_CRITICAL_T2_OFFSET * THERMAL_RECOVERY_RAPID_FACTOR, THERMAL_RECOVERY_T2_MIN_OFFSET
        )
        assert decision.offset == pytest.approx(expected_offset, abs=0.01)
        assert "rapid warming" in decision.reason.lower()

    def test_no_warming_no_damping(
        self, decision_engine, stable_outdoor_trend, stable_weather_forecast
    ):
        """Below 0.3°C/h warming → no damping."""
        decision_engine.predictor.get_current_trend.return_value = {
            "trend": "stable",
            "rate_per_hour": 0.2,  # Below threshold
            "confidence": 0.5,
            "samples": 16,
        }
        decision_engine.predictor.get_outdoor_trend.return_value = stable_outdoor_trend

        state = NibeState(
            outdoor_temp=-5.0,
            indoor_temp=21.5,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-850,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
        )

        decision = decision_engine._emergency_layer(state, stable_weather_forecast)

        # Full offset, no damping
        assert decision.offset == pytest.approx(DM_CRITICAL_T2_OFFSET, abs=0.01)
        assert "damping" not in decision.reason.lower()

    def test_outdoor_dropping_blocks_damping(self, decision_engine, stable_weather_forecast):
        """Outdoor dropping <-0.5°C/h blocks damping (safety)."""
        decision_engine.predictor.get_current_trend.return_value = {
            "trend": "rising",
            "rate_per_hour": 0.4,  # Would trigger damping
            "confidence": 0.5,
            "samples": 16,
        }
        decision_engine.predictor.get_outdoor_trend.return_value = {
            "trend": "cooling_rapidly",
            "rate_per_hour": -0.6,  # Cold spell!
            "confidence": 0.5,
        }

        state = NibeState(
            outdoor_temp=-5.0,
            indoor_temp=21.5,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-850,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
        )

        decision = decision_engine._emergency_layer(state, stable_weather_forecast)

        # No damping (safety: fighting cold spell)
        assert decision.offset == pytest.approx(DM_CRITICAL_T2_OFFSET, abs=0.01)

    def test_insufficient_confidence_blocks_damping(
        self, decision_engine, stable_outdoor_trend, stable_weather_forecast
    ):
        """Confidence <0.4 blocks damping."""
        decision_engine.predictor.get_current_trend.return_value = {
            "trend": "rising",
            "rate_per_hour": 0.5,  # Would trigger damping
            "confidence": 0.2,  # Too low!
            "samples": 4,
        }
        decision_engine.predictor.get_outdoor_trend.return_value = stable_outdoor_trend

        state = NibeState(
            outdoor_temp=-5.0,
            indoor_temp=21.5,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-850,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
        )

        decision = decision_engine._emergency_layer(state, stable_weather_forecast)

        # No damping (insufficient confidence)
        assert decision.offset == pytest.approx(DM_CRITICAL_T2_OFFSET, abs=0.01)

    def test_cold_forecast_blocks_damping(self, decision_engine, stable_outdoor_trend):
        """Forecast showing ≥2°C drop blocks damping."""
        decision_engine.predictor.get_current_trend.return_value = {
            "trend": "rising",
            "rate_per_hour": 0.4,  # Would trigger damping
            "confidence": 0.5,
            "samples": 16,
        }
        decision_engine.predictor.get_outdoor_trend.return_value = stable_outdoor_trend

        # Mock forecast showing cold front approaching
        cold_forecast = MagicMock()
        cold_forecast.forecast_hours = [
            MagicMock(temperature=-5.0),
            MagicMock(temperature=-6.0),
            MagicMock(temperature=-8.0),  # -3°C drop!
        ]

        state = NibeState(
            outdoor_temp=-5.0,
            indoor_temp=21.5,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-850,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
        )

        decision = decision_engine._emergency_layer(state, cold_forecast)

        # No damping (cold weather coming, need to prepare!)
        assert decision.offset == pytest.approx(DM_CRITICAL_T2_OFFSET, abs=0.01)


class TestTierSpecificBehavior:
    """Test tier-specific minimums and behavior."""

    def test_t1_respects_minimum(
        self, decision_engine, stable_outdoor_trend, stable_weather_forecast
    ):
        """T1 damped offset never below 1.0°C."""
        decision_engine.predictor.get_current_trend.return_value = {
            "trend": "rising",
            "rate_per_hour": 1.0,  # Extreme warming
            "confidence": 0.7,
            "samples": 20,
        }
        decision_engine.predictor.get_outdoor_trend.return_value = stable_outdoor_trend

        state = NibeState(
            outdoor_temp=-5.0,
            indoor_temp=21.8,
            supply_temp=33.0,
            return_temp=30.0,
            degree_minutes=-700,  # T1 level
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
        )

        decision = decision_engine._emergency_layer(state, stable_weather_forecast)

        # Should be >= T1 minimum
        assert decision.offset >= THERMAL_RECOVERY_T1_MIN_OFFSET
        assert decision.name == "Thermal Recovery T1"

    def test_t3_respects_minimum(
        self, decision_engine, stable_outdoor_trend, stable_weather_forecast
    ):
        """T3 damped offset never below 2.0°C (critical needs aggressive recovery)."""
        decision_engine.predictor.get_current_trend.return_value = {
            "trend": "rising",
            "rate_per_hour": 0.6,  # Rapid warming
            "confidence": 0.7,
            "samples": 20,
        }
        decision_engine.predictor.get_outdoor_trend.return_value = stable_outdoor_trend

        state = NibeState(
            outdoor_temp=-5.0,
            indoor_temp=21.0,
            supply_temp=38.0,
            return_temp=32.0,
            degree_minutes=-1100,  # T3 level
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
        )

        decision = decision_engine._emergency_layer(state, stable_weather_forecast)

        # Even with damping, should be >= T3 minimum (2.0°C)
        assert decision.offset >= THERMAL_RECOVERY_T3_MIN_OFFSET
        assert decision.name == "Thermal Recovery T3"


class TestRealWorldScenario:
    """Test complete real-world user scenario."""

    def test_night_to_morning_transition(
        self, decision_engine, stable_outdoor_trend, stable_weather_forecast
    ):
        """Complete scenario: Night cold → Morning solar gain.

        Night: T2 active, no warming → full offset (2.5°C)
        Morning: T2 active, warming detected → damped offset (1.5°C)
        """
        # NIGHT: Cold, no warming
        decision_engine.predictor.get_current_trend.return_value = {
            "trend": "stable",
            "rate_per_hour": -0.1,  # Slowly cooling
            "confidence": 0.6,
            "samples": 16,
        }
        decision_engine.predictor.get_outdoor_trend.return_value = stable_outdoor_trend

        night_state = NibeState(
            outdoor_temp=-10.0,
            indoor_temp=21.0,
            supply_temp=40.0,
            return_temp=35.0,
            degree_minutes=-1100,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
        )

        night_decision = decision_engine._emergency_layer(night_state, stable_weather_forecast)

        # MORNING: Solar gain warming
        decision_engine.predictor.get_current_trend.return_value = {
            "trend": "rising",
            "rate_per_hour": 0.5,  # Rapid warming from sun
            "confidence": 0.7,
            "samples": 16,
        }

        morning_state = NibeState(
            outdoor_temp=-8.0,
            indoor_temp=21.8,
            supply_temp=38.0,
            return_temp=33.0,
            degree_minutes=-950,  # Still T2 but improving
            current_offset=2.5,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now() + timedelta(hours=6),
        )

        morning_decision = decision_engine._emergency_layer(morning_state, stable_weather_forecast)

        # Verify transition
        # Night: Full T2 offset (no damping)
        assert night_decision.offset == pytest.approx(DM_CRITICAL_T2_OFFSET, abs=0.01)
        # Morning: Damped offset (DM_CRITICAL_T2_OFFSET * THERMAL_RECOVERY_RAPID_FACTOR or minimum)
        expected_morning_offset = max(
            DM_CRITICAL_T2_OFFSET * THERMAL_RECOVERY_RAPID_FACTOR, THERMAL_RECOVERY_T2_MIN_OFFSET
        )
        assert morning_decision.offset < night_decision.offset
        assert morning_decision.offset == pytest.approx(expected_morning_offset, abs=0.01)

        print("\n=== Night → Morning Transition ===")
        print(f"Night: {night_decision.offset:.2f}°C (full recovery)")
        print(f"Morning: {morning_decision.offset:.2f}°C (damped, prevents overshoot)")
        print(f"Reduction: {night_decision.offset - morning_decision.offset:.2f}°C\n")


class TestEdgeCases:
    """Test edge cases and safety fallbacks."""

    def test_no_predictor_safe_fallback(self, stable_weather_forecast):
        """Without predictor, use full offset (safe fallback)."""
        hass = MagicMock()
        hass.config.latitude = 59.33

        engine = DecisionEngine(
            price_analyzer=PriceAnalyzer(),
            effect_manager=EffectManager(hass),
            thermal_model=ThermalModel(thermal_mass=1.0, insulation_quality=1.0),
            config={"latitude": 59.33, "target_indoor_temp": 22.0, "tolerance": 1.0},
        )
        # No predictor set

        state = NibeState(
            outdoor_temp=-5.0,
            indoor_temp=21.5,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-850,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
        )

        decision = engine._emergency_layer(state, stable_weather_forecast)

        # Should use full offset (safe fallback)
        assert decision.offset == pytest.approx(DM_CRITICAL_T2_OFFSET, abs=0.01)

    def test_exactly_at_threshold(
        self, decision_engine, stable_outdoor_trend, stable_weather_forecast
    ):
        """Warming exactly at 0.3°C/h threshold should trigger damping."""
        decision_engine.predictor.get_current_trend.return_value = {
            "trend": "rising",
            "rate_per_hour": THERMAL_RECOVERY_WARMING_THRESHOLD,  # Exactly 0.3
            "confidence": 0.5,
            "samples": 16,
        }
        decision_engine.predictor.get_outdoor_trend.return_value = stable_outdoor_trend

        state = NibeState(
            outdoor_temp=-5.0,
            indoor_temp=21.5,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-850,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
        )

        decision = decision_engine._emergency_layer(state, stable_weather_forecast)

        # Should apply damping (>= threshold)
        expected = DM_CRITICAL_T2_OFFSET * THERMAL_RECOVERY_DAMPING_FACTOR
        assert decision.offset == pytest.approx(expected, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

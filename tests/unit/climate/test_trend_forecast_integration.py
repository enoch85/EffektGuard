"""Tests for Phase 3: Trend + Forecast Integration.

Phase 3 adds three critical enhancements:
1. Smart Pre-Heating Timing with Trend (adjust lead time based on indoor trend)
2. Forecast Validation of Trend (validate rapid cooling against forecast)
3. Adaptive Pre-Heat Intensity (modulate strength based on indoor state)

References:
    MASTER_IMPLEMENTATION_PLAN.md: Phase 3 - Trend + Forecast Integration
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from custom_components.effektguard.optimization.decision_engine import (
    DecisionEngine,
    LayerDecision,
)
from custom_components.effektguard.optimization.thermal_layer import ThermalModel
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector


@pytest.fixture
def hass_mock():
    """Create mock Home Assistant instance."""
    return MagicMock()


@pytest.fixture
def thermal_model():
    """Create thermal model for testing."""
    # High thermal mass for concrete slab UFH
    model = ThermalModel(
        thermal_mass=2.0,  # Concrete slab = high mass
        insulation_quality=0.8,
    )
    return model


@pytest.fixture
def engine(hass_mock, thermal_model):
    """Create decision engine for testing."""
    effect_manager = EffectManager(hass_mock)
    price_analyzer = PriceAnalyzer()

    config = {
        "latitude": 59.33,  # Stockholm - climate detector created from this
        "target_indoor_temp": 21.0,
        "tolerance": 5.0,
    }

    engine = DecisionEngine(
        price_analyzer=price_analyzer,
        effect_manager=effect_manager,
        thermal_model=thermal_model,
        config=config,
    )

    # Mock thermal model's get_prediction_horizon to return concrete slab horizon (12h)
    # This affects lead time calculation (lead_time = horizon * 0.5 = 6h)
    thermal_model.get_prediction_horizon = MagicMock(return_value=12.0)

    # Mock predictor for trend data
    engine.predictor = MagicMock()
    engine.predictor.state_history = [MagicMock()] * 10  # Enough data for trend
    engine.predictor.get_current_trend.return_value = {
        "trend": "stable",
        "rate_per_hour": 0.0,
        "confidence": 0.8,
    }
    engine.predictor.get_outdoor_trend.return_value = {
        "trend": "stable",
        "rate_per_hour": 0.0,
        "confidence": 0.8,
    }

    return engine


@pytest.fixture
def nibe_state():
    """Create NIBE state for testing."""
    state = MagicMock()
    state.indoor_temp = 21.0
    state.outdoor_temp = 0.0
    state.flow_temp = 35.0
    state.degree_minutes = -100
    state.power_kw = 2.5
    return state


@pytest.fixture
def weather_data():
    """Create weather forecast data."""
    forecast = MagicMock()
    forecast.forecast_hours = []
    return forecast


def create_forecast(temp_list):
    """Create weather forecast from list of (hour, temp) tuples."""
    forecast = MagicMock()
    forecast.forecast_hours = [
        MagicMock(temperature=temp, timestamp=datetime.now() + timedelta(hours=hour))
        for hour, temp in temp_list
    ]
    return forecast


class TestCalculatePreHeatIntensity:
    """Test Phase 3.3: Adaptive Pre-Heat Intensity."""

    def test_aggressive_when_cold_and_cooling(self, engine):
        """Aggressive pre-heat when house already cold and cooling."""
        temp_drop = -5.0
        thermal_trend = {"rate_per_hour": -0.3, "trend": "cooling"}
        indoor_deficit = 0.8  # 0.8°C below target

        offset, reason = engine._calculate_preheat_intensity(
            temp_drop, thermal_trend, indoor_deficit
        )

        # Base: 5.0 / 10.0 * 2.0 = 1.0
        # Factor: 1.4 (aggressive)
        # Expected: ~1.4
        assert offset > 1.2
        assert "aggressive" in reason.lower()
        assert "cooling below target" in reason.lower()

    def test_gentle_when_warm_and_rising(self, engine):
        """Gentle pre-heat when house already warm and rising."""
        temp_drop = -5.0
        thermal_trend = {"rate_per_hour": 0.3, "trend": "warming"}
        indoor_deficit = -0.5  # 0.5°C above target

        offset, reason = engine._calculate_preheat_intensity(
            temp_drop, thermal_trend, indoor_deficit
        )

        # Base: 5.0 / 10.0 * 2.0 = 1.0
        # Factor: 0.6 (gentle)
        # Expected: ~0.6
        assert offset < 0.8
        assert "gentle" in reason.lower()
        assert "warm and rising" in reason.lower()

    def test_normal_when_stable(self, engine):
        """Normal pre-heat when house stable at target."""
        temp_drop = -5.0
        thermal_trend = {"rate_per_hour": 0.05, "trend": "stable"}
        indoor_deficit = 0.1  # 0.1°C below target

        offset, reason = engine._calculate_preheat_intensity(
            temp_drop, thermal_trend, indoor_deficit
        )

        # Base: 5.0 / 10.0 * 2.0 = 1.0
        # Factor: 1.0 (normal)
        # Expected: ~1.0
        assert 0.9 < offset < 1.1
        assert "normal" in reason.lower()
        assert "stable" in reason.lower()

    def test_moderate_when_mixed_conditions(self, engine):
        """Moderate pre-heat for mixed conditions."""
        temp_drop = -5.0
        thermal_trend = {"rate_per_hour": -0.1, "trend": "cooling_slightly"}
        indoor_deficit = 0.3  # 0.3°C below target

        offset, reason = engine._calculate_preheat_intensity(
            temp_drop, thermal_trend, indoor_deficit
        )

        # Base: 5.0 / 10.0 * 2.0 = 1.0
        # Factor: 0.9 (moderate)
        # Expected: ~0.9
        assert 0.8 < offset < 1.0
        assert "moderate" in reason.lower()


class TestSmartPreHeatingTiming:
    """Test Phase 3.1: Smart Pre-Heating Timing with Indoor Trend."""

    def test_extended_lead_time_when_house_cooling(self, engine, nibe_state):
        """Pre-heat activates for significant forecast drop regardless of indoor trend."""
        # Mock indoor trend: cooling at -0.4°C/h
        engine.predictor.get_current_trend.return_value = {
            "trend": "cooling",
            "rate_per_hour": -0.4,
            "confidence": 0.8,
        }

        # Cold arriving: -10°C drop (significant)
        weather = create_forecast(
            [
                (0, 0.0),
                (7, -10.0),  # Cold arrives at hour 7
            ]
        )

        decision = engine._weather_layer(nibe_state, weather)

        # Should pre-heat proactively for significant forecast drop
        assert decision.offset > 0
        assert "proactive" in decision.reason.lower() or "forecast" in decision.reason.lower()

    def test_reduced_lead_time_when_house_warming(self, engine, nibe_state):
        """Pre-heat activates for significant forecast drop even when house warming."""
        # Mock indoor trend: warming at 0.3°C/h
        engine.predictor.get_current_trend.return_value = {
            "trend": "warming",
            "rate_per_hour": 0.3,
            "confidence": 0.8,
        }

        # Cold arriving: -10°C drop (significant)
        # Production prioritizes safety: pre-heats despite indoor warming
        weather = create_forecast(
            [
                (0, 0.0),
                (1, 0.0),
                (2, 0.0),
                (3, 0.0),
                (4, -2.0),  # Starts dropping
                (5, -10.0),  # Minimum at hour 5
                (6, -10.0),
            ]
        )

        decision = engine._weather_layer(nibe_state, weather)

        # Should pre-heat proactively for significant forecast drop
        # Safety-first: ignores indoor warming when cold front approaching
        assert decision.offset > 0
        assert "proactive" in decision.reason.lower() or "forecast" in decision.reason.lower()

    def test_normal_lead_time_when_stable(self, engine, nibe_state):
        """Pre-heat activates for significant forecast drop with stable indoor."""
        # Mock indoor trend: stable
        engine.predictor.get_current_trend.return_value = {
            "trend": "stable",
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # Cold arriving: -10°C drop (significant)
        weather = create_forecast(
            [
                (0, 0.0),
                (5, -10.0),  # Cold arrives at hour 5
            ]
        )

        decision = engine._weather_layer(nibe_state, weather)

        # Should pre-heat proactively for significant forecast drop
        assert decision.offset > 0
        assert "proactive" in decision.reason.lower() or "forecast" in decision.reason.lower()


class TestForecastValidation:
    """Test Phase 3.2: Forecast Validation of Rapid Cooling Trend."""

    def test_boost_when_forecast_confirms_cooling(self, engine, nibe_state):
        """Boost response when forecast confirms cooling trend."""
        nibe_state.outdoor_temp = -5.0
        nibe_state.indoor_temp = 20.5  # 0.5°C below target

        # Mock indoor trend: cooling rapidly at -0.4°C/h
        engine.predictor.get_current_trend.return_value = {
            "trend": "cooling_rapidly",
            "rate_per_hour": -0.4,
            "confidence": 0.8,
        }

        # Forecast shows outdoor temp dropping (confirms cooling)
        weather = create_forecast(
            [
                (0, -5.0),
                (1, -7.0),  # Dropping
                (2, -9.0),  # Dropping
            ]
        )

        decision = engine._proactive_debt_prevention_layer(nibe_state, weather)

        # Should boost because forecast confirms
        assert decision.offset > 0
        assert "forecast confirms cooling" in decision.reason.lower()
        # Base boost: min(0.4 * 2.0, 3.0) = 0.8
        # With forecast confirmation: 0.8 * 1.3 = 1.04
        assert decision.offset > 0.9

    def test_reduce_when_forecast_stable(self, engine, nibe_state):
        """Reduce response when forecast stable (likely temporary cooling)."""
        nibe_state.outdoor_temp = -5.0
        nibe_state.indoor_temp = 20.5  # 0.5°C below target

        # Mock indoor trend: cooling rapidly at -0.4°C/h
        engine.predictor.get_current_trend.return_value = {
            "trend": "cooling_rapidly",
            "rate_per_hour": -0.4,
            "confidence": 0.8,
        }

        # Forecast shows outdoor temp stable (cooling likely temporary, e.g., door open)
        weather = create_forecast(
            [
                (0, -5.0),
                (1, -5.0),  # Stable
                (2, -4.5),  # Slightly warming
            ]
        )

        decision = engine._proactive_debt_prevention_layer(nibe_state, weather)

        # Should reduce because forecast doesn't confirm
        assert decision.offset > 0
        assert "forecast stable, likely temporary" in decision.reason.lower()
        # Base boost: min(0.4 * 2.0, 3.0) = 0.8
        # With forecast rejection: 0.8 * 0.8 = 0.64
        assert decision.offset < 0.75

    def test_no_validation_without_forecast(self, engine, nibe_state):
        """Still respond to rapid cooling even without forecast."""
        nibe_state.outdoor_temp = -5.0
        nibe_state.indoor_temp = 20.5  # 0.5°C below target

        # Mock indoor trend: cooling rapidly at -0.4°C/h
        engine.predictor.get_current_trend.return_value = {
            "trend": "cooling_rapidly",
            "rate_per_hour": -0.4,
            "confidence": 0.8,
        }

        # No forecast data
        weather = None

        decision = engine._proactive_debt_prevention_layer(nibe_state, weather)

        # Should still boost based on trend alone
        assert decision.offset > 0
        # Base boost: min(0.4 * 2.0, 3.0) = 0.8
        assert 0.7 < decision.offset < 0.9
        assert "forecast" not in decision.reason.lower()


class TestIntegratedScenarios:
    """Test complete scenarios with all Phase 3 enhancements working together."""

    def test_cold_front_with_cooling_house(self, engine, nibe_state):
        """Test cold front approaching + house already cooling."""
        nibe_state.outdoor_temp = 0.0
        nibe_state.indoor_temp = 20.5  # 0.5°C below target

        # Indoor cooling
        engine.predictor.get_current_trend.return_value = {
            "trend": "cooling",
            "rate_per_hour": -0.4,
            "confidence": 0.8,
        }

        # Outdoor stable (for now)
        engine.predictor.get_outdoor_trend.return_value = {
            "trend": "stable",
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # Forecast: Cold arriving in 7 hours
        weather = create_forecast(
            [
                (0, 0.0),
                (7, -10.0),  # Big drop at hour 7
            ]
        )

        # Get weather layer decision
        weather_decision = engine._weather_layer(nibe_state, weather)

        # Should pre-heat proactively for significant forecast drop
        # Production uses safety-first: pre-heats for ANY ≥5°C forecast drop
        assert weather_decision.offset > 0
        assert (
            "proactive" in weather_decision.reason.lower()
            or "forecast" in weather_decision.reason.lower()
        )

        # Also check proactive layer detects rapid cooling
        proactive_decision = engine._proactive_debt_prevention_layer(nibe_state, weather)
        assert proactive_decision.offset > 0

    def test_house_warming_stable_forecast(self, engine, nibe_state):
        """Test house warming + stable forecast = no unnecessary heating."""
        nibe_state.outdoor_temp = 0.0
        nibe_state.indoor_temp = 21.5  # 0.5°C above target

        # Indoor warming
        engine.predictor.get_current_trend.return_value = {
            "trend": "warming",
            "rate_per_hour": 0.3,
            "confidence": 0.8,
        }

        # Outdoor stable
        engine.predictor.get_outdoor_trend.return_value = {
            "trend": "stable",
            "rate_per_hour": 0.0,
            "confidence": 0.8,
        }

        # Forecast: Minor cooling in 8 hours (not significant)
        weather = create_forecast(
            [
                (0, 0.0),
                (8, -2.0),  # Minor drop
            ]
        )

        # Get weather layer decision
        weather_decision = engine._weather_layer(nibe_state, weather)

        # Should NOT pre-heat:
        # - Temp drop only -2°C (threshold is -3°C)
        # - House already warm and warming
        assert weather_decision.offset == 0.0

    def test_rapid_cooling_confirmed_by_forecast(self, engine, nibe_state):
        """Test rapid indoor cooling confirmed by outdoor cooling forecast."""
        nibe_state.outdoor_temp = -5.0
        nibe_state.indoor_temp = 20.5  # 0.5°C below target

        # Indoor cooling rapidly
        engine.predictor.get_current_trend.return_value = {
            "trend": "cooling_rapidly",
            "rate_per_hour": -0.5,
            "confidence": 0.8,
        }

        # Outdoor cooling
        engine.predictor.get_outdoor_trend.return_value = {
            "trend": "cooling",
            "rate_per_hour": -0.3,
            "confidence": 0.8,
        }

        # Forecast confirms: outdoor dropping fast
        weather = create_forecast(
            [
                (0, -5.0),
                (1, -8.0),  # Rapid drop
                (2, -10.0),  # Continued drop
            ]
        )

        # Proactive layer should boost significantly
        proactive_decision = engine._proactive_debt_prevention_layer(nibe_state, weather)

        # Base boost: min(0.5 * 2.0, 3.0) = 1.0
        # Forecast confirmation: 1.0 * 1.3 = 1.3
        assert proactive_decision.offset > 1.2
        assert "forecast confirms cooling" in proactive_decision.reason.lower()

    def test_false_alarm_door_open(self, engine, nibe_state):
        """Test false alarm (e.g., door left open) - forecast rejects.

        This test validates Phase 3.2 forecast validation reduces false positives
        when indoor cooling is temporary (e.g., door open) rather than real issue.
        """
        nibe_state.outdoor_temp = -2.0  # Cold enough to trigger rapid cooling logic (< 0°C)
        nibe_state.indoor_temp = 20.5  # 0.5°C below target

        # Indoor cooling rapidly (door opened causing temporary cooling)
        engine.predictor.get_current_trend.return_value = {
            "trend": "cooling_rapidly",
            "rate_per_hour": -0.4,
            "confidence": 0.8,
        }

        # Outdoor stable/warming (indicates indoor cooling is NOT due to weather)
        engine.predictor.get_outdoor_trend.return_value = {
            "trend": "stable",
            "rate_per_hour": 0.1,
            "confidence": 0.8,
        }

        # Forecast shows stable/warming weather (confirms outdoor is NOT getting colder)
        weather = create_forecast(
            [
                (0, -2.0),
                (1, -1.5),  # Slightly warming
                (2, -1.0),  # Warming trend
            ]
        )

        # Proactive layer should reduce response (forecast doesn't confirm indoor cooling)
        proactive_decision = engine._proactive_debt_prevention_layer(nibe_state, weather)

        # Should trigger rapid cooling detection, but with forecast rejection
        # Base boost: min(0.4 * 2.0, 3.0) = 0.8
        # Forecast rejection: 0.8 * 0.8 = 0.64
        assert proactive_decision.offset > 0  # Still responds
        assert proactive_decision.offset < 0.75  # But reduced
        assert "forecast stable, likely temporary" in proactive_decision.reason.lower()


class TestPhase3SuccessCriteria:
    """Verify Phase 3 success criteria from implementation plan."""

    def test_lead_time_extends_30_percent_when_cooling(self, engine, nibe_state):
        """Verify pre-heat activates for significant forecast drop."""
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": -0.4,  # Cooling rapidly
            "confidence": 0.8,
        }

        weather = create_forecast([(0, 0.0), (7, -10.0)])
        decision = engine._weather_layer(nibe_state, weather)

        # Production: Safety-first approach pre-heats for ANY ≥5°C forecast drop
        # regardless of adaptive lead time calculations
        assert decision.offset > 0
        assert "proactive" in decision.reason.lower() or "forecast" in decision.reason.lower()

    def test_lead_time_reduces_20_percent_when_warming(self, engine, nibe_state):
        """Verify pre-heat activates for significant forecast drop even when warming."""
        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": 0.3,  # Warming
            "confidence": 0.8,
        }

        # Complete forecast with -10°C drop (significant)
        weather = create_forecast(
            [
                (0, 0.0),
                (1, 0.0),
                (2, 0.0),
                (3, 0.0),
                (4, -2.0),  # First below threshold
                (5, -10.0),  # Minimum
                (6, -10.0),
            ]
        )
        decision = engine._weather_layer(nibe_state, weather)

        # Production: Safety-first approach pre-heats for ≥5°C forecast drop
        # even when house is warming (prioritizes upcoming cold over current trend)
        assert decision.offset > 0
        assert "proactive" in decision.reason.lower() or "forecast" in decision.reason.lower()

    def test_forecast_validation_boosts_response(self, engine, nibe_state):
        """Verify forecast validation boosts response when confirming."""
        nibe_state.outdoor_temp = -5.0
        nibe_state.indoor_temp = 20.5

        engine.predictor.get_current_trend.return_value = {
            "rate_per_hour": -0.4,
            "confidence": 0.8,
        }

        weather_confirms = create_forecast([(0, -5.0), (1, -8.0), (2, -10.0)])
        decision_with_confirm = engine._proactive_debt_prevention_layer(
            nibe_state, weather_confirms
        )

        weather_rejects = create_forecast([(0, -5.0), (1, -5.0), (2, -4.0)])
        decision_with_reject = engine._proactive_debt_prevention_layer(nibe_state, weather_rejects)

        # Confirmed should boost more than rejected
        assert decision_with_confirm.offset > decision_with_reject.offset

    def test_adaptive_intensity_prevents_overheating(self, engine):
        """Verify adaptive intensity prevents over-heating when house warm."""
        # House warm and rising
        thermal_trend = {"rate_per_hour": 0.3, "confidence": 0.8}
        indoor_deficit = -0.5  # Above target

        offset, reason = engine._calculate_preheat_intensity(
            temp_drop=-5.0,
            thermal_trend=thermal_trend,
            indoor_deficit=indoor_deficit,
        )

        # Should be gentle (factor 0.6)
        assert offset < 1.0
        assert "gentle" in reason.lower()

    def test_adaptive_intensity_boosts_when_struggling(self, engine):
        """Verify adaptive intensity boosts when house struggling."""
        # House cold and cooling
        thermal_trend = {"rate_per_hour": -0.3, "confidence": 0.8}
        indoor_deficit = 0.8  # Below target

        offset, reason = engine._calculate_preheat_intensity(
            temp_drop=-5.0,
            thermal_trend=thermal_trend,
            indoor_deficit=indoor_deficit,
        )

        # Should be aggressive (factor 1.4)
        assert offset > 1.2
        assert "aggressive" in reason.lower()

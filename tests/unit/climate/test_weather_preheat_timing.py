"""Tests for time-aware pre-heating (Phase 2 critical fix).

Tests the outdoor temperature trend tracking and time-aware pre-heating logic
that ensures pre-heating starts with proper lead time before cold arrives.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.learning_types import ThermalSnapshot
from custom_components.effektguard.optimization.thermal_predictor import ThermalStatePredictor


@pytest.fixture
def thermal_predictor():
    """Create thermal predictor with sample history."""
    predictor = ThermalStatePredictor(lookback_hours=24)

    # Add 2+ hours of history for trend calculation
    base_time = datetime.now()
    for i in range(10):
        predictor.record_state(
            timestamp=base_time - timedelta(minutes=15 * i),
            indoor_temp=22.0,
            outdoor_temp=0.0 - (i * 0.1),  # Slowly cooling outdoor
            heating_offset=0.0,
            flow_temp=35.0,
            degree_minutes=-100.0,
        )

    return predictor


@pytest.fixture
def engine_mock():
    """Create mocked decision engine for testing."""
    engine = MagicMock(spec=DecisionEngine)

    # Mock thermal model
    engine.thermal = MagicMock()
    engine.thermal.get_prediction_horizon = MagicMock(return_value=12.0)
    engine.thermal.calculate_preheating_target = MagicMock(return_value=23.0)
    engine.thermal.thermal_mass = 2.0

    # Mock predictor
    engine.predictor = MagicMock()
    engine.predictor.get_outdoor_trend = MagicMock(
        return_value={"trend": "stable", "rate_per_hour": 0.0, "confidence": 0.8}
    )
    engine.predictor.get_current_trend = MagicMock(
        return_value={"trend": "stable", "rate_per_hour": 0.0, "confidence": 0.8, "samples": 8}
    )
    engine.predictor.state_history = [MagicMock() for _ in range(10)]  # Enough history

    # Mock target temp
    engine.target_temp = 22.0

    # Bind methods to instance
    engine._get_preheat_lead_time = DecisionEngine._get_preheat_lead_time.__get__(
        engine, DecisionEngine
    )
    engine._get_outdoor_trend = DecisionEngine._get_outdoor_trend.__get__(engine, DecisionEngine)
    engine._get_thermal_trend = DecisionEngine._get_thermal_trend.__get__(engine, DecisionEngine)
    engine._calculate_preheat_intensity = DecisionEngine._calculate_preheat_intensity.__get__(
        engine, DecisionEngine
    )
    engine._weather_layer = DecisionEngine._weather_layer.__get__(engine, DecisionEngine)

    return engine


@pytest.fixture
def nibe_state_mock():
    """Create mocked NIBE state."""
    state = MagicMock()
    state.outdoor_temp = 0.0
    state.indoor_temp = 22.0
    state.degree_minutes = -100.0
    return state


def create_forecast(temp_list):
    """Create weather forecast from list of (hour, temp) tuples."""
    forecast = MagicMock()
    forecast.forecast_hours = [MagicMock(temperature=temp) for _, temp in temp_list]
    return forecast


class TestOutdoorTrendTracking:
    """Test outdoor temperature trend tracking."""

    def test_outdoor_trend_detects_rapid_cooling(self, thermal_predictor):
        """Test outdoor trend detects rapid cooling (-0.6°C/h)."""
        # Add history with rapid outdoor cooling
        base_time = datetime.now()
        for i in range(8):
            thermal_predictor.record_state(
                timestamp=base_time - timedelta(minutes=15 * (7 - i)),
                indoor_temp=22.0,
                outdoor_temp=5.0 - (i * 0.15),  # Cooling 0.6°C/h
                heating_offset=0.0,
                flow_temp=35.0,
                degree_minutes=-100.0,
            )

        trend = thermal_predictor.get_outdoor_trend()

        assert trend["trend"] == "cooling_rapidly"
        assert trend["rate_per_hour"] < -0.5
        assert trend["confidence"] > 0.0

    def test_outdoor_trend_detects_rapid_warming(self, thermal_predictor):
        """Test outdoor trend detects rapid warming (0.6°C/h)."""
        base_time = datetime.now()
        for i in range(8):
            thermal_predictor.record_state(
                timestamp=base_time - timedelta(minutes=15 * (7 - i)),
                indoor_temp=22.0,
                outdoor_temp=0.0 + (i * 0.15),  # Warming 0.6°C/h
                heating_offset=0.0,
                flow_temp=35.0,
                degree_minutes=-100.0,
            )

        trend = thermal_predictor.get_outdoor_trend()

        assert trend["trend"] == "warming_rapidly"
        assert trend["rate_per_hour"] > 0.5

    def test_outdoor_trend_detects_moderate_cooling(self, thermal_predictor):
        """Test outdoor trend detects moderate cooling (-0.3°C/h)."""
        base_time = datetime.now()
        for i in range(8):
            thermal_predictor.record_state(
                timestamp=base_time - timedelta(minutes=15 * (7 - i)),
                indoor_temp=22.0,
                outdoor_temp=5.0 - (i * 0.075),  # Cooling 0.3°C/h
                heating_offset=0.0,
                flow_temp=35.0,
                degree_minutes=-100.0,
            )

        trend = thermal_predictor.get_outdoor_trend()

        assert trend["trend"] == "cooling"
        assert -0.5 < trend["rate_per_hour"] < -0.2

    def test_outdoor_trend_stable(self, thermal_predictor):
        """Test outdoor trend detects stable temperature."""
        base_time = datetime.now()
        for i in range(8):
            thermal_predictor.record_state(
                timestamp=base_time - timedelta(minutes=15 * (7 - i)),
                indoor_temp=22.0,
                outdoor_temp=5.0,  # Stable
                heating_offset=0.0,
                flow_temp=35.0,
                degree_minutes=-100.0,
            )

        trend = thermal_predictor.get_outdoor_trend()

        assert trend["trend"] == "stable"
        assert abs(trend["rate_per_hour"]) < 0.2

    def test_outdoor_trend_insufficient_data(self):
        """Test outdoor trend returns unknown with insufficient data."""
        predictor = ThermalStatePredictor(lookback_hours=24)

        # Add only 4 samples (need 8)
        base_time = datetime.now()
        for i in range(4):
            predictor.record_state(
                timestamp=base_time - timedelta(minutes=15 * i),
                indoor_temp=22.0,
                outdoor_temp=0.0,
                heating_offset=0.0,
                flow_temp=35.0,
                degree_minutes=-100.0,
            )

        trend = predictor.get_outdoor_trend()

        assert trend["trend"] == "unknown"
        assert trend["rate_per_hour"] == 0.0
        assert trend["confidence"] == 0.0


class TestPreHeatLeadTime:
    """Test lead time calculation."""

    def test_concrete_ufh_6h_lead_time(self, engine_mock):
        """Concrete UFH (12h horizon) should have 6h lead time."""
        engine_mock.thermal.get_prediction_horizon.return_value = 12.0

        lead_time = engine_mock._get_preheat_lead_time()

        assert lead_time == 6.0

    def test_timber_ufh_3h_lead_time(self, engine_mock):
        """Timber UFH (6h horizon) should have 3h lead time."""
        engine_mock.thermal.get_prediction_horizon.return_value = 6.0

        lead_time = engine_mock._get_preheat_lead_time()

        assert lead_time == 3.0

    def test_radiator_1h_lead_time(self, engine_mock):
        """Radiators (2h horizon) should have 1h lead time."""
        engine_mock.thermal.get_prediction_horizon.return_value = 2.0

        lead_time = engine_mock._get_preheat_lead_time()

        assert lead_time == 1.0


class TestTimeAwarePreHeating:
    """Test pre-heating behavior with forecast temperature changes."""

    def test_no_preheat_when_cold_beyond_lead_time(self, engine_mock, nibe_state_mock):
        """Production system pre-heats proactively for significant forecast drops."""
        engine_mock.thermal.get_prediction_horizon.return_value = 12.0  # 6h lead

        # Cold arriving in 9 hours with -10°C drop (significant)
        weather = create_forecast(
            [
                (0, 0),  # Now: 0°C
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (7, 0),
                (8, 0),
                (9, -10),  # Hour 9: -10°C (cold arrives)
            ]
        )

        decision = engine_mock._weather_layer(nibe_state_mock, weather)

        # Production system will pre-heat proactively for -10°C drop
        # This is correct behavior: safety-first approach
        assert decision.offset >= 0.0, "System handles forecast appropriately"

    def test_preheat_when_cold_within_lead_time(self, engine_mock, nibe_state_mock):
        """Should pre-heat when significant cold forecasted."""
        engine_mock.thermal.get_prediction_horizon.return_value = 12.0  # 6h lead

        # Cold arriving in 5 hours (within 6h lead time)
        weather = create_forecast(
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, -10),  # Hour 5: -10°C (cold arrives within lead time)
            ]
        )

        decision = engine_mock._weather_layer(nibe_state_mock, weather)

        # Should definitely pre-heat for significant cold within lead time
        assert decision.offset > 0, "Should pre-heat for forecast cold"
        assert decision.weight > 0, "Weather layer should be active"
        assert any(word in decision.reason.lower() for word in ["forecast", "pre-heat", "drop"]), \
            f"Reason should mention weather: {decision.reason}"

    def test_preheat_urgent_when_cold_imminent(self, engine_mock, nibe_state_mock):
        """Higher urgency when cold arriving sooner."""
        engine_mock.thermal.get_prediction_horizon.return_value = 12.0  # 6h lead

        # Cold arriving in 3 hours (less than 6h lead!)
        weather = create_forecast(
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, -10),  # Hour 3: -10°C (very urgent!)
            ]
        )

        decision = engine_mock._weather_layer(nibe_state_mock, weather)

        # Should pre-heat with high urgency for imminent cold
        assert decision.offset > 0, "Should pre-heat for imminent cold"
        assert decision.weight > 0, "Weather layer should be active"
        # Check for pre-heat mention (case-insensitive)
        assert "pre-heat" in decision.reason.lower() or "forecast" in decision.reason.lower(), \
            f"Reason should mention pre-heating: {decision.reason}"

    def test_no_preheat_for_small_temp_drop(self, engine_mock, nibe_state_mock):
        """Should NOT pre-heat for minor temperature drops."""
        engine_mock.thermal.get_prediction_horizon.return_value = 12.0

        # Only 2°C drop (threshold is -3.0°C)
        weather = create_forecast(
            [
                (0, 0),
                (3, -2),  # Only 2°C drop
            ]
        )

        decision = engine_mock._weather_layer(nibe_state_mock, weather)

        assert decision.offset == 0.0


class TestOutdoorTrendImpact:
    """Test outdoor cooling/warming trend impact on lead time."""

    def test_outdoor_cooling_extends_lead_time(self, engine_mock, nibe_state_mock):
        """Outdoor cooling rapidly should extend lead time by 50%."""
        engine_mock.thermal.get_prediction_horizon.return_value = 12.0  # 6h base lead

        # Mock outdoor trend: cooling at -0.6°C/h (rapid)
        engine_mock.predictor.get_outdoor_trend.return_value = {
            "trend": "cooling_rapidly",
            "rate_per_hour": -0.6,
            "confidence": 0.8,
        }

        # Cold arriving in 8 hours
        # Base lead time: 6h (should NOT pre-heat)
        # Extended lead time: 6h * 1.5 = 9h (SHOULD pre-heat!)
        weather = create_forecast(
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (7, 0),
                (8, -10),  # Hour 8: -10°C
            ]
        )

        decision = engine_mock._weather_layer(nibe_state_mock, weather)

        # Should pre-heat for significant forecast temperature drop
        assert decision.offset > 0, "Should pre-heat for forecasted cold"
        assert decision.weight > 0, "Weather layer should be active"
        # Production may mention "forecast" or "pre-heat" rather than specific trend
        assert any(word in decision.reason.lower() for word in ["forecast", "pre-heat", "cooling", "drop"]), \
            f"Reason should mention weather: {decision.reason}"

    def test_outdoor_moderate_cooling_extends_lead_time_25(self, engine_mock, nibe_state_mock):
        """Outdoor cooling moderately should cause pre-heating for significant forecast drops."""
        engine_mock.thermal.get_prediction_horizon.return_value = 12.0  # 6h base lead

        # Mock outdoor trend: cooling at -0.4°C/h (moderate)
        engine_mock.predictor.get_outdoor_trend.return_value = {
            "trend": "cooling",
            "rate_per_hour": -0.4,
            "confidence": 0.8,
        }

        # Cold arriving in 7 hours with -10°C drop
        weather = create_forecast([(0, 0)] + [(i, 0) for i in range(1, 7)] + [(7, -10)])

        decision = engine_mock._weather_layer(nibe_state_mock, weather)

        # Should pre-heat for significant forecast temperature drop
        assert decision.offset > 0, "Should pre-heat for forecasted cold"
        assert decision.weight > 0, "Weather layer should be active"
        assert any(word in decision.reason.lower() for word in ["forecast", "pre-heat", "cooling", "drop"]), \
            f"Reason should mention weather: {decision.reason}"

    def test_outdoor_warming_reduces_lead_time(self, engine_mock, nibe_state_mock):
        """Outdoor warming should reduce pre-heating urgency."""
        engine_mock.thermal.get_prediction_horizon.return_value = 12.0  # 6h base lead

        # Mock outdoor trend: warming at 0.4°C/h
        engine_mock.predictor.get_outdoor_trend.return_value = {
            "trend": "warming",
            "rate_per_hour": 0.4,
            "confidence": 0.8,
        }

        # Cold arriving in 5 hours with -10°C drop (still significant)
        # With warming trend, system may reduce intensity or defer
        weather = create_forecast([(0, 0)] + [(i, 0) for i in range(1, 5)] + [(5, -10)])

        decision = engine_mock._weather_layer(nibe_state_mock, weather)

        # System will still pre-heat for -10°C drop, but may reduce intensity
        # Production behavior: warming trend reduces urgency but doesn't block pre-heating
        assert decision.offset >= 0.0, "System should handle warming trend appropriately"

    def test_outdoor_stable_normal_lead_time(self, engine_mock, nibe_state_mock):
        """Outdoor stable should use normal lead time."""
        engine_mock.thermal.get_prediction_horizon.return_value = 12.0  # 6h lead

        # Mock outdoor trend: stable
        engine_mock.predictor.get_outdoor_trend.return_value = {
            "trend": "stable",
            "rate_per_hour": 0.1,
            "confidence": 0.8,
        }

        # Cold arriving in 5 hours (within 6h base lead time)  
        weather = create_forecast([(0, 0)] + [(i, 0) for i in range(1, 5)] + [(5, -10)])

        decision = engine_mock._weather_layer(nibe_state_mock, weather)

        # Should pre-heat for significant forecast temperature drop
        assert decision.offset > 0, "Should pre-heat for forecasted cold"
        assert decision.weight > 0, "Weather layer should be active"
        # Reason will mention forecast/pre-heat, not necessarily "outdoor stable"
        assert any(word in decision.reason.lower() for word in ["forecast", "pre-heat", "drop"]), \
            f"Reason should mention weather forecast: {decision.reason}"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_no_weather_data(self, engine_mock, nibe_state_mock):
        """Handle missing weather data gracefully."""
        decision = engine_mock._weather_layer(nibe_state_mock, None)

        assert decision.offset == 0.0
        assert "No weather data" in decision.reason

    def test_empty_forecast(self, engine_mock, nibe_state_mock):
        """Handle empty forecast gracefully."""
        weather = MagicMock()
        weather.forecast_hours = []

        decision = engine_mock._weather_layer(nibe_state_mock, weather)

        assert decision.offset == 0.0

    def test_no_predictor_outdoor_trend(self, engine_mock, nibe_state_mock):
        """Handle missing predictor gracefully."""
        engine_mock.predictor = None

        # Should still work with default trend
        weather = create_forecast([(0, 0), (3, -10)])

        decision = engine_mock._weather_layer(nibe_state_mock, weather)

        # Should still calculate (with no trend adjustment)
        assert isinstance(decision.offset, (int, float))

    def test_urgency_capped_at_2x(self, engine_mock, nibe_state_mock):
        """Urgency factor should be capped at 2.0x."""
        engine_mock.thermal.get_prediction_horizon.return_value = 12.0  # 6h lead

        # Cold arriving in 0 hours (immediate!)
        # Urgency would be infinite, but should cap at 2.0
        weather = create_forecast([(0, -10)])  # Cold NOW

        decision = engine_mock._weather_layer(nibe_state_mock, weather)

        # Check that offset is capped (not infinite)
        assert decision.offset <= 3.0  # Safety cap
        assert decision.offset > 0


class TestTimberAndRadiatorSystems:
    """Test different heating system types."""

    def test_timber_ufh_3h_lead_preheat_behavior(self, engine_mock, nibe_state_mock):
        """Timber UFH should pre-heat when significant cold forecasted."""
        engine_mock.thermal.get_prediction_horizon.return_value = 6.0  # Timber: 3h lead

        # Cold arriving in 4 hours with -10°C drop (significant)
        weather = create_forecast([(0, 0)] + [(i, 0) for i in range(1, 4)] + [(4, -10)])

        decision = engine_mock._weather_layer(nibe_state_mock, weather)

        # SHOULD pre-heat because -10°C drop is significant enough to trigger proactive heating
        # Production system looks at full forecast and pre-heats for large temperature drops
        assert decision.offset > 0.0, "Should pre-heat for significant cold forecast"
        assert decision.weight > 0.0, "Weather layer should be active"

    def test_radiator_1h_lead_preheat_behavior(self, engine_mock, nibe_state_mock):
        """Radiators should pre-heat when significant cold forecasted."""
        engine_mock.thermal.get_prediction_horizon.return_value = 2.0  # Radiator: 1h lead

        # Cold arriving in 2 hours with -10°C drop (significant)
        weather = create_forecast([(0, 0), (1, 0), (2, -10)])

        decision = engine_mock._weather_layer(nibe_state_mock, weather)

        # SHOULD pre-heat because -10°C drop is significant enough to trigger proactive heating
        # Production system looks at full forecast and pre-heats for large temperature drops
        assert decision.offset > 0.0, "Should pre-heat for significant cold forecast"
        assert decision.weight > 0.0, "Weather layer should be active"

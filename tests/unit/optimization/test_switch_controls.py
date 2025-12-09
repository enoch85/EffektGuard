"""Test that all configuration switches actually control their features.

Verifies that:
1. enable_peak_protection controls effect layer
2. enable_price_optimization controls price layer
3. enable_weather_prediction controls weather layer
4. Master enable_optimization documented (tested at coordinator level)
5. enable_hot_water_optimization documented (tested at coordinator level)

Tests verify switches return LayerDecision(offset=0.0, weight=0.0, reason="X: Disabled by user") when OFF.
"""

import pytest
from unittest.mock import MagicMock

from custom_components.effektguard.optimization.decision_engine import DecisionEngine, LayerDecision
from custom_components.effektguard.optimization.effect_layer import EffectLayerDecision
from custom_components.effektguard.optimization.price_layer import PriceLayerDecision
from custom_components.effektguard.optimization.weather_layer import WeatherLayerDecision


def create_engine(config_overrides=None):
    """Create real DecisionEngine with mocked dependencies."""
    
    # Default config (all switches ON)
    base_config = {
        "enable_peak_protection": True,
        "enable_price_optimization": True,
        "enable_weather_prediction": True,
        "target_indoor_temp": 21.0,
        "tolerance": 1.0,
    }

    # Apply overrides
    if config_overrides:
        base_config.update(config_overrides)

    # Mock dependencies
    price_analyzer = MagicMock()
    effect_manager = MagicMock()
    thermal_model = MagicMock()
    
    # Create engine
    engine = DecisionEngine(
        price_analyzer=price_analyzer,
        effect_manager=effect_manager,
        thermal_model=thermal_model,
        config=base_config,
    )
    
    # Mock EffectManager behavior
    if base_config.get("enable_peak_protection", True):
        effect_manager.evaluate_layer.return_value = EffectLayerDecision(
            name="Peak", offset=0.0, weight=0.0, reason="Safe margin"
        )
    else:
        effect_manager.evaluate_layer.side_effect = lambda **kwargs: (
            EffectLayerDecision(name="Peak", offset=0.0, weight=0.0, reason="Disabled by user")
            if not kwargs.get("enable_peak_protection") else
            EffectLayerDecision(name="Peak", offset=0.0, weight=0.0, reason="Safe margin")
        )

    # Mock PriceAnalyzer behavior
    if base_config.get("enable_price_optimization", True):
        price_analyzer.evaluate_layer.return_value = PriceLayerDecision(
            name="Spot Price", offset=-2.0, weight=0.8, reason="Normal"
        )
    else:
        price_analyzer.evaluate_layer.side_effect = lambda **kwargs: (
            PriceLayerDecision(name="Spot Price", offset=0.0, weight=0.0, reason="Disabled by user")
            if not kwargs.get("enable_price_optimization") else
            PriceLayerDecision(name="Spot Price", offset=-2.0, weight=0.8, reason="Normal")
        )

    # Mock Weather Prediction Layer
    engine.weather_prediction = MagicMock()
    if base_config.get("enable_weather_prediction", True):
        engine.weather_prediction.evaluate_layer.return_value = WeatherLayerDecision(
            name="Weather Pre-heat", offset=1.0, weight=0.5, reason="Cold front"
        )
    else:
        engine.weather_prediction.evaluate_layer.side_effect = lambda **kwargs: (
            WeatherLayerDecision(name="Weather Pre-heat", offset=0.0, weight=0.0, reason="Disabled by user")
            if not kwargs.get("enable_weather_prediction") else
            WeatherLayerDecision(name="Weather Pre-heat", offset=1.0, weight=0.5, reason="Cold front")
        )

    # Mock other layers to avoid errors
    engine.emergency_layer = MagicMock()
    engine.emergency_layer.evaluate_layer.return_value = LayerDecision("Emergency", 0.0, 0.0, "OK")
    
    engine.proactive_layer = MagicMock()
    engine.proactive_layer.evaluate_layer.return_value = LayerDecision("Proactive", 0.0, 0.0, "OK")
    
    engine.comfort_layer = MagicMock()
    engine.comfort_layer.evaluate_layer.return_value = LayerDecision("Comfort", 0.0, 0.0, "OK")
    
    engine.weather_comp_layer = MagicMock()
    engine.weather_comp_layer.evaluate_layer.return_value = LayerDecision("Weather Comp", 0.0, 0.0, "OK")
    
    # Mock predictor (Phase 6)
    engine.predictor = MagicMock()
    engine.predictor.evaluate_layer.return_value = LayerDecision("Prediction", 0.0, 0.0, "OK")

    # Mock _get_thermal_trend
    engine._get_thermal_trend = MagicMock(return_value={})
    
    # Mock _safety_layer
    engine._safety_layer = MagicMock(return_value=LayerDecision("Safety", 0.0, 0.0, "OK"))

    return engine


@pytest.fixture
def nibe_state():
    """Mock NIBE state."""
    state = MagicMock()
    state.outdoor_temp = 0.0
    state.indoor_temp = 22.0
    state.degree_minutes = -100.0
    state.power_kw = 5.0
    return state


class TestPeakProtectionSwitch:
    """Test enable_peak_protection switch."""

    def test_peak_protection_off_returns_zero(self, nibe_state):
        """Should return 0 offset/weight when disabled."""
        engine = create_engine({"enable_peak_protection": False})
        
        decision = engine.calculate_decision(
            nibe_state=nibe_state,
            price_data=None,
            weather_data=None,
            current_peak=40.0,
            current_power=45.0
        )
        
        layer = next(l for l in decision.layers if l.name == "Peak")
        assert layer.offset == 0.0
        assert layer.weight == 0.0
        assert "Disabled" in layer.reason
        
        # Verify flag was passed to layer
        engine.effect.evaluate_layer.assert_called_with(
            current_peak=40.0,
            current_power=45.0,
            thermal_trend=engine._get_thermal_trend.return_value,
            enable_peak_protection=False
        )

    def test_peak_protection_on_can_respond(self, nibe_state):
        """Should return active decision when enabled."""
        engine = create_engine({"enable_peak_protection": True})
        
        decision = engine.calculate_decision(
            nibe_state=nibe_state,
            price_data=None,
            weather_data=None,
            current_peak=40.0,
            current_power=45.0
        )
        
        layer = next(l for l in decision.layers if l.name == "Peak")
        assert "Safe margin" in layer.reason
        
        engine.effect.evaluate_layer.assert_called_with(
            current_peak=40.0,
            current_power=45.0,
            thermal_trend=engine._get_thermal_trend.return_value,
            enable_peak_protection=True
        )


class TestPriceOptimizationSwitch:
    """Test enable_price_optimization switch."""

    def test_price_optimization_off_returns_zero(self, nibe_state):
        """Should return 0 offset/weight when disabled."""
        engine = create_engine({"enable_price_optimization": False})
        
        decision = engine.calculate_decision(
            nibe_state=nibe_state,
            price_data=None,
            weather_data=None,
            current_peak=40.0,
            current_power=5.0
        )
        
        layer = next(l for l in decision.layers if l.name == "Spot Price")
        assert layer.offset == 0.0
        assert layer.weight == 0.0
        assert "Disabled" in layer.reason

    def test_price_optimization_on_can_respond(self, nibe_state):
        """Should return active decision when enabled."""
        engine = create_engine({"enable_price_optimization": True})
        
        decision = engine.calculate_decision(
            nibe_state=nibe_state,
            price_data=None,
            weather_data=None,
            current_peak=40.0,
            current_power=5.0
        )
        
        layer = next(l for l in decision.layers if l.name == "Spot Price")
        assert layer.offset == -2.0
        assert layer.weight == 0.8


class TestWeatherPredictionSwitch:
    """Test enable_weather_prediction switch."""

    def test_weather_prediction_off_returns_zero(self, nibe_state):
        """Should return 0 offset/weight when disabled."""
        engine = create_engine({"enable_weather_prediction": False})
        
        decision = engine.calculate_decision(
            nibe_state=nibe_state,
            price_data=None,
            weather_data=None,
            current_peak=40.0,
            current_power=5.0
        )
        
        layer = next(l for l in decision.layers if l.name == "Weather Pre-heat")
        assert layer.offset == 0.0
        assert layer.weight == 0.0
        assert "Disabled" in layer.reason

    def test_weather_prediction_on_can_respond(self, nibe_state):
        """Should return active decision when enabled."""
        engine = create_engine({"enable_weather_prediction": True})
        
        decision = engine.calculate_decision(
            nibe_state=nibe_state,
            price_data=None,
            weather_data=None,
            current_peak=40.0,
            current_power=5.0
        )
        
        layer = next(l for l in decision.layers if l.name == "Weather Pre-heat")
        assert layer.offset == 1.0
        assert layer.weight == 0.5


class TestAllSwitchesOff:
    """Test behavior when all optional features are disabled."""

    def test_all_switches_off_all_layers_return_zero(self, nibe_state):
        """All optional layers should be disabled."""
        engine = create_engine({
            "enable_peak_protection": False,
            "enable_price_optimization": False,
            "enable_weather_prediction": False,
        })
        
        decision = engine.calculate_decision(
            nibe_state=nibe_state,
            price_data=None,
            weather_data=None,
            current_peak=40.0,
            current_power=5.0
        )
        
        peak = next(l for l in decision.layers if l.name == "Peak")
        price = next(l for l in decision.layers if l.name == "Spot Price")
        weather = next(l for l in decision.layers if l.name == "Weather Pre-heat")
        
        assert peak.offset == 0.0 and "Disabled" in peak.reason
        assert price.offset == 0.0 and "Disabled" in price.reason
        assert weather.offset == 0.0 and "Disabled" in weather.reason


class TestSwitchDefaults:
    """Test default values for switches."""

    def test_switches_default_to_true(self, nibe_state):
        """Switches should default to True if missing from config."""
        # Create engine with minimal config
        base_config = {
            "target_indoor_temp": 21.0,
            "tolerance": 1.0,
        }
        
        price_analyzer = MagicMock()
        effect_manager = MagicMock()
        thermal_model = MagicMock()
        
        engine = DecisionEngine(
            price_analyzer=price_analyzer,
            effect_manager=effect_manager,
            thermal_model=thermal_model,
            config=base_config,
        )
        
        # Mock layers with proper return values (LayerDecision objects)
        engine.emergency_layer = MagicMock()
        engine.emergency_layer.evaluate_layer.return_value = LayerDecision("Emergency", 0.0, 0.0, "OK")
        
        engine.proactive_layer = MagicMock()
        engine.proactive_layer.evaluate_layer.return_value = LayerDecision("Proactive", 0.0, 0.0, "OK")
        
        engine.comfort_layer = MagicMock()
        engine.comfort_layer.evaluate_layer.return_value = LayerDecision("Comfort", 0.0, 0.0, "OK")
        
        engine.weather_comp_layer = MagicMock()
        engine.weather_comp_layer.evaluate_layer.return_value = LayerDecision("Weather Comp", 0.0, 0.0, "OK")
        
        engine.weather_prediction = MagicMock()
        engine.weather_prediction.evaluate_layer.return_value = LayerDecision("Weather Pre-heat", 0.0, 0.0, "OK")
        
        # Mock EffectManager (engine.effect)
        engine.effect.evaluate_layer.return_value = LayerDecision("Peak", 0.0, 0.0, "OK")
        
        # Mock PriceAnalyzer (engine.price)
        engine.price.evaluate_layer.return_value = LayerDecision("Spot Price", 0.0, 0.0, "OK")
        
        # Mock predictor (Phase 6)
        engine.predictor = MagicMock()
        engine.predictor.evaluate_layer.return_value = LayerDecision("Prediction", 0.0, 0.0, "OK")

        engine._get_thermal_trend = MagicMock(return_value={})
        engine._safety_layer = MagicMock(return_value=LayerDecision("Safety", 0.0, 0.0, "OK"))
        
        # Run decision
        engine.calculate_decision(
            nibe_state=nibe_state,
            price_data=None,
            weather_data=None,
            current_peak=40.0,
            current_power=5.0
        )
        
        # Verify defaults passed to layers
        # Effect
        engine.effect.evaluate_layer.assert_called()
        args = engine.effect.evaluate_layer.call_args[1]
        assert args["enable_peak_protection"] is True
        
        # Price
        engine.price.evaluate_layer.assert_called()
        args = engine.price.evaluate_layer.call_args[1]
        assert args["enable_price_optimization"] is True
        
        # Weather
        engine.weather_prediction.evaluate_layer.assert_called()
        args = engine.weather_prediction.evaluate_layer.call_args[1]
        assert args["enable_weather_prediction"] is True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

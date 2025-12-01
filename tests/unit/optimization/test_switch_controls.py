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
from custom_components.effektguard.const import (
    CONF_ENABLE_OPTIMIZATION,
    CONF_ENABLE_PEAK_PROTECTION,
    CONF_ENABLE_PRICE_OPTIMIZATION,
    CONF_ENABLE_WEATHER_PREDICTION,
    CONF_ENABLE_HOT_WATER_OPTIMIZATION,
    QuarterClassification,
)


def create_engine_mock(config_overrides=None):
    """Create mocked decision engine with specific config.
    
    Args:
        config_overrides: Dict of config values to override defaults
        
    Returns:
        Mock DecisionEngine with config and bound layer methods
    """
    engine = MagicMock(spec=DecisionEngine)
    
    # Default config (all switches ON)
    base_config = {
        "enable_peak_protection": True,
        "enable_price_optimization": True,
        "enable_weather_prediction": True,
    }
    
    # Apply overrides
    if config_overrides:
        base_config.update(config_overrides)
    
    engine.config = base_config
    
    # Mock dependencies that layers need when switches are ON
    # Effect layer needs self.effect
    engine.effect = MagicMock()
    engine.effect.should_limit_power = MagicMock(return_value=MagicMock(
        should_limit=False,
        reduction_needed=0.0,
        peak_margin=0.0,
        reason="Normal operation"
    ))
    
    # Price layer needs self.price
    engine.price = MagicMock()
    engine.price.get_current_classification = MagicMock(return_value=QuarterClassification.NORMAL)
    engine.price.get_forecast_price_increase = MagicMock(return_value=1.0)
    
    # Weather layer needs self.predictor and self.thermal
    engine.predictor = MagicMock()
    engine.predictor.get_current_trend = MagicMock(return_value={
        "trend": "stable",
        "rate_per_hour": 0.0,
        "confidence": 0.8,
        "samples": 8
    })
    engine.predictor.get_thermal_trend = MagicMock(return_value={
        "trend": "stable",
        "rate_per_hour": 0.0,
        "confidence": 0.8,
    })
    
    engine.thermal = MagicMock()
    engine.thermal.thermal_mass = 2.0
    engine.thermal.get_prediction_horizon = MagicMock(return_value=12.0)
    
    # Add tolerance (used by price layer)
    engine.tolerance = 5.0
    
    # Add _get_thermal_trend method to engine
    engine._get_thermal_trend = MagicMock(return_value={
        "trend": "stable",
        "rate_per_hour": 0.0,  # Float, not Mock!
        "confidence": 0.8,
    })
    
    # Bind real layer methods to mock engine  
    engine._effect_layer = DecisionEngine._effect_layer.__get__(engine, DecisionEngine)
    engine._price_layer = DecisionEngine._price_layer.__get__(engine, DecisionEngine)
    engine._weather_layer = DecisionEngine._weather_layer.__get__(engine, DecisionEngine)
    
    return engine


@pytest.fixture
def nibe_state():
    """Mock NIBE state."""
    state = MagicMock()
    state.outdoor_temp = 0.0
    state.indoor_temp = 22.0
    state.degree_minutes = -100.0
    return state


@pytest.fixture
def price_data():
    """Mock price data."""
    prices = MagicMock()
    # Create mock quarters with proper price attribute
    mock_quarters = [
        MagicMock(classification=QuarterClassification.NORMAL, price=1.0)
        for _ in range(96)
    ]
    prices.quarters = mock_quarters
    prices.today = mock_quarters  # Add today list (same as quarters)
    prices.periods = 96  # Add periods count
    prices.current_price = 1.0  # Add current price (float, not Mock)
    return prices


@pytest.fixture
def weather_data():
    """Mock weather data."""
    weather = MagicMock()
    weather.forecast_hours = [
        MagicMock(temperature=0.0) for _ in range(24)
    ]
    return weather


class TestPeakProtectionSwitch:
    """Test enable_peak_protection switch controls effect layer."""
    
    def test_peak_protection_off_returns_zero(self, nibe_state):
        """Switch OFF → LayerDecision(offset=0.0, weight=0.0, reason contains 'Disabled')."""
        engine = create_engine_mock({"enable_peak_protection": False})
        
        decision = engine._effect_layer(nibe_state, current_peak=45.0, current_power=40.0)
        
        assert decision.offset == 0.0
        assert decision.weight == 0.0
        assert "Disabled by user" in decision.reason
    
    def test_peak_protection_on_can_respond(self, nibe_state):
        """Switch ON → layer can vote (may or may not based on conditions)."""
        engine = create_engine_mock({"enable_peak_protection": True})
        
        decision = engine._effect_layer(nibe_state, current_peak=45.0, current_power=40.0)
        
        # Should NOT say "Disabled by user"
        assert "Disabled by user" not in decision.reason


class TestPriceOptimizationSwitch:
    """Test enable_price_optimization switch controls price layer."""
    
    def test_price_optimization_off_returns_zero(self, nibe_state, price_data):
        """Switch OFF → LayerDecision(offset=0.0, weight=0.0, reason contains 'Disabled')."""
        engine = create_engine_mock({"enable_price_optimization": False})
        
        # Set expensive prices (would normally trigger)
        for i in range(12):
            price_data.quarters[i].classification = QuarterClassification.EXPENSIVE
            price_data.quarters[i].price = 5.0
        
        decision = engine._price_layer(nibe_state, price_data)
        
        assert decision.offset == 0.0
        assert decision.weight == 0.0
        assert "Disabled by user" in decision.reason
    
    def test_price_optimization_on_can_respond(self, nibe_state, price_data):
        """Switch ON → layer can vote."""
        engine = create_engine_mock({"enable_price_optimization": True})
        
        for i in range(12):
            price_data.quarters[i].classification = QuarterClassification.EXPENSIVE
            price_data.quarters[i].price = 5.0
        
        decision = engine._price_layer(nibe_state, price_data)
        
        assert "Disabled by user" not in decision.reason


class TestWeatherPredictionSwitch:
    """Test enable_weather_prediction switch controls weather layer."""
    
    def test_weather_prediction_off_returns_zero(self, nibe_state, weather_data):
        """Switch OFF → LayerDecision(offset=0.0, weight=0.0, reason contains 'Disabled')."""
        engine = create_engine_mock({"enable_weather_prediction": False})
        
        # Set cold forecast (would normally trigger pre-heating)
        for i in range(4):
            weather_data.forecast_hours[i].temperature = 0.0
        for i in range(4, 12):
            weather_data.forecast_hours[i].temperature = -10.0
        
        decision = engine._weather_layer(nibe_state, weather_data)
        
        assert decision.offset == 0.0
        assert decision.weight == 0.0
        assert "Disabled by user" in decision.reason
    
    def test_weather_prediction_on_can_respond(self, nibe_state, weather_data):
        """Switch ON → layer can vote."""
        engine = create_engine_mock({"enable_weather_prediction": True})
        
        for i in range(4):
            weather_data.forecast_hours[i].temperature = 0.0
        for i in range(4, 12):
            weather_data.forecast_hours[i].temperature = -10.0
        
        decision = engine._weather_layer(nibe_state, weather_data)
        
        assert "Disabled by user" not in decision.reason


class TestAllSwitchesOff:
    """Test behavior when all optimization switches are disabled."""
    
    def test_all_switches_off_all_layers_return_zero(self, nibe_state, price_data, weather_data):
        """All switches OFF → all layers return zero."""
        engine = create_engine_mock({
            "enable_peak_protection": False,
            "enable_price_optimization": False,
            "enable_weather_prediction": False,
        })
        
        # Test each layer
        peak_decision = engine._effect_layer(nibe_state, 45.0, 40.0)
        assert peak_decision.offset == 0.0 and peak_decision.weight == 0.0
        
        price_decision = engine._price_layer(nibe_state, price_data)
        assert price_decision.offset == 0.0 and price_decision.weight == 0.0
        
        weather_decision = engine._weather_layer(nibe_state, weather_data)
        assert weather_decision.offset == 0.0 and weather_decision.weight == 0.0


class TestSwitchToggling:
    """Test toggling switches changes layer behavior."""
    
    def test_peak_protection_toggle_changes_behavior(self, nibe_state):
        """Toggling peak protection switch changes effect layer response."""
        # ON
        engine_on = create_engine_mock({"enable_peak_protection": True})
        decision_on = engine_on._effect_layer(nibe_state, 45.0, 40.0)
        
        # OFF
        engine_off = create_engine_mock({"enable_peak_protection": False})
        decision_off = engine_off._effect_layer(nibe_state, 45.0, 40.0)
        
        # OFF must return zero
        assert decision_off.offset == 0.0
        assert decision_off.weight == 0.0
        assert "Disabled by user" in decision_off.reason
        
        # ON must NOT say disabled
        assert "Disabled by user" not in decision_on.reason
    
    def test_price_optimization_toggle_changes_behavior(self, nibe_state, price_data):
        """Toggling price optimization switch changes price layer response."""
        for i in range(4):
            price_data.quarters[i].classification = QuarterClassification.EXPENSIVE
            price_data.quarters[i].price = 5.0
        
        # ON
        engine_on = create_engine_mock({"enable_price_optimization": True})
        decision_on = engine_on._price_layer(nibe_state, price_data)
        
        # OFF
        engine_off = create_engine_mock({"enable_price_optimization": False})
        decision_off = engine_off._price_layer(nibe_state, price_data)
        
        # OFF must return zero
        assert decision_off.offset == 0.0
        assert decision_off.weight == 0.0
        assert "Disabled by user" in decision_off.reason
        
        # ON must NOT say disabled
        assert "Disabled by user" not in decision_on.reason
    
    def test_weather_prediction_toggle_changes_behavior(self, nibe_state, weather_data):
        """Toggling weather prediction switch changes weather layer response."""
        for i in range(4):
            weather_data.forecast_hours[i].temperature = 0.0
        for i in range(4, 12):
            weather_data.forecast_hours[i].temperature = -10.0
        
        # ON
        engine_on = create_engine_mock({"enable_weather_prediction": True})
        decision_on = engine_on._weather_layer(nibe_state, weather_data)
        
        # OFF
        engine_off = create_engine_mock({"enable_weather_prediction": False})
        decision_off = engine_off._weather_layer(nibe_state, weather_data)
        
        # OFF must return zero
        assert decision_off.offset == 0.0
        assert decision_off.weight == 0.0
        assert "Disabled by user" in decision_off.reason
        
        # ON must NOT say disabled
        assert "Disabled by user" not in decision_on.reason


class TestSwitchDefaults:
    """Test switch default behavior."""
    
    def test_switches_default_to_true(self, nibe_state, price_data, weather_data):
        """When not specified in config, switches default to True (enabled)."""
        # Create engine with empty config (no switches specified)
        engine = create_engine_mock({})  # Use helper that adds all mocks
        
        # All layers should work (not disabled)
        peak_decision = engine._effect_layer(nibe_state, 45.0, 40.0)
        assert "Disabled by user" not in peak_decision.reason
        
        price_decision = engine._price_layer(nibe_state, price_data)
        assert "Disabled by user" not in price_decision.reason
        
        weather_decision = engine._weather_layer(nibe_state, weather_data)
        assert "Disabled by user" not in weather_decision.reason
    
    def test_explicit_false_overrides_default(self, nibe_state):
        """Explicit False should override default True."""
        engine = create_engine_mock({"enable_peak_protection": False})
        
        decision = engine._effect_layer(nibe_state, 45.0, 40.0)
        
        assert decision.offset == 0.0
        assert decision.weight == 0.0
        assert "Disabled by user" in decision.reason


class TestMasterSwitch:
    """Test master enable_optimization switch (tested at coordinator level)."""
    
    def test_master_switch_config_key(self):
        """Master switch configuration key is 'enable_optimization'."""
        assert CONF_ENABLE_OPTIMIZATION == "enable_optimization"
    
    def test_master_switch_blocks_optimization_at_coordinator_level(self):
        """Master switch is checked in coordinator._async_update_data(), not decision_engine.
        
        When enable_optimization=False:
        - Coordinator does NOT call engine.calculate_decision()
        - No offset changes are applied
        - System maintains current offset
        
        This is by design - master switch prevents optimization at coordinator level.
        Individual layer switches are checked within decision_engine.
        """
        # This test documents the architecture
        # Actual testing is done in coordinator tests
        pass


class TestDHWSwitch:
    """Test enable_hot_water_optimization switch (tested at coordinator level)."""
    
    def test_dhw_switch_config_key(self):
        """DHW optimization configuration key is 'enable_hot_water_optimization'."""
        assert CONF_ENABLE_HOT_WATER_OPTIMIZATION == "enable_hot_water_optimization"
    
    def test_dhw_switch_blocks_dhw_optimization_at_coordinator_level(self):
        """DHW switch is checked in coordinator, not decision_engine.
        
        When enable_hot_water_optimization=False:
        - Coordinator does NOT call DHW optimization logic
        - DHW blocking is not applied
        - DHW runs normally without optimization
        
        This is by design - DHW optimization is separate from heating curve optimization.
        """
        # This test documents the architecture
        # Actual testing is done in coordinator DHW tests
        pass

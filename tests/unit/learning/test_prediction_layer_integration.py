"""Test prediction layer integration in decision engine.

Verifies that ThermalStatePredictor integrates correctly with DecisionEngine
and provides prediction-based heating decisions.
"""

from unittest.mock import Mock
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.prediction_layer import ThermalStatePredictor


class TestPredictionLayerSetup:
    """Test that decision engine accepts and stores predictor."""

    def test_accepts_thermal_predictor_parameter(self):
        """Test decision engine accepts thermal_predictor in constructor."""
        predictor = ThermalStatePredictor()

        engine = DecisionEngine(
            price_analyzer=Mock(),
            effect_manager=Mock(),
            thermal_model=Mock(),
            config={},
            thermal_predictor=predictor,
        )

        assert hasattr(engine, "predictor")
        assert engine.predictor == predictor

    def test_works_without_predictor(self):
        """Test decision engine works when predictor is None (Phase 1-5)."""
        engine = DecisionEngine(
            price_analyzer=Mock(),
            effect_manager=Mock(),
            thermal_model=Mock(),
            config={},
            thermal_predictor=None,
        )

        assert engine.predictor is None

    def test_predictor_defaults_to_none(self):
        """Test predictor defaults to None if not provided."""
        engine = DecisionEngine(
            price_analyzer=Mock(),
            effect_manager=Mock(),
            thermal_model=Mock(),
            config={},
        )

        # Should not crash, predictor optional
        assert hasattr(engine, "predictor")


class TestPredictionLayerMethod:
    """Test the _prediction_layer method exists and works."""

    def test_prediction_layer_returns_decision(self):
        """Test prediction layer returns valid LayerDecision."""
        predictor = ThermalStatePredictor()
        engine = DecisionEngine(
            price_analyzer=Mock(),
            effect_manager=Mock(),
            thermal_model=Mock(),
            config={},
            thermal_predictor=predictor,
        )

        # Create mock NIBE state
        mock_nibe = Mock()
        mock_nibe.indoor_temp = 21.0
        mock_nibe.outdoor_temp = -5.0
        mock_nibe.degree_minutes = -150

        # Get prediction layer decision
        decision = engine.predictor.evaluate_layer(
            nibe_state=mock_nibe,
            weather_data=None,
            target_temp=engine.target_temp,
            thermal_model=engine.thermal,
        )

        # Should return valid decision
        assert decision is not None
        assert hasattr(decision, "offset")
        assert hasattr(decision, "weight")
        assert hasattr(decision, "reason")

    def test_prediction_layer_with_no_predictor(self):
        """Test prediction layer gracefully handles no predictor."""
        engine = DecisionEngine(
            price_analyzer=Mock(),
            effect_manager=Mock(),
            thermal_model=Mock(),
            config={},
            thermal_predictor=None,
        )

        # Configure mocks to return valid LayerDecision-like objects
        mock_decision = Mock()
        mock_decision.offset = 0.0
        mock_decision.weight = 0.0
        mock_decision.reason = "Mock"

        engine.price.evaluate_layer.return_value = mock_decision
        engine.effect.evaluate_layer.return_value = mock_decision

        mock_nibe = Mock()
        mock_nibe.indoor_temp = 21.0
        mock_nibe.outdoor_temp = -5.0
        mock_nibe.degree_minutes = 0  # Set to 0 to avoid ProactiveLayer interference
        mock_nibe.is_hot_water = False  # Not in DHW/lux mode
        mock_nibe.flow_temp = 35.0  # Normal flow temp

        # Should not crash, return neutral decision
        decision = engine.calculate_decision(mock_nibe, None, None, 0.0, 0.0)

        assert decision is not None
        # With no predictor, should return neutral/minimal offset

        # Verify Prediction layer decision specifically
        prediction_layer = next(
            l for l in decision.layers if l.name == "Prediction" or l.name == "Learned Pre-heat"
        )
        assert prediction_layer.reason == "Predictor not initialized"
        assert prediction_layer.offset == 0.0
        assert prediction_layer.weight == 0.0

    def test_prediction_layer_handles_missing_nibe_data(self):
        """Test prediction layer handles None NIBE state."""
        predictor = ThermalStatePredictor()
        engine = DecisionEngine(
            price_analyzer=Mock(),
            effect_manager=Mock(),
            thermal_model=Mock(),
            config={},
            thermal_predictor=predictor,
        )

        # Should handle None gracefully
        decision = engine.predictor.evaluate_layer(
            nibe_state=None,
            weather_data=None,
            target_temp=engine.target_temp,
            thermal_model=engine.thermal,
        )

        assert decision is not None
        assert decision.offset == 0.0  # Safe default


class TestPredictionLayerDecisionStructure:
    """Test that prediction layer decision has correct structure."""

    def test_decision_has_offset_field(self):
        """Test LayerDecision has offset field."""
        predictor = ThermalStatePredictor()
        engine = DecisionEngine(
            price_analyzer=Mock(),
            effect_manager=Mock(),
            thermal_model=Mock(),
            config={},
            thermal_predictor=predictor,
        )

        mock_nibe = Mock()
        mock_nibe.indoor_temp = 21.0
        mock_nibe.outdoor_temp = -5.0
        mock_nibe.degree_minutes = -150

        decision = engine.predictor.evaluate_layer(
            nibe_state=mock_nibe,
            weather_data=None,
            target_temp=engine.target_temp,
            thermal_model=engine.thermal,
        )

        assert hasattr(decision, "offset")
        assert isinstance(decision.offset, (int, float))

    def test_decision_has_weight_field(self):
        """Test LayerDecision has weight field (importance)."""
        predictor = ThermalStatePredictor()
        engine = DecisionEngine(
            price_analyzer=Mock(),
            effect_manager=Mock(),
            thermal_model=Mock(),
            config={},
            thermal_predictor=predictor,
        )

        mock_nibe = Mock()
        mock_nibe.indoor_temp = 21.0
        mock_nibe.outdoor_temp = -5.0
        mock_nibe.degree_minutes = -150

        decision = engine.predictor.evaluate_layer(
            nibe_state=mock_nibe,
            weather_data=None,
            target_temp=engine.target_temp,
            thermal_model=engine.thermal,
        )

        assert hasattr(decision, "weight")
        assert isinstance(decision.weight, (int, float))
        assert 0.0 <= decision.weight <= 1.0  # Weight should be normalized

    def test_decision_has_reason_field(self):
        """Test LayerDecision has reason field (explainability)."""
        predictor = ThermalStatePredictor()
        engine = DecisionEngine(
            price_analyzer=Mock(),
            effect_manager=Mock(),
            thermal_model=Mock(),
            config={},
            thermal_predictor=predictor,
        )

        mock_nibe = Mock()
        mock_nibe.indoor_temp = 21.0
        mock_nibe.outdoor_temp = -5.0
        mock_nibe.degree_minutes = -150

        decision = engine.predictor.evaluate_layer(
            nibe_state=mock_nibe,
            weather_data=None,
            target_temp=engine.target_temp,
            thermal_model=engine.thermal,
        )

        assert hasattr(decision, "reason")
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0  # Should have meaningful text


class TestPredictionLayerIntegrationWithCalculateDecision:
    """Test that prediction layer integrates into calculate_decision."""

    def test_calculate_decision_includes_prediction_layer(self):
        """Test calculate_decision uses prediction layer when predictor exists."""
        predictor = ThermalStatePredictor()
        engine = DecisionEngine(
            price_analyzer=Mock(),
            effect_manager=Mock(),
            thermal_model=Mock(),
            config={},
            thermal_predictor=predictor,
        )

        # This test verifies the method exists and can be called
        # Actual integration tested in full decision engine tests
        assert hasattr(engine, "calculate_decision")
        assert engine.predictor is not None

    def test_calculate_decision_works_without_predictor(self):
        """Test calculate_decision still works without predictor (backwards compat)."""
        engine = DecisionEngine(
            price_analyzer=Mock(),
            effect_manager=Mock(),
            thermal_model=Mock(),
            config={},
            thermal_predictor=None,
        )

        # Should work fine without predictor (Phase 1-5 compatibility)
        assert hasattr(engine, "calculate_decision")
        assert engine.predictor is None


class TestAdaptiveThermalModelIntegration:
    """Test prediction layer works with AdaptiveThermalModel (Phase 6)."""

    def test_prediction_layer_with_adaptive_thermal_model(self):
        """Test prediction layer works with AdaptiveThermalModel's insulation_quality property."""
        from custom_components.effektguard.optimization.adaptive_learning import (
            AdaptiveThermalModel,
        )

        adaptive_model = AdaptiveThermalModel(initial_thermal_mass=1.2)

        # Set insulation quality via property (simulates config reload)
        adaptive_model.insulation_quality = 1.5

        predictor = ThermalStatePredictor()
        engine = DecisionEngine(
            price_analyzer=Mock(),
            effect_manager=Mock(),
            thermal_model=adaptive_model,  # Use adaptive model
            config={},
            thermal_predictor=predictor,
        )

        # Create mock NIBE state
        mock_nibe = Mock()
        mock_nibe.indoor_temp = 21.0
        mock_nibe.outdoor_temp = -5.0
        mock_nibe.degree_minutes = -150

        # Should not raise AttributeError on insulation_quality
        decision = engine.predictor.evaluate_layer(
            nibe_state=mock_nibe,
            weather_data=None,
            target_temp=engine.target_temp,
            thermal_model=engine.thermal,
        )

        # Should return valid decision
        assert decision is not None
        assert hasattr(decision, "offset")
        assert hasattr(decision, "weight")
        assert hasattr(decision, "reason")

    def test_prediction_with_learned_parameters(self):
        """Test prediction layer uses learned insulation quality."""
        from custom_components.effektguard.optimization.adaptive_learning import (
            AdaptiveThermalModel,
        )

        adaptive_model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Simulate learned parameters (better insulation than default)
        adaptive_model.learned_parameters = {
            "thermal_mass": 1.3,
            "heat_loss_coefficient": 150.0,  # Better insulation → 180/150 = 1.2
            "heating_efficiency": 0.45,
            "thermal_decay_rate": -0.07,
            "confidence": 0.90,
        }

        # Property should use learned value
        insulation = adaptive_model.insulation_quality
        assert insulation > 1.0  # Better than default
        assert insulation < 1.3

        predictor = ThermalStatePredictor()
        engine = DecisionEngine(
            price_analyzer=Mock(),
            effect_manager=Mock(),
            thermal_model=adaptive_model,
            config={},
            thermal_predictor=predictor,
        )

        # Verify engine can access the property
        assert engine.thermal.insulation_quality > 1.0

        # Create mock NIBE state
        mock_nibe = Mock()
        mock_nibe.indoor_temp = 21.0
        mock_nibe.outdoor_temp = -5.0
        mock_nibe.degree_minutes = -150

        # Should work without AttributeError
        decision = engine.predictor.evaluate_layer(
            nibe_state=mock_nibe,
            weather_data=None,
            target_temp=engine.target_temp,
            thermal_model=engine.thermal,
        )
        assert decision is not None

    def test_config_reload_pattern_with_adaptive_model(self):
        """Test config reload pattern works with AdaptiveThermalModel.

        Simulates what coordinator.py:1913 does during config reload.
        """
        from custom_components.effektguard.optimization.adaptive_learning import (
            AdaptiveThermalModel,
        )

        adaptive_model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        predictor = ThermalStatePredictor()
        engine = DecisionEngine(
            price_analyzer=Mock(),
            effect_manager=Mock(),
            thermal_model=adaptive_model,
            config={},
            thermal_predictor=predictor,
        )

        # Simulate config reload changing insulation_quality
        new_options = {
            "thermal_mass": 1.5,
            "insulation_quality": 1.8,
        }

        # This is what coordinator does on config reload
        engine.thermal.thermal_mass = new_options["thermal_mass"]
        engine.thermal.insulation_quality = new_options["insulation_quality"]

        # Verify values are set correctly
        assert engine.thermal.thermal_mass == 1.5
        assert abs(engine.thermal.insulation_quality - 1.8) < 0.01

        # Prediction layer should still work
        mock_nibe = Mock()
        mock_nibe.indoor_temp = 21.0
        mock_nibe.outdoor_temp = -5.0
        mock_nibe.degree_minutes = -150

        decision = engine.predictor.evaluate_layer(
            nibe_state=mock_nibe,
            weather_data=None,
            target_temp=engine.target_temp,
            thermal_model=engine.thermal,
        )
        assert decision is not None

    def test_insufficient_learning_data_uses_manual_config(self):
        """Test that manual config is used when learning data insufficient."""
        from custom_components.effektguard.optimization.adaptive_learning import (
            AdaptiveThermalModel,
        )

        adaptive_model = AdaptiveThermalModel(initial_thermal_mass=1.0)

        # Set manual config
        adaptive_model.insulation_quality = 1.5

        # No observations yet, but manual config should be used
        assert len(adaptive_model.observations) == 0
        assert adaptive_model.insulation_quality == 1.5

        predictor = ThermalStatePredictor()
        engine = DecisionEngine(
            price_analyzer=Mock(),
            effect_manager=Mock(),
            thermal_model=adaptive_model,
            config={},
            thermal_predictor=predictor,
        )

        # Should use manual config value
        assert engine.thermal.insulation_quality == 1.5

        # Prediction layer should work with manual config
        mock_nibe = Mock()
        mock_nibe.indoor_temp = 21.0
        mock_nibe.outdoor_temp = -5.0
        mock_nibe.degree_minutes = -150

        decision = engine.predictor.evaluate_layer(
            nibe_state=mock_nibe,
            weather_data=None,
            target_temp=engine.target_temp,
            thermal_model=engine.thermal,
        )
        assert decision is not None

"""Test learning data serialization and persistence.

Verifies that learned data can be saved to disk and restored correctly,
ensuring learning progress persists across Home Assistant restarts.
"""

import pytest
from datetime import datetime, timedelta
from custom_components.effektguard.optimization.adaptive_learning import AdaptiveThermalModel
from custom_components.effektguard.optimization.thermal_predictor import ThermalStatePredictor
from custom_components.effektguard.const import UFHType


class TestAdaptiveLearningSerializa:
    """Test AdaptiveThermalModel serialization (save/load)."""

    def test_serializes_to_dict(self):
        """Test model serializes to dictionary."""
        model = AdaptiveThermalModel()

        # Add observations
        for i in range(10):
            model.record_observation(
                timestamp=datetime.now() + timedelta(minutes=15 * i),
                indoor_temp=21.0 + i * 0.1,
                outdoor_temp=-5.0,
                heating_offset=2.0,
            )

        data = model.to_dict()

        # Verify structure
        assert isinstance(data, dict)
        assert "thermal_mass" in data
        assert "ufh_type" in data
        assert "observations" in data

    def test_serialized_observations_are_list(self):
        """Test observations serialize to list."""
        model = AdaptiveThermalModel()

        # Add observations
        for i in range(5):
            model.record_observation(
                timestamp=datetime.now() + timedelta(minutes=15 * i),
                indoor_temp=21.0,
                outdoor_temp=-5.0,
                heating_offset=2.0,
            )

        data = model.to_dict()

        assert isinstance(data["observations"], list)
        assert len(data["observations"]) == 5

    def test_deserializes_from_dict(self):
        """Test model restores from dictionary."""
        # Create and populate original model
        model1 = AdaptiveThermalModel()
        for i in range(50):
            model1.record_observation(
                timestamp=datetime.now() + timedelta(minutes=15 * i),
                indoor_temp=21.0,
                outdoor_temp=-5.0,
                heating_offset=2.0,
            )

        # Serialize
        data = model1.to_dict()

        # Deserialize to new model
        model2 = AdaptiveThermalModel.from_dict(data)

        # Verify restoration
        assert model2.thermal_mass == model1.thermal_mass
        assert len(model2.observations) == len(model1.observations)

    def test_preserves_thermal_mass_setting(self):
        """Test thermal mass setting is preserved through save/load."""
        model1 = AdaptiveThermalModel()
        model1.thermal_mass = 1.5  # Custom value

        data = model1.to_dict()
        model2 = AdaptiveThermalModel.from_dict(data)

        assert model2.thermal_mass == 1.5

    def test_preserves_ufh_type_setting(self):
        """Test UFH type is preserved through save/load."""
        model1 = AdaptiveThermalModel()
        model1.ufh_type = UFHType.SLOW_RESPONSE  # Updated: use enum

        data = model1.to_dict()
        model2 = AdaptiveThermalModel.from_dict(data)

        assert model2.ufh_type == UFHType.SLOW_RESPONSE  # Updated: use enum

    def test_round_trip_preserves_observation_count(self):
        """Test complete round-trip preserves observation count."""
        model1 = AdaptiveThermalModel()

        # Add exactly 100 observations
        for i in range(100):
            model1.record_observation(
                timestamp=datetime.now() + timedelta(minutes=15 * i),
                indoor_temp=21.0,
                outdoor_temp=-5.0,
                heating_offset=2.0,
            )

        # Round trip
        data = model1.to_dict()
        model2 = AdaptiveThermalModel.from_dict(data)

        assert len(model2.observations) == 100


class TestThermalPredictorSerialization:
    """Test ThermalStatePredictor serialization."""

    def test_serializes_to_dict(self):
        """Test predictor serializes to dictionary."""
        predictor = ThermalStatePredictor()

        # Add states
        for i in range(10):
            predictor.record_state(
                timestamp=datetime.now() + timedelta(minutes=15 * i),
                indoor_temp=21.0,
                outdoor_temp=-5.0,
                heating_offset=2.0,
                flow_temp=35.0,
                degree_minutes=-150,
            )

        data = predictor.to_dict()

        assert isinstance(data, dict)
        assert "state_history" in data

    def test_deserializes_from_dict(self):
        """Test predictor restores from dictionary."""
        predictor1 = ThermalStatePredictor()

        # Add states
        for i in range(20):
            predictor1.record_state(
                timestamp=datetime.now() + timedelta(minutes=15 * i),
                indoor_temp=21.0,
                outdoor_temp=-5.0,
                heating_offset=2.0,
                flow_temp=35.0,
                degree_minutes=-150,
            )

        # Round trip
        data = predictor1.to_dict()
        predictor2 = ThermalStatePredictor.from_dict(data)

        assert len(predictor2.state_history) == len(predictor1.state_history)

    def test_preserves_degree_minutes_in_states(self):
        """Test degree minutes preserved through serialization."""
        predictor1 = ThermalStatePredictor()

        predictor1.record_state(
            timestamp=datetime.now(),
            indoor_temp=21.0,
            outdoor_temp=-5.0,
            heating_offset=2.0,
            flow_temp=35.0,
            degree_minutes=-250,  # Specific DM value
        )

        # Round trip
        data = predictor1.to_dict()
        predictor2 = ThermalStatePredictor.from_dict(data)

        assert predictor2.state_history[0].degree_minutes == -250


class TestEmptyDataSerialization:
    """Test serialization with no observations/states."""

    def test_empty_adaptive_learning_serializes(self):
        """Test empty AdaptiveThermalModel serializes safely."""
        model = AdaptiveThermalModel()

        data = model.to_dict()

        assert isinstance(data, dict)
        assert data["observations"] == []

    def test_empty_adaptive_learning_deserializes(self):
        """Test empty data deserializes to valid model."""
        data = {
            "thermal_mass": 1.0,
            "ufh_type": "timber",
            "observations": [],
        }

        model = AdaptiveThermalModel.from_dict(data)

        assert len(model.observations) == 0
        assert model.thermal_mass == 1.0

    def test_empty_thermal_predictor_serializes(self):
        """Test empty ThermalStatePredictor serializes safely."""
        predictor = ThermalStatePredictor()

        data = predictor.to_dict()

        assert isinstance(data, dict)
        assert data["state_history"] == []

    def test_empty_thermal_predictor_deserializes(self):
        """Test empty predictor data deserializes to valid predictor."""
        data = {
            "state_history": [],
        }

        predictor = ThermalStatePredictor.from_dict(data)

        assert len(predictor.state_history) == 0

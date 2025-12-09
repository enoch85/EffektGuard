"""Test learning module observation recording.

Verifies that AdaptiveThermalModel and ThermalStatePredictor correctly
record observations during coordinator updates.
"""

from datetime import datetime, timedelta
from custom_components.effektguard.optimization.adaptive_learning import AdaptiveThermalModel
from custom_components.effektguard.optimization.prediction_layer import ThermalStatePredictor


class TestAdaptiveLearningObservations:
    """Test AdaptiveThermalModel observation recording."""

    def test_records_single_observation(self):
        """Test recording a single observation."""
        model = AdaptiveThermalModel()
        timestamp = datetime.now()

        model.record_observation(
            timestamp=timestamp,
            indoor_temp=21.5,
            outdoor_temp=-5.0,
            heating_offset=2.0,
        )

        assert len(model.observations) == 1
        assert model.observations[0].indoor_temp == 21.5
        assert model.observations[0].outdoor_temp == -5.0
        assert model.observations[0].heating_offset == 2.0

    def test_records_multiple_observations(self):
        """Test recording multiple observations over time."""
        model = AdaptiveThermalModel()
        base_time = datetime.now()

        # Record 10 observations at 15-minute intervals
        for i in range(10):
            model.record_observation(
                timestamp=base_time + timedelta(minutes=15 * i),
                indoor_temp=21.0 + i * 0.1,
                outdoor_temp=-5.0,
                heating_offset=2.0,
            )

        assert len(model.observations) == 10
        assert model.observations[0].indoor_temp == 21.0
        assert model.observations[9].indoor_temp == 21.9

    def test_observation_has_correct_attributes(self):
        """Test observation contains all required attributes."""
        model = AdaptiveThermalModel()
        timestamp = datetime.now()

        model.record_observation(
            timestamp=timestamp,
            indoor_temp=21.5,
            outdoor_temp=-5.0,
            heating_offset=2.0,
        )

        observation = model.observations[0]
        assert hasattr(observation, "timestamp")
        assert hasattr(observation, "indoor_temp")
        assert hasattr(observation, "outdoor_temp")
        assert hasattr(observation, "heating_offset")

    def test_observations_maintain_chronological_order(self):
        """Test observations are stored in chronological order."""
        model = AdaptiveThermalModel()
        base_time = datetime.now()

        # Record observations in order
        timestamps = []
        for i in range(5):
            timestamp = base_time + timedelta(minutes=15 * i)
            timestamps.append(timestamp)
            model.record_observation(
                timestamp=timestamp,
                indoor_temp=21.0,
                outdoor_temp=-5.0,
                heating_offset=2.0,
            )

        # Verify order is maintained
        for i in range(5):
            assert model.observations[i].timestamp == timestamps[i]


class TestThermalPredictorStateRecording:
    """Test ThermalStatePredictor state recording."""

    def test_records_single_state(self):
        """Test recording a single thermal state."""
        predictor = ThermalStatePredictor()
        timestamp = datetime.now()

        predictor.record_state(
            timestamp=timestamp,
            indoor_temp=21.5,
            outdoor_temp=-5.0,
            heating_offset=2.0,
            flow_temp=35.0,
            degree_minutes=-150,
        )

        assert len(predictor.state_history) == 1

    def test_records_multiple_states(self):
        """Test recording multiple thermal states."""
        predictor = ThermalStatePredictor()
        base_time = datetime.now()

        # Record 20 states
        for i in range(20):
            predictor.record_state(
                timestamp=base_time + timedelta(minutes=15 * i),
                indoor_temp=21.0,
                outdoor_temp=-5.0 - i * 0.5,
                heating_offset=2.0,
                flow_temp=35.0,
                degree_minutes=-150,
            )

        assert len(predictor.state_history) == 20

    def test_state_has_degree_minutes(self):
        """Test state includes degree minutes (critical NIBE metric)."""
        predictor = ThermalStatePredictor()
        timestamp = datetime.now()

        predictor.record_state(
            timestamp=timestamp,
            indoor_temp=21.5,
            outdoor_temp=-5.0,
            heating_offset=2.0,
            flow_temp=35.0,
            degree_minutes=-150,
        )

        state = predictor.state_history[0]
        assert hasattr(state, "degree_minutes")
        assert state.degree_minutes == -150

    def test_state_has_flow_temperature(self):
        """Test state includes flow temperature (BT25)."""
        predictor = ThermalStatePredictor()
        timestamp = datetime.now()

        predictor.record_state(
            timestamp=timestamp,
            indoor_temp=21.5,
            outdoor_temp=-5.0,
            heating_offset=2.0,
            flow_temp=35.0,
            degree_minutes=-150,
        )

        state = predictor.state_history[0]
        assert hasattr(state, "flow_temp")
        assert state.flow_temp == 35.0


class TestObservationLimits:
    """Test observation history limits (prevent memory growth)."""

    def test_adaptive_learning_limits_observations(self):
        """Test AdaptiveThermalModel limits observation history."""
        model = AdaptiveThermalModel()
        base_time = datetime.now()

        # Try to record more observations than max (2016 = 3 weeks)
        for i in range(3000):
            model.record_observation(
                timestamp=base_time + timedelta(minutes=15 * i),
                indoor_temp=21.0,
                outdoor_temp=-5.0,
                heating_offset=2.0,
            )

        # Should not exceed reasonable limit
        assert len(model.observations) <= 2016  # 3 weeks of 15-min intervals

    def test_thermal_predictor_limits_state_history(self):
        """Test ThermalStatePredictor limits state history."""
        predictor = ThermalStatePredictor()
        base_time = datetime.now()

        # Try to record more states than max
        for i in range(1000):
            predictor.record_state(
                timestamp=base_time + timedelta(minutes=15 * i),
                indoor_temp=21.0,
                outdoor_temp=-5.0,
                heating_offset=2.0,
                flow_temp=35.0,
                degree_minutes=-150,
            )

        # Should not exceed reasonable limit (typically 672 = 1 week)
        assert len(predictor.state_history) <= 672

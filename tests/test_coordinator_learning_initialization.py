"""Test learning module initialization in coordinator.

Verifies that AdaptiveThermalModel, ThermalStatePredictor, and WeatherPatternLearner
are properly created and initialized when the coordinator is set up.
"""

import pytest
import tempfile
from unittest.mock import AsyncMock, Mock
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.adaptive_learning import AdaptiveThermalModel
from custom_components.effektguard.optimization.thermal_predictor import ThermalStatePredictor
from custom_components.effektguard.optimization.weather_learning import WeatherPatternLearner


def create_mock_hass(latitude: float = 59.3):
    """Create a properly configured mock Home Assistant instance.

    Args:
        latitude: Latitude for climate region detection

    Returns:
        Mock hass with required attributes for Store initialization
    """
    mock_hass = Mock()
    mock_hass.data = {}  # Store requires hass.data to be a dict
    mock_hass.config.latitude = latitude
    mock_hass.config.config_dir = tempfile.mkdtemp()  # Store needs a valid path
    mock_hass.async_add_executor_job = AsyncMock()
    return mock_hass


class TestLearningModuleCreation:
    """Test that coordinator creates all required learning modules."""

    @pytest.mark.asyncio
    async def test_creates_adaptive_thermal_model(self):
        """Test coordinator creates AdaptiveThermalModel."""
        mock_hass = create_mock_hass(latitude=59.3)

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=Mock(),
        )

        assert isinstance(coordinator.adaptive_learning, AdaptiveThermalModel)

    @pytest.mark.asyncio
    async def test_creates_thermal_state_predictor(self):
        """Test coordinator creates ThermalStatePredictor."""
        mock_hass = create_mock_hass(latitude=59.3)

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=Mock(),
        )

        assert isinstance(coordinator.thermal_predictor, ThermalStatePredictor)

    @pytest.mark.asyncio
    async def test_creates_weather_pattern_learner(self):
        """Test coordinator creates WeatherPatternLearner."""
        mock_hass = create_mock_hass(latitude=59.3)

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=Mock(),
        )

        assert isinstance(coordinator.weather_learner, WeatherPatternLearner)

    @pytest.mark.asyncio
    async def test_creates_all_three_learning_modules(self):
        """Test coordinator creates all three learning modules together."""
        mock_hass = create_mock_hass(latitude=59.3)

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=Mock(),
        )

        # Verify all three modules exist and are correct types
        assert isinstance(coordinator.adaptive_learning, AdaptiveThermalModel)
        assert isinstance(coordinator.thermal_predictor, ThermalStatePredictor)
        assert isinstance(coordinator.weather_learner, WeatherPatternLearner)


class TestCoordinatorLearningMethods:
    """Test that coordinator has all required learning-related methods."""

    @pytest.mark.asyncio
    async def test_has_learning_initialization_method(self):
        """Test coordinator has async_initialize_learning method."""
        mock_hass = create_mock_hass(latitude=59.3)

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=Mock(),
        )

        assert hasattr(coordinator, "async_initialize_learning")
        assert callable(coordinator.async_initialize_learning)

    @pytest.mark.asyncio
    async def test_has_observation_recording_method(self):
        """Test coordinator has _record_learning_observations method."""
        mock_hass = create_mock_hass(latitude=59.3)

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=Mock(),
        )

        assert hasattr(coordinator, "_record_learning_observations")
        assert callable(coordinator._record_learning_observations)

    @pytest.mark.asyncio
    async def test_has_save_learned_data_method(self):
        """Test coordinator has _save_learned_data method."""
        mock_hass = create_mock_hass(latitude=59.3)

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=Mock(),
        )

        assert hasattr(coordinator, "_save_learned_data")
        assert callable(coordinator._save_learned_data)

    @pytest.mark.asyncio
    async def test_has_load_learned_data_method(self):
        """Test coordinator has _load_learned_data method."""
        mock_hass = create_mock_hass(latitude=59.3)

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=Mock(),
        )

        assert hasattr(coordinator, "_load_learned_data")
        assert callable(coordinator._load_learned_data)

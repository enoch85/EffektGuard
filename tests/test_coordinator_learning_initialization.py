"""Test learning module initialization in coordinator.

Verifies that the coordinator properly creates:
- AdaptiveThermalModel
- ThermalStatePredictor
- WeatherPatternLearner
"""

import pytest
from unittest.mock import Mock
from conftest import create_mock_hass, create_mock_entry
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.adaptive_learning import AdaptiveThermalModel
from custom_components.effektguard.optimization.thermal_predictor import (
    ThermalStatePredictor,
)
from custom_components.effektguard.optimization.weather_learning import (
    WeatherPatternLearner,
)

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
    mock_hass.async_add_executor_job = AsyncMock(side_effect=lambda func, *args: func(*args))
    mock_hass.loop = Mock()  # Add loop for DataUpdateCoordinator
    mock_hass.loop.call_soon_threadsafe = Mock()
    return mock_hass


def create_mock_entry():
    """Create a properly configured mock config entry.

    Returns:
        Mock entry with options that return actual values, not Mocks
    """
    mock_entry = Mock()
    mock_entry.data = Mock()
    # Configure options.get() to return actual values with defaults
    mock_entry.options.get.side_effect = lambda key, default=None: {
        "dhw_morning_enabled": True,
        "dhw_morning_hour": 7,
        "dhw_evening_enabled": True,
        "dhw_evening_hour": 18,
        "enable_dhw_optimization": False,
        "enable_airflow_optimization": False,
    }.get(key, default)
    # Configure data.get() to return defaults for values not in options
    mock_entry.data.get.side_effect = lambda key, default=None: default
    return mock_entry


class TestLearningModuleCreation:
    """Test that coordinator creates all required learning modules."""

    @pytest.mark.asyncio
    async def test_creates_adaptive_thermal_model(self):
        """Test coordinator creates AdaptiveThermalModel."""
        mock_hass = create_mock_hass(latitude=59.3)
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
        )

        assert isinstance(coordinator.adaptive_learning, AdaptiveThermalModel)

    @pytest.mark.asyncio
    async def test_creates_thermal_state_predictor(self):
        """Test coordinator creates ThermalStatePredictor."""
        mock_hass = create_mock_hass(latitude=59.3)
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
        )

        assert isinstance(coordinator.thermal_predictor, ThermalStatePredictor)

    @pytest.mark.asyncio
    async def test_creates_weather_pattern_learner(self):
        """Test coordinator creates WeatherPatternLearner."""
        mock_hass = create_mock_hass(latitude=59.3)
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
        )

        assert isinstance(coordinator.weather_learner, WeatherPatternLearner)

    @pytest.mark.asyncio
    async def test_creates_all_three_learning_modules(self):
        """Test coordinator creates all three learning modules together."""
        mock_hass = create_mock_hass(latitude=59.3)
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
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
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
        )

        assert hasattr(coordinator, "async_initialize_learning")
        assert callable(coordinator.async_initialize_learning)

    @pytest.mark.asyncio
    async def test_has_observation_recording_method(self):
        """Test coordinator has _record_learning_observations method."""
        mock_hass = create_mock_hass(latitude=59.3)
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
        )

        assert hasattr(coordinator, "_record_learning_observations")
        assert callable(coordinator._record_learning_observations)

    @pytest.mark.asyncio
    async def test_has_save_learned_data_method(self):
        """Test coordinator has _save_learned_data method."""
        mock_hass = create_mock_hass(latitude=59.3)
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
        )

        assert hasattr(coordinator, "_save_learned_data")
        assert callable(coordinator._save_learned_data)

    @pytest.mark.asyncio
    async def test_has_load_learned_data_method(self):
        """Test coordinator has _load_learned_data method."""
        mock_hass = create_mock_hass(latitude=59.3)
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
        )

        assert hasattr(coordinator, "_load_learned_data")
        assert callable(coordinator._load_learned_data)

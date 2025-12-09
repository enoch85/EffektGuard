"""Tests for thermal predictor persistence across reboots."""

import pytest
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from homeassistant.util import dt as dt_util


@pytest.fixture
def mock_config_entry():
    """Mock config entry for testing."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        "nibe_entity": "sensor.nibe_test",
        "gespot_entity": "sensor.gespot_test",
        "weather_entity": "weather.test",
        "enable_hot_water_optimization": False,
    }
    entry.options = {
        "target_indoor_temp": 21.0,
        "tolerance": 0.5,
        "thermal_mass": 1.0,
        "insulation_quality": 1.0,
        "dhw_morning_enabled": True,
        "dhw_morning_hour": 7,
        "dhw_evening_enabled": True,
        "dhw_evening_hour": 18,
        "dhw_target_temp": 50.0,
    }
    return entry


@pytest.fixture
def mock_storage():
    """Mock storage for testing."""
    storage_data = {}

    async def async_save(data):
        storage_data.clear()
        storage_data.update(data)

    async def async_load():
        return dict(storage_data) if storage_data else None

    store = MagicMock()
    store.async_save = AsyncMock(side_effect=async_save)
    store.async_load = AsyncMock(side_effect=async_load)

    return store, storage_data


class TestThermalPredictorPersistence:
    """Test thermal predictor state persistence."""

    @pytest.mark.asyncio
    async def test_thermal_predictor_saves_immediately_after_record(
        self, hass, mock_config_entry, mock_storage
    ):
        """Test that thermal predictor saves immediately after recording state."""
        store, storage_data = mock_storage

        # Create coordinator with mocked storage
        coordinator = EffektGuardCoordinator(
            hass,
            MagicMock(),  # nibe_adapter
            MagicMock(),  # gespot_adapter
            MagicMock(),  # weather_adapter
            MagicMock(),  # decision_engine
            MagicMock(),  # effect_manager
            mock_config_entry,
        )
        coordinator.learning_store = store

        # Record a thermal state
        nibe_data = NibeState(
            outdoor_temp=5.0,
            indoor_temp=21.0,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-100.0,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=dt_util.utcnow(),
            compressor_hz=50,
        )

        now = dt_util.utcnow()
        await coordinator._record_learning_observations(
            nibe_data=nibe_data,
            weather_data=None,
            current_offset=0.5,
        )

        # Verify that data was saved
        assert store.async_save.called
        assert "thermal_predictor" in storage_data
        assert storage_data["thermal_predictor"]["state_history"]
        assert len(storage_data["thermal_predictor"]["state_history"]) == 1

        # Verify the saved snapshot has correct data
        snapshot = storage_data["thermal_predictor"]["state_history"][0]
        assert snapshot["indoor_temp"] == 21.0
        assert snapshot["outdoor_temp"] == 5.0
        assert snapshot["flow_temp"] == 35.0
        assert snapshot["degree_minutes"] == -100.0
        assert snapshot["heating_offset"] == 0.5

    @pytest.mark.asyncio
    async def test_thermal_predictor_throttles_saves(self, hass, mock_config_entry, mock_storage):
        """Test that thermal predictor throttles saves to once per 5 minutes."""
        store, storage_data = mock_storage

        coordinator = EffektGuardCoordinator(
            hass,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            mock_config_entry,
        )
        coordinator.learning_store = store

        nibe_data = NibeState(
            outdoor_temp=5.0,
            indoor_temp=21.0,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-100.0,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=dt_util.utcnow(),
            compressor_hz=50,
        )

        # First save should go through
        await coordinator._record_learning_observations(
            nibe_data=nibe_data,
            weather_data=None,
            current_offset=0.5,
        )
        first_call_count = store.async_save.call_count
        assert first_call_count > 0

        # Second save within 5 minutes should be throttled
        await coordinator._record_learning_observations(
            nibe_data=nibe_data,
            weather_data=None,
            current_offset=0.6,
        )
        second_call_count = store.async_save.call_count
        assert second_call_count == first_call_count  # No additional save

        # Mock time passing (301 seconds = 5 minutes + 1 second)
        coordinator._last_predictor_save = dt_util.utcnow() - timedelta(seconds=301)

        # Third save after throttle period should go through
        await coordinator._record_learning_observations(
            nibe_data=nibe_data,
            weather_data=None,
            current_offset=0.7,
        )
        third_call_count = store.async_save.call_count
        assert third_call_count > second_call_count  # New save happened

    @pytest.mark.asyncio
    async def test_thermal_predictor_preserves_other_learning_data(
        self, hass, mock_config_entry, mock_storage
    ):
        """Test that thermal predictor save preserves other learning module data."""
        store, storage_data = mock_storage

        # Pre-populate storage with existing learning data
        storage_data["thermal_model"] = {"thermal_mass": 1.5, "ufh_type": "concrete_slab"}
        storage_data["weather_patterns"] = {"patterns": []}

        coordinator = EffektGuardCoordinator(
            hass,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            mock_config_entry,
        )
        coordinator.learning_store = store

        nibe_data = NibeState(
            outdoor_temp=5.0,
            indoor_temp=21.0,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-100.0,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=dt_util.utcnow(),
            compressor_hz=50,
        )

        # Record thermal state (should update thermal_predictor only)
        await coordinator._record_learning_observations(
            nibe_data=nibe_data,
            weather_data=None,
            current_offset=0.5,
        )

        # Verify other learning data is preserved
        assert "thermal_model" in storage_data
        assert storage_data["thermal_model"]["thermal_mass"] == 1.5
        assert "weather_patterns" in storage_data
        assert "thermal_predictor" in storage_data

    @pytest.mark.asyncio
    async def test_thermal_predictor_accumulates_history(
        self, hass, mock_config_entry, mock_storage
    ):
        """Test that thermal predictor accumulates 24 hours of history."""
        store, storage_data = mock_storage

        coordinator = EffektGuardCoordinator(
            hass,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            mock_config_entry,
        )
        coordinator.learning_store = store

        # Record 10 states at 15-minute intervals
        base_time = dt_util.utcnow()
        for i in range(10):
            nibe_data = NibeState(
                indoor_temp=21.0 + (i * 0.1),
                outdoor_temp=5.0 - (i * 0.2),
                supply_temp=35.0,
                return_temp=30.0,
                current_offset=0.0,
                is_heating=True,
                is_hot_water=False,
                timestamp=dt_util.utcnow(),
                degree_minutes=-100.0 - (i * 10),
                compressor_hz=50.0,
            )

            # Mock time progressing
            with patch("homeassistant.util.dt.utcnow") as mock_time:
                mock_time.return_value = base_time + timedelta(minutes=15 * i)

                # Force save by resetting throttle
                coordinator._last_predictor_save = None

                await coordinator._record_learning_observations(
                    nibe_data=nibe_data,
                    weather_data=None,
                    current_offset=0.5,
                )

        # Verify all 10 snapshots are in the predictor
        assert len(coordinator.thermal_predictor.state_history) == 10

        # Verify the data is saved
        assert "thermal_predictor" in storage_data
        assert len(storage_data["thermal_predictor"]["state_history"]) == 10

        # Verify temperature progression
        first_snapshot = storage_data["thermal_predictor"]["state_history"][0]
        last_snapshot = storage_data["thermal_predictor"]["state_history"][-1]
        assert first_snapshot["indoor_temp"] == 21.0
        assert last_snapshot["indoor_temp"] == pytest.approx(21.9, rel=0.01)

    @pytest.mark.asyncio
    async def test_thermal_predictor_restores_from_storage(
        self, hass, mock_config_entry, mock_storage
    ):
        """Test that thermal predictor restores state from storage on init."""
        store, storage_data = mock_storage

        # Pre-populate storage with thermal predictor data
        storage_data["thermal_predictor"] = {
            "lookback_hours": 24,
            "state_history": [
                {
                    "timestamp": "2025-10-18T12:00:00+00:00",
                    "indoor_temp": 21.5,
                    "outdoor_temp": 5.0,
                    "heating_offset": 0.5,
                    "flow_temp": 35.0,
                    "degree_minutes": -150.0,
                },
                {
                    "timestamp": "2025-10-18T12:15:00+00:00",
                    "indoor_temp": 21.6,
                    "outdoor_temp": 4.8,
                    "heating_offset": 0.5,
                    "flow_temp": 35.2,
                    "degree_minutes": -160.0,
                },
            ],
            "thermal_responsiveness": None,
        }

        # Create coordinator (will load from storage during init)
        coordinator = EffektGuardCoordinator(
            hass,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            mock_config_entry,
        )
        coordinator.learning_store = store

        # Manually call initialization (normally done in async_setup)
        await coordinator.async_initialize_learning()

        # Verify thermal predictor loaded the state history
        assert len(coordinator.thermal_predictor.state_history) == 2
        assert coordinator.thermal_predictor.state_history[0].indoor_temp == 21.5
        assert coordinator.thermal_predictor.state_history[1].indoor_temp == 21.6

        # Verify trends can be calculated from restored data
        trend = coordinator.thermal_predictor.get_current_trend()
        assert trend["trend"] in ["rising", "falling", "stable", "unknown"]

    @pytest.mark.asyncio
    async def test_thermal_predictor_handles_save_errors_gracefully(self, hass, mock_config_entry):
        """Test that save errors don't crash the coordinator."""
        # Create store that fails on save
        store = MagicMock()
        store.async_save = AsyncMock(side_effect=OSError("Disk full"))
        store.async_load = AsyncMock(return_value=None)

        coordinator = EffektGuardCoordinator(
            hass,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            mock_config_entry,
        )
        coordinator.learning_store = store

        nibe_data = NibeState(
            outdoor_temp=5.0,
            indoor_temp=21.0,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-100.0,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=dt_util.utcnow(),
            compressor_hz=50,
        )

        # Should not raise exception even though save fails
        await coordinator._record_learning_observations(
            nibe_data=nibe_data,
            weather_data=None,
            current_offset=0.5,
        )

        # Verify thermal predictor still recorded the state in memory
        assert len(coordinator.thermal_predictor.state_history) == 1

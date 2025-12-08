"""Tests for event listener integration and recorder usage.

Validates:
- Event listener setup without deprecated event_filter parameter
- Power sensor state change listeners
- DHW history retrieval using recorder executor
- Graceful handling when recorder unavailable
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta


def create_mock_coordinator_dependencies(mock_hass, entry):
    """Create all required dependencies for EffektGuardCoordinator."""
    # Mock NIBE adapter
    nibe_adapter = MagicMock()
    nibe_adapter.get_current_state = AsyncMock()

    # Mock spot price adapter
    gespot_adapter = MagicMock()
    gespot_adapter.get_prices = AsyncMock()

    # Mock Weather adapter
    weather_adapter = MagicMock()
    weather_adapter.get_forecast = AsyncMock()

    # Mock Decision Engine with climate detector
    decision_engine = MagicMock()
    decision_engine.climate_detector = MagicMock()
    decision_engine.climate_detector.get_expected_dm_range = MagicMock(
        return_value={"critical": -1500, "warning": -700}
    )
    decision_engine.calculate_decision = AsyncMock()

    # Mock Effect Manager
    effect_manager = MagicMock()

    return nibe_adapter, gespot_adapter, weather_adapter, decision_engine, effect_manager


@pytest.mark.asyncio
async def test_event_listener_no_event_filter():
    """Test that power sensor listener works without event_filter parameter.

    Fix 1.1: Event listener should filter inside callback, not via event_filter parameter.
    """
    from custom_components.effektguard.coordinator import EffektGuardCoordinator

    # Create mock hass
    mock_hass = MagicMock()
    mock_hass.data = {}
    mock_hass.bus = MagicMock()
    mock_hass.states = MagicMock()
    mock_hass.states.get = MagicMock(return_value=None)
    mock_hass.config = MagicMock()
    mock_hass.config.latitude = 59.3
    mock_hass.config.config_dir = "/tmp/test"

    # Create mock config entry
    entry = MagicMock()
    entry.data = {
        "nibe_degree_minutes_entity": "sensor.degree_minutes",
        "nibe_indoor_temp_entity": "sensor.indoor_temp",
        "nibe_outdoor_temp_entity": "sensor.outdoor_temp",
        "nibe_curve_offset_entity": "number.curve_offset",
        "external_power_sensor": "sensor.power",
    }
    entry.options = {}

    # Track if async_listen is called correctly
    listen_called = False

    def mock_async_listen(event_type, callback, **kwargs):
        nonlocal listen_called
        # Should NOT have event_filter parameter
        if "event_filter" in kwargs:
            pytest.fail("async_listen received event_filter parameter - FIX FAILED")
        listen_called = True
        return MagicMock()  # Return unsubscribe callable

    mock_hass.bus.async_listen = mock_async_listen

    # Create coordinator with all dependencies
    nibe, gespot, weather, engine, effect = create_mock_coordinator_dependencies(mock_hass, entry)
    coordinator = EffektGuardCoordinator(mock_hass, nibe, gespot, weather, engine, effect, entry)

    # The setup_power_sensor_listener should not raise TypeError about event_filter
    coordinator.setup_power_sensor_listener()

    assert listen_called, "async_listen should have been called"


@pytest.mark.asyncio
async def test_power_sensor_listener_no_event_filter_param():
    """Test that event listener is registered without event_filter parameter.

    Fix 1.1: async_listen should be called without event_filter kwarg.
    """
    from custom_components.effektguard.coordinator import EffektGuardCoordinator

    # Create mock hass
    mock_hass = MagicMock()
    mock_hass.data = {}
    mock_hass.bus = MagicMock()
    mock_hass.states = MagicMock()
    # Return a power sensor state that's unavailable so listener is set up
    mock_hass.states.get = MagicMock(return_value=MagicMock(state="unavailable"))
    mock_hass.config = MagicMock()
    mock_hass.config.latitude = 59.3
    mock_hass.config.config_dir = "/tmp/test"

    entry = MagicMock()
    entry.data = {
        "nibe_degree_minutes_entity": "sensor.degree_minutes",
        "nibe_indoor_temp_entity": "sensor.indoor_temp",
        "nibe_outdoor_temp_entity": "sensor.outdoor_temp",
        "nibe_curve_offset_entity": "number.curve_offset",
        "external_power_sensor": "sensor.power",
    }
    entry.options = {}

    # Create coordinator with all dependencies
    nibe, gespot, weather, engine, effect = create_mock_coordinator_dependencies(mock_hass, entry)

    # Configure nibe adapter to have power sensor (so listener is registered)
    nibe._power_sensor_entity = "sensor.power"

    coordinator = EffektGuardCoordinator(mock_hass, nibe, gespot, weather, engine, effect, entry)

    # Track calls to async_listen
    listen_calls = []

    def mock_async_listen(event_type, callback, **kwargs):
        """Mock async_listen to capture callback."""
        listen_calls.append(
            {
                "event_type": event_type,
                "callback": callback,
                "kwargs": kwargs,
            }
        )
        return MagicMock()  # Return unsubscribe callable

    mock_hass.bus.async_listen = mock_async_listen

    # Setup listener
    coordinator.setup_power_sensor_listener()

    # Verify async_listen was called
    assert len(listen_calls) == 1, "async_listen should be called once"

    # Verify NO event_filter parameter was passed
    assert (
        "event_filter" not in listen_calls[0]["kwargs"]
    ), "FAIL: event_filter parameter should not be used (Phase 1 Fix 1.1)"

    # Verify event_type is correct
    assert listen_calls[0]["event_type"] == "state_changed"


@pytest.mark.asyncio
async def test_dhw_history_uses_recorder_executor():
    """Test that DHW history lookup uses recorder instance executor.

    Fix 1.2: Should use recorder.get_instance().async_add_executor_job
    instead of hass.async_add_executor_job for database queries.
    """
    from custom_components.effektguard.coordinator import EffektGuardCoordinator

    # Create mock hass
    mock_hass = MagicMock()
    mock_hass.data = {}

    # Mock entity state to be OFF (so history lookup is triggered)
    mock_entity_state = MagicMock()
    mock_entity_state.state = "off"
    mock_entity_state.last_changed = datetime.now()

    mock_hass.states = MagicMock()
    mock_hass.states.get = MagicMock(return_value=mock_entity_state)
    mock_hass.config = MagicMock()
    mock_hass.config.latitude = 59.3
    mock_hass.config.config_dir = "/tmp/test"

    entry = MagicMock()
    entry.data = {
        "nibe_degree_minutes_entity": "sensor.degree_minutes",
        "nibe_indoor_temp_entity": "sensor.indoor_temp",
        "nibe_outdoor_temp_entity": "sensor.outdoor_temp",
        "nibe_curve_offset_entity": "number.curve_offset",
        "nibe_temp_lux_entity": "binary_sensor.dhw_active",
    }
    entry.options = {}

    # Create coordinator with all dependencies
    nibe, gespot, weather, engine, effect = create_mock_coordinator_dependencies(mock_hass, entry)
    coordinator = EffektGuardCoordinator(mock_hass, nibe, gespot, weather, engine, effect, entry)

    # Mock recorder module and instance
    mock_recorder_instance = MagicMock()
    mock_recorder_instance.async_add_executor_job = AsyncMock(
        return_value={"binary_sensor.dhw_active": []}
    )

    mock_recorder = MagicMock()
    mock_recorder.get_instance = MagicMock(return_value=mock_recorder_instance)
    mock_recorder.history = MagicMock()
    mock_recorder.history.state_changes_during_period = MagicMock(return_value={})

    # Patch recorder import where it's used (inside the function)
    with patch("homeassistant.components.recorder", mock_recorder):
        # This should use recorder instance executor, not hass executor
        result = await coordinator._get_last_dhw_heating_time()

        # Verify recorder.get_instance was called
        mock_recorder.get_instance.assert_called_once_with(mock_hass)

        # Verify recorder executor was used
        assert (
            mock_recorder_instance.async_add_executor_job.called
        ), "FAIL: Should use recorder.get_instance().async_add_executor_job (Phase 1 Fix 1.2)"


@pytest.mark.asyncio
async def test_dhw_history_handles_no_recorder():
    """Test that DHW history handles recorder being unavailable gracefully.

    Fix 1.2: Should log warning and return None when recorder not available.
    """
    from custom_components.effektguard.coordinator import EffektGuardCoordinator

    # Create mock hass
    mock_hass = MagicMock()
    mock_hass.data = {}
    mock_hass.states = MagicMock()
    mock_hass.states.get = MagicMock(return_value=MagicMock(state="off"))
    mock_hass.config = MagicMock()
    mock_hass.config.latitude = 59.3
    mock_hass.config.config_dir = "/tmp/test"

    entry = MagicMock()
    entry.data = {
        "nibe_degree_minutes_entity": "sensor.degree_minutes",
        "nibe_indoor_temp_entity": "sensor.indoor_temp",
        "nibe_outdoor_temp_entity": "sensor.outdoor_temp",
        "nibe_curve_offset_entity": "number.curve_offset",
        "nibe_temp_lux_entity": "binary_sensor.dhw_active",
    }
    entry.options = {}

    # Create coordinator with all dependencies
    nibe, gespot, weather, engine, effect = create_mock_coordinator_dependencies(mock_hass, entry)
    coordinator = EffektGuardCoordinator(mock_hass, nibe, gespot, weather, engine, effect, entry)

    # Mock recorder module returning None for get_instance
    mock_recorder_module = MagicMock()
    mock_recorder_module.get_instance = MagicMock(return_value=None)

    # Patch the recorder import in coordinator module
    with patch.dict(
        "sys.modules",
        {"homeassistant.components.recorder": mock_recorder_module},
    ):
        result = await coordinator._get_last_dhw_heating_time()

        # Should return None gracefully
        assert result is None

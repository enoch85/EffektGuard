"""Tests for Phase 1 critical fixes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant, Event
from homeassistant.util import dt as dt_util


async def test_event_listener_no_event_filter(hass: HomeAssistant):
    """Test that power sensor listener works without event_filter parameter.
    
    Fix 1.1: Event listener should filter inside callback, not via event_filter parameter.
    """
    from custom_components.effektguard.coordinator import EffektGuardCoordinator
    
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
    
    # Create coordinator
    coordinator = EffektGuardCoordinator(hass, entry)
    
    # The setup_power_sensor_listener should not raise TypeError about event_filter
    try:
        await coordinator._setup_power_sensor_listener()
        success = True
    except TypeError as e:
        if "event_filter" in str(e):
            success = False
            pytest.fail(f"event_filter parameter error: {e}")
        else:
            raise
    
    assert success, "Power sensor listener should setup without event_filter error"


async def test_power_sensor_callback_filters_entity_id(hass: HomeAssistant):
    """Test that power sensor callback properly filters events by entity_id.
    
    Fix 1.1: Callback should ignore events for other entities.
    """
    from custom_components.effektguard.coordinator import EffektGuardCoordinator
    
    entry = MagicMock()
    entry.data = {
        "nibe_degree_minutes_entity": "sensor.degree_minutes",
        "nibe_indoor_temp_entity": "sensor.indoor_temp",
        "nibe_outdoor_temp_entity": "sensor.outdoor_temp",
        "nibe_curve_offset_entity": "number.curve_offset",
        "external_power_sensor": "sensor.power",
    }
    entry.options = {}
    
    # Mock hass.states.get to return unavailable state
    hass.states.get = MagicMock(return_value=MagicMock(state="unavailable"))
    
    coordinator = EffektGuardCoordinator(hass, entry)
    
    # Track calls to async_listen
    listen_calls = []
    
    def mock_async_listen(event_type, callback, **kwargs):
        """Mock async_listen to capture callback."""
        listen_calls.append({
            "event_type": event_type,
            "callback": callback,
            "kwargs": kwargs,
        })
        return MagicMock()  # Return unsubscribe callable
    
    hass.bus.async_listen = mock_async_listen
    
    # Setup listener
    await coordinator._setup_power_sensor_listener()
    
    # Verify no event_filter parameter was used
    assert len(listen_calls) == 1
    assert "event_filter" not in listen_calls[0]["kwargs"]
    
    # Get the callback
    callback = listen_calls[0]["callback"]
    
    # Test callback filters correctly
    # Event for wrong entity should be ignored
    wrong_event = MagicMock(spec=Event)
    wrong_event.data = {
        "entity_id": "sensor.other_entity",
        "new_state": MagicMock(state="10.0"),
    }
    
    # This should not trigger availability flag
    callback(wrong_event)
    assert not coordinator._power_sensor_available
    
    # Event for correct entity should be processed
    correct_event = MagicMock(spec=Event)
    correct_event.data = {
        "entity_id": "sensor.power",
        "new_state": MagicMock(state="10.0"),
    }
    
    callback(correct_event)
    assert coordinator._power_sensor_available


@pytest.mark.skipif(
    "recorder" not in dir(),
    reason="Recorder not available in test environment",
)
async def test_dhw_history_uses_recorder_executor(hass: HomeAssistant):
    """Test that DHW history lookup uses recorder instance executor.
    
    Fix 1.2: Should use recorder.get_instance().async_add_executor_job
    instead of hass.async_add_executor_job for database queries.
    """
    from custom_components.effektguard.coordinator import EffektGuardCoordinator
    
    entry = MagicMock()
    entry.data = {
        "nibe_degree_minutes_entity": "sensor.degree_minutes",
        "nibe_indoor_temp_entity": "sensor.indoor_temp",
        "nibe_outdoor_temp_entity": "sensor.outdoor_temp",
        "nibe_curve_offset_entity": "number.curve_offset",
        "nibe_temp_lux_entity": "binary_sensor.dhw_active",
    }
    entry.options = {}
    
    coordinator = EffektGuardCoordinator(hass, entry)
    
    # Mock entity state as OFF
    hass.states.get = MagicMock(return_value=MagicMock(state="off"))
    
    # Mock recorder.get_instance
    mock_recorder = MagicMock()
    mock_recorder.async_add_executor_job = AsyncMock(return_value={})
    
    with patch(
        "custom_components.effektguard.coordinator.recorder.get_instance",
        return_value=mock_recorder,
    ):
        # This should use recorder instance executor, not hass executor
        result = await coordinator._get_hours_since_last_dhw_sync()
        
        # Verify recorder executor was used
        assert mock_recorder.async_add_executor_job.called


async def test_dhw_history_handles_no_recorder(hass: HomeAssistant):
    """Test that DHW history handles recorder being unavailable gracefully.
    
    Fix 1.2: Should log warning and return None when recorder not available.
    """
    from custom_components.effektguard.coordinator import EffektGuardCoordinator
    
    entry = MagicMock()
    entry.data = {
        "nibe_degree_minutes_entity": "sensor.degree_minutes",
        "nibe_indoor_temp_entity": "sensor.indoor_temp",
        "nibe_outdoor_temp_entity": "sensor.outdoor_temp",
        "nibe_curve_offset_entity": "number.curve_offset",
        "nibe_temp_lux_entity": "binary_sensor.dhw_active",
    }
    entry.options = {}
    
    coordinator = EffektGuardCoordinator(hass, entry)
    
    # Mock entity state as OFF
    hass.states.get = MagicMock(return_value=MagicMock(state="off"))
    
    # Mock recorder.get_instance returning None
    with patch(
        "custom_components.effektguard.coordinator.recorder.get_instance",
        return_value=None,
    ):
        result = await coordinator._get_hours_since_last_dhw_sync()
        
        # Should return None gracefully
        assert result is None

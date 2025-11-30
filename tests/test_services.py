"""Tests for EffektGuard services.

Tests Phase 5 service implementations:
- force_offset: Manual heating curve override
- reset_peak_tracking: Reset monthly peak data
- boost_heating: Emergency comfort boost
- calculate_optimal_schedule: Preview 24h optimization
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError

from custom_components.effektguard.const import (
    ATTR_DURATION,
    ATTR_OFFSET,
    DOMAIN,
    MAX_OFFSET,
    MIN_OFFSET,
    SERVICE_BOOST_HEATING,
    SERVICE_CALCULATE_OPTIMAL_SCHEDULE,
    SERVICE_FORCE_OFFSET,
    SERVICE_RESET_PEAK_TRACKING,
)
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_manager import EffectManager


@pytest.fixture
def mock_hass():
    """Create mock Home Assistant instance."""
    hass = MagicMock(spec=HomeAssistant)
    hass.data = {DOMAIN: {}}
    hass.services = MagicMock()
    hass.services.has_service = MagicMock(return_value=False)
    hass.services.async_register = AsyncMock()
    return hass


@pytest.fixture
def mock_coordinator(mock_hass):
    """Create mock coordinator with decision engine and effect manager."""
    from custom_components.effektguard.coordinator import EffektGuardCoordinator

    coordinator = MagicMock(spec=EffektGuardCoordinator)
    coordinator.hass = mock_hass

    # Mock decision engine
    coordinator.engine = MagicMock(spec=DecisionEngine)
    coordinator.engine.set_manual_override = MagicMock()

    # Mock effect manager
    coordinator.effect = MagicMock(spec=EffectManager)
    coordinator.effect.reset_monthly_peaks = MagicMock()
    coordinator.effect.async_save = AsyncMock()

    # Mock coordinator methods
    coordinator.async_request_refresh = AsyncMock()

    # Mock data for calculate_optimal_schedule
    coordinator.data = {
        "nibe": MagicMock(
            indoor_temp=21.5,
            outdoor_temp=5.0,
        ),
        "price": MagicMock(
            today=[
                MagicMock(
                    price=1.0 + (i * 0.01),
                    quarter_of_day=i,
                    is_daytime=(24 <= i <= 87),
                )
                for i in range(96)
            ],
            tomorrow=None,
        ),
        "weather": MagicMock(),
    }

    # Add to hass.data
    mock_hass.data[DOMAIN]["test_entry"] = coordinator

    return coordinator


# ============================================================================
# force_offset service tests
# ============================================================================


async def test_force_offset_service_registration(mock_hass):
    """Test force_offset service is registered correctly."""
    from custom_components.effektguard import _async_register_services

    await _async_register_services(mock_hass)

    # Verify service was registered
    calls = mock_hass.services.async_register.call_args_list
    service_names = [call[0][1] for call in calls]
    assert SERVICE_FORCE_OFFSET in service_names


async def test_force_offset_sets_override(mock_hass, mock_coordinator):
    """Test force_offset sets manual override in decision engine."""
    from custom_components.effektguard import _async_register_services

    await _async_register_services(mock_hass)

    # Get the registered handler
    calls = mock_hass.services.async_register.call_args_list
    force_offset_call = next(call for call in calls if call[0][1] == SERVICE_FORCE_OFFSET)
    handler = force_offset_call[0][2]

    # Create service call
    call = MagicMock()
    call.data = {ATTR_OFFSET: 2.5, ATTR_DURATION: 60}

    # Execute handler
    await handler(call)

    # Verify override was set
    mock_coordinator.engine.set_manual_override.assert_called_once_with(2.5, 60)
    mock_coordinator.async_request_refresh.assert_called_once()


async def test_force_offset_with_zero_duration(mock_hass, mock_coordinator):
    """Test force_offset with duration=0 (until next cycle)."""
    from custom_components.effektguard import _async_register_services, _service_last_called

    # Clear service cooldown tracking
    _service_last_called.clear()

    # Add engine mock
    mock_coordinator.engine = MagicMock()
    mock_coordinator.engine.set_manual_override = MagicMock()

    await _async_register_services(mock_hass)

    calls = mock_hass.services.async_register.call_args_list
    force_offset_call = next(call for call in calls if call[0][1] == SERVICE_FORCE_OFFSET)
    handler = force_offset_call[0][2]

    call = MagicMock()
    call.data = {ATTR_OFFSET: -3.0, ATTR_DURATION: 0}

    await handler(call)

    mock_coordinator.engine.set_manual_override.assert_called_once_with(-3.0, 0)


async def test_force_offset_validates_range(mock_hass, mock_coordinator):
    """Test force_offset validates offset is within valid range."""
    from custom_components.effektguard import _async_register_services
    from homeassistant.exceptions import ServiceValidationError

    # Patch cooldown tracker to allow the call
    with patch("custom_components.effektguard._service_last_called", {}):
        await _async_register_services(mock_hass)

        calls = mock_hass.services.async_register.call_args_list
        force_offset_call = next(call for call in calls if call[0][1] == SERVICE_FORCE_OFFSET)
        handler = force_offset_call[0][2]

        # Test with out-of-range offset
        call = MagicMock()
        call.data = {ATTR_OFFSET: 15.0, ATTR_DURATION: 60}  # > MAX_OFFSET

        # Should raise ServiceValidationError for invalid offset
        with pytest.raises(ServiceValidationError, match="outside valid range"):
            await handler(call)

        # Should not set override for invalid offset
        mock_coordinator.engine.set_manual_override.assert_not_called()


# ============================================================================
# reset_peak_tracking service tests
# ============================================================================


async def test_reset_peak_tracking_service_registration(mock_hass):
    """Test reset_peak_tracking service is registered correctly."""
    from custom_components.effektguard import _async_register_services

    await _async_register_services(mock_hass)

    calls = mock_hass.services.async_register.call_args_list
    service_names = [call[0][1] for call in calls]
    assert SERVICE_RESET_PEAK_TRACKING in service_names


async def test_reset_peak_tracking_clears_peaks(mock_hass, mock_coordinator):
    """Test reset_peak_tracking clears monthly peaks."""
    from custom_components.effektguard import _async_register_services

    await _async_register_services(mock_hass)

    calls = mock_hass.services.async_register.call_args_list
    reset_call = next(call for call in calls if call[0][1] == SERVICE_RESET_PEAK_TRACKING)
    handler = reset_call[0][2]

    call = MagicMock()
    call.data = {}

    await handler(call)

    # Verify peaks were reset
    mock_coordinator.effect.reset_monthly_peaks.assert_called_once()
    mock_coordinator.effect.async_save.assert_called_once()
    mock_coordinator.async_request_refresh.assert_called_once()


# ============================================================================
# boost_heating service tests
# ============================================================================


async def test_boost_heating_service_registration(mock_hass):
    """Test boost_heating service is registered correctly."""
    from custom_components.effektguard import _async_register_services

    await _async_register_services(mock_hass)

    calls = mock_hass.services.async_register.call_args_list
    service_names = [call[0][1] for call in calls]
    assert SERVICE_BOOST_HEATING in service_names


async def test_boost_heating_sets_max_offset(mock_hass, mock_coordinator):
    """Test boost_heating sets maximum positive offset."""
    from custom_components.effektguard import _async_register_services

    await _async_register_services(mock_hass)

    calls = mock_hass.services.async_register.call_args_list
    boost_call = next(call for call in calls if call[0][1] == SERVICE_BOOST_HEATING)
    handler = boost_call[0][2]

    call = MagicMock()
    call.data = {ATTR_DURATION: 120}

    await handler(call)

    # Should set MAX_OFFSET (+10°C)
    mock_coordinator.engine.set_manual_override.assert_called_once_with(MAX_OFFSET, 120)
    mock_coordinator.async_request_refresh.assert_called_once()


async def test_boost_heating_default_duration(mock_hass, mock_coordinator):
    """Test boost_heating uses default duration if not specified."""
    from custom_components.effektguard import _async_register_services, _service_last_called

    # Clear service cooldown tracking
    _service_last_called.clear()

    # Add engine mock
    mock_coordinator.engine = MagicMock()
    mock_coordinator.engine.set_manual_override = MagicMock()

    await _async_register_services(mock_hass)

    calls = mock_hass.services.async_register.call_args_list
    boost_call = next(call for call in calls if call[0][1] == SERVICE_BOOST_HEATING)
    handler = boost_call[0][2]

    call = MagicMock()
    call.data = {}  # No duration specified

    await handler(call)

    # Should use default duration (120 minutes)
    mock_coordinator.engine.set_manual_override.assert_called_once_with(MAX_OFFSET, 120)


# ============================================================================
# calculate_optimal_schedule service tests
# ============================================================================


async def test_calculate_optimal_schedule_service_registration(mock_hass):
    """Test calculate_optimal_schedule service is registered correctly."""
    from custom_components.effektguard import _async_register_services

    await _async_register_services(mock_hass)

    # Should be registered with supports_response=True
    calls = mock_hass.services.async_register.call_args_list
    schedule_call = next(call for call in calls if call[0][1] == SERVICE_CALCULATE_OPTIMAL_SCHEDULE)

    # Verify supports_response is True
    assert "supports_response" in schedule_call[1]
    assert schedule_call[1]["supports_response"] is True


async def test_calculate_optimal_schedule_returns_24h_schedule(mock_hass, mock_coordinator):
    """Test calculate_optimal_schedule returns 24-hour schedule."""
    from custom_components.effektguard import _async_register_services

    # Setup price analyzer mock
    mock_coordinator.engine.price = MagicMock()
    mock_coordinator.engine.price.get_current_classification = MagicMock(return_value="normal")
    mock_coordinator.engine.price.get_base_offset = MagicMock(return_value=0.5)

    await _async_register_services(mock_hass)

    calls = mock_hass.services.async_register.call_args_list
    schedule_call = next(call for call in calls if call[0][1] == SERVICE_CALCULATE_OPTIMAL_SCHEDULE)
    handler = schedule_call[0][2]

    call = MagicMock()
    call.data = {}

    result = await handler(call)

    # Verify response structure
    assert "schedule" in result
    assert "generated_at" in result
    assert len(result["schedule"]) == 24  # 24 hours

    # Verify schedule entry structure
    entry = result["schedule"][0]
    assert "time" in entry
    assert "hour" in entry
    assert "quarter" in entry
    assert "classification" in entry
    assert "estimated_offset" in entry
    assert "price" in entry
    assert "is_daytime" in entry


async def test_calculate_optimal_schedule_handles_missing_data(mock_hass, mock_coordinator):
    """Test calculate_optimal_schedule handles missing coordinator data."""
    from custom_components.effektguard import _async_register_services

    # Remove coordinator data
    mock_coordinator.data = None

    await _async_register_services(mock_hass)

    calls = mock_hass.services.async_register.call_args_list
    schedule_call = next(call for call in calls if call[0][1] == SERVICE_CALCULATE_OPTIMAL_SCHEDULE)
    handler = schedule_call[0][2]

    call = MagicMock()
    call.data = {}

    result = await handler(call)

    # Should return error
    assert "error" in result


async def test_calculate_optimal_schedule_no_coordinator(mock_hass):
    """Test calculate_optimal_schedule when no coordinator available."""
    from custom_components.effektguard import _async_register_services

    # Empty domain data
    mock_hass.data[DOMAIN] = {}

    await _async_register_services(mock_hass)

    calls = mock_hass.services.async_register.call_args_list
    schedule_call = next(call for call in calls if call[0][1] == SERVICE_CALCULATE_OPTIMAL_SCHEDULE)
    handler = schedule_call[0][2]

    call = MagicMock()
    call.data = {}

    result = await handler(call)

    # Should return error when coordinator not found
    assert "error" in result
    assert "coordinator" in result["error"].lower()


# ============================================================================
# Decision engine manual override tests
# ============================================================================


def test_decision_engine_set_manual_override():
    """Test decision engine set_manual_override method."""
    from custom_components.effektguard.optimization.decision_engine import (
        DecisionEngine,
    )

    engine = DecisionEngine(
        price_analyzer=MagicMock(),
        effect_manager=MagicMock(),
        thermal_model=MagicMock(),
        config={},
    )

    # Set override
    engine.set_manual_override(2.5, 60)

    assert engine._manual_override_offset == 2.5
    assert engine._manual_override_until is not None


def test_decision_engine_clear_manual_override():
    """Test decision engine clear_manual_override method."""
    from custom_components.effektguard.optimization.decision_engine import (
        DecisionEngine,
    )

    engine = DecisionEngine(
        price_analyzer=MagicMock(),
        effect_manager=MagicMock(),
        thermal_model=MagicMock(),
        config={},
    )

    # Set then clear
    engine.set_manual_override(2.5, 60)
    engine.clear_manual_override()

    assert engine._manual_override_offset is None
    assert engine._manual_override_until is None


def test_decision_engine_check_manual_override_active():
    """Test _check_manual_override returns override when active."""
    from custom_components.effektguard.optimization.decision_engine import (
        DecisionEngine,
    )

    engine = DecisionEngine(
        price_analyzer=MagicMock(),
        effect_manager=MagicMock(),
        thermal_model=MagicMock(),
        config={},
    )

    engine.set_manual_override(2.5, 60)

    result = engine._check_manual_override()
    assert result == 2.5


def test_decision_engine_check_manual_override_expired():
    """Test _check_manual_override clears expired override."""
    from custom_components.effektguard.optimization.decision_engine import (
        DecisionEngine,
    )
    from homeassistant.util import dt as dt_util

    engine = DecisionEngine(
        price_analyzer=MagicMock(),
        effect_manager=MagicMock(),
        thermal_model=MagicMock(),
        config={},
    )

    # Set override with past expiry
    engine._manual_override_offset = 2.5
    engine._manual_override_until = dt_util.now() - timedelta(minutes=1)

    result = engine._check_manual_override()

    # Should clear and return None
    assert result is None
    assert engine._manual_override_offset is None


def test_decision_engine_calculate_with_manual_override():
    """Test calculate_decision uses manual override when active."""
    from custom_components.effektguard.optimization.decision_engine import (
        DecisionEngine,
    )

    engine = DecisionEngine(
        price_analyzer=MagicMock(),
        effect_manager=MagicMock(),
        thermal_model=MagicMock(),
        config={},
    )

    # Set manual override
    engine.set_manual_override(5.0, 60)

    # Mock state
    nibe_state = MagicMock(indoor_temp=21.0, outdoor_temp=5.0, degree_minutes=-100)
    price_data = MagicMock()
    weather_data = MagicMock()

    decision = engine.calculate_decision(nibe_state, price_data, weather_data, 0.0, 0.0)

    # Should return manual override
    assert decision.offset == 5.0
    assert "manual override" in decision.reasoning.lower()


# ============================================================================
# Effect manager reset tests
# ============================================================================


def test_effect_manager_reset_monthly_peaks():
    """Test EffectManager.reset_monthly_peaks clears all peaks."""
    from custom_components.effektguard.optimization.effect_manager import (
        EffectManager,
        PeakEvent,
    )

    hass = MagicMock()
    hass.data = {}
    effect = EffectManager(hass)

    # Add some peaks
    effect._monthly_peaks = [
        PeakEvent(
            timestamp=datetime.now(),
            quarter_of_day=50,
            actual_power=5.0,
            effective_power=5.0,
            is_daytime=True,
        ),
        PeakEvent(
            timestamp=datetime.now(),
            quarter_of_day=60,
            actual_power=4.5,
            effective_power=4.5,
            is_daytime=True,
        ),
    ]
    effect._current_peak = 5.0

    # Reset
    effect.reset_monthly_peaks()

    # Verify cleared
    assert len(effect._monthly_peaks) == 0
    assert effect._current_peak == 0.0


# ============================================================================
# Service error handling tests
# ============================================================================


async def test_service_handles_no_coordinator_gracefully(mock_hass):
    """Test services handle missing coordinator gracefully."""
    from custom_components.effektguard import _async_register_services
    from homeassistant.exceptions import ServiceValidationError

    # Empty domain data
    mock_hass.data[DOMAIN] = {}

    # Patch cooldown tracker to allow the calls
    with patch("custom_components.effektguard._service_last_called", {}):
        await _async_register_services(mock_hass)

        # Get handlers
        calls = mock_hass.services.async_register.call_args_list

        force_offset_handler = next(
            call[0][2] for call in calls if call[0][1] == SERVICE_FORCE_OFFSET
        )
        reset_handler = next(
            call[0][2] for call in calls if call[0][1] == SERVICE_RESET_PEAK_TRACKING
        )
        boost_handler = next(call[0][2] for call in calls if call[0][1] == SERVICE_BOOST_HEATING)

        # Test each handler with no coordinator - should raise ServiceValidationError
        call = MagicMock()
        call.data = {ATTR_OFFSET: 2.5, ATTR_DURATION: 60}

        with pytest.raises(ServiceValidationError, match="No EffektGuard coordinator found"):
            await force_offset_handler(call)

        call.data = {}
        with pytest.raises(ServiceValidationError, match="No EffektGuard coordinator found"):
            await reset_handler(call)

        call.data = {ATTR_DURATION: 120}
        with pytest.raises(ServiceValidationError, match="No EffektGuard coordinator found"):
            await boost_handler(call)


# ============================================================================
# Integration tests
# ============================================================================


async def test_all_services_registered(mock_hass):
    """Test all four Phase 5 services are registered."""
    from custom_components.effektguard import _async_register_services

    await _async_register_services(mock_hass)

    # Verify all services registered
    calls = [call[0][1] for call in mock_hass.services.async_register.call_args_list]

    assert SERVICE_FORCE_OFFSET in calls
    assert SERVICE_RESET_PEAK_TRACKING in calls
    assert SERVICE_BOOST_HEATING in calls
    assert SERVICE_CALCULATE_OPTIMAL_SCHEDULE in calls

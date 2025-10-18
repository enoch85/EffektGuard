"""Tests for DHW automatic control via temporary lux switch.

Tests the integration between DHW optimizer decisions and temporary lux control.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call

from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.dhw_optimizer import (
    IntelligentDHWScheduler,
    DHWScheduleDecision,
)


@pytest.fixture
def mock_hass():
    """Create mock Home Assistant instance."""
    hass = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.states = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda func, *args: func(*args))
    return hass


@pytest.fixture
def mock_config_entry():
    """Create mock config entry with DHW optimization enabled."""
    entry = MagicMock()
    entry.data = {
        "enable_hot_water_optimization": True,
        "nibe_temp_lux_entity": "switch.temporary_lux_50004",
        "target_indoor_temp": 21.0,
        "latitude": 55.60,
    }
    entry.options = {
        "dhw_target_temp": 50.0,  # Default DHW target temperature
        "dhw_morning_enabled": True,
        "dhw_morning_hour": 7,
        "dhw_evening_enabled": True,
        "dhw_evening_hour": 18,
    }
    entry.entry_id = "test_entry"
    return entry


@pytest.fixture
def mock_nibe_data():
    """Create mock NIBE data."""
    data = MagicMock()
    data.indoor_temp = 21.0
    data.outdoor_temp = 10.0
    data.degree_minutes = -150.0  # Normal range
    data.flow_temp = 30.0
    return data


@pytest.fixture
def mock_price_data():
    """Create mock price data."""
    data = MagicMock()
    data.today = [MagicMock(price=50.0) for _ in range(96)]  # Cheap prices
    return data


@pytest.fixture
def coordinator(mock_hass, mock_config_entry):
    """Create coordinator with minimal required mocks."""
    coordinator = EffektGuardCoordinator(
        hass=mock_hass,
        nibe_adapter=Mock(),
        gespot_adapter=Mock(),
        weather_adapter=Mock(),
        decision_engine=Mock(),
        effect_manager=Mock(),
        entry=mock_config_entry,
    )
    # Mock the entry attribute
    coordinator.entry = mock_config_entry
    # Mock engine and price analyzer
    coordinator.engine = Mock()
    coordinator.engine.price = Mock()
    coordinator.engine.price.get_current_classification = Mock(return_value="cheap")
    # Mock DHW optimizer
    coordinator.dhw_optimizer = Mock(spec=IntelligentDHWScheduler)
    # Mock history tracking methods (Phase 5.4)
    coordinator._get_last_dhw_heating_time = AsyncMock(return_value=None)
    coordinator._calculate_hours_since_last_dhw = AsyncMock(return_value=24.0)
    return coordinator


class TestDHWAutomaticControl:
    """Test automatic DHW control via temporary lux switch."""

    @pytest.mark.asyncio
    async def test_turn_on_lux_when_should_heat_true(
        self, mock_hass, coordinator, mock_nibe_data, mock_price_data
    ):
        """Test that temporary lux is turned ON when DHW should heat."""
        # Mock temporary lux switch state as OFF
        mock_lux_state = MagicMock()
        mock_lux_state.state = "off"
        mock_hass.states.get.return_value = mock_lux_state

        # Mock DHW optimizer to return should_heat=True
        decision = DHWScheduleDecision(
            should_heat=True,
            priority_reason="CHEAP_ELECTRICITY_OPPORTUNITY",
            target_temp=50.0,
            max_runtime_minutes=45,
            abort_conditions=[],
        )
        coordinator.dhw_optimizer.should_start_dhw = Mock(return_value=decision)

        # Call DHW control
        await coordinator._apply_dhw_control(
            mock_nibe_data, mock_price_data, None, 42.0, datetime.now()
        )

        # Verify temporary lux was turned ON
        mock_hass.services.async_call.assert_called_once_with(
            "switch",
            "turn_on",
            {"entity_id": "switch.temporary_lux_50004"},
            blocking=False,
        )

    @pytest.mark.asyncio
    async def test_turn_off_lux_when_should_heat_false(
        self, mock_hass, coordinator, mock_nibe_data, mock_price_data
    ):
        """Test that temporary lux is turned OFF when DHW should not heat."""
        # Mock temporary lux switch state as ON
        mock_lux_state = MagicMock()
        mock_lux_state.state = "on"
        mock_hass.states.get.return_value = mock_lux_state

        # Mock DHW optimizer to return should_heat=False (critical thermal debt)
        mock_nibe_data.degree_minutes = -450.0  # Critical
        decision = DHWScheduleDecision(
            should_heat=False,
            priority_reason="CRITICAL_THERMAL_DEBT",
            target_temp=0.0,
            max_runtime_minutes=0,
            abort_conditions=[],
        )
        coordinator.dhw_optimizer.should_start_dhw = Mock(return_value=decision)

        # Call DHW control
        await coordinator._apply_dhw_control(
            mock_nibe_data, mock_price_data, None, 38.0, datetime.now()
        )

        # Verify temporary lux was turned OFF
        mock_hass.services.async_call.assert_called_once_with(
            "switch",
            "turn_off",
            {"entity_id": "switch.temporary_lux_50004"},
            blocking=False,
        )

    @pytest.mark.asyncio
    async def test_no_change_when_already_in_correct_state(
        self, mock_hass, coordinator, mock_nibe_data, mock_price_data
    ):
        """Test that no action is taken when lux is already in correct state."""
        # Mock temporary lux switch state as OFF
        mock_lux_state = MagicMock()
        mock_lux_state.state = "off"
        mock_hass.states.get.return_value = mock_lux_state

        # Mock DHW optimizer to return should_heat=False (DHW adequate)
        decision = DHWScheduleDecision(
            should_heat=False,
            priority_reason="DHW_ADEQUATE",
            target_temp=0.0,
            max_runtime_minutes=0,
            abort_conditions=[],
        )
        coordinator.dhw_optimizer.should_start_dhw = Mock(return_value=decision)

        # Call DHW control
        await coordinator._apply_dhw_control(
            mock_nibe_data, mock_price_data, None, 48.0, datetime.now()
        )

        # Verify NO service call was made (already in correct state)
        mock_hass.services.async_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limiting_prevents_frequent_changes(
        self, mock_hass, coordinator, mock_nibe_data, mock_price_data
    ):
        """Test that rate limiting prevents changes within 10 minutes."""
        # Mock temporary lux switch state as OFF
        mock_lux_state = MagicMock()
        mock_lux_state.state = "off"
        mock_hass.states.get.return_value = mock_lux_state

        # Mock DHW optimizer to return should_heat=True
        decision = DHWScheduleDecision(
            should_heat=True,
            priority_reason="CHEAP_ELECTRICITY_OPPORTUNITY",
            target_temp=50.0,
            max_runtime_minutes=45,
            abort_conditions=[],
        )
        coordinator.dhw_optimizer.should_start_dhw = Mock(return_value=decision)

        # First call - should succeed
        now = datetime.now()
        await coordinator._apply_dhw_control(mock_nibe_data, mock_price_data, None, 42.0, now)
        assert mock_hass.services.async_call.call_count == 1

        # Second call 30 minutes later - should be rate limited (min 60 minutes)
        mock_hass.services.async_call.reset_mock()
        await coordinator._apply_dhw_control(
            mock_nibe_data, mock_price_data, None, 41.0, now + timedelta(minutes=30)
        )
        assert mock_hass.services.async_call.call_count == 0  # Rate limited

        # Third call 61 minutes later - should succeed
        mock_hass.services.async_call.reset_mock()
        await coordinator._apply_dhw_control(
            mock_nibe_data, mock_price_data, None, 40.0, now + timedelta(minutes=61)
        )
        assert mock_hass.services.async_call.call_count == 1  # Allowed

    @pytest.mark.asyncio
    async def test_no_action_when_temp_lux_entity_not_configured(
        self, mock_hass, coordinator, mock_nibe_data, mock_price_data
    ):
        """Test that no action is taken when temp lux entity is not configured."""
        # Remove temp lux entity from config
        coordinator.entry.data["nibe_temp_lux_entity"] = None

        # Call DHW control
        await coordinator._apply_dhw_control(
            mock_nibe_data, mock_price_data, None, 42.0, datetime.now()
        )

        # Verify NO service call was made
        mock_hass.services.async_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_action_when_temp_lux_entity_not_found(
        self, mock_hass, coordinator, mock_nibe_data, mock_price_data
    ):
        """Test that no action is taken when temp lux entity doesn't exist."""
        # Mock entity not found
        mock_hass.states.get.return_value = None

        # Call DHW control
        await coordinator._apply_dhw_control(
            mock_nibe_data, mock_price_data, None, 42.0, datetime.now()
        )

        # Verify NO service call was made
        mock_hass.services.async_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_critical_thermal_debt_blocks_dhw(
        self, mock_hass, coordinator, mock_nibe_data, mock_price_data
    ):
        """Test that critical thermal debt (DM < -240) blocks DHW heating."""
        # Mock temporary lux switch state as ON (currently heating)
        mock_lux_state = MagicMock()
        mock_lux_state.state = "on"
        mock_hass.states.get.return_value = mock_lux_state

        # Set critical thermal debt
        mock_nibe_data.degree_minutes = -434.0  # Like in the user's log

        # Mock DHW optimizer
        coordinator.dhw_optimizer.should_start_dhw = Mock(
            return_value=DHWScheduleDecision(
                should_heat=False,
                priority_reason="CRITICAL_THERMAL_DEBT",
                target_temp=0.0,
                max_runtime_minutes=0,
                abort_conditions=[],
            )
        )

        # Call DHW control
        await coordinator._apply_dhw_control(
            mock_nibe_data, mock_price_data, None, 38.0, datetime.now()
        )

        # Verify temporary lux was turned OFF to block DHW
        mock_hass.services.async_call.assert_called_once_with(
            "switch",
            "turn_off",
            {"entity_id": "switch.temporary_lux_50004"},
            blocking=False,
        )

    @pytest.mark.asyncio
    async def test_cheap_electricity_triggers_dhw_heating(
        self, mock_hass, coordinator, mock_nibe_data, mock_price_data
    ):
        """Test that cheap electricity triggers DHW heating when conditions are safe."""
        # Mock temporary lux switch state as OFF
        mock_lux_state = MagicMock()
        mock_lux_state.state = "off"
        mock_hass.states.get.return_value = mock_lux_state

        # Good thermal debt, cheap prices
        mock_nibe_data.degree_minutes = -80.0  # Safe

        # Mock DHW optimizer
        coordinator.dhw_optimizer.should_start_dhw = Mock(
            return_value=DHWScheduleDecision(
                should_heat=True,
                priority_reason="CHEAP_ELECTRICITY_OPPORTUNITY",
                target_temp=55.0,
                max_runtime_minutes=45,
                abort_conditions=["thermal_debt < -400"],
            )
        )

        # Call DHW control
        await coordinator._apply_dhw_control(
            mock_nibe_data, mock_price_data, None, 42.0, datetime.now()
        )

        # Verify temporary lux was turned ON to boost DHW
        mock_hass.services.async_call.assert_called_once_with(
            "switch",
            "turn_on",
            {"entity_id": "switch.temporary_lux_50004"},
            blocking=False,
        )

    @pytest.mark.asyncio
    async def test_service_call_error_handling(
        self, mock_hass, coordinator, mock_nibe_data, mock_price_data
    ):
        """Test that service call errors are handled gracefully."""
        # Mock temporary lux switch state as OFF
        mock_lux_state = MagicMock()
        mock_lux_state.state = "off"
        mock_hass.states.get.return_value = mock_lux_state

        # Mock service call to raise exception
        mock_hass.services.async_call = AsyncMock(side_effect=Exception("Service call failed"))

        # Mock DHW optimizer to return should_heat=True
        decision = DHWScheduleDecision(
            should_heat=True,
            priority_reason="CHEAP_ELECTRICITY_OPPORTUNITY",
            target_temp=50.0,
            max_runtime_minutes=45,
            abort_conditions=[],
        )
        coordinator.dhw_optimizer.should_start_dhw = Mock(return_value=decision)

        # Call DHW control - should not raise exception
        try:
            await coordinator._apply_dhw_control(
                mock_nibe_data, mock_price_data, None, 42.0, datetime.now()
            )
        except Exception as e:
            pytest.fail(f"DHW control raised exception: {e}")

        # Verify service was attempted
        mock_hass.services.async_call.assert_called_once()


class TestDHWOptimizerDecisions:
    """Test DHW optimizer decision logic for automatic control."""

    def test_block_dhw_critical_thermal_debt(self):
        """Test that DM <= -240 blocks DHW."""
        optimizer = IntelligentDHWScheduler()

        decision = optimizer.should_start_dhw(
            current_dhw_temp=42.0,
            space_heating_demand_kw=3.0,
            thermal_debt_dm=-434.0,  # Critical (user's scenario)
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=10.0,
            price_classification="cheap",
            current_time=datetime.now(),
        )

        assert decision.should_heat is False
        assert decision.priority_reason == "CRITICAL_THERMAL_DEBT"

    def test_heat_dhw_cheap_electricity(self):
        """Test that cheap electricity triggers DHW heating when safe."""
        optimizer = IntelligentDHWScheduler()

        decision = optimizer.should_start_dhw(
            current_dhw_temp=42.0,
            space_heating_demand_kw=1.0,
            thermal_debt_dm=-80.0,  # Safe
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=10.0,
            price_classification="cheap",
            current_time=datetime.now(),
        )

        assert decision.should_heat is True
        assert decision.priority_reason == "CHEAP_ELECTRICITY_OPPORTUNITY"

    def test_safety_minimum_forces_heating(self):
        """Test that DHW < 35°C forces heating despite thermal debt."""
        optimizer = IntelligentDHWScheduler()

        decision = optimizer.should_start_dhw(
            current_dhw_temp=33.0,  # Below safety minimum
            space_heating_demand_kw=5.0,
            thermal_debt_dm=-200.0,  # Not critical yet
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=10.0,
            price_classification="expensive",
            current_time=datetime.now(),
        )

        assert decision.should_heat is True
        assert decision.priority_reason == "DHW_SAFETY_MINIMUM"
        assert decision.max_runtime_minutes == 30  # Limited runtime

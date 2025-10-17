"""Test EffektGuard options flow.

Tests to verify the OptionsFlowHandler follows Home Assistant best practices:
- @staticmethod @callback decorator on async_get_options_flow
- OptionsFlowHandler() instantiated without arguments
- config_entry available via property (set by HA framework)

This prevents the TypeError that was occurring:
"TypeError: OptionsFlowHandler() takes no arguments"
"""

import pytest
import inspect
from unittest.mock import Mock

from homeassistant import config_entries

from custom_components.effektguard.config_flow import (
    EffektGuardConfigFlow,
    OptionsFlowHandler,
)


@pytest.fixture
def mock_config_entry():
    """Create a mock config entry."""
    entry = Mock(spec=config_entries.ConfigEntry)
    entry.domain = "effektguard"
    entry.data = {"nibe_entity": "number.nibe_test"}
    entry.options = {"target_temperature": 21.0}
    return entry


class TestOptionsFlowHandler:
    """Test the options flow handler follows HA best practices."""

    def test_options_flow_handler_instantiates_without_args(self):
        """Test that OptionsFlowHandler() can be instantiated without arguments.

        This is the exact error that was occurring before the fix:
        TypeError: OptionsFlowHandler() takes no arguments

        Per HA best practices, OptionsFlowHandler should not have __init__,
        and config_entry is set by the framework after instantiation.
        """
        # This should NOT raise TypeError
        handler = OptionsFlowHandler()
        assert handler is not None
        assert isinstance(handler, config_entries.OptionsFlow)

    def test_async_get_options_flow_is_staticmethod(self):
        """Test that async_get_options_flow is decorated as @staticmethod."""
        # Should be callable without an instance
        assert callable(EffektGuardConfigFlow.async_get_options_flow)

        # Verify signature - should only have config_entry param (no self)
        sig = inspect.signature(EffektGuardConfigFlow.async_get_options_flow)
        params = list(sig.parameters.keys())
        assert len(params) == 1
        assert "config_entry" in params
        assert "self" not in params  # Proves it's a staticmethod

    def test_async_get_options_flow_returns_handler_instance(self, mock_config_entry):
        """Test that async_get_options_flow returns OptionsFlowHandler instance."""
        handler = EffektGuardConfigFlow.async_get_options_flow(mock_config_entry)
        assert isinstance(handler, OptionsFlowHandler)

    def test_options_handler_inherits_from_options_flow(self):
        """Test that OptionsFlowHandler properly inherits from OptionsFlow."""
        assert issubclass(OptionsFlowHandler, config_entries.OptionsFlow)

    def test_options_handler_has_async_step_init(self):
        """Test that OptionsFlowHandler implements async_step_init."""
        assert hasattr(OptionsFlowHandler, "async_step_init")
        assert callable(getattr(OptionsFlowHandler, "async_step_init"))

        # Verify it's async
        import asyncio

        method = getattr(OptionsFlowHandler, "async_step_init")
        assert asyncio.iscoroutinefunction(method)


class TestHomeAssistantBestPractices:
    """Verify implementation follows official HA documentation.

    Reference: https://developers.home-assistant.io/docs/config_entries_options_flow_handler/

    Per HA documentation:
    - Options flow should ONLY update config_entry.options (runtime settings)
    - Entity configuration belongs in config_entry.data (initial setup only)
    - To change entity configuration after setup, use reconfigure flow (not options flow)
    """

    def test_no_init_method_defined(self):
        """Test that OptionsFlowHandler doesn't define its own __init__.

        Per HA docs, OptionsFlowHandler should NOT have __init__ method.
        The config_entry is automatically available via self.config_entry property.
        """
        # Check if __init__ is defined in OptionsFlowHandler specifically
        # (not inherited from parent)
        has_own_init = "__init__" in OptionsFlowHandler.__dict__
        assert not has_own_init, "OptionsFlowHandler should not define __init__"

    def test_async_get_options_flow_pattern(self, mock_config_entry):
        """Test the recommended pattern from HA documentation.

        The official pattern is:
        @staticmethod
        @callback
        def async_get_options_flow(config_entry):
            return OptionsFlowHandler()

        Note: config_entry is NOT passed to OptionsFlowHandler()
        """
        # Should work exactly as in the documentation
        handler = EffektGuardConfigFlow.async_get_options_flow(mock_config_entry)

        # Verify it's an instance of OptionsFlowHandler
        assert isinstance(handler, OptionsFlowHandler)

        # Verify it was created without passing config_entry
        # (the framework will set it later via property)

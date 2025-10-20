"""Pytest configuration for EffektGuard tests."""

import asyncio
import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add custom_components to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)

# Filter warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="josepy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="acme")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="homeassistant")


@pytest.fixture(autouse=True)
async def setup_frame_helper():
    """Set up the frame helper for all tests."""
    from homeassistant.helpers import frame
    from homeassistant.core import HomeAssistant

    # Create a minimal hass instance for frame setup
    hass = HomeAssistant("/tmp")
    frame.async_setup(hass)  # This is not actually async, just named that way
    yield
    # Teardown
    await hass.async_stop()


# Common mock helper functions
def create_mock_hass(latitude: float = 59.3):
    """Create a properly configured mock Home Assistant instance.

    Args:
        latitude: Latitude for climate region detection

    Returns:
        Mock hass with required attributes for Store initialization
    """
    mock_hass = Mock()
    mock_hass.data = {}
    mock_hass.config.latitude = latitude
    mock_hass.config.config_dir = tempfile.mkdtemp()
    mock_hass.async_add_executor_job = AsyncMock(side_effect=lambda func, *args: func(*args))
    mock_hass.loop = Mock()  # Add loop for DataUpdateCoordinator
    mock_hass.loop.call_soon_threadsafe = Mock()
    return mock_hass


def create_mock_entry():
    """Create a properly configured mock config entry.

    Returns:
        Mock entry with proper options.get() side_effect
    """
    mock_entry = Mock()
    mock_entry.options = Mock()
    mock_entry.options.get.side_effect = lambda key, default=None: {
        "dhw_morning_hour": 7,
        "dhw_evening_hour": 18,
        "enable_dhw_optimization": False,
    }.get(key, default)
    return mock_entry

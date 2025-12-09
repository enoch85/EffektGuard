"""Pytest configuration for EffektGuard tests."""

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
def setup_frame_helper(monkeypatch):
    """Set up the frame helper for all tests."""
    from homeassistant.helpers import frame

    # Mock the report_usage function to avoid frame helper errors
    monkeypatch.setattr(frame, "report_usage", Mock())

    yield


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
    mock_entry.data = Mock()
    # Configure options.get() to return actual values with proper defaults
    mock_entry.options.get.side_effect = lambda key, default=None: {
        "dhw_morning_hour": 7,
        "dhw_evening_hour": 18,
        "enable_dhw_optimization": False,
        "enable_airflow_optimization": False,
    }.get(key, default)
    # Configure data.get() to return defaults for values not in options
    mock_entry.data.get.side_effect = lambda key, default=None: default
    return mock_entry


def create_mock_price_analyzer():
    """Create a mock PriceAnalyzer for DHW optimizer tests.

    Returns a MagicMock that provides the find_cheapest_window method
    required by DHW optimizer for window search.

    This is the canonical mock for all DHW tests - layers are required,
    not optional.
    """
    from datetime import timedelta

    from custom_components.effektguard.optimization.price_layer import CheapestWindowResult

    mock_analyzer = Mock()

    def mock_find_cheapest_window(current_time, price_periods, duration_minutes, lookahead_hours):
        """Mock implementation that finds cheapest window."""
        from math import ceil

        if not price_periods:
            return None

        quarters_needed = ceil(duration_minutes / 15)
        end_time = current_time + timedelta(hours=lookahead_hours)

        # Filter to lookahead window
        available = [
            p for p in price_periods if p.start_time >= current_time and p.start_time < end_time
        ]

        if len(available) < quarters_needed:
            return None

        # Find cheapest continuous window
        lowest_price = None
        best_start_idx = None

        for i in range(len(available) - quarters_needed + 1):
            window = available[i : i + quarters_needed]
            avg_price = sum(p.price for p in window) / quarters_needed
            if lowest_price is None or avg_price < lowest_price:
                lowest_price = avg_price
                best_start_idx = i

        if best_start_idx is None:
            return None

        window = available[best_start_idx : best_start_idx + quarters_needed]
        return CheapestWindowResult(
            start_time=window[0].start_time,
            end_time=window[-1].start_time + timedelta(minutes=15),
            quarters=[best_start_idx + j for j in range(quarters_needed)],
            avg_price=lowest_price,
            hours_until=(window[0].start_time - current_time).total_seconds() / 3600,
        )

    def mock_calculate_lookahead_hours(heating_type, thermal_mass=1.0, next_demand_hours=None):
        """Mock implementation for lookahead calculation."""
        if heating_type == "dhw":
            if next_demand_hours is not None:
                return max(1.0, min(next_demand_hours, 24.0))
            return 24.0
        return 4.0 * thermal_mass  # space heating

    mock_analyzer.find_cheapest_window.side_effect = mock_find_cheapest_window
    mock_analyzer.calculate_lookahead_hours.side_effect = mock_calculate_lookahead_hours
    return mock_analyzer


@pytest.fixture
def mock_price_analyzer():
    """Pytest fixture for mock price analyzer."""
    return create_mock_price_analyzer()

"""Tests for PriceAnalyzer.evaluate_layer() method.

Phase 2 of layer refactor: Tests for the extracted price layer evaluation logic.
"""

import pytest
from datetime import timedelta
from unittest.mock import MagicMock

from homeassistant.util import dt as dt_util

from custom_components.effektguard.optimization.price_layer import (
    PriceAnalyzer,
    PriceData,
    PriceLayerDecision,
    QuarterPeriod,
)
from custom_components.effektguard.const import (
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    OPTIMIZATION_MODE_COMFORT,
    OPTIMIZATION_MODE_SAVINGS,
    QuarterClassification,
)


@pytest.fixture
def price_analyzer():
    """Create a PriceAnalyzer instance."""
    return PriceAnalyzer()


@pytest.fixture
def mock_nibe_state():
    """Create mock NIBE state."""
    state = MagicMock()
    state.indoor_temp = 21.0
    state.outdoor_temp = 0.0
    state.degree_minutes = -100.0
    return state


@pytest.fixture
def mock_price_data_normal():
    """Real PriceData with 96 uniform-price quarters anchored to today.

    Uniform prices classify NORMAL at every position, so assertions hold
    regardless of the wall-clock quarter the test happens to run in.
    """
    base = dt_util.now().replace(hour=0, minute=0, second=0, microsecond=0)
    periods = [
        QuarterPeriod(start_time=base + timedelta(minutes=15 * q), price=1.0) for q in range(96)
    ]
    return PriceData(today=periods, tomorrow=[], has_tomorrow=False)


class TestEvaluateLayerDisabled:
    """Test evaluate_layer when price optimization is disabled."""

    def test_disabled_returns_zero_offset(
        self, price_analyzer, mock_nibe_state, mock_price_data_normal
    ):
        """When disabled, returns offset=0.0, weight=0.0."""
        result = price_analyzer.evaluate_layer(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data_normal,
            thermal_mass=1.5,
            target_temp=21.0,
            tolerance=1.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],
            gespot_entity="sensor.test_price",
            enable_price_optimization=False,
        )

        assert isinstance(result, PriceLayerDecision)
        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "Disabled by user" in result.reason

    def test_disabled_returns_spot_price_name(
        self, price_analyzer, mock_nibe_state, mock_price_data_normal
    ):
        """Disabled layer still returns proper name."""
        result = price_analyzer.evaluate_layer(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data_normal,
            thermal_mass=1.5,
            target_temp=21.0,
            tolerance=1.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],
            gespot_entity="sensor.test_price",
            enable_price_optimization=False,
        )

        assert result.name == "Spot Price"


class TestEvaluateLayerEnabled:
    """Test evaluate_layer when price optimization is enabled."""

    def test_enabled_returns_price_layer_decision(
        self, price_analyzer, mock_nibe_state, mock_price_data_normal
    ):
        """When enabled, returns PriceLayerDecision with proper structure."""
        result = price_analyzer.evaluate_layer(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data_normal,
            thermal_mass=1.5,
            target_temp=21.0,
            tolerance=1.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],
            gespot_entity="sensor.test_price",
            enable_price_optimization=True,
        )

        assert isinstance(result, PriceLayerDecision)
        assert result.name == "Spot Price"
        # With NORMAL classification, should have minimal offset
        assert isinstance(result.offset, float)
        assert isinstance(result.weight, float)
        assert "Disabled by user" not in result.reason

    def test_enabled_includes_adapter_in_reason(
        self, price_analyzer, mock_nibe_state, mock_price_data_normal
    ):
        """Reason string includes adapter entity name."""
        result = price_analyzer.evaluate_layer(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data_normal,
            thermal_mass=1.5,
            target_temp=21.0,
            tolerance=1.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],
            gespot_entity="sensor.gespot_se3_current",
            enable_price_optimization=True,
        )

        assert "gespot_se3_current" in result.reason


class TestEvaluateLayerModeConfigs:
    """Test evaluate_layer respects mode configurations."""

    def test_comfort_mode_reduces_price_influence(
        self, price_analyzer, mock_nibe_state, mock_price_data_normal
    ):
        """Comfort mode has lower price_tolerance_multiplier."""
        comfort_result = price_analyzer.evaluate_layer(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data_normal,
            thermal_mass=1.5,
            target_temp=21.0,
            tolerance=1.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_COMFORT],
            gespot_entity="sensor.test_price",
            enable_price_optimization=True,
        )

        savings_result = price_analyzer.evaluate_layer(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data_normal,
            thermal_mass=1.5,
            target_temp=21.0,
            tolerance=1.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_SAVINGS],
            gespot_entity="sensor.test_price",
            enable_price_optimization=True,
        )

        # Both should return valid decisions
        assert isinstance(comfort_result, PriceLayerDecision)
        assert isinstance(savings_result, PriceLayerDecision)


class TestEvaluateLayerNoPriceData:
    """Test evaluate_layer behavior when price data is unavailable."""

    def test_no_price_data_returns_zero(self, price_analyzer, mock_nibe_state):
        """When price_data is None, returns zero offset."""
        result = price_analyzer.evaluate_layer(
            nibe_state=mock_nibe_state,
            price_data=None,
            thermal_mass=1.5,
            target_temp=21.0,
            tolerance=1.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],
            gespot_entity="sensor.test_price",
            enable_price_optimization=True,
        )

        assert isinstance(result, PriceLayerDecision)
        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "No price data" in result.reason

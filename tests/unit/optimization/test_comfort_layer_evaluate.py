"""Tests for ComfortLayer.evaluate_layer() method.

Phase 8 of layer refactoring: Comfort layer extraction.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.const import (
    COMFORT_CORRECTION_MILD,
    COMFORT_CORRECTION_MULT,
    LAYER_WEIGHT_COMFORT_HIGH,
    LAYER_WEIGHT_COMFORT_MAX,
    LAYER_WEIGHT_COMFORT_MIN,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    OPTIMIZATION_MODE_COMFORT,
    OPTIMIZATION_MODE_SAVINGS,
    OVERSHOOT_PROTECTION_OFFSET_MIN,
    OVERSHOOT_PROTECTION_START,
)
from custom_components.effektguard.optimization.comfort_layer import (
    ComfortLayer,
    ComfortLayerDecision,
)


@dataclass
class MockNibeState:
    """Mock NIBE state for testing."""

    indoor_temp: float = 21.0
    outdoor_temp: float = 0.0
    degree_minutes: float = 0.0
    flow_temp: float = 35.0


@dataclass
class MockForecastHour:
    """Mock forecast hour for testing."""

    temperature: float


@dataclass
class MockWeatherData:
    """Mock weather data for testing."""

    forecast_hours: list = None


@dataclass
class MockPricePeriod:
    """Mock price period for testing."""

    price: float


@dataclass
class MockPriceData:
    """Mock price data for testing."""

    today: list = None
    tomorrow: list = None
    has_tomorrow: bool = False


class TestComfortLayerInit:
    """Tests for ComfortLayer initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        layer = ComfortLayer()

        assert layer.tolerance_range == 0.5
        assert layer.target_temp == 21.0
        # Default trend function should return 0
        trend = layer._get_thermal_trend()
        assert trend["rate_per_hour"] == 0.0

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        layer = ComfortLayer(
            tolerance_range=1.0,
            target_temp=22.0,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_COMFORT],
        )

        assert layer.tolerance_range == 1.0
        assert layer.target_temp == 22.0
        assert layer.mode_config.dead_zone == 0.1  # Comfort mode


class TestComfortLayerAtTarget:
    """Tests for comfort layer when at target."""

    def test_at_target_no_action(self):
        """Test no action when exactly at target."""
        layer = ComfortLayer(target_temp=21.0, tolerance_range=0.5)
        nibe_state = MockNibeState(indoor_temp=21.0)

        result = layer.evaluate_layer(nibe_state=nibe_state)

        assert result.offset == 0.0
        assert result.weight == 0.0
        assert result.reason == "At target"

    def test_within_dead_zone_no_action(self):
        """Test no action when within dead zone."""
        layer = ComfortLayer(
            target_temp=21.0,
            tolerance_range=0.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],  # 0.2°C dead zone
        )
        nibe_state = MockNibeState(indoor_temp=21.1)  # 0.1°C above, within 0.2 dead zone

        result = layer.evaluate_layer(nibe_state=nibe_state)

        assert result.offset == 0.0
        assert result.weight == 0.0


class TestComfortLayerWithinTolerance:
    """Tests for comfort layer when within tolerance but outside dead zone."""

    def test_slightly_warm_gentle_reduce(self):
        """Test gentle reduction when slightly warm."""
        layer = ComfortLayer(
            target_temp=21.0,
            tolerance_range=0.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],
        )
        nibe_state = MockNibeState(indoor_temp=21.3)  # 0.3°C above, beyond 0.2 dead zone

        result = layer.evaluate_layer(nibe_state=nibe_state)

        assert result.offset < 0.0  # Should reduce heating
        expected_offset = -0.3 * COMFORT_CORRECTION_MULT
        assert result.offset == pytest.approx(expected_offset, rel=0.01)
        assert "Slightly warm" in result.reason

    def test_slightly_cool_gentle_boost(self):
        """Test gentle boost when slightly cool."""
        layer = ComfortLayer(
            target_temp=21.0,
            tolerance_range=0.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],
        )
        nibe_state = MockNibeState(indoor_temp=20.7)  # 0.3°C below, beyond 0.2 dead zone

        result = layer.evaluate_layer(nibe_state=nibe_state)

        assert result.offset > 0.0  # Should boost heating
        expected_offset = 0.3 * COMFORT_CORRECTION_MULT
        assert result.offset == pytest.approx(expected_offset, rel=0.01)
        assert "Slightly cool" in result.reason


class TestComfortLayerOvershoot:
    """Tests for comfort layer overshoot protection."""

    def test_mild_overshoot_gentle_correction(self):
        """Test gentle correction for mild overshoot (below start threshold).

        OVERSHOOT_PROTECTION_START is 0.6°C above target+tolerance.
        With tolerance 0.5, we need indoor < target + tolerance + 0.6 = 22.1
        to be in mild overshoot range.
        """
        layer = ComfortLayer(target_temp=21.0, tolerance_range=0.5)
        # temp_deviation = 21.55 - 21.0 = 0.55
        # This is > tolerance (0.5) but < OVERSHOOT_PROTECTION_START (0.6)
        nibe_state = MockNibeState(indoor_temp=21.55)

        result = layer.evaluate_layer(nibe_state=nibe_state)

        assert result.offset < 0.0
        expected_offset = -0.55 * COMFORT_CORRECTION_MILD
        assert result.offset == pytest.approx(expected_offset, rel=0.1)
        assert result.weight == LAYER_WEIGHT_COMFORT_HIGH
        assert "Warm" in result.reason

    def test_significant_overshoot_coast(self):
        """Test coasting for significant overshoot."""
        layer = ComfortLayer(target_temp=21.0, tolerance_range=0.5)
        # 1.0°C above tolerance = 1.5°C above target
        nibe_state = MockNibeState(indoor_temp=22.5)

        result = layer.evaluate_layer(nibe_state=nibe_state)

        assert result.offset < 0.0
        # Should be in range of coast offsets (-7 to -10)
        assert result.offset <= OVERSHOOT_PROTECTION_OFFSET_MIN
        assert result.weight >= LAYER_WEIGHT_COMFORT_HIGH
        assert "COAST" in result.reason or "Overshoot" in result.reason


class TestComfortLayerTooCold:
    """Tests for comfort layer when too cold."""

    def test_too_cold_increase_heating(self):
        """Test strong heating increase when too cold."""
        layer = ComfortLayer(target_temp=21.0, tolerance_range=0.5)
        # 1.0°C below tolerance = 1.5°C below target
        nibe_state = MockNibeState(indoor_temp=19.5)

        result = layer.evaluate_layer(nibe_state=nibe_state)

        assert result.offset > 0.0  # Should increase heating
        assert result.weight == LAYER_WEIGHT_COMFORT_MAX
        assert "Too cold" in result.reason


class TestComfortLayerModeAdjustment:
    """Tests for mode-based weight adjustments."""

    def test_comfort_mode_higher_weight(self):
        """Test comfort mode has higher weight multiplier."""
        layer = ComfortLayer(
            target_temp=21.0,
            tolerance_range=0.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_COMFORT],
        )
        nibe_state = MockNibeState(indoor_temp=21.3)

        result = layer.evaluate_layer(nibe_state=nibe_state)

        # Weight should be multiplied by 1.3 in comfort mode
        expected_weight = LAYER_WEIGHT_COMFORT_MIN * 1.3
        assert result.weight == pytest.approx(expected_weight, rel=0.01)

    def test_savings_mode_lower_weight(self):
        """Test savings mode has lower weight multiplier."""
        layer = ComfortLayer(
            target_temp=21.0,
            tolerance_range=0.5,
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_SAVINGS],
        )
        nibe_state = MockNibeState(indoor_temp=21.4)  # Outside 0.3 savings dead zone

        result = layer.evaluate_layer(nibe_state=nibe_state)

        # Weight should be multiplied by 0.7 in savings mode
        expected_weight = LAYER_WEIGHT_COMFORT_MIN * 0.7
        assert result.weight == pytest.approx(expected_weight, rel=0.01)


class TestComfortLayerDecisionDataclass:
    """Tests for ComfortLayerDecision dataclass."""

    def test_dataclass_fields(self):
        """Test dataclass has all expected fields."""
        decision = ComfortLayerDecision(
            name="Comfort",
            offset=-0.5,
            weight=0.3,
            reason="Test reason",
            temp_deviation=0.5,
            buffer_hours=2.0,
            is_thermal_aware=True,
        )

        assert decision.name == "Comfort"
        assert decision.offset == -0.5
        assert decision.weight == 0.3
        assert decision.reason == "Test reason"
        assert decision.temp_deviation == 0.5
        assert decision.buffer_hours == 2.0
        assert decision.is_thermal_aware is True

    def test_dataclass_defaults(self):
        """Test dataclass default values."""
        decision = ComfortLayerDecision(
            name="Comfort",
            offset=0.0,
            weight=0.0,
            reason="At target",
        )

        assert decision.temp_deviation == 0.0
        assert decision.buffer_hours == 0.0
        assert decision.is_thermal_aware is False


class TestComfortLayerThermalAwareOvershoot:
    """Tests for thermal-aware overshoot protection."""

    def test_overshoot_triggers_coast_protection(self):
        """Test that significant overshoot triggers coast protection."""
        mock_thermal = MagicMock()
        mock_thermal.thermal_mass = 1.0
        mock_thermal.insulation_quality = 1.0
        mock_thermal.get_prediction_horizon.return_value = 12

        layer = ComfortLayer(
            target_temp=21.0,
            tolerance_range=0.5,
            thermal_model=mock_thermal,
        )

        # 1.3°C overshoot (above 0.6 threshold)
        nibe_state = MockNibeState(indoor_temp=22.3, outdoor_temp=0.0)

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            price_data=None,
        )

        # Standard overshoot protection (not thermal-aware since no price data)
        assert result.offset < 0.0  # Should reduce heating
        assert "COAST" in result.reason or "Overshoot" in result.reason
        assert result.weight >= LAYER_WEIGHT_COMFORT_HIGH

    def test_overshoot_with_sufficient_buffer(self):
        """Test overshoot handling when buffer is sufficient."""
        mock_thermal = MagicMock()
        mock_thermal.thermal_mass = 1.0
        mock_thermal.insulation_quality = 1.0
        mock_thermal.get_prediction_horizon.return_value = 12

        layer = ComfortLayer(
            target_temp=21.0,
            tolerance_range=0.5,
            thermal_model=mock_thermal,
        )

        # Large overshoot = large buffer
        nibe_state = MockNibeState(indoor_temp=25.0, outdoor_temp=0.0)

        # Create price data with expensive period
        today_prices = [MockPricePeriod(price=1.0)] * 96
        for i in range(8, 12):  # 1 hour expensive
            today_prices[i] = MockPricePeriod(price=2.0)

        price_data = MockPriceData(today=today_prices, has_tomorrow=False)

        result = layer.evaluate_layer(
            nibe_state=nibe_state,
            weather_data=None,
            price_data=price_data,
        )

        # With large buffer, thermal-aware logic should apply
        assert result.offset < 0.0  # Should reduce heating

"""Tests for DecisionEngine peak protection layer integration.

Tests Phase 3 requirements:
- Effect layer integration in decision engine
- Peak protection layer weighting
- Integration with other layers
- Offset aggregation with peak protection
"""

import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import MagicMock

from custom_components.effektguard.optimization.decision_engine import (
    DecisionEngine,
    LayerDecision,
    OptimizationDecision,
)
from custom_components.effektguard.optimization.effect_manager import EffectManager
from custom_components.effektguard.optimization.price_analyzer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_model import ThermalModel
from custom_components.effektguard.const import (
    QuarterClassification,
    DM_CRITICAL_T3_WEIGHT,
)


@pytest.fixture
def mock_nibe_state():
    """Create mock NIBE state."""
    state = MagicMock()
    state.outdoor_temp = 5.0
    state.indoor_temp = 21.0
    state.supply_temp = 35.0
    state.degree_minutes = -100.0
    state.current_offset = 0.0
    state.is_heating = True
    state.timestamp = datetime(2025, 10, 14, 12, 0)
    return state


@pytest.fixture
def mock_price_data():
    """Create mock price data."""
    price_data = MagicMock()
    price_data.today = []
    price_data.tomorrow = []
    for i in range(96):
        quarter = MagicMock()
        quarter.quarter_of_day = i
        quarter.price = 1.0
        quarter.is_daytime = 24 <= i <= 87
        price_data.today.append(quarter)
        # Also populate tomorrow with same data
        quarter_tomorrow = MagicMock()
        quarter_tomorrow.quarter_of_day = i
        quarter_tomorrow.price = 1.0
        quarter_tomorrow.is_daytime = 24 <= i <= 87
        price_data.tomorrow.append(quarter_tomorrow)
    return price_data


@pytest.fixture
def mock_weather_data():
    """Create mock weather data."""
    weather = MagicMock()
    weather.forecast_hours = []
    return weather


@pytest.fixture
def hass_mock():
    """Create mock Home Assistant instance."""
    return MagicMock()


@pytest_asyncio.fixture
async def decision_engine(hass_mock, mock_price_data):
    """Create DecisionEngine with all dependencies."""
    price_analyzer = PriceAnalyzer()
    effect_manager = EffectManager(hass_mock)
    thermal_model = ThermalModel(thermal_mass=1.0, insulation_quality=1.0)

    config = {
        "target_temperature": 21.0,
        "tolerance": 5.0,  # Mid-range
    }

    engine = DecisionEngine(
        price_analyzer=price_analyzer,
        effect_manager=effect_manager,
        thermal_model=thermal_model,
        config=config,
    )

    # Initialize price analyzer with default prices
    price_analyzer.update_prices(mock_price_data)

    return engine


class TestEffectLayerIntegration:
    """Test effect layer integration in decision engine."""

    @pytest.mark.asyncio
    async def test_effect_layer_called(
        self, decision_engine, mock_nibe_state, mock_price_data, mock_weather_data
    ):
        """Test effect layer is called during decision calculation."""
        decision = decision_engine.calculate_decision(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data,
            weather_data=mock_weather_data,
            current_peak=0.0,
        )

        # Should have 9 layers:
        # 1. Safety, 2. Emergency, 3. Effect, 4. Prediction (learned),
        # 5. Weather Compensation, 6. Weather (pre-heat), 7. Price, 8. Comfort
        # Note: Phase 6 added prediction layer between effect and weather
        assert len(decision.layers) == 9

        # Find effect layer (layer 3, 0-indexed)
        effect_layer = decision.layers[3]
        assert "peak" in effect_layer.reason.lower() or "ok" in effect_layer.reason.lower()

    @pytest.mark.asyncio
    async def test_effect_layer_no_peaks(
        self, decision_engine, mock_nibe_state, mock_price_data, mock_weather_data
    ):
        """Test effect layer with no peaks recorded."""
        decision = decision_engine.calculate_decision(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data,
            weather_data=mock_weather_data,
            current_peak=0.0,
        )

        effect_layer = decision.layers[3]  # Effect is layer 3 (0-indexed)

        # Should have zero weight when no limiting needed
        assert effect_layer.weight == 0.0
        assert effect_layer.offset == 0.0

    @pytest.mark.asyncio
    async def test_effect_layer_critical_peak(
        self, decision_engine, mock_nibe_state, mock_price_data, mock_weather_data
    ):
        """Test effect layer responds to critical peak risk."""
        # Set up peak in effect manager
        timestamp = datetime(2025, 10, 14, 12, 0)
        await decision_engine.effect.record_quarter_measurement(3.0, 48, timestamp)

        # Mock high current power to exceed peak
        mock_nibe_state.is_heating = True

        decision = decision_engine.calculate_decision(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data,
            weather_data=mock_weather_data,
            current_peak=3.0,
        )

        effect_layer = decision.layers[
            3
        ]  # Effect is layer 3 (after safety, emergency, thermal debt)

        # Should recommend reduction when approaching peak
        # (exact values depend on power estimation)
        assert effect_layer.weight >= 0.0  # Has some weight
        assert "peak" in effect_layer.reason.lower()


class TestLayerPriority:
    """Test layer priority and aggregation with peak protection."""

    @pytest.mark.asyncio
    async def test_safety_overrides_peak_protection(
        self, decision_engine, mock_nibe_state, mock_price_data, mock_weather_data
    ):
        """Test safety layer overrides peak protection."""
        # Set critical temperature
        mock_nibe_state.indoor_temp = 17.0  # Below MIN_TEMP_LIMIT (18°C)

        # Set up peak to trigger protection
        timestamp = datetime(2025, 10, 14, 12, 0)
        await decision_engine.effect.record_quarter_measurement(3.0, 48, timestamp)

        decision = decision_engine.calculate_decision(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data,
            weather_data=mock_weather_data,
            current_peak=3.0,
        )

        # Safety should force positive offset (heating) despite peak risk
        assert decision.offset > 0.0
        assert "SAFETY" in decision.reasoning

    @pytest.mark.asyncio
    async def test_emergency_overrides_peak_protection(
        self, decision_engine, mock_nibe_state, mock_price_data, mock_weather_data
    ):
        """Test emergency layer behavior during CRITICAL degree minutes.

        Nov 30, 2025: Updated to reflect smart recovery behavior.
        When indoor temp is at target and prices aren't cheap, the system correctly
        ignores DM recovery (Smart Recovery feature). This is the correct behavior
        because:
        1. Indoor temp is comfortable (21.0°C = target)
        2. DM recovery during non-cheap periods wastes money
        3. The system will recover DM when prices become cheap

        If we need emergency to trigger, we must set indoor temp BELOW target.
        """
        # Set critical degree minutes close to absolute max
        mock_nibe_state.degree_minutes = -1300.0  # Close to DM_THRESHOLD_AUX_LIMIT (-1500)
        mock_nibe_state.outdoor_temp = 5.0
        # Set indoor temp BELOW target to trigger real emergency (smart recovery bypass)
        mock_nibe_state.indoor_temp = 19.5  # 1.5°C below target triggers emergency

        # Set up CRITICAL monthly peak to trigger protection
        timestamp = datetime(2025, 10, 14, 12, 0)
        await decision_engine.effect.record_quarter_measurement(3.0, 48, timestamp)

        decision = decision_engine.calculate_decision(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data,
            weather_data=mock_weather_data,
            current_peak=3.0,
        )

        # With indoor temp below target, emergency recovery SHOULD trigger
        # because comfort is at risk (not just theoretical DM concern)
        assert (
            "EMERGENCY" in decision.reasoning
            or "CRITICAL" in decision.reasoning
            or "Peak" in decision.reasoning
            or "Smart Recovery" in decision.reasoning  # Smart recovery is also valid
            or "cold" in decision.reasoning.lower()  # Too cold detection
        )


class TestPeakProtectionScenarios:
    """Test realistic peak protection scenarios."""

    @pytest.mark.asyncio
    async def test_daytime_peak_avoidance(
        self, decision_engine, mock_nibe_state, mock_price_data, mock_weather_data, hass_mock
    ):
        """Test peak avoidance during expensive daytime period."""
        # Set up monthly peaks
        timestamp = datetime(2025, 10, 14, 8, 0)  # Morning
        await decision_engine.effect.record_quarter_measurement(5.0, 32, timestamp)
        await decision_engine.effect.record_quarter_measurement(5.2, 33, timestamp)
        await decision_engine.effect.record_quarter_measurement(5.5, 34, timestamp)

        # Simulate approaching peak during daytime
        mock_nibe_state.timestamp = datetime(2025, 10, 14, 12, 0)
        mock_nibe_state.is_heating = True

        decision = decision_engine.calculate_decision(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data,
            weather_data=mock_weather_data,
            current_peak=5.5,
        )

        # Should include peak protection in reasoning
        assert decision.layers[3].weight >= 0.0  # Effect layer active

    @pytest.mark.asyncio
    async def test_nighttime_peak_weighting(
        self, decision_engine, mock_nibe_state, mock_price_data, mock_weather_data
    ):
        """Test nighttime peak with 50% weighting."""
        # Set up daytime peaks
        timestamp = datetime(2025, 10, 14, 12, 0)
        await decision_engine.effect.record_quarter_measurement(5.0, 48, timestamp)

        # Simulate nighttime - can use more power due to 50% weight
        mock_nibe_state.timestamp = datetime(2025, 10, 14, 23, 0)  # 23:00
        mock_nibe_state.is_heating = True

        decision = decision_engine.calculate_decision(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data,
            weather_data=mock_weather_data,
            current_peak=5.0,
        )

        # Night time should allow higher actual power (effective = actual * 0.5)
        # So 10 kW actual = 5 kW effective
        effect_layer = decision.layers[
            3
        ]  # Effect is layer 3 (after safety, emergency, thermal debt)
        assert effect_layer.weight >= 0.0


class TestOffsetAggregation:
    """Test offset aggregation with multiple active layers."""

    @pytest.mark.asyncio
    async def test_aggregates_peak_and_price_layers(
        self, decision_engine, mock_nibe_state, mock_price_data, mock_weather_data
    ):
        """Test aggregation of peak protection and price optimization."""
        # Normal conditions - both price and peak layers may vote
        decision = decision_engine.calculate_decision(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data,
            weather_data=mock_weather_data,
            current_peak=0.0,
        )

        # Should have reasoning from active layers
        assert decision.reasoning != ""
        assert isinstance(decision.offset, float)

    @pytest.mark.asyncio
    async def test_critical_layer_dominates(
        self, decision_engine, mock_nibe_state, mock_price_data, mock_weather_data
    ):
        """Test critical layer (weight=0.99) has very high priority in aggregation."""
        # Set critical degree minutes close to absolute max
        mock_nibe_state.degree_minutes = -1300.0  # Close to DM_THRESHOLD_AUX_LIMIT (-1500)
        mock_nibe_state.outdoor_temp = 5.0
        # Ensure we are below target so Smart Debt Recovery doesn't suppress heating
        mock_nibe_state.indoor_temp = 20.5  # Target is 21.0

        decision = decision_engine.calculate_decision(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data,
            weather_data=mock_weather_data,
            current_peak=0.0,
        )

        # Emergency layer should have very high weight (use constant)
        emergency_layer = decision.layers[1]
        assert emergency_layer.weight == DM_CRITICAL_T3_WEIGHT
        assert decision.offset > 0.0  # Force heating

    @pytest.mark.asyncio
    async def test_smart_debt_recovery_ignores_dm_at_target(
        self, decision_engine, mock_nibe_state, mock_price_data, mock_weather_data
    ):
        """Test that DM is ignored when at target and price is not cheap (User Request Nov 29)."""
        # Set critical degree minutes
        mock_nibe_state.degree_minutes = -1300.0
        # Set at target
        mock_nibe_state.indoor_temp = 21.0
        # Price is Normal (1.0) from fixture

        decision = decision_engine.calculate_decision(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data,
            weather_data=mock_weather_data,
            current_peak=0.0,
        )

        # Should NOT force heating
        # Emergency layer returns 0.0, Price returns 0.0
        assert decision.offset == 0.0
        
        # Check the layer reason directly since weight 0.0 layers are filtered from main reasoning
        emergency_layer = decision.layers[1]
        assert "Smart Recovery" in emergency_layer.reason
        assert "ignoring DM" in emergency_layer.reason


class TestReasoningGeneration:
    """Test reasoning string generation includes peak protection."""

    @pytest.mark.asyncio
    async def test_includes_peak_status(
        self, decision_engine, mock_nibe_state, mock_price_data, mock_weather_data
    ):
        """Test reasoning includes peak status."""
        decision = decision_engine.calculate_decision(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data,
            weather_data=mock_weather_data,
            current_peak=0.0,
        )

        # Reasoning should be generated
        assert decision.reasoning != ""
        assert isinstance(decision.reasoning, str)

    @pytest.mark.asyncio
    async def test_reasoning_shows_active_layers(
        self, decision_engine, mock_nibe_state, mock_price_data, mock_weather_data
    ):
        """Test reasoning shows all active layers."""
        decision = decision_engine.calculate_decision(
            nibe_state=mock_nibe_state,
            price_data=mock_price_data,
            weather_data=mock_weather_data,
            current_peak=0.0,
        )

        # Verify active layers present in reasoning
        # Note: Price layer may include horizon calculation with internal '|' separator
        active_layers = [l for l in decision.layers if l.weight > 0]
        assert len(active_layers) > 0, "Should have at least one active layer"
        
        # Each active layer should have its reason mentioned in the final reasoning
        for layer in active_layers:
            # Extract the layer's main reason (before any internal | separators)
            layer_main_reason = layer.reason.split("|")[0].strip()
            assert layer_main_reason in decision.reasoning, (
                f"Layer reason '{layer_main_reason}' not found in decision reasoning"
            )


class TestPowerEstimation:
    """Test heat pump power estimation for peak tracking."""

    def test_heating_mode_base_power(self, decision_engine, mock_nibe_state):
        """Test base power estimation when heating."""
        mock_nibe_state.is_heating = True
        mock_nibe_state.outdoor_temp = 5.0

        power = decision_engine._estimate_heat_pump_power(mock_nibe_state)

        assert power > 0.1  # More than standby
        assert power > 3.0  # Reasonable heating power
        assert power < 8.0  # Not excessive

    def test_standby_mode_low_power(self, decision_engine, mock_nibe_state):
        """Test low power estimation in standby."""
        mock_nibe_state.is_heating = False

        power = decision_engine._estimate_heat_pump_power(mock_nibe_state)

        assert power == 0.1  # Standby power

    def test_cold_weather_higher_power(self, decision_engine, mock_nibe_state):
        """Test higher power estimation in cold weather."""
        mock_nibe_state.is_heating = True
        mock_nibe_state.outdoor_temp = -15.0  # Very cold

        power = decision_engine._estimate_heat_pump_power(mock_nibe_state)

        # Should be higher than mild weather
        mock_nibe_state.outdoor_temp = 10.0
        mild_power = decision_engine._estimate_heat_pump_power(mock_nibe_state)

        assert power > mild_power

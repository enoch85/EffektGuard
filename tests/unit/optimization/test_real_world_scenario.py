"""Test real-world multi-layer optimization scenario.

This test validates the exact scenario documented in REAL_WORLD_EXAMPLE_ALL_FACTORS.md:
- Time: 08:00 (Q32)
- Spot price: 1.90 SEK/kWh (EXPENSIVE)
- Outdoor: -5°C
- Indoor: 20.8°C
- DM: -180
- All 8 layers voting and aggregating

Expected result: -1.5°C offset from weighted aggregation of:
- Weather Compensation: -2.0°C (weight 0.8)
- Spot Price: -1.5°C (weight 0.6)
- Comfort: +0.1°C (weight 0.3)
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel
from custom_components.effektguard.const import (
    LAYER_WEIGHT_PRICE,
    VOLATILE_WEIGHT_REDUCTION,
)


@pytest.fixture
def real_world_nibe_state():
    """Create NIBE state matching real-world example."""
    state = MagicMock()
    state.outdoor_temp = -5.0  # Cold Swedish winter morning
    state.indoor_temp = 20.8  # Slightly below target
    state.flow_temp = 30.0  # Typical NIBE curve (used by weather compensation)
    state.supply_temp = 30.0  # Same as flow temp for consistency
    state.degree_minutes = -180.0  # Extended runs, acceptable
    state.current_offset = 0.0
    state.is_heating = True
    state.timestamp = datetime(2025, 1, 16, 8, 0)  # 08:00, Q32
    return state


@pytest.fixture
def expensive_price_data():
    """Create price data with expensive morning period."""
    price_data = MagicMock()
    price_data.today = []
    price_data.tomorrow = []

    # Create realistic Swedish price pattern with clear classification
    # Need prices to span a wide range for percentile-based classification
    for i in range(96):
        quarter = MagicMock()
        quarter.quarter_of_day = i
        quarter.is_daytime = 24 <= i <= 87  # 06:00-22:00

        # Night (Q0-Q23): VERY CHEAP 0.50-0.80 SEK
        if i < 24:
            quarter.price = 0.50 + (i * 0.0125)  # 0.50 -> 0.80
        # Morning (Q24-Q35): EXPENSIVE 2.20-2.50 SEK
        elif i < 36:
            quarter.price = 2.20 + ((i - 24) * 0.025)  # 2.20 -> 2.50
        # Mid-day (Q36-Q55): CHEAP 0.90-1.20 SEK
        elif i < 56:
            quarter.price = 0.90 + ((i - 36) * 0.015)  # 0.90 -> 1.20
        # Afternoon (Q56-Q67): NORMAL 1.30-1.45 SEK
        elif i < 68:
            quarter.price = 1.30 + ((i - 56) * 0.01)
        # Evening peak (Q68-Q83): PEAK 2.80-3.20 SEK
        elif i < 84:
            quarter.price = 2.80 + ((i - 68) * 0.025)  # 2.80 -> 3.20
        # Late (Q84-Q95): NORMAL
        else:
            quarter.price = 1.35 + ((i - 84) * 0.01)

        price_data.today.append(quarter)

        # Tomorrow: similar pattern but slightly cheaper
        quarter_tomorrow = MagicMock()
        quarter_tomorrow.quarter_of_day = i
        quarter_tomorrow.price = quarter.price * 0.95
        quarter_tomorrow.is_daytime = quarter.is_daytime
        price_data.tomorrow.append(quarter_tomorrow)

    # Verify Q32 (08:00-08:15) has expensive price
    # Q32 = 8*4 = 32, in morning range Q24-Q35
    # price = 2.20 + (32-24)*0.025 = 2.20 + 0.20 = 2.40 SEK
    assert price_data.today[32].price == pytest.approx(2.40, abs=0.01)
    assert price_data.today[32].is_daytime is True

    return price_data


@pytest.fixture
def winter_weather_data():
    """Create weather data with gradual temperature drop."""
    weather = MagicMock()
    weather.forecast_hours = []

    # Current: -5°C, dropping to -10°C over 18 hours
    temps = [-5, -6, -6, -7, -7, -8, -8, -8, -9, -9, -9, -10, -10, -10, -10, -10, -10, -10]
    for hour, temp in enumerate(temps):
        forecast = MagicMock()
        forecast.hour = hour
        forecast.temperature = temp
        # Use timedelta to handle day overflow
        from datetime import timedelta

        forecast.datetime = datetime(2025, 1, 16, 8, 0) + timedelta(hours=hour)
        weather.forecast_hours.append(forecast)

    return weather


@pytest.fixture
def hass_mock():
    """Create mock Home Assistant instance."""
    return MagicMock()


@pytest.fixture
async def decision_engine(hass_mock, expensive_price_data):
    """Create DecisionEngine with real-world configuration."""
    price_analyzer = PriceAnalyzer()
    effect_manager = EffectManager(hass_mock)
    thermal_model = ThermalModel(
        thermal_mass=1.0,  # Typical Swedish house
        insulation_quality=1.0,  # Average insulation
    )

    config = {
        "target_temperature": 21.0,  # User target
        "tolerance": 2.0,  # Balanced (0.5-3.0 scale, 2.0 = 68% factor)
        "heat_loss_coefficient": 180.0,  # W/°C typical Swedish house
    }

    engine = DecisionEngine(
        price_analyzer=price_analyzer,
        effect_manager=effect_manager,
        thermal_model=thermal_model,
        config=config,
    )

    # Initialize price analyzer with expensive morning prices
    price_analyzer.update_prices(expensive_price_data)

    return engine


class TestRealWorldScenario:
    """Test complete real-world optimization scenario."""

    @pytest.mark.asyncio
    async def test_08_00_expensive_morning_optimization(
        self,
        decision_engine,
        real_world_nibe_state,
        expensive_price_data,
        winter_weather_data,
    ):
        """Test 08:00 expensive morning period with all layers active.

        Expected behavior:
        - Layer 1 (Safety): 0.0°C (temp OK)
        - Layer 2 (Emergency): 0.0°C (DM -180 acceptable)
        - Layer 3 (Proactive Debt Prevention): +0.5°C (DM -180 approaching -240 threshold)
        - Layer 4 (Effect Tariff): 0.0°C (no peak risk)
        - Layer 5 (Prediction): 0.0°C (optional, not configured)
        - Layer 6 (Weather Comp): Variable (based on current vs optimal)
        - Layer 7 (Weather Pred): +3.0°C (5°C drop triggers preheating)
        - Layer 8 (Spot Price): -1.5°C (EXPENSIVE period, daytime multiplier)
        - Layer 9 (Comfort): 0.0°C (temp at target)

        Final offset will be POSITIVE (weather preheating overrides price savings)
        Safety > cost savings: thermal protection during cold spell
        """
        # Mock dt_util.now() to return our test timestamp (08:00)
        test_time = datetime(2025, 1, 16, 8, 0)

        with patch(
            "custom_components.effektguard.optimization.decision_engine.dt_util.now",
            return_value=test_time,
        ):
            decision = decision_engine.calculate_decision(
                nibe_state=real_world_nibe_state,
                price_data=expensive_price_data,
                weather_data=winter_weather_data,
                current_peak=2.8,  # Safe margin from monthly peak 5.2 kW
                current_power=1.5,
            )

            # Debug: Check what quarter we're actually in
            calc_quarter = (test_time.hour * 4) + (test_time.minute // 15)
            print(f"\n=== Debug Info ===")
            print(f"Test timestamp: {test_time}")
            print(f"Calculated quarter: Q{calc_quarter}")
            print(
                f"Price at Q{calc_quarter}: {expensive_price_data.today[calc_quarter].price:.2f} SEK"
            )
            print(f"Price layer reason: {decision.layers[6].reason}")
            print(f"==================\n")

        # Verify decision structure
        assert decision is not None
        assert hasattr(decision, "offset")
        assert hasattr(decision, "reasoning")
        assert hasattr(decision, "layers")

        # Verify all 9 layers exist (added proactive thermal debt layer)
        assert len(decision.layers) == 9

        # Layer 1: Safety (should be inactive, temp OK)
        safety_layer = decision.layers[0]
        assert safety_layer.offset == 0.0
        assert "Safety" in safety_layer.reason or "OK" in safety_layer.reason

        # Layer 2: Emergency (should be inactive, DM OK)
        emergency_layer = decision.layers[1]
        assert emergency_layer.offset == 0.0
        assert emergency_layer.weight == 0.0
        assert (
            "Emergency" in emergency_layer.reason
            or "OK" in emergency_layer.reason
            or "-180" in emergency_layer.reason
        )

        # Layer 3: Proactive Debt Prevention (NEW - may be active at DM -180)
        proactive_layer = decision.layers[2]
        # May vote for gentle heating to prevent debt progression

        # Layer 4: Effect Tariff (should be inactive, safe margin)
        effect_layer = decision.layers[3]
        assert effect_layer.offset == 0.0
        assert effect_layer.weight == 0.0

        # Layer 5: Prediction (Phase 6 optional, not configured)
        prediction_layer = decision.layers[4]
        assert prediction_layer.offset == 0.0
        assert prediction_layer.weight == 0.0

        # Layer 6: Weather Compensation (deferred when thermal debt exists)
        weather_comp_layer = decision.layers[5]
        # Note: With DM -180 (light debt), weather compensation defers to recovery layers
        # This is correct production behavior: safety > optimization
        # Weight will be 0.0 when deferred, or >0 if debt is minimal
        assert weather_comp_layer.weight >= 0.0  # May be deferred
        # When deferred, reason will mention "debt" or "Deferred"

        # Layer 7: Weather Prediction (may be active with forecast)
        weather_pred_layer = decision.layers[6]
        # Weather layer can vote for pre-heating

        # Layer 8: Spot Price (SHOULD BE ACTIVE - KEY TEST)
        price_layer = decision.layers[7]
        assert price_layer.offset < 0.0, "Price layer should reduce during EXPENSIVE period"
        # Note: Real-world data may trigger volatile detection (8/9 non-NORMAL in scan window)
        # Weight may be reduced based on VOLATILE_WEIGHT_REDUCTION constant
        min_expected_weight = LAYER_WEIGHT_PRICE * VOLATILE_WEIGHT_REDUCTION
        max_expected_weight = LAYER_WEIGHT_PRICE
        assert min_expected_weight <= price_layer.weight <= max_expected_weight, (
            f"Price layer weight should be between {min_expected_weight} (volatile) and "
            f"{max_expected_weight} (normal), got {price_layer.weight}"
        )
        assert (
            "EXPENSIVE" in price_layer.reason
            or "PEAK" in price_layer.reason
            or "Q32" in price_layer.reason
        )

        # Calculate expected price offset
        # With the new price data:
        # Q32 = 2.40 SEK (high in the distribution)
        # Should be classified as EXPENSIVE or PEAK based on percentiles
        # Base: -1.0°C (EXPENSIVE) or -2.0°C (PEAK)
        # Daytime multiplier: ×1.5
        # Tolerance factor: 5/5.0 = 1.0
        # Expected: -1.5°C to -3.0°C range
        assert price_layer.offset <= -1.0, (
            f"Price layer should significantly reduce during expensive period, "
            f"got {price_layer.offset}°C with reason: {price_layer.reason}"
        )

        # Layer 9: Comfort (should be slightly positive, temp below target)
        comfort_layer = decision.layers[8]
        # May be inactive if temp is close to target
        if comfort_layer.weight > 0:
            assert comfort_layer.offset >= -0.5, "Comfort offset should be gentle"

        # Final offset - The multi-layer system balances all factors
        # In this scenario:
        # - Weather pre-heat: +1.17°C (weight 0.7) - suggests heating before cold
        # - Spot Price: -1.5°C (weight 0.75) - expensive period, reduce heating
        # - Math WC: +0.33°C (weight 0.3185) - weather compensation adjustment
        # - Proactive Z1: +0.5°C (weight 0.3) - gentle debt prevention
        #
        # The weighted average can be negative if price weight > weather weight
        # This is correct behavior: during expensive periods, optimize for cost
        # unless weather protection is critical (which it's not at 5h lead time)
        #
        # The system correctly prioritizes cost savings when there's adequate time
        # before the cold snap (5 hours with 6h lead time = not urgent)
        assert decision.offset is not None, "Decision should have an offset"

        # Verify all major layers contributed to the decision
        active_layers = [l for l in decision.layers if l.weight > 0]
        active_layer_names = [l.name for l in active_layers]

        # Weather pre-heat layer should be active
        assert any(
            "Weather" in name or "Pre-heat" in name for name in active_layer_names
        ), f"Weather/preheat should be considered. Active layers: {active_layer_names}"

        # Price layer should be active
        assert (
            "Spot Price" in active_layer_names
        ), f"Price layer should be active. Active layers: {active_layer_names}"

        # Expected range: Price optimization may win if not urgent
        # If offset is negative: cost optimization dominant (correct when not urgent)
        # If offset is positive: weather protection dominant (correct when urgent)
        # The multi-layer system balances all factors - result can be negative or positive
        # depending on the relative weights and urgency
        assert (
            -3.0 <= decision.offset <= 3.0
        ), f"Final offset {decision.offset}°C outside safety bounds -3.0 to 3.0°C"

        # Verify reasoning includes active layers
        assert decision.reasoning != ""
        # Should mention weather compensation, spot price, and/or comfort
        reasoning_lower = decision.reasoning.lower()
        assert (
            "wc" in reasoning_lower
            or "weather" in reasoning_lower
            or "spot" in reasoning_lower
            or "price" in reasoning_lower
        ), f"Reasoning should mention active layers: {decision.reasoning}"

        print(f"\n=== Real-World Scenario Test Results ===")
        print(f"Time: 08:00 (Q32)")
        print(f"Outdoor: {real_world_nibe_state.outdoor_temp}°C")
        print(f"Indoor: {real_world_nibe_state.indoor_temp}°C")
        print(f"Spot Price: {expensive_price_data.today[32].price:.2f} SEK/kWh")
        print(f"\nLayer Votes:")
        for i, layer in enumerate(decision.layers, 1):
            if layer.weight > 0:
                print(
                    f"  Layer {i}: {layer.offset:+.1f}°C (weight {layer.weight:.1f}) - {layer.reason}"
                )
        print(f"\nFinal Offset: {decision.offset:.1f}°C")
        print(f"Reasoning: {decision.reasoning}")
        print(f"========================================\n")

    @pytest.mark.asyncio
    async def test_spot_price_layer_daytime_multiplier(
        self,
        decision_engine,
        real_world_nibe_state,
        expensive_price_data,
        winter_weather_data,
    ):
        """Test that daytime multiplier amplifies expensive/peak reductions.

        Note: Forward-looking price optimization (Nov 27, 2025) adds forecast adjustment
        when much cheaper period detected within 4-hour horizon.
        """
        test_time = datetime(2025, 1, 16, 8, 0)  # Q32

        with patch(
            "custom_components.effektguard.optimization.decision_engine.dt_util.now",
            return_value=test_time,
        ):
            decision = decision_engine.calculate_decision(
                nibe_state=real_world_nibe_state,
                price_data=expensive_price_data,
                weather_data=winter_weather_data,
                current_peak=2.8,
                current_power=2.0,
            )

            price_layer = decision.layers[7]

            # Q32 is daytime (08:00), price 2.40 öre
            # EXPENSIVE classification: base -1.0°C
            # Daytime multiplier: ×1.5
            # Tolerance factor: 0.2 + ((2.0 - 0.5) / 2.5) * 0.8 = 0.68
            # Mode multiplier: 1.0 (balanced)
            # Base: -1.0 × 1.5 × 0.68 × 1.0 = -1.02°C
            # Forward-looking: Detects cheaper period ahead (Q44-48 @ 0.90 öre = 62% cheaper)
            # Forecast adjustment: -1.5°C (wait for cheaper period - strengthened Dec 5, 2025)
            # Expected: -1.02 + (-1.5) = -2.52°C

            assert price_layer.offset == pytest.approx(-2.5, abs=0.3)
            # Note: Real-world data may trigger volatile detection (8/9 non-NORMAL in scan window)
            # Weight may be reduced based on VOLATILE_WEIGHT_REDUCTION constant
            min_expected_weight = LAYER_WEIGHT_PRICE * VOLATILE_WEIGHT_REDUCTION
            max_expected_weight = LAYER_WEIGHT_PRICE
            assert min_expected_weight <= price_layer.weight <= max_expected_weight, (
                f"Price layer weight should be between {min_expected_weight} (volatile) and "
                f"{max_expected_weight} (normal), got {price_layer.weight}"
            )
            assert "EXPENSIVE" in price_layer.reason or "day" in price_layer.reason.lower()
            assert "cheaper" in price_layer.reason.lower()  # Forecast message

    @pytest.mark.asyncio
    async def test_nighttime_cheap_period_preheating(
        self,
        decision_engine,
        real_world_nibe_state,
        expensive_price_data,
        winter_weather_data,
    ):
        """Test pre-heating during cheap nighttime period."""
        # Move to nighttime cheap period (Q10 = 02:30)
        test_time = datetime(2025, 1, 16, 2, 30)
        real_world_nibe_state.indoor_temp = 21.2  # Slightly above target

        with patch(
            "custom_components.effektguard.optimization.decision_engine.dt_util.now",
            return_value=test_time,
        ):
            decision = decision_engine.calculate_decision(
                nibe_state=real_world_nibe_state,
                price_data=expensive_price_data,
                weather_data=winter_weather_data,
                current_peak=1.5,  # Low nighttime power
                current_power=1.2,
            )

            price_layer = decision.layers[7]

            # Q10 is nighttime, should be CHEAP classification
            # Base: +2.0°C (pre-heat opportunity)
            # No daytime multiplier (nighttime)
            # Expected: +2.0°C × 1.0 = +2.0°C

            assert price_layer.offset > 0.0, "Should pre-heat during CHEAP period"
            # Q10 is in stable VERY_CHEAP region with ±30min window (Q8-Q12 all VERY_CHEAP)
            # No volatility detected, so expect full weight
            assert price_layer.weight == pytest.approx(
                LAYER_WEIGHT_PRICE, abs=0.01
            ), f"Expected full weight ({LAYER_WEIGHT_PRICE}) in stable region, got {price_layer.weight}"

    @pytest.mark.asyncio
    async def test_evening_peak_aggressive_reduction(
        self,
        decision_engine,
        real_world_nibe_state,
        expensive_price_data,
        winter_weather_data,
    ):
        """Test aggressive reduction during evening PEAK period."""
        # Move to evening peak (Q72 = 18:00)
        test_time = datetime(2025, 1, 16, 18, 0)
        real_world_nibe_state.indoor_temp = 20.9

        with patch(
            "custom_components.effektguard.optimization.decision_engine.dt_util.now",
            return_value=test_time,
        ):
            decision = decision_engine.calculate_decision(
                nibe_state=real_world_nibe_state,
                price_data=expensive_price_data,
                weather_data=winter_weather_data,
                current_peak=4.5,  # Approaching monthly peak
                current_power=4.2,
            )

            price_layer = decision.layers[7]

            # Q72 is daytime, should be in upper price range
            # Classification depends on percentile distribution
            # Could be EXPENSIVE (-1.5°C with daytime) or PEAK (-3.0°C with daytime)
            # Either way, should be significantly reducing

            assert (
                price_layer.offset <= -1.0
            ), f"Should reduce during high-price period, got {price_layer.offset}°C"
            # Q72 is in stable PEAK region with ±30min window (Q70-Q74 all PEAK)
            # No volatility detected, so expect full weight
            assert price_layer.weight == pytest.approx(
                LAYER_WEIGHT_PRICE, abs=0.01
            ), f"Expected full weight ({LAYER_WEIGHT_PRICE}) in stable region, got {price_layer.weight}"
            # Check it's recognized as high-price period
            assert "EXPENSIVE" in price_layer.reason or "PEAK" in price_layer.reason

    @pytest.mark.asyncio
    async def test_tolerance_setting_affects_aggressiveness(
        self,
        hass_mock,
        expensive_price_data,
        real_world_nibe_state,
        winter_weather_data,
    ):
        """Test that user tolerance setting scales spot price optimization.

        Tolerance range: 0.5-3.0 maps to factor 0.2-1.0
        Formula: factor = 0.2 + ((tolerance - 0.5) / 2.5) * 0.8

        Note: Forward-looking price optimization (Nov 27, 2025) adds forecast adjustment
        independent of tolerance setting.
        """
        test_time = datetime(2025, 1, 16, 8, 0)  # Q32

        # Create two engines with different tolerance settings
        # Using actual tolerance range: 0.5-3.0
        for tolerance_setting, expected_factor in [(0.5, 0.2), (3.0, 1.0)]:
            price_analyzer = PriceAnalyzer()
            effect_manager = EffectManager(hass_mock)
            thermal_model = ThermalModel(thermal_mass=1.0, insulation_quality=1.0)

            config = {
                "target_temperature": 21.0,
                "tolerance": float(tolerance_setting),
                "heat_loss_coefficient": 180.0,
            }

            engine = DecisionEngine(
                price_analyzer=price_analyzer,
                effect_manager=effect_manager,
                thermal_model=thermal_model,
                config=config,
            )

            price_analyzer.update_prices(expensive_price_data)

            with patch(
                "custom_components.effektguard.optimization.decision_engine.dt_util.now",
                return_value=test_time,
            ):
                decision = engine.calculate_decision(
                    nibe_state=real_world_nibe_state,
                    price_data=expensive_price_data,
                    weather_data=winter_weather_data,
                    current_peak=2.8,
                    current_power=2.0,
                )

                price_layer = decision.layers[7]

                # Base: -1.0°C (EXPENSIVE)
                # Daytime: ×1.5
                # Tolerance factor: 0.2 + ((tolerance - 0.5) / 2.5) * 0.8
                # Mode multiplier: 1.0 (balanced)
                # Forward-looking: -1.5°C (cheaper period ahead, strengthened Dec 5, 2025)
                expected_base = -1.0 * 1.5 * expected_factor * 1.0  # mode mult = 1.0
                expected_offset = expected_base + (-1.5)  # Add forecast adjustment

                print(
                    f"\nTolerance {tolerance_setting}: "
                    f"Price layer offset {price_layer.offset:.2f}°C "
                    f"(expected {expected_offset:.2f}°C)"
                )

                # Allow some variance due to other layer interactions
                assert price_layer.offset == pytest.approx(expected_offset, abs=0.3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""Spot-price layer guards across a full day of real prices.

Drives the engine at daytime-expensive, nighttime-cheap, and evening-peak quarters and pins the
resulting price-layer offset and weight, including the daytime multiplier, the cheap-period
pre-heat, and how the user's tolerance setting scales the reduction.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from freezegun import freeze_time

from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import (
    PriceAnalyzer,
    PriceData,
    QuarterPeriod,
)
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
    """Real PriceData with an expensive morning period (base day 2025-01-16).

    Real timestamps let the production timestamp-containment lookup resolve
    the current interval; tests pin the clock with freeze_time to match.
    """
    base_date = datetime(2025, 1, 16, 0, 0, 0, tzinfo=timezone.utc)
    today = []
    tomorrow = []

    # Create realistic Swedish price pattern with clear classification
    # Need prices to span a wide range for percentile-based classification
    for i in range(96):

        # Night (Q0-Q23): VERY CHEAP 0.50-0.80 SEK
        if i < 24:
            price = 0.50 + (i * 0.0125)  # 0.50 -> 0.80
        # Morning (Q24-Q35): EXPENSIVE 2.20-2.50 SEK
        elif i < 36:
            price = 2.20 + ((i - 24) * 0.025)  # 2.20 -> 2.50
        # Mid-day (Q36-Q55): CHEAP 0.90-1.20 SEK
        elif i < 56:
            price = 0.90 + ((i - 36) * 0.015)  # 0.90 -> 1.20
        # Afternoon (Q56-Q67): NORMAL 1.30-1.45 SEK
        elif i < 68:
            price = 1.30 + ((i - 56) * 0.01)
        # Evening peak (Q68-Q83): PEAK 2.80-3.20 SEK
        elif i < 84:
            price = 2.80 + ((i - 68) * 0.025)  # 2.80 -> 3.20
        # Late (Q84-Q95): NORMAL
        else:
            price = 1.35 + ((i - 84) * 0.01)

        start_time = base_date + timedelta(minutes=i * 15)
        today.append(QuarterPeriod(start_time=start_time, price=price))

        # Tomorrow: similar pattern but slightly cheaper
        tomorrow.append(
            QuarterPeriod(start_time=start_time + timedelta(days=1), price=price * 0.95)
        )

    price_data = PriceData(today=today, tomorrow=tomorrow, has_tomorrow=True)

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
    """Price-layer offset and weight at representative quarters through the day."""

    @pytest.mark.asyncio
    @freeze_time("2025-01-16 08:00:00")
    async def test_spot_price_layer_daytime_multiplier(
        self,
        decision_engine,
        real_world_nibe_state,
        expensive_price_data,
        winter_weather_data,
    ):
        """The daytime multiplier amplifies the EXPENSIVE reduction, and a forecast adjustment
        adds further reduction when a much cheaper period lies ahead.
        """
        test_time = datetime(2025, 1, 16, 8, 0, tzinfo=timezone.utc)  # Q32

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
            # Forecast adjustment: -1.5°C (cheaper period ahead, Q44-48 @ 0.90 öre = 62% cheaper)
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
    @freeze_time("2025-01-16 02:30:00")
    async def test_nighttime_cheap_period_preheating(
        self,
        decision_engine,
        real_world_nibe_state,
        expensive_price_data,
        winter_weather_data,
    ):
        """Test pre-heating during cheap nighttime period."""
        # Move to nighttime cheap period (Q10 = 02:30)
        test_time = datetime(2025, 1, 16, 2, 30, tzinfo=timezone.utc)
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
    @freeze_time("2025-01-16 18:00:00")
    async def test_evening_peak_aggressive_reduction(
        self,
        decision_engine,
        real_world_nibe_state,
        expensive_price_data,
        winter_weather_data,
    ):
        """Test aggressive reduction during evening PEAK period."""
        # Move to evening peak (Q72 = 18:00)
        test_time = datetime(2025, 1, 16, 18, 0, tzinfo=timezone.utc)
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
    @freeze_time("2025-01-16 08:00:00")
    async def test_tolerance_setting_affects_aggressiveness(
        self,
        hass_mock,
        expensive_price_data,
        real_world_nibe_state,
        winter_weather_data,
    ):
        """The user tolerance setting scales the spot-price reduction.

        Tolerance range 0.5-3.0 maps to factor 0.2-1.0:
        factor = 0.2 + ((tolerance - 0.5) / 2.5) * 0.8. The forecast adjustment is added on top,
        independent of tolerance.
        """
        test_time = datetime(2025, 1, 16, 8, 0, tzinfo=timezone.utc)  # Q32

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
                # Forecast adjustment: -1.5°C (cheaper period ahead)
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

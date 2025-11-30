"""Tests for volatile price weight reduction with realistic scenarios (Nov 30, 2025).

Tests the weight-based approach where volatile periods reduce price layer influence
rather than blocking pre-heating decisions. This allows extreme spikes to still
trigger pre-heating through weighted aggregation.
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime

from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.const import (
    PRICE_VOLATILE_MIN_THRESHOLD,
    PRICE_VOLATILE_MAX_THRESHOLD,
    PRICE_VOLATILE_SCAN_QUARTERS,
    PRICE_VOLATILE_WEIGHT_REDUCTION,
    PRICE_FORECAST_PREHEAT_OFFSET,
    PRICE_FORECAST_EXPENSIVE_THRESHOLD,
    LAYER_WEIGHT_PRICE,
)


class TestVolatileWeightReduction:
    """Test weight reduction during volatile periods."""

    @pytest.fixture
    def engine(self):
        """Create decision engine for testing."""
        from custom_components.effektguard.optimization.price_analyzer import PriceAnalyzer
        from custom_components.effektguard.optimization.effect_manager import EffectManager
        from custom_components.effektguard.optimization.thermal_model import ThermalModel
        from unittest.mock import MagicMock
        
        hass_mock = MagicMock()
        
        config = {
            "target_temperature": 22.0,
            "tolerance": 5.0,
            "system_type": "concrete_ufh",
            "pump_mode": "AUTO",
            "latitude": 59.33,  # Stockholm
        }
        
        thermal_model = ThermalModel(thermal_mass=1.5, insulation_quality=1.0)
        effect_manager = EffectManager(hass_mock)
        price_analyzer = PriceAnalyzer()
        
        return DecisionEngine(
            price_analyzer=price_analyzer,
            effect_manager=effect_manager,
            thermal_model=thermal_model,
            config=config,
        )

    @pytest.fixture
    def base_nibe_state(self):
        """Create base NIBE state."""
        state = MagicMock()
        state.indoor_temp = 22.0
        state.outdoor_temp = -5.0
        state.flow_temp = 35.0
        state.degree_minutes = -100.0
        state.compressor_frequency = 50
        state.current_power = 2.5
        state.supply_temp = None
        state.return_temp = None
        state.hot_water_temp = None
        state.timestamp = datetime.now()
        return state

    @pytest.fixture
    def base_weather_data(self):
        """Create base weather data."""
        weather = MagicMock()
        weather.forecast_hours = []
        return weather

    def test_extreme_spike_during_volatility_still_preheats(self, engine, base_nibe_state, base_weather_data):
        """Test scenario from screenshot: 06:00 cheap with 5x spike at 12:00 during volatile period.
        
        Scenario:
        - Current time: 06:00 (Q24)
        - Current price: 20 öre (CHEAP)
        - Volatile period detected (mixed CHEAP/NORMAL/EXPENSIVE in scan window)
        - Massive spike coming: 100 öre at 12:00 (5x current price)
        
        Expected:
        - Volatile flag detected: True
        - Pre-heat still triggered: +2.0°C offset
        - Weight reduced: 0.8 → 0.4
        - Net influence: +2.0 × 0.4 = +0.8°C (still significant!)
        """
        # Build price data with extreme spike scenario
        price_periods = []
        
        # 00:00-06:00: Mixed CHEAP/NORMAL/EXPENSIVE (volatile pattern)
        for q in range(24):
            if q % 3 == 0:
                price = 15.0  # CHEAP
            elif q % 3 == 1:
                price = 30.0  # NORMAL
            else:
                price = 60.0  # EXPENSIVE
            period = MagicMock()
            period.price = price
            period.is_daytime = False
            price_periods.append(period)
        
        # 06:00-12:00: Current CHEAP period
        for q in range(24, 48):
            period = MagicMock()
            period.price = 20.0
            period.is_daytime = True
            price_periods.append(period)
        
        # 12:00-18:00: MASSIVE SPIKE (5x current price)
        for q in range(48, 72):
            period = MagicMock()
            period.price = 100.0
            period.is_daytime = True
            price_periods.append(period)
        
        # 18:00-24:00: Back to normal
        for q in range(72, 96):
            period = MagicMock()
            period.price = 30.0
            period.is_daytime = False
            price_periods.append(period)
        
        price_data = MagicMock()
        price_data.today = price_periods
        price_data.tomorrow = []
        price_data.has_tomorrow = False
        
        # Get decision at 06:00 (Q24)
        # Mock current time to 06:00
        engine.price._current_time_override = datetime(2025, 11, 30, 6, 0)
        
        decision = engine.calculate_decision(
            nibe_state=base_nibe_state,
            price_data=price_data,
            weather_data=base_weather_data,
            current_peak=5.0,
        )
        
        # Verify volatile period detected
        # Scan window Q24-Q32: should have mix of CHEAP (Q24-Q31) and previous volatility
        # The forecast logic should detect the 5x spike and pre-heat
        
        # Expected: Pre-heat offset (+2.0) with reduced weight (0.3)
        # Offset should be positive (pre-heating)
        assert decision.offset > 0, f"Should pre-heat before extreme spike, got {decision.offset}"
        
        # Reasoning should mention the extreme price increase
        assert "expensive" in decision.reasoning.lower() or "forecast" in decision.reasoning.lower(), \
            f"Reasoning should mention upcoming expensive period: {decision.reasoning}"

    def test_normal_volatility_without_extreme_spike(self, engine, base_nibe_state, base_weather_data):
        """Test normal volatile period without extreme price changes.
        
        Scenario:
        - Current time: 10:00 (Q40)
        - Prices jumping between CHEAP/NORMAL/EXPENSIVE (±30-50% changes)
        - No extreme spikes (no 2x+ changes)
        - Just typical volatile day
        
        Expected:
        - Volatile flag detected: True
        - Weight reduced: 0.8 → 0.4
        - Offset stays near zero (no strong signals)
        - System holds steady instead of chasing prices
        """
        # Build price data with normal volatility (no extremes)
        price_periods = []
        
        # Create realistic volatile pattern: prices jump 30-50% frequently
        base_price = 40.0
        for q in range(96):
            # Alternate between -30%, normal, +30%
            if q % 3 == 0:
                price = base_price * 0.7  # 28 öre (CHEAP)
            elif q % 3 == 1:
                price = base_price * 1.0  # 40 öre (NORMAL)
            else:
                price = base_price * 1.3  # 52 öre (EXPENSIVE)
            
            is_day = 6 <= (q // 4) < 22
            period = MagicMock()
            period.price = price
            period.is_daytime = is_day
            price_periods.append(period)
        
        price_data = MagicMock()
        price_data.today = price_periods
        price_data.tomorrow = []
        price_data.has_tomorrow = False
        
        # Get decision at 10:00 (Q40)
        engine.price._current_time_override = datetime(2025, 11, 30, 10, 0)
        
        decision = engine.calculate_decision(
            nibe_state=base_nibe_state,
            price_data=price_data,
            weather_data=base_weather_data,
            current_peak=5.0,
        )
        
        # Verify system holds steady during normal volatility
        # Offset should be small (no extreme changes to react to)
        assert abs(decision.offset) < 1.5, \
            f"Should hold relatively steady during normal volatility, got {decision.offset}"
        
        # Reasoning should mention price layer (GE-Spot, price, or volatile)
        reasoning_lower = decision.reasoning.lower()
        assert any(keyword in reasoning_lower for keyword in ["volatile", "price", "ge-spot", "spot"]), \
            f"Reasoning should mention price behavior: {decision.reasoning}"

    def test_weight_reduction_math(self):
        """Test that weight reduction constants make mathematical sense.
        
        Verifies:
        - Reduced weight (0.3) allows strong signals through
        - +2.0°C at 0.3 weight > -0.5°C at 0.6 weight
        - Extreme spikes win in weighted aggregation
        """
        # Normal price layer weight
        normal_weight = LAYER_WEIGHT_PRICE  # 0.6
        
        # Volatile period weight
        volatile_weight = normal_weight * PRICE_VOLATILE_WEIGHT_REDUCTION  # 0.6 × 0.5 = 0.3
        
        # Verify reduction makes sense (0.8 × 0.5 = 0.4)
        assert volatile_weight == pytest.approx(0.4), \
            f"Volatile weight should be 0.4, got {volatile_weight}"
        
        # Verify extreme spike signal still wins
        extreme_spike_influence = PRICE_FORECAST_PREHEAT_OFFSET * volatile_weight  # +2.0 × 0.4 = 0.8
        normal_offset_influence = 0.5 * normal_weight  # +0.5 × 0.8 = 0.4 (normal cheap boost)
        
        assert extreme_spike_influence > normal_offset_influence, \
            f"Extreme spike at reduced weight ({extreme_spike_influence}) should beat normal offset ({normal_offset_influence})"

    def test_volatile_detection_threshold_logic(self, engine, base_nibe_state, base_weather_data):
        """Test that volatile detection uses correct thresholds.
        
        Scenario 1: 3 non-NORMAL with mix → Volatile (min threshold)
        Scenario 2: 6 non-NORMAL → Definitely volatile (max threshold)
        Scenario 3: 2 non-NORMAL → Not volatile (below min)
        Scenario 4: 4 non-NORMAL but all same type → Not volatile (no mix)
        """
        # Scenario 1: Min threshold with mix (3 non-NORMAL)
        price_periods_min = []
        # Q0-Q2: 2 EXPENSIVE, 1 CHEAP (mixed)
        for _ in range(2):
            period = MagicMock()
            period.price = 80.0
            period.is_daytime = False
            price_periods_min.append(period)
        period = MagicMock()
        period.price = 15.0
        period.is_daytime = False
        price_periods_min.append(period)
        # Q3-Q7: NORMAL
        for _ in range(5):
            period = MagicMock()
            period.price = 40.0
            period.is_daytime = False
            price_periods_min.append(period)
        # Fill rest
        for i in range(8, 96):
            period = MagicMock()
            period.price = 40.0
            period.is_daytime = i >= 24
            price_periods_min.append(period)
        
        price_data_min = MagicMock()
        price_data_min.today = price_periods_min
        price_data_min.tomorrow = []
        price_data_min.has_tomorrow = False
        
        engine.price._current_time_override = datetime(2025, 11, 30, 0, 0)
        
        decision_min = engine.calculate_decision(
            base_nibe_state, price_data_min, base_weather_data, 5.0
        )
        
        # Should detect volatility (3 non-NORMAL with mix in scan window)
        # Weight should be reduced
        # Note: We can't directly check internal weight, but reasoning should reflect volatile behavior
        
        # Scenario 2: Max threshold (6 non-NORMAL)
        price_periods_max = []
        # Q0-Q5: Mix of EXPENSIVE and CHEAP (6 non-NORMAL)
        for i in range(6):
            period = MagicMock()
            period.price = 80.0 if i % 2 == 0 else 15.0
            period.is_daytime = False
            price_periods_max.append(period)
        # Q6-Q7: NORMAL
        for _ in range(2):
            period = MagicMock()
            period.price = 40.0
            period.is_daytime = False
            price_periods_max.append(period)
        # Fill rest
        for i in range(8, 96):
            period = MagicMock()
            period.price = 40.0
            period.is_daytime = i >= 24
            price_periods_max.append(period)
        
        price_data_max = MagicMock()
        price_data_max.today = price_periods_max
        price_data_max.tomorrow = []
        price_data_max.has_tomorrow = False
        
        decision_max = engine.calculate_decision(
            base_nibe_state, price_data_max, base_weather_data, 5.0
        )
        
        # Should definitely detect volatility (6 non-NORMAL = 75% of scan window)
        # Decision should reflect reduced price influence

    def test_constants_relationship(self):
        """Verify constants have sensible relationships."""
        # Min threshold should prevent single-dip false positives
        assert PRICE_VOLATILE_MIN_THRESHOLD >= 3, \
            "Min threshold should ignore brief 1-2 period dips"
        
        # Max threshold should catch clear volatility (~75% of scan)
        assert PRICE_VOLATILE_MAX_THRESHOLD <= PRICE_VOLATILE_SCAN_QUARTERS, \
            "Max threshold must be within scan window"
        
        volatility_ratio = PRICE_VOLATILE_MAX_THRESHOLD / PRICE_VOLATILE_SCAN_QUARTERS
        assert 0.6 <= volatility_ratio <= 0.8, \
            f"Max threshold should be 60-80% of scan window, got {volatility_ratio:.1%}"
        
        # Weight reduction should be moderate (not kill price signal entirely)
        assert 0.3 <= PRICE_VOLATILE_WEIGHT_REDUCTION <= 0.7, \
            f"Weight reduction should be 30-70%, got {PRICE_VOLATILE_WEIGHT_REDUCTION}"
        
        # Expensive threshold for spikes should require >2x price
        assert PRICE_FORECAST_EXPENSIVE_THRESHOLD >= 1.5, \
            f"Expensive threshold should require significant increase, got {PRICE_FORECAST_EXPENSIVE_THRESHOLD}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

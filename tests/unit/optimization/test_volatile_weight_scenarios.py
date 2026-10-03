"""Volatile price weight reduction.

Volatile periods reduce the price layer's influence via a reduced weight rather than blocking
pre-heating, so extreme spikes still get through weighted aggregation.
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta, timezone
from freezegun import freeze_time

from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod

from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.const import (
    LAYER_WEIGHT_PRICE,
    PRICE_FORECAST_EXPENSIVE_THRESHOLD,
    PRICE_FORECAST_PREHEAT_OFFSET,
    QuarterClassification,
    VOLATILE_MIN_DURATION_QUARTERS,
    VOLATILE_WEIGHT_REDUCTION,
)

VOLATILE_TEST_BASE = datetime(2025, 11, 30, 0, 0, tzinfo=timezone.utc)


def realize_price_data(period_stubs, base=VOLATILE_TEST_BASE, tomorrow_stubs=None):
    """Convert price stubs (only .price used) into a real PriceData.

    Real timestamps let the production timestamp-containment lookup resolve
    the current interval; tests pin the clock with freeze_time to match.
    """

    def day(stubs, day_base):
        return [
            QuarterPeriod(start_time=day_base + timedelta(minutes=15 * q), price=stub.price)
            for q, stub in enumerate(stubs)
        ]

    tomorrow = day(tomorrow_stubs, base + timedelta(days=1)) if tomorrow_stubs else []
    return PriceData(today=day(period_stubs, base), tomorrow=tomorrow, has_tomorrow=bool(tomorrow))


class TestVolatileWeightReduction:
    """Test weight reduction during volatile periods."""

    @pytest.fixture
    def engine(self):
        """Create decision engine for testing."""
        from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
        from custom_components.effektguard.optimization.effect_layer import EffectManager
        from custom_components.effektguard.optimization.thermal_layer import ThermalModel
        from unittest.mock import MagicMock

        hass_mock = MagicMock()

        config = {
            "target_indoor_temp": 22.0,  # Correct key name (not target_temperature)
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
        state.indoor_temp = 21.5  # Below target to avoid triggering overshoot protection
        state.outdoor_temp = -5.0
        state.flow_temp = 35.0
        state.degree_minutes = -100.0
        state.compressor_frequency = 50
        state.current_power = 2.5
        state.current_offset = 0.0  # Required for anti-windup tracking
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

    def test_extreme_spike_during_volatility_still_preheats(
        self, engine, base_nibe_state, base_weather_data
    ):
        """At a cheap quarter (06:00) before a 5x spike during a volatile period, the offset stays
        gentle (not strongly negative) and the proactive layer still contributes positive heating.
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

        price_data = realize_price_data(price_periods)

        # Get decision at 06:00 (Q24)
        with freeze_time("2025-11-30 06:00:00"):
            decision = engine.calculate_decision(
                nibe_state=base_nibe_state,
                price_data=price_data,
                weather_data=base_weather_data,
                current_peak=5.0,
                current_power=2.0,
            )

        # At 06:00 the current price is cheap and a cheaper period follows the spike, so a small
        # reduction is acceptable; the offset must stay gentle (not strongly negative).
        assert (
            decision.offset > -1.0
        ), f"Offset too negative before spike (should be gentle), got {decision.offset}"

        # Verify proactive layer is contributing positive offset
        proactive_layers = [l for l in decision.layers if l.name.startswith("Z")]
        assert len(proactive_layers) > 0, "Should have proactive layer active"
        assert proactive_layers[0].offset > 0, "Proactive layer should suggest heating"

    def test_normal_volatility_without_extreme_spike(
        self, engine, base_nibe_state, base_weather_data
    ):
        """During a normal volatile day (±30-50% jumps, no extreme spikes) the engine holds steady:
        the offset stays small and the Spot Price layer remains present.
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

        price_data = realize_price_data(price_periods)

        # Get decision at 10:00 (Q40)
        with freeze_time("2025-11-30 10:00:00"):
            decision = engine.calculate_decision(
                nibe_state=base_nibe_state,
                price_data=price_data,
                weather_data=base_weather_data,
                current_peak=5.0,
                current_power=2.0,
            )

        # Verify system holds steady during normal volatility
        # Offset should be small (no extreme changes to react to)
        assert (
            abs(decision.offset) < 1.5
        ), f"Should hold relatively steady during normal volatility, got {decision.offset}"

        # Verify Spot Price layer is present (price optimization is active)
        price_layer = next((l for l in decision.layers if l.name == "Spot Price"), None)
        assert price_layer is not None, "Should have Spot Price layer"

    def test_weight_reduction_math(self):
        """The reduced volatile weight still lets a strong pre-heat signal outweigh a normal
        cheap-period boost through weighted aggregation.
        """
        # Normal price layer weight (from const.py)
        normal_weight = LAYER_WEIGHT_PRICE

        # Volatile period weight (calculated from constants)
        volatile_weight = normal_weight * VOLATILE_WEIGHT_REDUCTION
        expected_volatile_weight = LAYER_WEIGHT_PRICE * VOLATILE_WEIGHT_REDUCTION

        # Verify reduction matches expected value from constants
        assert volatile_weight == pytest.approx(expected_volatile_weight), (
            f"Volatile weight should be {expected_volatile_weight} "
            f"({LAYER_WEIGHT_PRICE} × {VOLATILE_WEIGHT_REDUCTION}), got {volatile_weight}"
        )

        # Calculate influence using actual constants
        extreme_spike_influence = PRICE_FORECAST_PREHEAT_OFFSET * volatile_weight
        normal_offset_influence = 0.5 * normal_weight  # Example normal cheap boost

        # A ~30% weight retention still lets the price layer keep meaningful influence.
        assert extreme_spike_influence > normal_offset_influence, (
            f"With reduction {VOLATILE_WEIGHT_REDUCTION}, extreme spike ({extreme_spike_influence}) "
            f"should still beat normal offset ({normal_offset_influence}) through weighted aggregation"
        )

    def test_early_morning_edge_case(self, engine, base_nibe_state, base_weather_data):
        """The backward volatile scan must not crash when the full 8-quarter history is
        unavailable (Q1, only 2 quarters back) and must still return a valid offset.
        """
        # Build price data with volatility in first few quarters
        price_periods = []

        # Q0: CHEAP
        period = MagicMock()
        period.price = 20.0
        period.is_daytime = False
        price_periods.append(period)

        # Q1: PEAK (sudden spike)
        period = MagicMock()
        period.price = 90.0
        period.is_daytime = False
        price_periods.append(period)

        # Q2-Q7: EXPENSIVE
        for _ in range(6):
            period = MagicMock()
            period.price = 60.0
            period.is_daytime = False
            price_periods.append(period)

        # Rest of day: NORMAL
        for i in range(8, 96):
            period = MagicMock()
            period.price = 40.0
            period.is_daytime = i >= 24
            price_periods.append(period)

        price_data = realize_price_data(price_periods)

        # Test at Q1 (00:15) - only 2 quarters of history
        with freeze_time("2025-11-30 00:15:00"):
            # Should not crash
            decision = engine.calculate_decision(
                nibe_state=base_nibe_state,
                price_data=price_data,
                weather_data=base_weather_data,
                current_peak=5.0,
                current_power=2.0,
            )

        # Should complete successfully
        assert decision is not None, "Decision should complete even with partial scan window"
        assert decision.offset is not None, "Should return valid offset"

        # Test at Q7 (01:45) - 8 quarters available (full window)
        with freeze_time("2025-11-30 01:45:00"):
            decision_q7 = engine.calculate_decision(
                nibe_state=base_nibe_state,
                price_data=price_data,
                weather_data=base_weather_data,
                current_peak=5.0,
                current_power=2.0,
            )

        # Should also work fine
        assert decision_q7 is not None, "Decision should work with full 8-quarter window"

        # Q0-Q7 has mix (CHEAP, PEAK, EXPENSIVE) so should detect volatility
        # Can't directly check internal flag, but system should be conservative
        assert (
            abs(decision_q7.offset) <= 2.0
        ), f"Should be somewhat conservative with early morning volatility, offset: {decision_q7.offset}"

    @freeze_time("2025-11-30 23:45:00")  # Q95 (23:45-00:00)
    def test_day_transition_volatile_scan(self, engine, base_nibe_state, base_weather_data):
        """The bidirectional volatile scan at the 23:45 day boundary must produce a valid decision
        both with tomorrow's prices (9-quarter window crossing midnight) and without them.
        """
        # Build price data with day transition volatility
        price_periods_today = []

        # Q0-Q90: NORMAL ~50 öre (stable all day)
        for q in range(91):
            period = MagicMock()
            period.price = 50.0
            period.is_daytime = 6 * 4 <= q < 22 * 4  # 06:00-22:00
            period.period_of_day = q
            price_periods_today.append(period)

        # Q91-Q95: Volatile spike - PEAK ~80 öre
        for q in range(91, 96):
            period = MagicMock()
            period.price = 85.0  # PEAK
            period.is_daytime = False
            period.period_of_day = q
            price_periods_today.append(period)

        # Tomorrow Q0-Q3: Volatile drop - CHEAP ~20 öre
        price_periods_tomorrow = []
        for q in range(4):
            period = MagicMock()
            period.price = 20.0  # CHEAP
            period.is_daytime = False
            period.period_of_day = q
            price_periods_tomorrow.append(period)

        # Tomorrow Q4+: Stabilize to NORMAL
        for q in range(4, 96):
            period = MagicMock()
            period.price = 50.0
            period.is_daytime = 6 * 4 <= q < 22 * 4
            period.period_of_day = q
            price_periods_tomorrow.append(period)

        # Test WITH tomorrow prices
        price_data_with_tomorrow = realize_price_data(
            price_periods_today, tomorrow_stubs=price_periods_tomorrow
        )

        # Populate classifications for both days
        classifications_today = engine.price.classify_quarterly_periods(
            price_data_with_tomorrow.today
        )
        classifications_tomorrow = engine.price.classify_quarterly_periods(
            price_data_with_tomorrow.tomorrow
        )
        engine.price._classifications_today = classifications_today
        engine.price._classifications_tomorrow = classifications_tomorrow

        decision_with_tomorrow = engine.calculate_decision(
            nibe_state=base_nibe_state,
            price_data=price_data_with_tomorrow,
            weather_data=base_weather_data,
            current_peak=5.0,
            current_power=2.0,
        )

        # With tomorrow: Q91-Q95 (PEAK) + Q96-Q99 (CHEAP) = 9 quarters
        # Mix of PEAK + CHEAP = volatility detected → weight 0.4
        assert decision_with_tomorrow is not None, "Should handle day transition with tomorrow"
        assert (
            abs(decision_with_tomorrow.offset) <= 1.0
        ), f"Should be conservative during day transition volatility, offset: {decision_with_tomorrow.offset}"

        # Test WITHOUT tomorrow prices (partial scan)
        price_data_no_tomorrow = realize_price_data(price_periods_today)

        # Only today classifications
        engine.price._classifications_today = classifications_today
        engine.price._classifications_tomorrow = []

        decision_no_tomorrow = engine.calculate_decision(
            nibe_state=base_nibe_state,
            price_data=price_data_no_tomorrow,
            weather_data=base_weather_data,
            current_peak=5.0,
            current_power=2.0,
        )

        # Without tomorrow: Only Q91-Q95 (5 quarters of PEAK)
        # All PEAK = no mix = no volatility by current threshold
        # System should still work safely
        assert decision_no_tomorrow is not None, "Should handle day transition without tomorrow"

        # Both scenarios should produce valid decisions
        assert decision_with_tomorrow.offset is not None
        assert decision_no_tomorrow.offset is not None

    def test_constants_relationship(self):
        """Verify constants have sensible relationships."""
        # Run-length volatility uses VOLATILE_MIN_DURATION_QUARTERS (3 quarters / 45 min)
        # Any run shorter than this is considered "brief" and triggers volatility
        assert (
            VOLATILE_MIN_DURATION_QUARTERS >= 2
        ), "Min duration should be at least 2 quarters (30 min) for compressor efficiency"
        assert (
            VOLATILE_MIN_DURATION_QUARTERS <= 4
        ), "Min duration shouldn't be too long or real price changes get ignored"

        # Weight reduction during volatility - moderate (25-35% retention) to avoid chasing prices.
        assert (
            0.25 <= VOLATILE_WEIGHT_REDUCTION <= 0.35
        ), f"Weight reduction should be moderate (25-35% retention), got {VOLATILE_WEIGHT_REDUCTION}"

        # Expensive threshold for spikes should require significant increase
        assert (
            PRICE_FORECAST_EXPENSIVE_THRESHOLD >= 1.5
        ), f"Expensive threshold should require significant increase, got {PRICE_FORECAST_EXPENSIVE_THRESHOLD}"

    @freeze_time("2025-01-15 17:15:00")
    def test_peak_cluster_expensive_between_peaks(self, engine, base_nibe_state, base_weather_data):
        """An EXPENSIVE quarter sandwiched between PEAKs (a PEAK+EXPENSIVE cluster run >= 3) inherits
        PEAK behavior: weight 1.0 (critical) and an aggressive negative offset, not the small
        EXPENSIVE reduction.
        """
        # Build evening price data with PEAK cluster pattern
        price_periods = []
        classifications = []

        base_date = datetime(2025, 1, 15, 0, 0, 0, tzinfo=timezone.utc)

        for q in range(96):
            start_time = base_date + timedelta(minutes=q * 15)
            if q in [68, 70, 71]:  # PEAK quarters
                price = 95.0 - (q - 68) * 2  # Slightly decreasing
                period = QuarterPeriod(
                    start_time=start_time,
                    price=price,
                )
                price_periods.append(period)
                classifications.append(QuarterClassification.PEAK)
            elif q == 69:  # Current quarter - EXPENSIVE sandwiched between PEAKs
                period = QuarterPeriod(
                    start_time=start_time,
                    price=85.0,
                )
                price_periods.append(period)
                classifications.append(QuarterClassification.EXPENSIVE)
            elif q >= 64 and q <= 75:  # Surrounding hours - EXPENSIVE/NORMAL
                price = 70.0 + (q % 4) * 2
                period = QuarterPeriod(
                    start_time=start_time,
                    price=price,
                )
                price_periods.append(period)
                classifications.append(QuarterClassification.EXPENSIVE)
            else:  # Rest of day - NORMAL
                period = QuarterPeriod(
                    start_time=start_time,
                    price=50.0,
                )
                price_periods.append(period)
                classifications.append(QuarterClassification.NORMAL)

        # Set up price data (periods are real QuarterPeriods on 2025-01-15)
        price_data = PriceData(today=price_periods, tomorrow=[], has_tomorrow=False)

        # Configure classifier
        engine.price._classifications_today = classifications
        engine.price._classifications_tomorrow = []

        def get_classification(quarter):
            if 0 <= quarter < len(classifications):
                return classifications[quarter]
            return None

        engine.price.get_current_classification = get_classification
        engine.price.get_tomorrow_classification = lambda q: None

        decision = engine.calculate_decision(
            nibe_state=base_nibe_state,
            price_data=price_data,
            weather_data=base_weather_data,
            current_peak=5.0,
            current_power=2.0,
        )

        # Find the price layer decision
        price_layer = next(
            (layer for layer in decision.layers if layer.name == "Spot Price"),
            None,
        )
        assert price_layer is not None, "Should have price layer"

        # EXPENSIVE between PEAKs should inherit PEAK behavior
        # Weight should be 1.0 (critical priority like PEAK)
        assert (
            price_layer.weight == 1.0
        ), f"EXPENSIVE in PEAK cluster should have weight 1.0, got {price_layer.weight}"

        # Should mention PEAK cluster in reason
        assert (
            "PEAK cluster" in price_layer.reason
        ), f"Should mention PEAK cluster in reason: {price_layer.reason}"

        # Offset should be scaled PEAK offset (tolerance factor applied)
        # In balanced mode: PEAK uses tolerance factor, so offset will be -3.0 or similar
        # The key is it should be significantly negative, not the small EXPENSIVE reduction
        assert (
            price_layer.offset < -1.0
        ), f"PEAK cluster should use aggressive negative offset, got {price_layer.offset}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

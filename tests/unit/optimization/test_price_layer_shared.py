"""Tests for shared PriceAnalyzer methods.

Phase 11: Tests for find_cheapest_window(), get_price_forecast(), and
calculate_lookahead_hours() shared methods.
"""

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.effektguard.optimization.price_layer import (
    CheapestWindowResult,
    PriceAnalyzer,
    PriceData,
    PriceForecast,
    QuarterPeriod,
)
from custom_components.effektguard.const import (
    DHW_NORMAL_RUNTIME_MINUTES,
    PRICE_FORECAST_BASE_HORIZON,
    PRICE_FORECAST_CHEAP_THRESHOLD,
    PRICE_FORECAST_EXPENSIVE_THRESHOLD,
    PRICE_FORECAST_MIN_DURATION,
)


class TestCheapestWindowResult:
    """Tests for CheapestWindowResult dataclass."""

    def test_dataclass_fields(self):
        """Test CheapestWindowResult has all required fields."""
        now = datetime.now(timezone.utc)
        result = CheapestWindowResult(
            start_time=now,
            end_time=now + timedelta(minutes=45),
            quarters=[30, 31, 32],
            avg_price=15.5,
            hours_until=2.0,
            savings_vs_current=25.0,
        )

        assert result.start_time == now
        assert result.end_time == now + timedelta(minutes=45)
        assert result.quarters == [30, 31, 32]
        assert result.avg_price == 15.5
        assert result.hours_until == 2.0
        assert result.savings_vs_current == 25.0

    def test_savings_optional(self):
        """Test savings_vs_current is optional."""
        now = datetime.now(timezone.utc)
        result = CheapestWindowResult(
            start_time=now,
            end_time=now + timedelta(minutes=45),
            quarters=[30, 31, 32],
            avg_price=15.5,
            hours_until=2.0,
        )

        assert result.savings_vs_current is None


class TestPriceForecast:
    """Tests for PriceForecast dataclass."""

    def test_dataclass_fields(self):
        """Test PriceForecast has all required fields."""
        forecast = PriceForecast(
            next_cheap_quarters_away=5,
            cheap_period_duration=4,
            cheap_price_ratio=0.6,
            next_expensive_quarters_away=10,
            expensive_period_duration=3,
            expensive_price_ratio=1.8,
            is_volatile=True,
            current_run_length=2,
            volatile_reason="Brief run",
            in_peak_cluster=False,
        )

        assert forecast.next_cheap_quarters_away == 5
        assert forecast.cheap_period_duration == 4
        assert forecast.cheap_price_ratio == 0.6
        assert forecast.next_expensive_quarters_away == 10
        assert forecast.expensive_period_duration == 3
        assert forecast.expensive_price_ratio == 1.8
        assert forecast.is_volatile is True
        assert forecast.current_run_length == 2
        assert forecast.volatile_reason == "Brief run"
        assert forecast.in_peak_cluster is False


class TestFindCheapestWindow:
    """Tests for PriceAnalyzer.find_cheapest_window()."""

    def _create_price_periods(
        self, base_time: datetime, prices: list[float]
    ) -> list[QuarterPeriod]:
        """Helper to create QuarterPeriod list from prices."""
        periods = []
        for i, price in enumerate(prices):
            start_time = base_time + timedelta(minutes=15 * i)
            periods.append(
                QuarterPeriod(
                    start_time=start_time,
                    price=price,
                )
            )
        return periods

    def test_finds_cheapest_window(self):
        """Test finds absolute cheapest continuous window."""
        analyzer = PriceAnalyzer()
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        # Prices: 20, 25, 30, 10, 11, 12, 35, 40 (cheapest at Q3-5)
        prices = [20.0, 25.0, 30.0, 10.0, 11.0, 12.0, 35.0, 40.0]
        periods = self._create_price_periods(now, prices)

        result = analyzer.find_cheapest_window(
            current_time=now,
            price_periods=periods,
            duration_minutes=DHW_NORMAL_RUNTIME_MINUTES,  # 3 quarters
            lookahead_hours=2.0,
        )

        assert result is not None
        assert result.avg_price == pytest.approx(11.0, rel=0.01)  # avg(10, 11, 12)
        assert len(result.quarters) == 3

    def test_returns_none_insufficient_data(self):
        """Test returns None when not enough quarters available."""
        analyzer = PriceAnalyzer()
        now = datetime.now(timezone.utc)

        # Only 2 quarters but need 3
        prices = [20.0, 25.0]
        periods = self._create_price_periods(now, prices)

        result = analyzer.find_cheapest_window(
            current_time=now,
            price_periods=periods,
            duration_minutes=DHW_NORMAL_RUNTIME_MINUTES,  # 3 quarters
            lookahead_hours=1.0,
        )

        assert result is None

    def test_returns_none_empty_periods(self):
        """Test returns None when no price periods provided."""
        analyzer = PriceAnalyzer()
        now = datetime.now(timezone.utc)

        result = analyzer.find_cheapest_window(
            current_time=now,
            price_periods=[],
            duration_minutes=DHW_NORMAL_RUNTIME_MINUTES,
            lookahead_hours=2.0,
        )

        assert result is None

    def test_respects_lookahead_window(self):
        """Test only considers periods within lookahead window."""
        analyzer = PriceAnalyzer()
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        # Cheapest window is at hour 3, but lookahead is only 1 hour
        prices = [50.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0, 10.0]  # 8 quarters = 2 hours
        periods = self._create_price_periods(now, prices)

        result = analyzer.find_cheapest_window(
            current_time=now,
            price_periods=periods,
            duration_minutes=DHW_NORMAL_RUNTIME_MINUTES,
            lookahead_hours=1.0,  # Only 1 hour = 4 quarters
        )

        assert result is not None
        # Should use first available window since cheap ones are outside lookahead
        assert result.avg_price == pytest.approx(50.0, rel=0.01)

    def test_calculates_savings_vs_current(self):
        """Test calculates savings percentage when current_price provided."""
        analyzer = PriceAnalyzer()
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        # Current price is 100, cheapest window avg is 50 = 50% savings
        prices = [100.0, 100.0, 50.0, 50.0, 50.0, 100.0]
        periods = self._create_price_periods(now, prices)

        result = analyzer.find_cheapest_window(
            current_time=now,
            price_periods=periods,
            duration_minutes=DHW_NORMAL_RUNTIME_MINUTES,
            lookahead_hours=2.0,
            current_price=100.0,
        )

        assert result is not None
        assert result.savings_vs_current == pytest.approx(50.0, rel=0.01)

    def test_hours_until_calculated_correctly(self):
        """Test hours_until is calculated correctly."""
        analyzer = PriceAnalyzer()
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        # Cheapest at quarters 4-6 (1 hour away)
        prices = [50.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0, 50.0]
        periods = self._create_price_periods(now, prices)

        result = analyzer.find_cheapest_window(
            current_time=now,
            price_periods=periods,
            duration_minutes=DHW_NORMAL_RUNTIME_MINUTES,
            lookahead_hours=2.0,
        )

        assert result is not None
        assert result.hours_until == pytest.approx(1.0, rel=0.01)


class TestCalculateLookaheadHours:
    """Tests for PriceAnalyzer.calculate_lookahead_hours()."""

    def test_space_heating_scales_with_thermal_mass(self):
        """Test space heating lookahead scales with thermal mass."""
        analyzer = PriceAnalyzer()

        # Low thermal mass = shorter lookahead
        result_low = analyzer.calculate_lookahead_hours("space", thermal_mass=0.5)
        assert result_low == PRICE_FORECAST_BASE_HORIZON * 0.5

        # High thermal mass = longer lookahead
        result_high = analyzer.calculate_lookahead_hours("space", thermal_mass=2.0)
        assert result_high == PRICE_FORECAST_BASE_HORIZON * 2.0

    def test_dhw_uses_demand_hours(self):
        """Test DHW lookahead uses hours until next demand."""
        analyzer = PriceAnalyzer()

        result = analyzer.calculate_lookahead_hours("dhw", next_demand_hours=3.0)
        assert result == 3.0

    def test_dhw_caps_at_24_hours(self):
        """Test DHW lookahead is capped at 24 hours."""
        analyzer = PriceAnalyzer()

        result = analyzer.calculate_lookahead_hours("dhw", next_demand_hours=48.0)
        assert result == 24.0

    def test_dhw_min_1_hour(self):
        """Test DHW lookahead has minimum of 1 hour."""
        analyzer = PriceAnalyzer()

        result = analyzer.calculate_lookahead_hours("dhw", next_demand_hours=0.5)
        assert result == 1.0

    def test_dhw_default_24_hours(self):
        """Test DHW lookahead defaults to 24 hours when no demand specified."""
        analyzer = PriceAnalyzer()

        result = analyzer.calculate_lookahead_hours("dhw")
        assert result == 24.0


class TestGetPriceForecast:
    """Tests for PriceAnalyzer.get_price_forecast()."""

    def _create_price_data(
        self, prices_today: list[float], prices_tomorrow: list[float] | None = None
    ) -> PriceData:
        """Helper to create PriceData from price lists."""
        base_today = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today = []
        for i, price in enumerate(prices_today):
            start_time = base_today + timedelta(minutes=15 * i)
            today.append(QuarterPeriod(start_time=start_time, price=price))

        tomorrow = []
        if prices_tomorrow:
            base_tomorrow = base_today + timedelta(days=1)
            for i, price in enumerate(prices_tomorrow):
                start_time = base_tomorrow + timedelta(minutes=15 * i)
                tomorrow.append(QuarterPeriod(start_time=start_time, price=price))

        return PriceData(
            today=today,
            tomorrow=tomorrow,
            has_tomorrow=len(tomorrow) > 0,
        )

    def test_returns_empty_forecast_no_data(self):
        """Test returns empty forecast when no price data."""
        analyzer = PriceAnalyzer()

        forecast = analyzer.get_price_forecast(
            current_quarter=0,
            price_data=None,
            lookahead_hours=4.0,
        )

        assert forecast.next_cheap_quarters_away is None
        assert forecast.next_expensive_quarters_away is None
        assert forecast.is_volatile is False

    def test_detects_upcoming_cheap_period(self):
        """Test detects upcoming cheap period within threshold."""
        analyzer = PriceAnalyzer()

        # Create prices: high now, cheap soon
        # Need 96 prices for full day, with cheap period in lookahead
        prices = [100.0] * 10 + [30.0] * 6 + [100.0] * 80  # Cheap at Q10-15

        price_data = self._create_price_data(prices)
        analyzer.update_prices(price_data)

        forecast = analyzer.get_price_forecast(
            current_quarter=5,  # Start at Q5
            price_data=price_data,
            lookahead_hours=4.0,  # Look ahead 16 quarters
        )

        # Cheap ratio should be < PRICE_FORECAST_CHEAP_THRESHOLD
        # 30/100 = 0.3, which is < 0.75 threshold
        assert forecast.cheap_price_ratio < PRICE_FORECAST_CHEAP_THRESHOLD

    def test_detects_upcoming_expensive_period(self):
        """Test detects upcoming expensive period within threshold."""
        analyzer = PriceAnalyzer()

        # Create prices: normal now, expensive soon
        prices = [50.0] * 10 + [150.0] * 6 + [50.0] * 80  # Expensive at Q10-15

        price_data = self._create_price_data(prices)
        analyzer.update_prices(price_data)

        forecast = analyzer.get_price_forecast(
            current_quarter=5,  # Start at Q5
            price_data=price_data,
            lookahead_hours=4.0,  # Look ahead 16 quarters
        )

        # Expensive ratio should be > PRICE_FORECAST_EXPENSIVE_THRESHOLD
        # 150/50 = 3.0, which is > 1.25 threshold
        assert forecast.expensive_price_ratio > PRICE_FORECAST_EXPENSIVE_THRESHOLD

    def test_detects_volatility_brief_run(self):
        """Test detects volatility when current run is brief."""
        analyzer = PriceAnalyzer()

        # Create prices with clear variance and a short cheap period at end
        # Need variance to avoid "uniform prices" classification
        # Create pattern: expensive -> normal -> expensive -> single cheap
        prices = []
        for i in range(96):
            if i < 40:
                prices.append(150.0)  # Expensive
            elif i < 80:
                prices.append(100.0)  # Normal
            elif i < 94:
                prices.append(150.0)  # Expensive again
            elif i == 94:
                prices.append(50.0)  # Single cheap quarter
            else:
                prices.append(150.0)  # Back to expensive

        price_data = self._create_price_data(prices)
        analyzer.update_prices(price_data)

        forecast = analyzer.get_price_forecast(
            current_quarter=94,  # At the single cheap quarter
            price_data=price_data,
            lookahead_hours=4.0,
        )

        # Current run of 1 quarter < PRICE_FORECAST_MIN_DURATION (3)
        assert forecast.current_run_length < PRICE_FORECAST_MIN_DURATION
        assert forecast.is_volatile is True


class TestDHWOptimizerIntegration:
    """Tests for DHW optimizer using shared price_analyzer."""

    def test_price_analyzer_find_cheapest_window(self):
        """Test PriceAnalyzer.find_cheapest_window() finds cheapest period."""
        analyzer = PriceAnalyzer()
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        # Create test prices
        prices = [50.0, 50.0, 20.0, 20.0, 20.0, 50.0, 50.0, 50.0]
        periods = []
        for i, price in enumerate(prices):
            start_time = now + timedelta(minutes=15 * i)
            periods.append(QuarterPeriod(start_time=start_time, price=price))

        # Use PriceAnalyzer directly (the shared layer)
        result = analyzer.find_cheapest_window(
            current_time=now,
            price_periods=periods,
            duration_minutes=DHW_NORMAL_RUNTIME_MINUTES,
            lookahead_hours=2.0,
        )

        assert result is not None
        # Should find cheapest window at Q2-4 (prices 20, 20, 20)
        assert result.avg_price == pytest.approx(20.0, rel=0.01)

    def test_dhw_optimizer_uses_price_analyzer_internally(self):
        """Test DHW optimizer uses price_analyzer for window search in should_start_dhw()."""
        from unittest.mock import MagicMock
        from custom_components.effektguard.optimization.dhw_optimizer import (
            IntelligentDHWScheduler,
        )

        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        # Create test prices
        prices = [50.0, 50.0, 20.0, 20.0, 20.0, 50.0, 50.0, 50.0]
        periods = []
        for i, price in enumerate(prices):
            start_time = now + timedelta(minutes=15 * i)
            periods.append(QuarterPeriod(start_time=start_time, price=price))

        # Create mock price_analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.find_cheapest_window.return_value = CheapestWindowResult(
            start_time=now + timedelta(minutes=30),
            end_time=now + timedelta(minutes=75),
            quarters=[2, 3, 4],
            avg_price=20.0,
            hours_until=0.5,
        )

        # Initialize with mock price_analyzer
        scheduler = IntelligentDHWScheduler(price_analyzer=mock_analyzer)

        # Call should_start_dhw which internally uses price_analyzer
        _ = scheduler.should_start_dhw(
            current_dhw_temp=40.0,
            space_heating_demand_kw=1.0,
            thermal_debt_dm=-100,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            outdoor_temp=5.0,
            price_classification="normal",
            current_time=now,
            price_periods=periods,
        )

        # Verify price_analyzer.find_cheapest_window was called
        mock_analyzer.find_cheapest_window.assert_called()

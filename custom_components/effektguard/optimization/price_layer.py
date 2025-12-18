"""Price analyzer for spot price classification.

Analyzes spot prices with native 15-minute granularity to classify
periods for optimization decisions.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil

import numpy as np

from homeassistant.util import dt as dt_util

from ..const import (
    BENEFICIAL_CLASSIFICATIONS,
    OptimizationModeConfig,
    LAYER_WEIGHT_PRICE,
    PRICE_DAYTIME_MULTIPLIER,
    PRICE_FORECAST_BASE_HORIZON,
    PRICE_FORECAST_CHEAP_THRESHOLD,
    PRICE_FORECAST_DM_DEBT_OFFSET,
    PRICE_FORECAST_EXPENSIVE_THRESHOLD,
    PRICE_FORECAST_MIN_DURATION,
    PRICE_FORECAST_PREHEAT_OFFSET,
    PRICE_FORECAST_REDUCTION_OFFSET,
    PRICE_OFFSET_CHEAP,
    PRICE_OFFSET_EXPENSIVE,
    PRICE_OFFSET_NORMAL,
    PRICE_OFFSET_PEAK,
    PRICE_OFFSET_VERY_CHEAP,
    PRICE_PERCENTILE_CHEAP,
    PRICE_PERCENTILE_EXPENSIVE,
    PRICE_PERCENTILE_NORMAL,
    PRICE_PERCENTILE_VERY_CHEAP,
    PRICE_PRE_PEAK_OFFSET,
    PRICE_TOLERANCE_FACTOR_MAX,
    PRICE_TOLERANCE_FACTOR_MIN,
    PRICE_TOLERANCE_MAX,
    PRICE_TOLERANCE_MIN,
    QuarterClassification,
    VOLATILE_WEIGHT_REDUCTION,
    WEATHER_COMP_DEFER_DM_CRITICAL,
)
from ..utils.time_utils import get_current_quarter
from ..utils.volatile_helpers import get_volatile_info, should_skip_volatile_boost

_LOGGER = logging.getLogger(__name__)

# Re-export from adapters for convenience
from ..adapters.gespot_adapter import PriceData, QuarterPeriod

__all__ = [
    "CheapestWindowResult",
    "PriceAnalyzer",
    "PriceData",
    "PriceForecast",
    "PriceLayerDecision",
    "QuarterPeriod",
    "get_fallback_prices",
]


@dataclass
class CheapestWindowResult:
    """Result of cheapest continuous window search.

    Used by both DHW optimizer (45-min window) and space heating
    pre-heat planning (2-4h window).

    Extracted from dhw_optimizer.find_cheapest_dhw_window() for shared reuse.
    """

    start_time: datetime
    end_time: datetime
    quarters: list[int]  # List of quarter IDs in window
    avg_price: float  # Average price in window (öre/kWh)
    hours_until: float  # Hours from current_time until window starts
    savings_vs_current: float | None = None  # % cheaper than current price (optional)


@dataclass
class PriceForecast:
    """Forward-looking price analysis result.

    Consolidated forecast logic used by:
    - Space heating: Pre-heat before expensive, reduce before cheap
    - DHW: Wait for cheap if adequate, heat now if urgent

    Extracted from price_layer.evaluate_layer() for shared reuse.
    """

    # Cheap period info
    next_cheap_quarters_away: int | None  # Quarters until next cheap period (None if not found)
    cheap_period_duration: int  # Duration in quarters of the cheap period
    cheap_price_ratio: float  # Price ratio vs current (e.g., 0.6 = 40% cheaper)

    # Expensive period info
    next_expensive_quarters_away: int | None  # Quarters until next expensive period
    expensive_period_duration: int  # Duration in quarters of the expensive period
    expensive_price_ratio: float  # Price ratio vs current (e.g., 1.5 = 50% more expensive)

    # Volatility detection
    is_volatile: bool  # True if current run is too brief for effective heating
    current_run_length: int  # Duration of current classification run (quarters)
    remaining_quarters: int  # Quarters remaining forward with same classification
    volatile_reason: str  # Human-readable explanation
    is_ending_soon: bool  # True if favorable period ending soon (< 3 quarters remain)

    # Cluster detection
    in_peak_cluster: bool  # True when EXPENSIVE sandwiched between PEAKs


def get_fallback_prices() -> PriceData:
    """Get fallback price data when spot price unavailable.

    Returns neutral price classification to maintain safe operation
    without optimization. All periods are set to price=1.0 (normalized).

    Moved from coordinator._get_fallback_prices for shared reuse.

    Returns:
        PriceData with 96 neutral-priced quarters for today, empty tomorrow
    """
    _LOGGER.debug("Creating fallback price data (no optimization)")

    # Create neutral periods - all classified as "normal"
    fallback_periods = []
    base_date = dt_util.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for quarter in range(96):  # 96 quarters per day (15-min intervals)
        hour = quarter // 4
        minute = (quarter % 4) * 15
        start_time = base_date.replace(hour=hour, minute=minute)
        fallback_periods.append(QuarterPeriod(start_time=start_time, price=1.0))

    return PriceData(
        today=fallback_periods,
        tomorrow=[],
        has_tomorrow=False,
    )


@dataclass
class PriceLayerDecision:
    """Decision from price/spot optimization layer.

    Used to communicate layer decisions back to decision engine.
    """

    name: str  # Layer name for display (e.g., "Spot Price")
    offset: float  # Proposed heating curve offset (°C)
    weight: float  # Layer weight/priority (0.0-1.0)
    reason: str  # Human-readable explanation


class PriceAnalyzer:
    """Analyze electricity spot prices with native 15-minute granularity.

    Spot price integrations provide true quarterly data (96 intervals per day), which
    perfectly matches Swedish Effektavgift requirements.
    """

    def __init__(self):
        """Initialize price analyzer."""
        self._quarterly_periods_today: list[QuarterPeriod] = []
        self._quarterly_periods_tomorrow: list[QuarterPeriod] = []
        self._classifications_today: dict[int, QuarterClassification] = {}
        self._classifications_tomorrow: dict[int, QuarterClassification] = {}

    def update_prices(self, price_data: PriceData) -> None:
        """Update price data and recalculate classifications.

        Args:
            price_data: PriceData with today's and tomorrow's quarters
        """
        self._quarterly_periods_today = price_data.today
        self._quarterly_periods_tomorrow = price_data.tomorrow

        # Classify periods
        self._classifications_today = self.classify_quarterly_periods(self._quarterly_periods_today)

        if price_data.has_tomorrow:
            self._classifications_tomorrow = self.classify_quarterly_periods(
                self._quarterly_periods_tomorrow
            )

        _LOGGER.debug(
            "Classified %d today periods, %d tomorrow periods",
            len(self._classifications_today),
            len(self._classifications_tomorrow),
        )

    def classify_quarterly_periods(
        self,
        periods: list[QuarterPeriod],
    ) -> dict[int, QuarterClassification]:
        """Classify 15-minute periods from native spot price data.

        Algorithm:
        1. Calculate price percentiles across all 96 periods
        2. Classify each period based on price ranking
        3. Apply effect tariff day/night awareness
        4. Identify consecutive expensive periods for peak risk

        Special case: If all prices are uniform (e.g., fallback mode), classify
        all periods as NORMAL to avoid misleading optimization signals.

        Args:
            periods: List of 96 QuarterPeriod objects (native 15-min intervals)

        Returns:
            dict mapping quarter (0-95) to classification
        """
        if not periods:
            _LOGGER.warning("No periods to classify")
            return {}

        # Extract prices for percentile calculation
        prices = [p.price for p in periods]

        # Calculate percentiles using configurable thresholds
        p10 = np.percentile(prices, PRICE_PERCENTILE_VERY_CHEAP)
        p25 = np.percentile(prices, PRICE_PERCENTILE_CHEAP)
        p75 = np.percentile(prices, PRICE_PERCENTILE_NORMAL)
        p90 = np.percentile(prices, PRICE_PERCENTILE_EXPENSIVE)

        _LOGGER.debug(
            "Price percentiles - P10: %.3f, P25: %.3f, P75: %.3f, P90: %.3f",
            p10,
            p25,
            p75,
            p90,
        )

        # Special case: Uniform prices (all equal) - happens with fallback mode
        # When spot price unavailable, fallback creates 96 periods with price=1.0
        # Without variance, classification is meaningless - mark all as NORMAL
        if p25 == p90:  # No price variance
            _LOGGER.info(
                "Uniform prices detected (%.3f), classifying all periods as NORMAL (no optimization)",
                p25,
            )
            return {period.quarter_of_day: QuarterClassification.NORMAL for period in periods}

        # Classify each period
        # Order: VERY_CHEAP (bottom 10%) -> CHEAP (10-25%) -> NORMAL (25-75%) ->
        #        EXPENSIVE (75-90%) -> PEAK (top 10%)
        classifications = {}
        for period in periods:
            if period.price <= p10:
                classification = QuarterClassification.VERY_CHEAP
            elif period.price <= p25:
                classification = QuarterClassification.CHEAP
            elif period.price <= p75:
                classification = QuarterClassification.NORMAL
            elif period.price <= p90:
                classification = QuarterClassification.EXPENSIVE
            else:
                classification = QuarterClassification.PEAK

            classifications[period.quarter_of_day] = classification

        return classifications

    def get_base_offset(
        self,
        quarter: int,
        classification: QuarterClassification,
        is_daytime: bool,
    ) -> float:
        """Get base temperature offset for quarterly period classification.

        Original algorithm for Swedish effect tariff optimization with
        native 15-minute granularity.

        Args:
            quarter: Quarter of day (0-95, where 0 = 00:00-00:15)
            classification: VERY_CHEAP/CHEAP/NORMAL/EXPENSIVE/PEAK
            is_daytime: True if 06:00-22:00 (full effect tariff weight)

        Returns offset in °C (before tolerance scaling):
            VERY_CHEAP: +4.0 (exceptional prices, aggressive pre-heating!)
            CHEAP: +1.5 (pre-heat opportunity, charge thermal battery)
            NORMAL: 0.0 (maintain)
            EXPENSIVE: -1.0 (conserve), -1.5 during daytime
            PEAK: -10.0 (maximum reduction, coast through expensive period)

        Note: These are BASE offsets before tolerance factor is applied.
        Final offset depends on user tolerance setting and optimization mode:
        - Savings mode: PEAK bypasses tolerance (always full -10.0°C)
        - Balanced/Comfort modes: Tolerance factor applied (e.g., -3.0°C)

        Effect tariff weighting:
            - Daytime (06:00-22:00): Full weight, EXPENSIVE gets 1.5x multiplier
            - Nighttime (22:00-06:00): Standard weight

        Note: VERY_CHEAP includes negative prices (you get paid to heat!)
        """
        if classification == QuarterClassification.VERY_CHEAP:
            offset = PRICE_OFFSET_VERY_CHEAP
        elif classification == QuarterClassification.CHEAP:
            offset = PRICE_OFFSET_CHEAP
        elif classification == QuarterClassification.NORMAL:
            offset = PRICE_OFFSET_NORMAL
        elif classification == QuarterClassification.EXPENSIVE:
            offset = PRICE_OFFSET_EXPENSIVE
        else:  # PEAK
            offset = PRICE_OFFSET_PEAK

        # More aggressive during daytime (full effect tariff weight)
        # Nighttime peaks are less critical (50% tariff weight)
        # Note: PEAK already at -10.0 (NIBE max), don't multiply further
        if is_daytime and classification == QuarterClassification.EXPENSIVE:
            offset *= PRICE_DAYTIME_MULTIPLIER

        return offset

    def get_current_classification(self, quarter: int) -> QuarterClassification:
        """Get classification for current quarter.

        Args:
            quarter: Quarter of day (0-95)

        Returns:
            Classification for the quarter
        """
        return self._classifications_today.get(quarter, QuarterClassification.NORMAL)

    def get_tomorrow_classification(self, quarter: int) -> QuarterClassification:
        """Get classification for tomorrow's quarter.

        Args:
            quarter: Quarter of day (0-95)

        Returns:
            Classification for the quarter, or NORMAL if tomorrow not available
        """
        return self._classifications_tomorrow.get(quarter, QuarterClassification.NORMAL)

    def has_tomorrow_prices(self) -> bool:
        """Check if tomorrow's prices are available."""
        return len(self._classifications_tomorrow) > 0

    def get_next_cheap_period(self, current_quarter: int) -> int | None:
        """Find next cheap or very cheap period after current quarter.

        Useful for pre-heating scheduling.

        Args:
            current_quarter: Current quarter of day (0-95)

        Returns:
            Quarter number of next cheap/very cheap period, or None if none found
        """
        # Search today's periods
        for quarter in range(current_quarter + 1, 96):
            if self._classifications_today.get(quarter) in BENEFICIAL_CLASSIFICATIONS:
                return quarter

        # Search tomorrow's periods if available
        for quarter in range(96):
            if self._classifications_tomorrow.get(quarter) in BENEFICIAL_CLASSIFICATIONS:
                return 96 + quarter  # Offset by 96 for tomorrow

        return None

    def get_next_expensive_period(self, current_quarter: int) -> int | None:
        """Find next expensive/peak period after current quarter.

        Useful for pre-heating before expensive periods.

        Args:
            current_quarter: Current quarter of day (0-95)

        Returns:
            Quarter number of next expensive/peak period, or None if none found
        """
        # Search today's periods
        for quarter in range(current_quarter + 1, 96):
            classification = self._classifications_today.get(quarter)
            if classification in [
                QuarterClassification.EXPENSIVE,
                QuarterClassification.PEAK,
            ]:
                return quarter

        # Search tomorrow's periods if available
        for quarter in range(96):
            classification = self._classifications_tomorrow.get(quarter)
            if classification in [
                QuarterClassification.EXPENSIVE,
                QuarterClassification.PEAK,
            ]:
                return 96 + quarter  # Offset by 96 for tomorrow

        return None

    def find_cheapest_window(
        self,
        current_time: datetime,
        price_periods: list[QuarterPeriod],
        duration_minutes: int,
        lookahead_hours: float,
        current_price: float | None = None,
    ) -> CheapestWindowResult | None:
        """Find cheapest continuous window for heating.

        Sliding window algorithm extracted from dhw_optimizer.find_cheapest_dhw_window()
        for shared reuse by both DHW and space heating optimization.

        Implementation:
        - Converts duration to 15-minute quarters
        - Filters periods within lookahead window
        - Uses sliding window to find absolute cheapest continuous period
        - Calculates savings vs current price if provided

        Used by:
        - DHW: Find 45-min window (3 quarters) for DHW heating
        - Space heating: Find optimal pre-heating window (2-4h) before expensive period

        Args:
            current_time: Search start timestamp
            price_periods: Combined today + tomorrow QuarterPeriod objects
            duration_minutes: Required heating duration (e.g., 45 for DHW, 120 for pre-heat)
            lookahead_hours: How far ahead to search
            current_price: Current period price for savings calculation (optional)

        Returns:
            CheapestWindowResult or None if insufficient data
        """
        if not price_periods:
            _LOGGER.warning("No price data available for cheapest window search")
            return None

        # Convert duration to quarters (45 min = 3 quarters)
        quarters_needed = ceil(duration_minutes / 15)

        # Filter to lookahead window
        end_time = current_time + timedelta(hours=lookahead_hours)

        # Build available quarters from price periods
        # QuarterPeriod has: quarter_of_day, hour, minute, price, is_daytime, start_time
        available_quarters = []

        for period in price_periods:
            # Use actual datetime from spot price (already handles timezone and date correctly)
            period_time = period.start_time

            # Check if within lookahead window
            if period_time >= current_time and period_time < end_time:
                # Calculate quarter ID based on actual datetime to distinguish duplicates
                days_ahead = (period_time.date() - current_time.date()).days
                quarter_id = period.quarter_of_day + (days_ahead * 96)

                available_quarters.append(
                    {
                        "start": period_time,
                        "end": period_time + timedelta(minutes=15),
                        "quarter": quarter_id,
                        "price": period.price,
                    }
                )

        if len(available_quarters) < quarters_needed:
            _LOGGER.warning(
                "Not enough price data for window search: %d quarters available, %d needed",
                len(available_quarters),
                quarters_needed,
            )
            return None

        _LOGGER.debug(
            "Cheapest window search: %d quarters available, need %d quarters (%d min), "
            "lookahead %.1fh",
            len(available_quarters),
            quarters_needed,
            duration_minutes,
            lookahead_hours,
        )

        # Sliding window to find cheapest continuous period
        lowest_price = None
        lowest_index = None
        window_candidates = []  # Track all valid windows for debugging

        for i in range(len(available_quarters) - quarters_needed + 1):
            window = available_quarters[i : i + quarters_needed]

            # Verify continuity (15-min gaps)
            is_continuous = True
            for j in range(len(window) - 1):
                time_gap = (window[j + 1]["start"] - window[j]["end"]).total_seconds()
                if abs(time_gap) > 1:  # Allow 1 second tolerance
                    is_continuous = False
                    break

            if not is_continuous:
                continue

            window_avg = sum(q["price"] for q in window) / quarters_needed

            # Track this candidate
            window_candidates.append(
                {
                    "start": window[0]["start"],
                    "avg_price": window_avg,
                    "hours_until": (window[0]["start"] - current_time).total_seconds() / 3600,
                }
            )

            if lowest_price is None or window_avg < lowest_price:
                lowest_price = window_avg
                lowest_index = i

        # Log candidates for debugging
        if window_candidates:
            _LOGGER.debug(
                "Window candidates found: %d windows evaluated",
                len(window_candidates),
            )
            # Show top 5 cheapest
            sorted_candidates = sorted(window_candidates, key=lambda x: x["avg_price"])
            for idx, candidate in enumerate(sorted_candidates[:5], 1):
                _LOGGER.debug(
                    "  #%d: %.1före/kWh at %s (%.1fh away)%s",
                    idx,
                    candidate["avg_price"],
                    candidate["start"].strftime("%H:%M"),
                    candidate["hours_until"],
                    " ← SELECTED" if idx == 1 else "",
                )

        if lowest_index is None:
            _LOGGER.debug("Could not find continuous window")
            return None

        optimal_window = available_quarters[lowest_index : lowest_index + quarters_needed]

        # Calculate savings vs current price if provided
        savings_vs_current = None
        if current_price is not None and current_price > 0:
            savings_vs_current = (1 - lowest_price / current_price) * 100

        result = CheapestWindowResult(
            start_time=optimal_window[0]["start"],
            end_time=optimal_window[-1]["end"],
            quarters=[q.get("quarter", i + lowest_index) for i, q in enumerate(optimal_window)],
            avg_price=lowest_price,
            hours_until=(optimal_window[0]["start"] - current_time).total_seconds() / 3600,
            savings_vs_current=savings_vs_current,
        )

        _LOGGER.info(
            "Optimal window: %s (Q%d-Q%d) @ %.1före/kWh, %.1fh away%s",
            result.start_time.strftime("%H:%M"),
            result.quarters[0],
            result.quarters[-1],
            result.avg_price,
            result.hours_until,
            f", {savings_vs_current:.0f}% savings" if savings_vs_current else "",
        )

        return result

    def calculate_lookahead_hours(
        self,
        heating_type: str,
        thermal_mass: float = 1.0,
        next_demand_hours: float | None = None,
    ) -> float:
        """Calculate dynamic lookahead horizon for price forecast.

        Shared calculation for both space heating and DHW optimization.

        Args:
            heating_type: "space" for space heating, "dhw" for domestic hot water
            thermal_mass: Building thermal mass multiplier (0.5-2.0, space heating only)
            next_demand_hours: Hours until next DHW demand period (DHW only)

        Returns:
            Lookahead hours (adaptive based on context):
            - Space heating: Base hours × thermal_mass (2-8h range)
            - DHW: Hours until next demand period (capped at 24h)
        """
        if heating_type == "space":
            return PRICE_FORECAST_BASE_HORIZON * thermal_mass
        else:  # dhw
            if next_demand_hours is not None:
                return max(1.0, min(next_demand_hours, 24.0))
            return 24.0  # Default full day lookahead

    def get_price_forecast(
        self,
        current_quarter: int,
        price_data: PriceData,
        lookahead_hours: float = 4.0,
    ) -> PriceForecast:
        """Analyze upcoming price periods for optimization decisions.

        Consolidated forecast logic used by:
        - price_layer.evaluate_layer(): Space heating decisions
        - dhw_optimizer.should_start_dhw(): DHW decisions
        - coordinator._calculate_dhw_recommendation(): User-facing summary

        Extracted from evaluate_layer() for shared reuse.

        Args:
            current_quarter: Current 15-min period (0-95)
            price_data: Today + tomorrow prices
            lookahead_hours: How far ahead to analyze (scales with thermal_mass)

        Returns:
            PriceForecast with detailed upcoming period info
        """
        if not price_data or not price_data.today:
            return PriceForecast(
                next_cheap_quarters_away=None,
                cheap_period_duration=0,
                cheap_price_ratio=1.0,
                next_expensive_quarters_away=None,
                expensive_period_duration=0,
                expensive_price_ratio=1.0,
                is_volatile=False,
                current_run_length=0,
                remaining_quarters=0,
                volatile_reason="",
                is_ending_soon=False,
                in_peak_cluster=False,
            )

        # Bound check quarter index (safety)
        if current_quarter >= len(price_data.today):
            current_quarter = min(current_quarter, len(price_data.today) - 1)

        current_period = price_data.today[current_quarter]
        current_price = current_period.price
        current_classification = self.get_current_classification(current_quarter)

        # Initialize forecast result values
        cheap_quarters_away = None
        cheap_duration = 0
        cheap_ratio = 1.0
        expensive_quarters_away = None
        expensive_duration = 0
        expensive_ratio = 1.0

        # Build list of upcoming periods
        forecast_quarters = int(lookahead_hours * 4)
        lookahead_end = min(current_quarter + forecast_quarters, 96)
        upcoming_periods = price_data.today[current_quarter + 1 : lookahead_end]

        # Include tomorrow's periods if near end of day
        if price_data.has_tomorrow and lookahead_end >= 96:
            remaining_quarters = forecast_quarters - (96 - current_quarter - 1)
            upcoming_periods.extend(price_data.tomorrow[:remaining_quarters])

        # Analyze upcoming periods if we have data
        if upcoming_periods and current_price > 0:
            # Find cheap and expensive periods
            min_price = min(p.price for p in upcoming_periods)
            max_price = max(p.price for p in upcoming_periods)

            min_idx = next(i for i, p in enumerate(upcoming_periods) if p.price == min_price)
            max_idx = next(i for i, p in enumerate(upcoming_periods) if p.price == max_price)

            # Calculate price ratios
            cheap_ratio = min_price / current_price
            expensive_ratio = max_price / current_price

            # Count consecutive quarters around min price meeting CHEAP threshold
            if cheap_ratio < PRICE_FORECAST_CHEAP_THRESHOLD:
                cheap_quarters_away = min_idx + 1  # +1 because we skip current
                cheap_duration = 1
                for i in range(min_idx + 1, len(upcoming_periods)):
                    if upcoming_periods[i].price / current_price < PRICE_FORECAST_CHEAP_THRESHOLD:
                        cheap_duration += 1
                    else:
                        break
                for i in range(min_idx - 1, -1, -1):
                    if upcoming_periods[i].price / current_price < PRICE_FORECAST_CHEAP_THRESHOLD:
                        cheap_duration += 1
                    else:
                        break

            # Count consecutive quarters around max price meeting EXPENSIVE threshold
            if expensive_ratio > PRICE_FORECAST_EXPENSIVE_THRESHOLD:
                expensive_quarters_away = max_idx + 1  # +1 because we skip current
                expensive_duration = 1
                for i in range(max_idx + 1, len(upcoming_periods)):
                    if (
                        upcoming_periods[i].price / current_price
                        > PRICE_FORECAST_EXPENSIVE_THRESHOLD
                    ):
                        expensive_duration += 1
                    else:
                        break
                for i in range(max_idx - 1, -1, -1):
                    if (
                        upcoming_periods[i].price / current_price
                        > PRICE_FORECAST_EXPENSIVE_THRESHOLD
                    ):
                        expensive_duration += 1
                    else:
                        break

        # Get volatility and cluster info from shared helper
        volatile_info = get_volatile_info(self, price_data, current_quarter)
        is_volatile = volatile_info.is_volatile
        volatile_reason = volatile_info.reason
        current_run_length = volatile_info.run_length
        remaining_quarters = volatile_info.remaining_quarters
        in_peak_cluster = volatile_info.in_peak_cluster
        is_ending_soon = volatile_info.is_ending_soon

        return PriceForecast(
            next_cheap_quarters_away=cheap_quarters_away,
            cheap_period_duration=cheap_duration,
            cheap_price_ratio=cheap_ratio,
            next_expensive_quarters_away=expensive_quarters_away,
            expensive_period_duration=expensive_duration,
            expensive_price_ratio=expensive_ratio,
            is_volatile=is_volatile,
            current_run_length=current_run_length,
            remaining_quarters=remaining_quarters,
            volatile_reason=volatile_reason,
            is_ending_soon=is_ending_soon,
            in_peak_cluster=in_peak_cluster,
        )

    def evaluate_layer(
        self,
        nibe_state,
        price_data: PriceData,
        thermal_mass: float,
        target_temp: float,
        tolerance: float,
        mode_config: OptimizationModeConfig,
        gespot_entity: str = "unknown",
        enable_price_optimization: bool = True,
    ) -> PriceLayerDecision:
        """Spot price layer: Forward-looking optimization from native 15-minute spot price data.

        Enhanced Nov 27, 2025: Added forward-looking price analysis
        - Looks ahead 4 hours for significant price changes
        - Reduces heating when much cheaper period coming soon
        - Pre-heats when much more expensive period approaching

        Refactored Dec 2025: Uses shared get_price_forecast() for forecast analysis
        to reduce code duplication with DHW optimizer.

        Args:
            nibe_state: Current NIBE state (for strategic overshoot context)
            price_data: Spot price data with native 15-min intervals
            thermal_mass: Building thermal mass (0.5-2.0)
            target_temp: Target indoor temperature
            tolerance: User tolerance setting (0.5-3.0)
            mode_config: Optimization mode configuration
            gespot_entity: Name of the price entity for display
            enable_price_optimization: Whether price optimization is enabled

        Returns:
            PriceLayerDecision with price-based offset
        """
        # Check if price optimization is enabled
        if not enable_price_optimization:
            return PriceLayerDecision(
                name="Spot Price",
                offset=0.0,
                weight=0.0,
                reason="Disabled by user",
            )

        if not price_data or not price_data.today:
            return PriceLayerDecision(
                name="Spot Price", offset=0.0, weight=0.0, reason="No price data"
            )

        now = dt_util.now()
        current_quarter = get_current_quarter(now)

        # Bound check quarter index (safety)
        if current_quarter >= len(price_data.today):
            _LOGGER.warning(
                "Current quarter %d exceeds available periods (%d)",
                current_quarter,
                len(price_data.today),
            )
            current_quarter = min(current_quarter, len(price_data.today) - 1)

        # Get current period classification and price
        classification = self.get_current_classification(current_quarter)
        current_period = price_data.today[current_quarter]
        current_price = current_period.price

        # Forward-looking price analysis - horizon scales with thermal mass
        # Base 4h × thermal_mass (0.5-2.0) → 2.0-8.0 hour adaptive horizon
        forecast_hours = self.calculate_lookahead_hours("space", thermal_mass=thermal_mass)

        # Use shared forecast method for cluster detection and price analysis
        forecast = self.get_price_forecast(current_quarter, price_data, forecast_hours)

        # Get volatility and cluster info from forecast (computed via helper)
        is_volatile = forecast.is_volatile
        is_ending_soon = forecast.is_ending_soon
        in_peak_cluster = forecast.in_peak_cluster
        volatile_reason = f" | {forecast.volatile_reason}" if forecast.volatile_reason else ""

        # Calculate forecast adjustment based on upcoming price periods
        forecast_adjustment = 0.0
        forecast_reason = ""

        # Use forecast data to determine adjustments
        cheap_quarters_away = forecast.next_cheap_quarters_away
        cheap_duration = forecast.cheap_period_duration
        cheap_ratio = forecast.cheap_price_ratio
        expensive_quarters_away = forecast.next_expensive_quarters_away
        expensive_duration = forecast.expensive_period_duration
        expensive_ratio = forecast.expensive_price_ratio

        if classification in BENEFICIAL_CLASSIFICATIONS:
            # Pre-heat before upcoming expensive periods (if sustained AND far enough away)
            if (
                expensive_quarters_away is not None
                and expensive_ratio > PRICE_FORECAST_EXPENSIVE_THRESHOLD
                and expensive_duration >= PRICE_FORECAST_MIN_DURATION
                and expensive_quarters_away >= PRICE_FORECAST_MIN_DURATION
            ):
                increase_percent = int((expensive_ratio - 1) * 100)
                forecast_adjustment = PRICE_FORECAST_PREHEAT_OFFSET
                forecast_reason = (
                    f" | Forecast: {increase_percent}% more expensive in "
                    f"{expensive_quarters_away // 4}h - pre-heat now"
                )
            elif expensive_ratio > PRICE_FORECAST_EXPENSIVE_THRESHOLD:
                # Expensive period exists but doesn't meet criteria
                if expensive_duration < PRICE_FORECAST_MIN_DURATION:
                    forecast_reason = (
                        f" | Forecast: Expensive period too brief "
                        f"({expensive_duration * 15}min < {PRICE_FORECAST_MIN_DURATION * 15}min)"
                    )
                elif (
                    expensive_quarters_away is not None
                    and expensive_quarters_away < PRICE_FORECAST_MIN_DURATION
                ):
                    forecast_reason = (
                        f" | Forecast: Expensive period too soon "
                        f"({expensive_quarters_away * 15}min < "
                        f"{PRICE_FORECAST_MIN_DURATION * 15}min lookahead)"
                    )

        elif classification in [
            QuarterClassification.EXPENSIVE,
            QuarterClassification.PEAK,
        ]:
            # Wait for upcoming cheap periods (if sustained AND far enough away)
            if (
                cheap_quarters_away is not None
                and cheap_ratio < PRICE_FORECAST_CHEAP_THRESHOLD
                and cheap_duration >= PRICE_FORECAST_MIN_DURATION
                and cheap_quarters_away >= PRICE_FORECAST_MIN_DURATION
            ):
                savings_percent = int((1 - cheap_ratio) * 100)
                forecast_adjustment = PRICE_FORECAST_REDUCTION_OFFSET
                forecast_reason = (
                    f" | Forecast: {savings_percent}% cheaper in "
                    f"{cheap_quarters_away // 4}h - reduce heating"
                )
            elif cheap_ratio < PRICE_FORECAST_CHEAP_THRESHOLD:
                # Cheap period exists but doesn't meet criteria
                if cheap_duration < PRICE_FORECAST_MIN_DURATION:
                    forecast_reason = (
                        f" | Forecast: Cheap period too brief "
                        f"({cheap_duration * 15}min < {PRICE_FORECAST_MIN_DURATION * 15}min)"
                    )
                elif (
                    cheap_quarters_away is not None
                    and cheap_quarters_away < PRICE_FORECAST_MIN_DURATION
                ):
                    forecast_reason = (
                        f" | Forecast: Cheap period too soon "
                        f"({cheap_quarters_away * 15}min < "
                        f"{PRICE_FORECAST_MIN_DURATION * 15}min lookahead)"
                    )

        else:  # NORMAL - check both directions, take most significant sustained change
            expensive_valid = (
                expensive_quarters_away is not None
                and expensive_ratio > PRICE_FORECAST_EXPENSIVE_THRESHOLD
                and expensive_duration >= PRICE_FORECAST_MIN_DURATION
                and expensive_quarters_away >= PRICE_FORECAST_MIN_DURATION
            )
            cheap_valid = (
                cheap_quarters_away is not None
                and cheap_ratio < PRICE_FORECAST_CHEAP_THRESHOLD
                and cheap_duration >= PRICE_FORECAST_MIN_DURATION
                and cheap_quarters_away >= PRICE_FORECAST_MIN_DURATION
            )

            if expensive_valid and cheap_valid:
                # Both valid - choose larger magnitude
                if (expensive_ratio - 1.0) > (1.0 - cheap_ratio):
                    increase_percent = int((expensive_ratio - 1) * 100)
                    forecast_adjustment = PRICE_FORECAST_PREHEAT_OFFSET
                    forecast_reason = (
                        f" | Forecast: {increase_percent}% more expensive in "
                        f"{expensive_quarters_away // 4}h - pre-heat now"
                    )
                else:
                    savings_percent = int((1 - cheap_ratio) * 100)
                    forecast_adjustment = PRICE_FORECAST_REDUCTION_OFFSET
                    forecast_reason = (
                        f" | Forecast: {savings_percent}% cheaper in "
                        f"{cheap_quarters_away // 4}h - reduce heating"
                    )
            elif expensive_valid:
                increase_percent = int((expensive_ratio - 1) * 100)
                forecast_adjustment = PRICE_FORECAST_PREHEAT_OFFSET
                forecast_reason = (
                    f" | Forecast: {increase_percent}% more expensive in "
                    f"{expensive_quarters_away // 4}h - pre-heat now"
                )
            elif cheap_valid:
                savings_percent = int((1 - cheap_ratio) * 100)
                forecast_adjustment = PRICE_FORECAST_REDUCTION_OFFSET
                forecast_reason = (
                    f" | Forecast: {savings_percent}% cheaper in "
                    f"{cheap_quarters_away // 4}h - reduce heating"
                )
            else:
                # Check if either direction had potential but didn't meet criteria
                if expensive_ratio > PRICE_FORECAST_EXPENSIVE_THRESHOLD:
                    if expensive_duration < PRICE_FORECAST_MIN_DURATION:
                        forecast_reason = (
                            f" | Forecast: Expensive period too brief "
                            f"({expensive_duration * 15}min)"
                        )
                    elif (
                        expensive_quarters_away is not None
                        and expensive_quarters_away < PRICE_FORECAST_MIN_DURATION
                    ):
                        forecast_reason = (
                            f" | Forecast: Expensive period too soon "
                            f"({expensive_quarters_away * 15}min)"
                        )
                elif cheap_ratio < PRICE_FORECAST_CHEAP_THRESHOLD:
                    if cheap_duration < PRICE_FORECAST_MIN_DURATION:
                        forecast_reason = (
                            f" | Forecast: Cheap period too brief ({cheap_duration * 15}min)"
                        )
                    elif (
                        cheap_quarters_away is not None
                        and cheap_quarters_away < PRICE_FORECAST_MIN_DURATION
                    ):
                        forecast_reason = (
                            f" | Forecast: Cheap period too soon "
                            f"({cheap_quarters_away * 15}min)"
                        )

        # DM debt gate: Don't suppress heating when thermal debt exists (Dec 13, 2025)
        # Price layer was fighting thermal recovery by applying -1.5°C during critical DM debt.
        # When in debt, switch to gentle positive offset to aid recovery while respecting savings intent.
        if forecast_adjustment < 0 and nibe_state.degree_minutes < WEATHER_COMP_DEFER_DM_CRITICAL:
            forecast_reason = (
                f" | DM debt ({nibe_state.degree_minutes:.0f}): "
                f"forecast {forecast_adjustment:+.1f}→{PRICE_FORECAST_DM_DEBT_OFFSET:+.1f}°C"
            )
            forecast_adjustment = PRICE_FORECAST_DM_DEBT_OFFSET

        # Get base offset for current classification
        base_offset = self.get_base_offset(
            current_quarter,
            classification,
            current_period.is_daytime,
        )

        # PEAK cluster: EXPENSIVE quarters between PEAKs use PEAK offset
        peak_cluster_reason = ""
        if in_peak_cluster:
            base_offset = PRICE_OFFSET_PEAK
            peak_cluster_reason = " | PEAK cluster: EXPENSIVE sandwiched between PEAKs"

        # Pre-PEAK detection (Dec 2, 2025)
        # Start reducing 1 quarter BEFORE peak to allow pump slowdown time
        pre_peak_reason = ""
        next_quarter = current_quarter + 1
        if next_quarter < 96:
            next_classification = self.get_current_classification(next_quarter)
            if (
                next_classification == QuarterClassification.PEAK
                and classification != QuarterClassification.PEAK
            ):
                base_offset = min(base_offset, PRICE_PRE_PEAK_OFFSET)
                pre_peak_reason = " | Pre-PEAK: reducing 1Q early for pump slowdown"
        elif price_data.has_tomorrow:
            next_classification = self.get_tomorrow_classification(0)
            if (
                next_classification == QuarterClassification.PEAK
                and classification != QuarterClassification.PEAK
            ):
                base_offset = min(base_offset, PRICE_PRE_PEAK_OFFSET)
                pre_peak_reason = " | Pre-PEAK: reducing 1Q early for pump slowdown"

        # Skip heating boost for volatile periods or when favorable period ending soon
        if should_skip_volatile_boost(is_volatile, base_offset, is_ending_soon):
            base_offset = 0.0  # Treat as NORMAL

        # Adjust for tolerance setting (0.5-3.0 scale → 0.2-1.0 factor)
        tolerance_range = PRICE_TOLERANCE_MAX - PRICE_TOLERANCE_MIN
        factor_range = PRICE_TOLERANCE_FACTOR_MAX - PRICE_TOLERANCE_FACTOR_MIN
        tolerance_factor = (
            PRICE_TOLERANCE_FACTOR_MIN
            + ((tolerance - PRICE_TOLERANCE_MIN) / tolerance_range) * factor_range
        )

        # Apply mode multiplier (comfort=0.7, balanced=1.0, savings=1.3)
        tolerance_factor *= mode_config.price_tolerance_multiplier

        # Savings mode: PEAK bypasses tolerance (always full reduction)
        if classification == QuarterClassification.PEAK and mode_config.peak_bypass_tolerance:
            adjusted_offset = base_offset
        else:
            adjusted_offset = base_offset * tolerance_factor

        # Apply forecast adjustment (additive to base classification)
        if classification == QuarterClassification.PEAK:
            final_offset = adjusted_offset  # Already -10 from base_offset
        else:
            final_offset = adjusted_offset + forecast_adjustment

        # Check for strategic overshoot context when pre-heating
        strategic_context = ""
        max_overshoot = mode_config.preheat_overshoot_allowed
        if forecast_adjustment > 0 and nibe_state.indoor_temp > target_temp:
            overshoot = nibe_state.indoor_temp - target_temp
            if overshoot <= max_overshoot:
                strategic_context = (
                    f" | Strategic storage: +{overshoot:.1f}°C overshoot OK "
                    f"(≤{max_overshoot:.1f}°C)"
                )
            else:
                final_offset = max(final_offset - (overshoot - max_overshoot), 0)
                strategic_context = f" | Overshoot {overshoot:.1f}°C > {max_overshoot:.1f}°C limit"

        # Apply weight based on classification and volatility
        if classification == QuarterClassification.PEAK or in_peak_cluster:
            price_weight = 1.0  # Critical priority
        elif is_volatile:
            price_weight = LAYER_WEIGHT_PRICE * VOLATILE_WEIGHT_REDUCTION
        else:
            price_weight = LAYER_WEIGHT_PRICE

        # DEBUG: Log price analysis
        _LOGGER.debug(
            "Price Q%d (%02d:%02d): %.2f öre → %s | Horizon: %.1fh | "
            "Base: %.1f°C | Forecast adj: %.1f°C | Final: %.1f°C | Weight: %.2f%s%s%s%s",
            current_quarter,
            now.hour,
            now.minute,
            current_price,
            classification.name,
            forecast_hours,
            adjusted_offset,
            forecast_adjustment,
            final_offset,
            price_weight,
            volatile_reason,
            strategic_context,
            pre_peak_reason,
            peak_cluster_reason,
        )

        adapter_name = gespot_entity.split(".")[-1] if gespot_entity else "unknown"

        return PriceLayerDecision(
            name="Spot Price",
            offset=final_offset,
            weight=price_weight,
            reason=(
                f"Q{current_quarter}: {classification.name} "
                f"({'day' if current_period.is_daytime else 'night'}) | "
                f"Adapter: {adapter_name} | Horizon: {forecast_hours:.1f}h"
                f"{forecast_reason}{volatile_reason}{strategic_context}"
                f"{pre_peak_reason}{peak_cluster_reason}"
            ),
        )

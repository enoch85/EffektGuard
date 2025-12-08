"""Price analyzer for spot price classification.

Analyzes spot prices with native 15-minute granularity to classify
periods for optimization decisions.
"""

import logging
from dataclasses import dataclass

import numpy as np

from homeassistant.util import dt as dt_util

from ..const import (
    OptimizationModeConfig,
    LAYER_WEIGHT_PRICE,
    PRICE_DAYTIME_MULTIPLIER,
    PRICE_FORECAST_BASE_HORIZON,
    PRICE_FORECAST_CHEAP_THRESHOLD,
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
    PRICE_VOLATILE_WEIGHT_REDUCTION,
    QuarterClassification,
)

_LOGGER = logging.getLogger(__name__)

# Re-export from adapters for convenience
from ..adapters.gespot_adapter import PriceData, QuarterPeriod

__all__ = ["PriceAnalyzer", "PriceData", "PriceLayerDecision", "QuarterPeriod"]


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
        beneficial_classifications = [
            QuarterClassification.VERY_CHEAP,
            QuarterClassification.CHEAP,
        ]
        # Search today's periods
        for quarter in range(current_quarter + 1, 96):
            if self._classifications_today.get(quarter) in beneficial_classifications:
                return quarter

        # Search tomorrow's periods if available
        for quarter in range(96):
            if self._classifications_tomorrow.get(quarter) in beneficial_classifications:
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
        current_quarter = (now.hour * 4) + (now.minute // 15)  # 0-95

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

        # Initialize variables (moved outside if-block to avoid UnboundLocalError)
        remaining_cheap_quarters = 0
        is_volatile = False
        volatile_reason = ""
        in_peak_cluster = False  # True when EXPENSIVE is sandwiched between PEAKs

        # Classifications that benefit from heating (used in multiple places)
        beneficial_classifications = [
            QuarterClassification.VERY_CHEAP,
            QuarterClassification.CHEAP,
        ]

        # Forward-looking price analysis - horizon scales with thermal mass
        # Base 4h × thermal_mass (0.5-2.0) → 2.0-8.0 hour adaptive horizon
        # Calculate horizon based on thermal mass (higher mass = longer lookahead)
        forecast_hours = PRICE_FORECAST_BASE_HORIZON * thermal_mass

        forecast_quarters = int(forecast_hours * 4)  # Convert hours to 15-min quarters

        lookahead_end = min(current_quarter + forecast_quarters, 96)
        upcoming_periods = price_data.today[current_quarter + 1 : lookahead_end]

        # Also check tomorrow's first periods if we're near end of day
        if price_data.has_tomorrow and lookahead_end >= 96:
            remaining_quarters = forecast_quarters - (96 - current_quarter - 1)
            upcoming_periods.extend(price_data.tomorrow[:remaining_quarters])

        forecast_adjustment = 0.0
        forecast_reason = ""

        if upcoming_periods and current_price > 0:
            # Find index and count duration of min/max prices
            min_idx = next(
                i
                for i, p in enumerate(upcoming_periods)
                if p.price == min(p.price for p in upcoming_periods)
            )
            max_idx = next(
                i
                for i, p in enumerate(upcoming_periods)
                if p.price == max(p.price for p in upcoming_periods)
            )

            # Count consecutive quarters around min price meeting CHEAP threshold
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
            expensive_duration = 1
            for i in range(max_idx + 1, len(upcoming_periods)):
                if upcoming_periods[i].price / current_price > PRICE_FORECAST_EXPENSIVE_THRESHOLD:
                    expensive_duration += 1
                else:
                    break
            for i in range(max_idx - 1, -1, -1):
                if upcoming_periods[i].price / current_price > PRICE_FORECAST_EXPENSIVE_THRESHOLD:
                    expensive_duration += 1
                else:
                    break

            # Calculate price ratios
            min_price_ratio = min(p.price for p in upcoming_periods) / current_price
            max_price_ratio = max(p.price for p in upcoming_periods) / current_price

            # Check if current CHEAP/VERY_CHEAP period is too brief for meaningful heating
            # Compressor needs ~45min to efficiently use cheap electricity
            if classification in beneficial_classifications:
                # Count remaining consecutive beneficial quarters (including current)
                remaining_cheap_quarters = 1
                for q in range(current_quarter + 1, 96):
                    if self.get_current_classification(q) in beneficial_classifications:
                        remaining_cheap_quarters += 1
                    else:
                        break

                # Check tomorrow if cheap continues to end of today
                if (
                    remaining_cheap_quarters < PRICE_FORECAST_MIN_DURATION
                    and current_quarter + remaining_cheap_quarters >= 96
                    and self.has_tomorrow_prices()
                ):
                    for q in range(96):
                        if self.get_tomorrow_classification(q) in beneficial_classifications:
                            remaining_cheap_quarters += 1
                        else:
                            break

            # Volatile detection: brief runs get reduced weight (see const.py for details)
            # Calculate current quarter's run length by scanning backwards and forwards
            current_run_length = 1  # Count current quarter

            # Scan backwards to find start of current run
            for offset in range(1, 96):  # Max scan to start of day
                check_quarter = current_quarter - offset
                if check_quarter < 0:
                    break
                check_class = self.get_current_classification(check_quarter)
                if check_class == classification:
                    current_run_length += 1
                else:
                    break

            # Scan forwards to find end of current run
            for offset in range(1, 96):  # Max scan to end of day
                check_quarter = current_quarter + offset
                if check_quarter < 96:
                    check_class = self.get_current_classification(check_quarter)
                elif price_data.has_tomorrow:
                    # Look into tomorrow if available
                    tomorrow_quarter = check_quarter - 96
                    if tomorrow_quarter < 96:
                        check_class = self.get_tomorrow_classification(tomorrow_quarter)
                    else:
                        break
                else:
                    break

                if check_class == classification:
                    current_run_length += 1
                else:
                    break

            # A period is volatile if the current run is brief (< 3 quarters = < 45 min)
            # PEAK is never volatile - it's always predictable grid stress, we always reduce
            is_brief_run = current_run_length < PRICE_FORECAST_MIN_DURATION
            is_ending_soon = remaining_cheap_quarters < PRICE_FORECAST_MIN_DURATION

            # Single volatility flag - either the whole run is short OR we're near the end
            # PEAK is never volatile - it's always predictable grid stress, we always reduce
            is_volatile = (is_brief_run and classification != QuarterClassification.PEAK) or (
                classification == QuarterClassification.CHEAP and is_ending_soon
            )

            # PEAK cluster detection: When EXPENSIVE or NORMAL quarters are sandwiched between PEAKs,
            # treat them as PEAK to prevent offset jumps during peak pricing periods.
            # Example: PEAK(17:00) → NORMAL(17:15) → PEAK(17:30) should all use PEAK offset.
            # CHEAP quarters are NOT included - they should break the cluster.
            # This prevents the heat pump from ramping up during brief dips in peak periods.
            in_peak_cluster = False
            if (
                classification in [QuarterClassification.EXPENSIVE, QuarterClassification.NORMAL]
                and is_volatile
            ):
                # Check if we're sandwiched between PEAK quarters
                # Scan backwards to find if there's a PEAK within the cluster window
                has_peak_before = False
                cluster_break_classifications = [
                    QuarterClassification.VERY_CHEAP,
                    QuarterClassification.CHEAP,
                ]
                for back_offset in range(1, PRICE_FORECAST_MIN_DURATION + 1):
                    check_quarter = current_quarter - back_offset
                    if check_quarter < 0:
                        break
                    check_class = self.get_current_classification(check_quarter)
                    if check_class == QuarterClassification.PEAK:
                        has_peak_before = True
                        break
                    # Stop if we hit CHEAP/VERY_CHEAP - that breaks the cluster
                    if check_class in cluster_break_classifications:
                        break

                # Scan forwards to find if there's a PEAK within the cluster window
                has_peak_after = False
                if has_peak_before:  # Only check forward if we found PEAK before
                    for fwd_offset in range(1, PRICE_FORECAST_MIN_DURATION + 1):
                        check_quarter = current_quarter + fwd_offset
                        if check_quarter >= 96:
                            # Check tomorrow if available
                            if price_data.has_tomorrow:
                                tomorrow_quarter = check_quarter - 96
                                check_class = self.get_tomorrow_classification(tomorrow_quarter)
                            else:
                                break
                        else:
                            check_class = self.get_current_classification(check_quarter)

                        if check_class is None:
                            break
                        if check_class == QuarterClassification.PEAK:
                            has_peak_after = True
                            break
                        # Stop if we hit CHEAP/VERY_CHEAP - that breaks the cluster
                        if check_class in cluster_break_classifications:
                            break

                # If sandwiched between PEAKs, inherit PEAK behavior
                if has_peak_before and has_peak_after:
                    in_peak_cluster = True
                    is_volatile = False  # No longer volatile when part of PEAK cluster

            if is_volatile:
                quarters_left = (
                    min(remaining_cheap_quarters, current_run_length)
                    if classification in beneficial_classifications
                    else current_run_length
                )
                volatile_reason = f" | Price volatile: {classification.name} {quarters_left * 15}min<{PRICE_FORECAST_MIN_DURATION * 15}min"

            # Apply normal forecast logic regardless of volatility
            # Volatility will reduce weight, not block smart decisions
            if classification in beneficial_classifications:
                # Pre-heat before upcoming expensive periods (if sustained AND far enough away)
                # Only act if expensive period is at least 45min in future (same as duration filter)
                if (
                    max_price_ratio > PRICE_FORECAST_EXPENSIVE_THRESHOLD
                    and expensive_duration >= PRICE_FORECAST_MIN_DURATION
                    and max_idx >= PRICE_FORECAST_MIN_DURATION
                ):
                    increase_percent = int((max_price_ratio - 1) * 100)
                    forecast_adjustment = PRICE_FORECAST_PREHEAT_OFFSET
                    forecast_reason = f" | Forecast: {increase_percent}% more expensive in {max_idx//4}h - pre-heat now"
                elif max_price_ratio > PRICE_FORECAST_EXPENSIVE_THRESHOLD:
                    # Expensive period exists but doesn't meet criteria - explain why
                    if expensive_duration < PRICE_FORECAST_MIN_DURATION:
                        forecast_reason = f" | Forecast: Expensive period too brief ({expensive_duration * 15}min < {PRICE_FORECAST_MIN_DURATION * 15}min)"
                    elif max_idx < PRICE_FORECAST_MIN_DURATION:
                        forecast_reason = f" | Forecast: Expensive period too soon ({max_idx * 15}min < {PRICE_FORECAST_MIN_DURATION * 15}min lookahead)"

            elif classification in [
                QuarterClassification.EXPENSIVE,
                QuarterClassification.PEAK,
            ]:
                # Wait for upcoming cheap periods (if sustained AND far enough away)
                # Only reduce if cheap period is at least 45min in future (same as duration filter)
                if (
                    min_price_ratio < PRICE_FORECAST_CHEAP_THRESHOLD
                    and cheap_duration >= PRICE_FORECAST_MIN_DURATION
                    and min_idx >= PRICE_FORECAST_MIN_DURATION
                ):
                    savings_percent = int((1 - min_price_ratio) * 100)
                    forecast_adjustment = PRICE_FORECAST_REDUCTION_OFFSET
                    forecast_reason = (
                        f" | Forecast: {savings_percent}% cheaper in {min_idx//4}h - reduce heating"
                    )
                elif min_price_ratio < PRICE_FORECAST_CHEAP_THRESHOLD:
                    # Cheap period exists but doesn't meet criteria - explain why
                    if cheap_duration < PRICE_FORECAST_MIN_DURATION:
                        forecast_reason = f" | Forecast: Cheap period too brief ({cheap_duration * 15}min < {PRICE_FORECAST_MIN_DURATION * 15}min)"
                    elif min_idx < PRICE_FORECAST_MIN_DURATION:
                        forecast_reason = f" | Forecast: Cheap period too soon ({min_idx * 15}min < {PRICE_FORECAST_MIN_DURATION * 15}min lookahead)"

            else:  # NORMAL - check both directions, take most significant sustained change
                # Apply same lookahead requirement: price change must be ≥45min away to act on
                expensive_valid = (
                    max_price_ratio > PRICE_FORECAST_EXPENSIVE_THRESHOLD
                    and expensive_duration >= PRICE_FORECAST_MIN_DURATION
                    and max_idx >= PRICE_FORECAST_MIN_DURATION
                )
                cheap_valid = (
                    min_price_ratio < PRICE_FORECAST_CHEAP_THRESHOLD
                    and cheap_duration >= PRICE_FORECAST_MIN_DURATION
                    and min_idx >= PRICE_FORECAST_MIN_DURATION
                )

                if expensive_valid and cheap_valid:
                    # Both valid - choose larger magnitude
                    if (max_price_ratio - 1.0) > (1.0 - min_price_ratio):
                        increase_percent = int((max_price_ratio - 1) * 100)
                        forecast_adjustment = PRICE_FORECAST_PREHEAT_OFFSET
                        forecast_reason = f" | Forecast: {increase_percent}% more expensive in {max_idx//4}h - pre-heat now"
                    else:
                        savings_percent = int((1 - min_price_ratio) * 100)
                        forecast_adjustment = PRICE_FORECAST_REDUCTION_OFFSET
                        forecast_reason = f" | Forecast: {savings_percent}% cheaper in {min_idx//4}h - reduce heating"
                elif expensive_valid:
                    increase_percent = int((max_price_ratio - 1) * 100)
                    forecast_adjustment = PRICE_FORECAST_PREHEAT_OFFSET
                    forecast_reason = f" | Forecast: {increase_percent}% more expensive in {max_idx//4}h - pre-heat now"
                elif cheap_valid:
                    savings_percent = int((1 - min_price_ratio) * 100)
                    forecast_adjustment = PRICE_FORECAST_REDUCTION_OFFSET
                    forecast_reason = (
                        f" | Forecast: {savings_percent}% cheaper in {min_idx//4}h - reduce heating"
                    )
                else:
                    # Check if either direction had potential but didn't meet criteria
                    if max_price_ratio > PRICE_FORECAST_EXPENSIVE_THRESHOLD:
                        if expensive_duration < PRICE_FORECAST_MIN_DURATION:
                            forecast_reason = f" | Forecast: Expensive period too brief ({expensive_duration * 15}min)"
                        elif max_idx < PRICE_FORECAST_MIN_DURATION:
                            forecast_reason = (
                                f" | Forecast: Expensive period too soon ({max_idx * 15}min)"
                            )
                    elif min_price_ratio < PRICE_FORECAST_CHEAP_THRESHOLD:
                        if cheap_duration < PRICE_FORECAST_MIN_DURATION:
                            forecast_reason = (
                                f" | Forecast: Cheap period too brief ({cheap_duration * 15}min)"
                            )
                        elif min_idx < PRICE_FORECAST_MIN_DURATION:
                            forecast_reason = (
                                f" | Forecast: Cheap period too soon ({min_idx * 15}min)"
                            )

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
        # Pump needs time to reduce output - acting at PEAK start is too late
        pre_peak_reason = ""
        next_quarter = current_quarter + 1
        if next_quarter < 96:
            next_classification = self.get_current_classification(next_quarter)
            if (
                next_classification == QuarterClassification.PEAK
                and classification != QuarterClassification.PEAK
            ):
                # Next quarter is PEAK but current is not - pre-act now
                base_offset = min(base_offset, PRICE_PRE_PEAK_OFFSET)
                pre_peak_reason = " | Pre-PEAK: reducing 1Q early for pump slowdown"
        elif price_data.has_tomorrow:
            # Check tomorrow Q0 if we're at Q95
            next_classification = self.get_tomorrow_classification(0)
            if (
                next_classification == QuarterClassification.PEAK
                and classification != QuarterClassification.PEAK
            ):
                base_offset = min(base_offset, PRICE_PRE_PEAK_OFFSET)
                pre_peak_reason = " | Pre-PEAK: reducing 1Q early for pump slowdown"

        # Skip heating boost for volatile CHEAP/VERY_CHEAP periods
        # Compressor needs ~45min to be efficient - don't boost for short dips
        if is_volatile and classification in beneficial_classifications and base_offset > 0:
            base_offset = 0.0  # Treat as NORMAL instead of CHEAP/VERY_CHEAP

        # Adjust for tolerance setting (0.5-3.0 scale → 0.2-1.0 factor)
        # Linear interpolation: 0.5 → 0.2 (conservative), 3.0 → 1.0 (full offset)
        tolerance_range = PRICE_TOLERANCE_MAX - PRICE_TOLERANCE_MIN  # 2.5
        factor_range = PRICE_TOLERANCE_FACTOR_MAX - PRICE_TOLERANCE_FACTOR_MIN  # 0.8
        tolerance_factor = (
            PRICE_TOLERANCE_FACTOR_MIN
            + ((tolerance - PRICE_TOLERANCE_MIN) / tolerance_range) * factor_range
        )

        # Apply mode multiplier (comfort=0.7, balanced=1.0, savings=1.3)
        tolerance_factor *= mode_config.price_tolerance_multiplier

        # Savings mode: PEAK bypasses tolerance (always full reduction)
        if classification == QuarterClassification.PEAK and mode_config.peak_bypass_tolerance:
            adjusted_offset = base_offset  # Skip tolerance scaling for PEAK in savings mode
        else:
            adjusted_offset = base_offset * tolerance_factor

        # Apply forecast adjustment (additive to base classification)
        # BUT: During PEAK, we want maximum reduction regardless of forecast
        if classification == QuarterClassification.PEAK:
            # PEAK periods get maximum reduction, no forecast adjustment
            # The goal is -10 offset to coast through the expensive period
            final_offset = adjusted_offset  # Already -10 from base_offset
        else:
            final_offset = adjusted_offset + forecast_adjustment

        # Check for strategic overshoot context when pre-heating
        strategic_context = ""
        max_overshoot = mode_config.preheat_overshoot_allowed
        if forecast_adjustment > 0 and nibe_state.indoor_temp > target_temp:
            overshoot = nibe_state.indoor_temp - target_temp
            if overshoot <= max_overshoot:
                # Calculate cost savings multiplier from price forecast
                if upcoming_periods and current_price > 0:
                    max_upcoming = max(p.price for p in upcoming_periods)
                    cost_multiplier = max_upcoming / current_price
                    strategic_context = f" | Strategic storage: +{overshoot:.1f}°C overshoot OK (≤{max_overshoot:.1f}°C)"
            else:
                # Overshoot exceeds mode limit - reduce pre-heating
                final_offset = max(final_offset - (overshoot - max_overshoot), 0)
                strategic_context = f" | Overshoot {overshoot:.1f}°C > {max_overshoot:.1f}°C limit"

        # Apply moderate volatility weight reduction (Dec 1, 2025)
        # After int accumulation fix, safe to allow stronger price influence during volatility
        # Math: 0.8 × 0.3 = 0.24 (price layer reduced to 30% of normal strength)
        # Was 0.1 (10%) before fix when decimal changes caused API oscillation
        # Effect: Thermal/comfort/weather layers still dominate, but price has meaningful input
        #
        # Dec 3, 2025: PEAK classification gets weight 1.0 (critical priority)
        # This ensures peak avoidance overrides all other layers except safety
        # PEAK cluster: EXPENSIVE quarters between PEAKs also get PEAK treatment
        #
        # Dec 5, 2025: Only reduce weight for volatile CHEAP/VERY_CHEAP periods, not EXPENSIVE
        # Volatility logic was designed to prevent oscillation during short cheap dips
        # But we WANT strong price influence during EXPENSIVE periods to reduce heating
        if classification == QuarterClassification.PEAK or in_peak_cluster:
            price_weight = 1.0  # Critical priority - override all other layers
        elif is_volatile and classification in beneficial_classifications:
            # Only reduce weight for volatile CHEAP/VERY_CHEAP periods (prevent ramp-up for brief dips)
            # EXPENSIVE periods keep full weight even if volatile - we want to reduce heating!
            price_weight = LAYER_WEIGHT_PRICE * PRICE_VOLATILE_WEIGHT_REDUCTION  # 0.8 → 0.24
        else:
            price_weight = LAYER_WEIGHT_PRICE

        # DEBUG: Log price analysis with thermal mass horizon calculation
        _LOGGER.debug(
            "Price Q%d (%02d:%02d): %.2f öre → %s | Horizon: %.1fh (%.1f base × %.1f thermal_mass) | Base: %.1f°C | Forecast adj: %.1f°C | Final: %.1f°C | Weight: %.2f%s%s%s%s",
            current_quarter,
            now.hour,
            now.minute,
            current_price,
            classification.name,
            forecast_hours,
            PRICE_FORECAST_BASE_HORIZON,
            thermal_mass,
            adjusted_offset,
            forecast_adjustment,
            final_offset,
            price_weight,
            volatile_reason,
            strategic_context,
            pre_peak_reason,
            peak_cluster_reason,
        )

        # Get adapter entity for display
        # Extract short name from entity_id (e.g., "sensor.gespot_current_price_se2" -> "gespot_current_price_se2")
        adapter_name = gespot_entity.split(".")[-1] if gespot_entity else "unknown"

        return PriceLayerDecision(
            name="Spot Price",
            offset=final_offset,
            weight=price_weight,
            reason=f"Q{current_quarter}: {classification.name} ({'day' if current_period.is_daytime else 'night'}) | Adapter: {adapter_name} | Horizon: {forecast_hours:.1f}h ({PRICE_FORECAST_BASE_HORIZON:.1f} × {thermal_mass:.1f}){forecast_reason}{volatile_reason}{strategic_context}{pre_peak_reason}{peak_cluster_reason}",
        )

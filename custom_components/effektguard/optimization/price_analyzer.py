"""Price analyzer for spot price classification.

Analyzes GE-Spot prices with native 15-minute granularity to classify
periods for optimization decisions.
"""

import logging
from dataclasses import dataclass

import numpy as np

from ..const import QuarterClassification

_LOGGER = logging.getLogger(__name__)

# Re-export from adapters for convenience
from ..adapters.gespot_adapter import PriceData, QuarterPeriod

__all__ = ["PriceAnalyzer", "PriceData", "QuarterPeriod"]


class PriceAnalyzer:
    """Analyze electricity spot prices with native 15-minute granularity.

    GE-Spot provides true quarterly data (96 intervals per day), which
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
        """Classify 15-minute periods from native GE-Spot data.

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

        # Calculate percentiles
        p25 = np.percentile(prices, 25)
        p50 = np.percentile(prices, 50)
        p75 = np.percentile(prices, 75)
        p90 = np.percentile(prices, 90)

        _LOGGER.debug(
            "Price percentiles - P25: %.3f, P50: %.3f, P75: %.3f, P90: %.3f",
            p25,
            p50,
            p75,
            p90,
        )

        # Special case: Uniform prices (all equal) - happens with fallback mode
        # When GE-Spot unavailable, fallback creates 96 periods with price=1.0
        # Without variance, classification is meaningless - mark all as NORMAL
        if p25 == p90:  # No price variance
            _LOGGER.info(
                "Uniform prices detected (%.3f), classifying all periods as NORMAL (no optimization)",
                p25,
            )
            return {period.quarter_of_day: QuarterClassification.NORMAL for period in periods}

        # Classify each period
        classifications = {}
        for period in periods:
            if period.price <= p25:
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
            classification: CHEAP/NORMAL/EXPENSIVE/PEAK
            is_daytime: True if 06:00-22:00 (full effect tariff weight)

        Returns offset in °C:
            CHEAP: +3.0 (pre-heat opportunity, charge thermal battery!)
            NORMAL: 0.0 (maintain)
            EXPENSIVE: -1.0 (conserve)
            PEAK: -2.0 (minimize)

        Effect tariff weighting:
            - Daytime (06:00-22:00): Full weight, more aggressive reductions
            - Nighttime (22:00-06:00): 50% weight, gentler reductions

        Note: CHEAP includes negative prices (you get paid to heat!)
        """
        # Base offsets for each classification
        base_offsets = {
            QuarterClassification.CHEAP: +3.0,  # Increased from +2.0 for better thermal storage
            QuarterClassification.NORMAL: 0.0,
            QuarterClassification.EXPENSIVE: -1.0,
            QuarterClassification.PEAK: -2.0,
        }

        offset = base_offsets[classification]

        # More aggressive during daytime (full effect tariff weight)
        # Nighttime peaks are less critical (50% tariff weight)
        if is_daytime and classification in [
            QuarterClassification.EXPENSIVE,
            QuarterClassification.PEAK,
        ]:
            offset *= 1.5  # More aggressive reduction during daytime

        return offset

    def get_current_classification(self, quarter: int) -> QuarterClassification:
        """Get classification for current quarter.

        Args:
            quarter: Quarter of day (0-95)

        Returns:
            Classification for the quarter
        """
        return self._classifications_today.get(quarter, QuarterClassification.NORMAL)

    def get_next_cheap_period(self, current_quarter: int) -> int | None:
        """Find next cheap period after current quarter.

        Useful for pre-heating scheduling.

        Args:
            current_quarter: Current quarter of day (0-95)

        Returns:
            Quarter number of next cheap period, or None if none found
        """
        # Search today's periods
        for quarter in range(current_quarter + 1, 96):
            if self._classifications_today.get(quarter) == QuarterClassification.CHEAP:
                return quarter

        # Search tomorrow's periods if available
        for quarter in range(96):
            if self._classifications_tomorrow.get(quarter) == QuarterClassification.CHEAP:
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

"""Whole-day volatility simulation tests.

These tests iterate all 96 quarters and validate the invariants of the shared
volatility helper against known classification patterns.

Goal: catch regressions where volatility detection flags stable multi-hour runs
or miscomputes run length / remaining quarters.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.effektguard.const import PRICE_FORECAST_MIN_DURATION, QuarterClassification
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer, PriceData, QuarterPeriod
from custom_components.effektguard.utils.volatile_helpers import get_volatile_info


def _create_price_data(prices: list[float], base_time: datetime) -> PriceData:
    assert len(prices) == 96
    today = [
        QuarterPeriod(
            start_time=base_time + timedelta(minutes=15 * i),
            price=price,
        )
        for i, price in enumerate(prices)
    ]
    return PriceData(today=today, tomorrow=[], has_tomorrow=False)


def _contiguous_run_info(
    analyzer: PriceAnalyzer, q: int
) -> tuple[QuarterClassification, int, int, int, int]:
    """Compute contiguous run info (same classification) around quarter q.

    Returns: (classification, start_q, end_q, run_length, remaining_quarters)
    """
    classification = analyzer.get_current_classification(q)

    start_q = q
    while start_q > 0 and analyzer.get_current_classification(start_q - 1) == classification:
        start_q -= 1

    end_q = q
    while end_q < 95 and analyzer.get_current_classification(end_q + 1) == classification:
        end_q += 1

    run_length = (end_q - start_q) + 1
    remaining_quarters = (end_q - q) + 1
    return classification, start_q, end_q, run_length, remaining_quarters


class TestVolatilityWholeDaySimulation:
    def test_stable_multi_hour_runs_are_not_volatile(self):
        """Stable multi-hour runs should never be marked volatile.

        This uses clear three-block pricing to force stable classifications:
        - Low block (0-23)
        - Medium block (24-71)
        - High block (72-95)

        Even if class labels shift due to percentiles, any contiguous run >= 3 quarters
        must not be considered volatile.
        """
        analyzer = PriceAnalyzer()
        base_time = datetime(2025, 12, 19, tzinfo=timezone.utc)

        prices = [10.0] * 24 + [50.0] * 48 + [100.0] * 24
        price_data = _create_price_data(prices, base_time)
        analyzer.update_prices(price_data)

        for q in range(96):
            info = get_volatile_info(analyzer, price_data, current_quarter=q)
            _, _, _, run_length, _ = _contiguous_run_info(analyzer, q)

            assert info.run_length == run_length
            if run_length >= PRICE_FORECAST_MIN_DURATION:
                assert info.is_volatile is False

    def test_short_beneficial_run_is_volatile_and_ending_soon(self):
        """A brief beneficial run should be volatile and also be flagged as ending soon.

        This sets classifications directly to avoid percentile sensitivity:
        - Q10-Q11: CHEAP (2 quarters)
        - Everything else: NORMAL

        Expected:
        - Q10/Q11: volatile (run_length=2 < 3)
        - Q10: remaining=2 => ending_soon=True
        - Q11: remaining=1 => ending_soon=True
        """
        analyzer = PriceAnalyzer()

        # Price data isn't used by get_volatile_info beyond bounds/tomorrow checks,
        # but it must have 96 entries.
        base_time = datetime(2025, 12, 19, tzinfo=timezone.utc)
        price_data = _create_price_data([1.0] * 96, base_time)

        analyzer._classifications_today = {q: QuarterClassification.NORMAL for q in range(96)}
        analyzer._classifications_today[10] = QuarterClassification.CHEAP
        analyzer._classifications_today[11] = QuarterClassification.CHEAP

        for q, expected_remaining in [(10, 2), (11, 1)]:
            info = get_volatile_info(analyzer, price_data, current_quarter=q)

            assert info.classification_name == QuarterClassification.CHEAP.name
            assert info.run_length == 2
            assert info.remaining_quarters == expected_remaining
            assert info.is_volatile is True
            assert info.is_ending_soon is True

        # Neighboring NORMAL quarters are run_length >= 3 (since we only changed 2 quarters)
        for q in [9, 12]:
            info = get_volatile_info(analyzer, price_data, current_quarter=q)
            assert info.is_volatile is False

    def test_single_quarter_island_between_same_class_is_merged(self):
        """A 1-quarter island between the same class should not cause volatility.

        Pattern:
        - Q13-Q14: CHEAP
        - Q15: NORMAL (1-quarter island)
        - Q16: CHEAP

        With island merging enabled, Q15 should be treated as part of a stable CHEAP context
        for volatility detection, and therefore NOT be volatile.
        """
        analyzer = PriceAnalyzer()

        base_time = datetime(2025, 12, 19, tzinfo=timezone.utc)
        price_data = _create_price_data([1.0] * 96, base_time)

        analyzer._classifications_today = {q: QuarterClassification.NORMAL for q in range(96)}
        analyzer._classifications_today[13] = QuarterClassification.CHEAP
        analyzer._classifications_today[14] = QuarterClassification.CHEAP
        analyzer._classifications_today[15] = QuarterClassification.NORMAL
        analyzer._classifications_today[16] = QuarterClassification.CHEAP

        info_q15 = get_volatile_info(analyzer, price_data, current_quarter=15)
        assert info_q15.is_volatile is False


@pytest.mark.parametrize("q", [0, 1, 50, 95])
def test_run_length_matches_contiguous_scan_on_any_day(q: int):
    """Sanity check: helper run-length equals contiguous same-classification length."""
    analyzer = PriceAnalyzer()
    base_time = datetime(2025, 12, 19, tzinfo=timezone.utc)

    # A deterministic day with gentle variation (avoids uniform special-case).
    prices = [float(i % 24) for i in range(96)]
    price_data = _create_price_data(prices, base_time)
    analyzer.update_prices(price_data)

    info = get_volatile_info(analyzer, price_data, current_quarter=q)
    _, _, _, run_length, remaining = _contiguous_run_info(analyzer, q)

    assert info.run_length == run_length
    assert info.remaining_quarters == remaining

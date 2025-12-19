"""Volatile price period detection.

This module contains common logic for handling volatile price periods.
All layers (price, thermal, decision_engine, dhw) should use these helpers
to ensure consistent behavior.

Volatile periods are brief price runs that may not last long enough to justify
heating adjustments. During these periods, we skip heating boosts and reduce
weights to prevent unnecessary wear on the heat pump.

Volatility is determined by the current price run length:
- If current classification (CHEAP, NORMAL, etc.) runs for < 3 quarters (45 min),
  it's considered volatile
- 45 min is based on compressor dynamics: 30 min ramp-up + 15 min cool-down
- Volatility scanning always checks the full day (96 quarters)

PEAK cluster detection: Short EXPENSIVE/NORMAL periods sandwiched between
PEAK periods are treated as part of the PEAK cluster (not volatile).
"""

from dataclasses import dataclass

from ..const import (
    BENEFICIAL_CLASSIFICATIONS,
    PRICE_FORECAST_MIN_DURATION,
    QuarterClassification,
    VOLATILE_ISLAND_MERGE_MAX_QUARTERS,
)
from .time_utils import get_current_quarter


def _build_classification_sequence(
    price_analyzer,
    price_data,
) -> list[QuarterClassification]:
    """Build a contiguous classification sequence for today (+tomorrow if available)."""

    today = [price_analyzer.get_current_classification(q) for q in range(96)]
    if not getattr(price_data, "has_tomorrow", False):
        return today

    tomorrow = [price_analyzer.get_tomorrow_classification(q) for q in range(96)]
    return today + tomorrow


def _merge_short_islands(
    classifications: list[QuarterClassification],
    max_island_quarters: int,
) -> list[QuarterClassification]:
    """Merge brief sandwiched runs into surrounding classification.

    This is used only for volatility run-length calculations to reduce false
    volatility from percentile boundary hopping.

    This intentionally only merges brief NORMAL islands; it does NOT merge
    brief CHEAP/VERY_CHEAP or EXPENSIVE islands, to avoid masking real price
    opportunities/spikes.

    PEAK is never merged and no island is merged *into* PEAK.
    """

    if max_island_quarters <= 0:
        return classifications

    merged = list(classifications)
    n = len(merged)
    if n < 3:
        return merged

    i = 0
    while i < n:
        cls = merged[i]
        j = i
        while j + 1 < n and merged[j + 1] == cls:
            j += 1

        run_len = (j - i) + 1
        if (
            run_len <= max_island_quarters
            and i > 0
            and j < n - 1
            and merged[i - 1] == merged[j + 1]
            and merged[i - 1] != cls
            and cls == QuarterClassification.NORMAL
            and cls != QuarterClassification.PEAK
            and merged[i - 1] != QuarterClassification.PEAK
        ):
            fill_cls = merged[i - 1]
            for k in range(i, j + 1):
                merged[k] = fill_cls

        i = j + 1

    return merged


@dataclass
class VolatileInfo:
    """Volatility detection result with details for logging."""

    is_volatile: bool
    run_length: int  # Total run length (backwards + forwards)
    remaining_quarters: int  # Quarters remaining forward with same classification
    classification_name: str
    reason: str  # Human-readable reason for logging
    in_peak_cluster: bool  # True if sandwiched between PEAK periods
    is_ending_soon: bool  # True if favorable period ending soon (< 3 quarters remain)


def get_volatile_info(
    price_analyzer,
    price_data,
    current_quarter: int | None = None,
) -> VolatileInfo:
    """Get detailed volatility info for current period.

    Counts total run length (backwards + forwards) with the same classification.
    If < 3 quarters (45 min) total, period is volatile (unless in PEAK cluster).

    PEAK cluster: Short EXPENSIVE/NORMAL periods between PEAK periods are
    not volatile - they should be treated as part of the expensive cluster.

    Args:
        price_analyzer: PriceAnalyzer instance with classification methods
        price_data: Current price data (may be None)
        current_quarter: Quarter to check (0-95). If None, uses current time.

    Returns:
        VolatileInfo with is_volatile flag, run length, cluster info, and reason
    """
    if not price_data or not price_data.today:
        return VolatileInfo(
            is_volatile=False,
            run_length=0,
            remaining_quarters=0,
            classification_name="UNKNOWN",
            reason="",
            in_peak_cluster=False,
            is_ending_soon=False,
        )

    if current_quarter is None:
        current_quarter = get_current_quarter()

    if current_quarter >= len(price_data.today):
        return VolatileInfo(
            is_volatile=False,
            run_length=0,
            remaining_quarters=0,
            classification_name="UNKNOWN",
            reason="",
            in_peak_cluster=False,
            is_ending_soon=False,
        )

    # Use a smoothed classification sequence for volatility only.
    # This reduces false positives from percentile boundary hopping (e.g., single-quarter islands).
    raw_classifications = _build_classification_sequence(price_analyzer, price_data)
    classifications = _merge_short_islands(
        raw_classifications,
        VOLATILE_ISLAND_MERGE_MAX_QUARTERS,
    )

    current_classification = classifications[current_quarter]
    classification_name = current_classification.name

    # Count total run length (current quarter = 1)
    run_length = 1
    remaining_quarters = 1  # Track forward quarters separately

    # Scan backwards
    for offset in range(1, 96):
        check_quarter = current_quarter - offset
        if check_quarter < 0:
            break
        if classifications[check_quarter] == current_classification:
            run_length += 1
        else:
            break

    # Scan forwards (count both run_length and remaining_quarters)
    for offset in range(1, 96):
        check_quarter = current_quarter + offset
        if check_quarter >= len(classifications):
            break
        check_class = classifications[check_quarter]

        if check_class == current_classification:
            run_length += 1
            remaining_quarters += 1
        else:
            break

    # Initial volatility check (short run, not PEAK)
    is_brief_run = run_length < PRICE_FORECAST_MIN_DURATION
    is_volatile = is_brief_run and current_classification != QuarterClassification.PEAK

    # Check if favorable period ending soon (for gradual transition)
    # This applies to CHEAP/VERY_CHEAP periods: stop boosting before expensive periods
    is_ending_soon = (
        current_classification in BENEFICIAL_CLASSIFICATIONS
        and remaining_quarters < PRICE_FORECAST_MIN_DURATION
    )

    # PEAK cluster detection - only for volatile EXPENSIVE/NORMAL periods
    in_peak_cluster = False

    if (
        current_classification in [QuarterClassification.EXPENSIVE, QuarterClassification.NORMAL]
        and is_volatile
    ):
        # Check for PEAK before (within min duration window)
        has_peak_before = False
        for back_offset in range(1, PRICE_FORECAST_MIN_DURATION + 1):
            check_quarter = current_quarter - back_offset
            if check_quarter < 0:
                break
            check_class = price_analyzer.get_current_classification(check_quarter)
            if check_class == QuarterClassification.PEAK:
                has_peak_before = True
                break
            if check_class in BENEFICIAL_CLASSIFICATIONS:
                break

        # Check for PEAK after (only if found before)
        has_peak_after = False
        if has_peak_before:
            for fwd_offset in range(1, PRICE_FORECAST_MIN_DURATION + 1):
                check_quarter = current_quarter + fwd_offset
                if check_quarter < 96:
                    check_class = price_analyzer.get_current_classification(check_quarter)
                elif price_data.has_tomorrow:
                    tomorrow_quarter = check_quarter - 96
                    if tomorrow_quarter < 96:
                        check_class = price_analyzer.get_tomorrow_classification(tomorrow_quarter)
                    else:
                        break
                else:
                    break

                if check_class is None:
                    break
                if check_class == QuarterClassification.PEAK:
                    has_peak_after = True
                    break
                if check_class in BENEFICIAL_CLASSIFICATIONS:
                    break

        if has_peak_before and has_peak_after:
            in_peak_cluster = True
            is_volatile = False  # Not volatile when part of PEAK cluster

    # Build reason string for logging
    reason = ""
    if is_volatile:
        run_min = run_length * 15
        min_required = PRICE_FORECAST_MIN_DURATION * 15
        reason = f"Price volatile: {classification_name} {run_min}min<{min_required}min"
    elif is_ending_soon:
        remaining_min = remaining_quarters * 15
        min_required = PRICE_FORECAST_MIN_DURATION * 15
        reason = f"Period ending soon: {classification_name} {remaining_min}min<{min_required}min"

    return VolatileInfo(
        is_volatile=is_volatile,
        run_length=run_length,
        remaining_quarters=remaining_quarters,
        classification_name=classification_name,
        reason=reason,
        in_peak_cluster=in_peak_cluster,
        is_ending_soon=is_ending_soon,
    )


def should_skip_volatile_boost(
    is_volatile: bool, offset: float, is_ending_soon: bool = False
) -> bool:
    """Check if heating boost should be skipped during volatile or ending period.

    Skip positive offsets (heating boosts) when:
    1. Period is volatile (too short to benefit)
    2. Favorable period is ending soon (allow DM recovery before expensive period)

    Args:
        is_volatile: True if in volatile period (short run)
        offset: The calculated offset (positive = heating boost)
        is_ending_soon: True if favorable period ending in < 3 quarters

    Returns:
        True if the boost should be skipped (offset should be set to 0.0)
    """
    return (is_volatile or is_ending_soon) and offset > 0

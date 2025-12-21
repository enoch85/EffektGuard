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
)
from .time_utils import get_current_quarter


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

    # Get current classification
    current_classification = price_analyzer.get_current_classification(current_quarter)
    classification_name = current_classification.name

    # Count total run length (current quarter = 1)
    run_length = 1
    remaining_quarters = 1  # Track forward quarters separately

    # Scan backwards
    for offset in range(1, 96):
        check_quarter = current_quarter - offset
        if check_quarter < 0:
            break
        if price_analyzer.get_current_classification(check_quarter) == current_classification:
            run_length += 1
        else:
            break

    # Scan forwards (count both run_length and remaining_quarters)
    for offset in range(1, 96):
        check_quarter = current_quarter + offset
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

        if check_class == current_classification:
            run_length += 1
            remaining_quarters += 1
        else:
            # Classification changed, count complete
            break

    # Initial volatility check (short run, not PEAK)
    is_brief_run = run_length < PRICE_FORECAST_MIN_DURATION
    is_volatile = is_brief_run and current_classification != QuarterClassification.PEAK

    # Check if CHEAP period is ending soon (v0.4.9 logic)
    # Only CHEAP periods should trigger "ending soon" to allow gradual cooldown before expensive.
    # VERY_CHEAP continues boosting, NORMAL doesn't have special ending logic.
    # This prevents oscillation at CHEAP→NORMAL transitions.
    is_ending_soon = (
        current_classification == QuarterClassification.CHEAP
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

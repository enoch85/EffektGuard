"""Shared volatile price handling helpers.

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
- Volatility scanning always checks the full day (96 quarters), independent of
  any lookahead parameter
"""

from .time_utils import get_current_quarter


def is_volatile_period(
    price_analyzer,
    price_data,
) -> bool:
    """Check if current period is volatile.

    Volatile means the current price run is too brief (< 3 quarters / 45 min)
    to justify heating adjustments. This is based on compressor cycle time:
    - 30 min ramp-up + 15 min cool-down = 45 min minimum effective cycle
    - If price classification changes faster than this, adjustments are wasteful

    The volatility check scans the full day (96 quarters) to count consecutive
    quarters with the same classification, so no lookahead parameter is needed.

    Args:
        price_analyzer: PriceAnalyzer instance with get_price_forecast
        price_data: Current price data (may be None)

    Returns:
        True if in a volatile (brief) period
    """
    if not price_data or not price_data.today:
        return False

    current_quarter = get_current_quarter()
    # Volatility detection scans full day (96 quarters) regardless of lookahead,
    # so we just use the default lookahead_hours value.
    forecast = price_analyzer.get_price_forecast(current_quarter, price_data)
    return forecast.is_volatile


def should_skip_volatile_boost(is_volatile: bool, offset: float) -> bool:
    """Check if heating boost should be skipped during volatile period.

    During volatile periods (brief price runs), skip positive offsets
    (heating boosts) because:
    1. The period may be too brief to benefit from the adjustment
    2. Ramping up/down heating for a short period causes unnecessary wear
    3. Better to maintain stable operation than chase brief price changes

    Args:
        is_volatile: True if in volatile period (from is_volatile_period)
        offset: The calculated offset (positive = heating boost)

    Returns:
        True if the boost should be skipped (offset should be set to 0.0)
    """
    return is_volatile and offset > 0
